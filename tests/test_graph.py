"""Graph Temporal JEPA (Part 6 #8) — GNN layers + graph spatial backbone.

Runs on synthetic (B,T,C,H,W) tensors, no PASTIS needed (mirrors tests/test_model_synthetic). Guards
grid-graph construction, message passing, the GraphSITSEncoder interface, JEPA grad routing with the
graph backbone, and that the default ViT path is untouched.
"""
import pytest
import torch

from models.graph_layers import grid_edge_index, scatter_mean, GridGraphEncoder
from models.graph_encoder import GraphSITSEncoder
from models.jepa import JEPA, SITSEncoder, build_model
from objectives.jepa_loss import jepa_latent_loss


def test_grid_edge_index():
    ei = grid_edge_index(2, 2, connectivity=4)                  # 2×2 grid
    assert ei.shape[0] == 2
    # every node has a self-loop; corners in a 2×2 grid have 2 grid neighbours + self = 3 in-edges
    dst = ei[1]
    for n in range(4):
        assert (dst == n).sum().item() == 3
    assert (ei[0] == ei[1]).sum().item() == 4                   # exactly 4 self-loops


def test_scatter_mean():
    src = torch.tensor([[[1.0], [3.0], [5.0]]])                 # (1, E=3, D=1)
    index = torch.tensor([0, 0, 1])                             # edges -> nodes {0,0,1}
    out = scatter_mean(src, index, dim_size=2)
    assert torch.allclose(out[0, 0], torch.tensor([2.0]))       # mean(1,3)
    assert torch.allclose(out[0, 1], torch.tensor([5.0]))


def test_grid_graph_encoder_shape():
    enc = GridGraphEncoder((4, 4), embed_dim=16, depth=2)
    x = torch.randn(3, 16, 16)                                  # (B, N=16, D)
    assert enc(x).shape == (3, 16, 16)


def _sat_batch(B=2, T=8, C=10, H=32):
    g = torch.Generator().manual_seed(0)
    return {"data": torch.randn(B, T, C, H, H, generator=g),
            "dates": torch.randint(1, 366, (B, T), generator=g),
            "pad_mask": torch.ones(B, T, dtype=torch.bool)}


def test_graph_encoder_interface_matches_sits():
    ge = GraphSITSEncoder(img_size=32, patch_size=16, in_chans=10, embed_dim=32, depth=2,
                          num_heads=4, temporal_depth=2)
    b = _sat_batch()
    assert ge.encode_full(b["data"][:, 0]).shape == (2, ge.num_patches, 32)
    assert ge.encode_temporal(b["data"], b["dates"], b["pad_mask"]).shape == (2, 8, ge.num_patches, 32)
    with pytest.raises(NotImplementedError):
        ge.encode_subset(b["data"][:, 0], torch.arange(2))      # spatial masking unsupported


def test_jepa_graph_forward_and_grad_routing():
    m = JEPA(objective="temporal_jepa", img_size=32, patch_size=16, in_chans=10, embed_dim=32,
             depth=2, num_heads=4, temporal_depth=2, pred_dim=16, pred_depth=2, pred_heads=4,
             horizon=1, min_context=3, spatial_backbone="graph")
    pred, target, ctx = m(_sat_batch())
    assert pred.shape == target.shape
    jepa_latent_loss(pred, target).backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())       # EMA target frozen
    assert any(p.grad is not None for p in m.context_encoder.parameters())  # graph encoder trains
    assert isinstance(m.context_encoder, GraphSITSEncoder)


def test_graph_rejects_spatial_jepa():
    with pytest.raises(AssertionError):
        JEPA(objective="spatial_jepa", img_size=32, patch_size=16, embed_dim=32, pred_dim=16,
             spatial_backbone="graph")


def test_build_model_graph_vs_vit():
    base = {"objective": "temporal_jepa",
            "encoder": {"patch_size": 16, "embed_dim": 32, "depth": 2, "num_heads": 4,
                        "temporal_depth": 2},
            "predictor": {"embed_dim": 16, "depth": 2, "num_heads": 4},
            "temporal": {"horizon": 1, "min_context": 3}}
    assert isinstance(build_model(base).context_encoder, SITSEncoder)       # default ViT unchanged
    base["encoder"]["spatial_backbone"] = "graph"
    assert isinstance(build_model(base).context_encoder, GraphSITSEncoder)
