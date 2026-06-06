"""End-to-end wiring tests on a tiny SYNTHETIC PASTIS-shaped batch — no download needed.

These cover the parts the data-dependent tests can't reach in CI:
    - JEPA spatial + temporal forward produce shape-aligned (pred, target).
    - a backward step touches the context encoder + predictor but NOT the target encoder.
    - the target encoder stays frozen (requires_grad False) and predictor is the bottleneck.
"""
import torch

from engine.ema import ema_update
from models.jepa import JEPA
from objectives.jepa_loss import jepa_latent_loss


def _batch(B=2, T=10, C=10, H=128, W=128):
    g = torch.Generator().manual_seed(0)
    data = torch.randn(B, T, C, H, W, generator=g)
    dates = torch.stack([torch.sort(torch.randint(1, 366, (T,), generator=g)).values
                         for _ in range(B)])
    pad = torch.ones(B, T, dtype=torch.bool)
    return {"data": data, "dates": dates, "pad_mask": pad, "label": None}


def _small(objective):
    return JEPA(objective=objective, embed_dim=64, depth=1, num_heads=4, temporal_depth=1,
                pred_dim=48, pred_depth=1, pred_heads=4, horizon=1, min_context=4)


def test_target_encoder_frozen():
    m = _small("temporal_jepa")
    assert all(not p.requires_grad for p in m.target_encoder.parameters())
    assert m.predictor.pred_dim < m.embed_dim, "predictor must be the narrow bottleneck"


def test_temporal_forward_shapes_and_grad():
    m = _small("temporal_jepa")
    pred, target = m(_batch())
    assert pred.shape == target.shape
    loss = jepa_latent_loss(pred, target)
    loss.backward()
    # context encoder + predictor get gradient; target encoder gets none.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.predictor.parameters())
    assert all(p.grad is None for p in m.target_encoder.parameters())


def test_spatial_forward_shapes():
    m = _small("spatial_jepa")
    pred, target = m(_batch())
    assert pred.shape == target.shape and pred.dim() == 3


def test_ema_moves_target_toward_context():
    m = _small("temporal_jepa")
    # perturb context so it differs from the (deep-copied) target
    with torch.no_grad():
        for p in m.context_encoder.parameters():
            p.add_(torch.randn_like(p))
    before = [p.clone() for p in m.target_encoder.parameters()]
    ema_update(m.context_encoder, m.target_encoder, momentum=0.5)
    moved = any(not torch.allclose(b, p) for b, p in zip(before, m.target_encoder.parameters()))
    assert moved, "EMA update did not move the target encoder"
