"""GraphSITSEncoder — the satellite encoder with a GNN spatial backbone (Part 6 #8).

Identical to models.jepa.SITSEncoder (per-frame PatchEmbed + 2D pos, then a factorized spatial→time
stack) except the per-frame SPATIAL encoder is a grid-graph GNN (local message passing) instead of the
global-attention spatial ViT. Same interface (encode_full / encode_temporal / spatial_pos / grid_hw /
num_patches / embed_dim), so it is a drop-in for the JEPA temporal objective; the temporal transformer
and everything downstream are reused unchanged.

Graph message passing over the full grid would leak target patches into the context under I-JEPA
spatial masking, so this backbone is TEMPORAL-objective only (encode_subset raises) — which is exactly
the Graph *Temporal* JEPA we want to test on PASTIS.
"""
from __future__ import annotations

import torch.nn as nn

from .graph_layers import GridGraphEncoder
from .patch_embed import PatchEmbed
from .pos_embed import build_2d_sincos_pos_embed
from .temporal_encoder import TemporalEncoder


class GraphSITSEncoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=10, embed_dim=256, depth=6, num_heads=8,
                 temporal_depth=4, mlp_ratio=4.0, grad_checkpoint=False, connectivity=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_hw = self.patch_embed.grid_hw
        self.num_patches = self.patch_embed.num_patches
        pos = build_2d_sincos_pos_embed(self.grid_hw, embed_dim)                 # (N, D)
        self.register_buffer("spatial_pos", pos, persistent=False)
        # GNN spatial backbone (8-connectivity grid graph by default) in place of the spatial ViT
        self.spatial_gnn = GridGraphEncoder(self.grid_hw, embed_dim, depth, num_heads, mlp_ratio,
                                            grad_checkpoint, connectivity=connectivity)
        self.temporal_enc = TemporalEncoder(embed_dim, temporal_depth, num_heads, mlp_ratio,
                                            grad_checkpoint)
        self.embed_dim = embed_dim

    def encode_full(self, frame):
        """frame (B, C, H, W) -> graph-encoded patch tokens (B, N, D)."""
        tok = self.patch_embed(frame)                                            # (B, N, D)
        return self.spatial_gnn(tok, pos_embed=self.spatial_pos)

    def encode_subset(self, frame, idx):
        raise NotImplementedError(
            "GraphSITSEncoder is temporal-objective only: message passing over the full grid graph "
            "would leak masked target patches into the context (spatial JEPA needs disjoint sets).")

    def encode_temporal(self, data, dates, pad_mask):
        """data (B, F, C, H, W) -> time-aware graph tokens (B, F, N, D)."""
        B, F, C, H, W = data.shape
        flat = data.reshape(B * F, C, H, W)
        tok = self.patch_embed(flat)                                             # (B*F, N, D)
        tok = self.spatial_gnn(tok, pos_embed=self.spatial_pos)                  # (B*F, N, D)
        tok = tok.reshape(B, F, self.num_patches, self.embed_dim)
        return self.temporal_enc(tok, dates, pad_mask)                           # (B, F, N, D)
