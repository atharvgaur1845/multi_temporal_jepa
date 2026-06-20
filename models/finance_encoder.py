"""PanelEncoder — the finance analogue of models/jepa.SITSEncoder (factorized assets->time).

Identical factorized structure as the satellite encoder, with the two image-specific pieces swapped
for their tabular counterparts:

    PASTIS                                   finance
    PatchEmbed: Conv2d(10 -> D, P x P)   ->  FrameEmbed: Linear(F_features -> D) per asset-day
    2D sin/cos spatial position          ->  LEARNED per-asset position (assets aren't ordered)
    ViTEncoder over pixel patches        ->  ViTEncoder over the N assets of one day  (REUSED)
    TemporalEncoder over acquisitions    ->  TemporalEncoder over trading days        (REUSED)

So a day's cross-section of N assets is encoded by cross-asset attention (the "spatial" ViT), then
each asset's token is integrated across days by the temporal transformer — exactly the SITS path.
The shared transformer stacks (models/vit.py, models/temporal_encoder.py) are reused unchanged.

The attribute named `spatial_pos` (the learned asset positions) mirrors SITSEncoder.spatial_pos so
the JEPA forward code that builds predictor target positions transfers without edits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .vit import ViTEncoder


class PanelEncoder(nn.Module):
    """Per-day cross-asset ViT + across-day temporal transformer over a financial panel.

    Used as BOTH the trainable context encoder and (deep-copied, frozen) the EMA target encoder,
    and as the shared backbone for the MAE/BYOL/SimCLR baselines — same as SITSEncoder.

    Shapes: frame (B, N, F) ; sequence (B, W, N, F) with N = num_assets, F = num_features.
    """

    def __init__(self, num_assets=9, num_features=4, embed_dim=128, depth=4, num_heads=4,
                 temporal_depth=4, mlp_ratio=4.0, grad_checkpoint=False):
        super().__init__()
        self.num_assets = num_assets
        self.num_patches = num_assets                      # name parity with SITSEncoder
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.frame_embed = nn.Linear(num_features, embed_dim)
        # learned per-asset positional embedding (assets have identity but no spatial order).
        self.spatial_pos = nn.Parameter(torch.zeros(num_assets, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.spatial_vit = ViTEncoder(embed_dim, depth, num_heads, mlp_ratio, grad_checkpoint)
        self.temporal_enc = TemporalEncoder(embed_dim, temporal_depth, num_heads, mlp_ratio,
                                            grad_checkpoint)

    # ---- per-frame (single day) paths ----
    def encode_full(self, frame):
        """frame (B, N, F) -> day cross-section tokens (B, N, D)."""
        tok = self.frame_embed(frame)                                  # (B, N, D)
        return self.spatial_vit(tok, pos_embed=self.spatial_pos)

    def encode_subset(self, frame, idx):
        """Encode only the assets in `idx` (Spatial-JEPA context). Returns (B, |idx|, D)."""
        tok = self.frame_embed(frame) + self.spatial_pos.unsqueeze(0)  # pos added pre-select
        tok = tok[:, idx]
        return self.spatial_vit(tok, pos_embed=None)

    # ---- spatiotemporal (window) path ----
    def encode_temporal(self, data, dates, pad_mask):
        """data (B, W, N, F) -> time-aware tokens (B, W, N, D)."""
        B, W, N, F = data.shape
        flat = data.reshape(B * W, N, F)
        tok = self.frame_embed(flat)                                   # (B*W, N, D)
        tok = self.spatial_vit(tok, pos_embed=self.spatial_pos)        # (B*W, N, D)
        tok = tok.reshape(B, W, N, self.embed_dim)
        return self.temporal_enc(tok, dates, pad_mask)                 # (B, W, N, D)
