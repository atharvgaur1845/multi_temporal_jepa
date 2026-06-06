"""Factorized temporal transformer — aggregates per-frame tokens across time.

Design choice (see plan §4.4): we factorize space then time instead of full 3D attention.
With ~64 spatial tokens/frame and up to 61 frames, full T·N self-attention (~3900 tokens) is
O(N²) heavy on one GPU. Factorization keeps it tractable:

    spatial ViT (per frame, shared)  ->  tokens  (B, T, N, D)
    temporal transformer (per spatial location, across T)  ->  time-aware tokens (B, T, N, D)

The temporal attention MUST honor the padding mask (variable-length series) and use the DOY
temporal positional encoding (models/pos_embed.doy_sincos_pos_embed).
"""
from __future__ import annotations

import torch.nn as nn

from .pos_embed import doy_sincos_pos_embed
from .vit import Block


class TemporalEncoder(nn.Module):
    """Transformer over the time axis, applied per spatial token position.

    forward(tokens, dates, pad_mask) where
        tokens   : (B, T, N, D)
        dates    : (B, T)        for DOY temporal pos
        pad_mask : (B, T)        True = real frame

    Returns time-aware tokens (B, T, N, D). The DOY embedding is added along T (broadcast over
    the N spatial positions); attention runs over T with the per-frame pad_mask so padded
    frames are ignored. We fold N into the batch so each spatial location attends across time.
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, dates, pad_mask=None):
        B, T, N, D = tokens.shape
        # DOY temporal positional encoding, added across the time axis (broadcast over N).
        temp_pos = doy_sincos_pos_embed(dates, D, pad_mask=pad_mask)  # (B, T, D)
        x = tokens + temp_pos.unsqueeze(2)  # (B, T, N, D)

        # fold N into batch so attention is over T: (B, T, N, D) -> (B*N, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        kpm = None
        if pad_mask is not None:
            kpm = pad_mask.unsqueeze(1).expand(B, N, T).reshape(B * N, T)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        x = self.norm(x)
        # back to (B, T, N, D)
        return x.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()
