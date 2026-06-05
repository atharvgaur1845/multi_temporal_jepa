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


class TemporalEncoder(nn.Module):
    """Transformer over the time axis, applied per spatial token position.

    forward(tokens, dates, pad_mask) where
        tokens   : (B, T, N, D)
        dates    : (B, T)        for DOY temporal pos
        pad_mask : (B, T)        True = real frame

    Returns time-aware tokens (B, T, N, D) (or a pooled (B, N, D) if you aggregate context).

    Implementation notes / TODO
        - Add DOY temporal positional encoding to tokens along T.
        - Reshape so attention runs over the T axis (e.g. fold N into batch).
        - Pass pad_mask as key_padding_mask so pad frames are ignored.
        - Decide how the CONTEXT path summarizes past frames for the predictor
          (e.g. take the tokens of the most recent context frame, or a learned query).
          Document your choice — it defines what "context representation" means.
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        raise NotImplementedError("M2")

    def forward(self, tokens, dates, pad_mask=None):
        raise NotImplementedError("M2")
