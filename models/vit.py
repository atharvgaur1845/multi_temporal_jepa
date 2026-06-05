"""Spatial ViT encoder — a standard transformer block stack over patch tokens.

Used as the per-frame spatial encoder. Nothing satellite-specific here; this is the piece
you most likely *can* implement from memory. It is included so the rest of the architecture
has a concrete dependency, but the attention/MLP math is still yours to write.
"""
from __future__ import annotations

import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention.

    Math: q,k,v = Linear(x) split into H heads; attn = softmax(qk^T / sqrt(d_head)); out = attn·v.
    Support an optional key-padding mask (needed when reused over padded temporal tokens).
    TODO: implement."""

    def __init__(self, dim, num_heads):
        super().__init__()
        raise NotImplementedError("M1")

    def forward(self, x, key_padding_mask=None):
        raise NotImplementedError("M1")


class Block(nn.Module):
    """Pre-norm transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x)).
    TODO: implement (LayerNorm, Attention, MLP with GELU, residuals)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        raise NotImplementedError("M1")

    def forward(self, x, key_padding_mask=None):
        raise NotImplementedError("M1")


class ViTEncoder(nn.Module):
    """Patch tokens (+ spatial pos) -> transformer blocks -> final LayerNorm.

    The final LayerNorm matters: it is the representation that the EMA target branch emits
    and that the JEPA loss is computed against (see objectives/jepa_loss.py).

    Parameters: embed_dim, depth, num_heads, mlp_ratio.
    forward(tokens, pos_embed, key_padding_mask=None) -> (..., N, D)

    TODO: stack `depth` Blocks; add pos_embed; apply final norm.
    """

    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        raise NotImplementedError("M1")

    def forward(self, tokens, pos_embed=None, key_padding_mask=None):
        raise NotImplementedError("M1")
