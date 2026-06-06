"""Spatial ViT encoder — a standard transformer block stack over patch tokens.

Used as the per-frame spatial encoder. Nothing satellite-specific here; this is the piece
you most likely can implement from memory. The attention block also accepts a key-padding mask
so the SAME Block can be reused over padded temporal tokens (see models/temporal_encoder.py).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head self-attention.

    q,k,v = Linear(x) split into H heads; attn = softmax(qk^T / sqrt(d_head)); out = attn·v.
    Uses F.scaled_dot_product_attention (fast, fused). An optional key_padding_mask (B, N) with
    True = real token masks out padded keys.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, head_dim)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, N) True=real -> (B, 1, 1, N) additive mask: keep True, -inf on padded keys.
            attn_mask = key_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Patch tokens (+ spatial pos) -> transformer blocks -> final LayerNorm.

    The final LayerNorm matters: it is the representation the EMA target branch emits and that
    the JEPA loss is computed against (objectives/jepa_loss.py).

    forward(tokens, pos_embed=None, key_padding_mask=None) -> (..., N, D)
        pos_embed : (N, D) or (B, N, D) added to tokens if provided.
    """

    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, pos_embed=None, key_padding_mask=None):
        x = tokens
        if pos_embed is not None:
            x = x + pos_embed.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)
