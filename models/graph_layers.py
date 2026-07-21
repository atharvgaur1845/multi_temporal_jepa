"""Graph message-passing layers for Graph Temporal JEPA (Part 6 #8).

The spatial ViT treats a frame's patches as an unordered set mixed by GLOBAL attention. For satellite
image time series the patches actually live on a grid with strong LOCAL structure (a parcel is a
contiguous blob), so a graph neural network with LOCAL message passing over the patch-grid graph is a
different (and arguably better-matched) spatial inductive bias — and it generalizes to an arbitrary
*parcel adjacency* graph, not just the grid.

Dependency-light (no torch_geometric): a graph is a precomputed `edge_index` (2, E) of (src, dst)
pairs; a GraphSAGE-style block aggregates neighbour features by scatter-mean and mixes them with a
residual MLP. `grid_edge_index` builds the 4/8-connectivity grid graph (+ self-loops).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def grid_edge_index(H, W, connectivity=4):
    """(2, E) long edge index (src, dst) for an H×W grid, undirected (both directions) + self-loops."""
    idx = np.arange(H * W).reshape(H, W)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    edges = []
    for i in range(H):
        for j in range(W):
            n = int(idx[i, j])
            edges.append((n, n))                                    # self-loop
            for di, dj in nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    edges.append((int(idx[ni, nj]), n))             # neighbour -> node (src, dst)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()   # (2, E)


def scatter_mean(src, index, dim_size):
    """Mean-aggregate edge messages to their destination nodes.
    src (B, E, D), index (E,) dst node per edge -> (B, dim_size, D)."""
    B, E, D = src.shape
    out = src.new_zeros(B, dim_size, D)
    out.scatter_add_(1, index.view(1, E, 1).expand(B, E, D), src)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index, torch.ones(E, device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0).view(1, dim_size, 1)


class GraphSAGEBlock(nn.Module):
    """Pre-norm GraphSAGE-mean block: h' = h + W_self·h + W_neigh·mean_{j∈N(i)} h_j ; then + MLP.
    Same residual structure as the ViT Block, so it drops into the encoder stack cleanly."""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, edge_index):
        h = self.norm1(x)                                           # (B, N, D)
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_mean(h[:, src], dst, x.shape[1])             # (B, N, D) neighbour aggregate
        x = x + self.lin_self(h) + self.lin_neigh(agg)             # message-passing residual
        x = x + self.mlp(self.norm2(x))
        return x


class GridGraphEncoder(nn.Module):
    """Stack of GraphSAGE blocks over a fixed grid graph (+ final LayerNorm) — a drop-in replacement
    for `models.vit.ViTEncoder` (same forward(tokens, pos_embed, key_padding_mask) signature; the
    key_padding_mask is unused — the graph is fixed)."""

    def __init__(self, grid_hw, embed_dim, depth, num_heads=8, mlp_ratio=4.0, grad_checkpoint=False,
                 connectivity=4):
        super().__init__()
        self.register_buffer("edge_index", grid_edge_index(grid_hw[0], grid_hw[1], connectivity),
                             persistent=False)
        self.blocks = nn.ModuleList([GraphSAGEBlock(embed_dim, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, pos_embed=None, key_padding_mask=None):
        x = tokens if pos_embed is None else tokens + pos_embed.to(tokens.dtype)
        for blk in self.blocks:
            x = blk(x, self.edge_index)
        return self.norm(x)
