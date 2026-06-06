"""Per-frame patch embedding for SITS.

Each acquisition (C=10, H=128, W=128) is split into non-overlapping P×P patches and linearly
projected to `embed_dim`. A Conv2d with kernel=stride=P does the patchify + projection in one op.

Shapes
    in : (B*T, C, H, W)
    out: (B*T, N, D)   where N = (H/P)*(W/P)   (P=16 -> N=64)

Note the 10-channel input (not 3): your Conv2d in_channels must be C, not 3.
"""
from __future__ import annotations

import torch.nn as nn


class PatchEmbed(nn.Module):
    """Conv2d(C -> D, kernel=P, stride=P), then flatten spatial grid to a token sequence."""

    def __init__(self, img_size=128, patch_size=16, in_chans=10, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_hw = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_hw[0] * self.grid_hw[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B*T, C, H, W) -> (B*T, D, H', W') -> (B*T, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
