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
    """Conv2d(C -> D, kernel=P, stride=P), then flatten spatial grid to a token sequence.

    Parameters: img_size, patch_size, in_chans (=10), embed_dim.

    TODO
    ----
    - Build the Conv2d projection.
    - forward(x): (B*T, C, H, W) -> (B*T, N, D); flatten + transpose so tokens are dim 1.
    - Expose self.num_patches and self.grid_hw = (H/P, W/P) for the positional encodings
      and the masking sampler.
    """

    def __init__(self, img_size=128, patch_size=16, in_chans=10, embed_dim=256):
        super().__init__()
        # TODO: self.proj = nn.Conv2d(...); self.grid_hw = (...); self.num_patches = ...
        raise NotImplementedError("M1")

    def forward(self, x):
        raise NotImplementedError("M1")
