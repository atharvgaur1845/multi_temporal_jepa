"""Raw-feature floor for PASTIS — the 'no learned representation' control.

The validity audit (scripts/audit_claims.py) surfaced that the satellite matrix had **no floor cell
at all**: every finance/C-MAPSS claim is bounded by a random-init and a raw-feature baseline, but the
project's strongest result (temporal JEPA on PASTIS) had an UNMEASURED floor. V1 (resolving power)
is unassessable without one — a benchmark that cannot show learning helped cannot support a
mechanistic claim about why it helped.

`RawPatchEncoder` is the honest floor: it emits, per patch token, the **mean raw spectral bands**
over that patch's pixels. No parameters, no training, no temporal model — just the input at the
token resolution the probe consumes. Anything a learned encoder achieves over this is attributable
to representation learning; anything it does not is not.

It duck-types the `SITSEncoder` surface the probe needs (`encode_full`, `encode_temporal`,
`num_patches`, `embed_dim`, `grid_hw`, `.eval()`), so it drops into `linear_probe_segmentation` and
`parcel_embeddings` unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RawPatchEncoder(nn.Module):
    """Patch-mean-pool of the raw bands. embed_dim == in_chans (10 for PASTIS Sentinel-2)."""

    def __init__(self, img_size=128, patch_size=8, in_chans=10):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = in_chans                      # the "representation" IS the raw bands
        g = img_size // patch_size
        self.grid_hw = (g, g)
        self.num_patches = g * g
        # a parameter-free module still needs a device/dtype anchor for .to(device)
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

    def encode_full(self, x):
        """(B, C, H, W) -> (B, N, C): mean of each patch's pixels, per band."""
        if x.dim() != 4:
            raise ValueError(f"expected (B,C,H,W), got {tuple(x.shape)}")
        pooled = F.avg_pool2d(x, kernel_size=self.patch_size, stride=self.patch_size)  # (B,C,g,g)
        return pooled.flatten(2).transpose(1, 2).contiguous()                          # (B,N,C)

    def encode_temporal(self, data, dates=None, pad_mask=None):
        """(B, T, C, H, W) -> (B, T, N, C). No temporal mixing — this is a floor, by design."""
        B, T = data.shape[:2]
        flat = data.reshape(B * T, *data.shape[2:])
        return self.encode_full(flat).reshape(B, T, self.num_patches, self.embed_dim)

    def encode_subset(self, x, idx):
        raise NotImplementedError("RawPatchEncoder is a floor; spatial masking is not defined")

    def forward(self, x):
        return self.encode_full(x)
