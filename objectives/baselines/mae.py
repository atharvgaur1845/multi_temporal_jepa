"""MAE baseline (SatMAE-flavored) — masked PIXEL reconstruction.

Contrast with JEPA: MAE reconstructs pixels via a decoder; JEPA predicts latents. Implementing
MAE here gives the "reconstruction objective" comparison point (M4).

Spec: mask a high fraction of patch tokens, encode the visible ones, decode (lightweight
decoder) to reconstruct the masked patches' PIXELS, loss = MSE on masked patches only.
"""
from __future__ import annotations

import torch


def random_patch_mask(num_patches, mask_ratio, batch_size, device=None, generator=None):
    """Per-sample random mask. Returns a bool tensor (B, N) with True = MASKED (to reconstruct)."""
    n_mask = int(round(mask_ratio * num_patches))
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for b in range(batch_size):
        idx = torch.randperm(num_patches, generator=generator, device=device)[:n_mask]
        mask[b, idx] = True
    return mask


def mae_loss(reconstruction, target_pixels, mask):
    """MSE over MASKED patches only.

    reconstruction, target_pixels : (B, N, patch_pixels)   per-patch flattened pixels
    mask : (B, N) bool, True where the patch was masked (and must be reconstructed).
    """
    per_patch = ((reconstruction - target_pixels) ** 2).mean(dim=-1)  # (B, N)
    denom = mask.sum().clamp_min(1)
    return (per_patch * mask).sum() / denom
