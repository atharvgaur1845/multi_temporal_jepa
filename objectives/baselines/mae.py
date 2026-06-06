"""MAE baseline (SatMAE-flavored) — masked PIXEL reconstruction.

Contrast with JEPA: MAE reconstructs pixels via a decoder; JEPA predicts latents. Implementing
MAE here gives the "reconstruction objective" comparison point (M4).

Spec: mask a high fraction of patch tokens, encode the visible ones, decode (lightweight
decoder) to reconstruct the masked patches' PIXELS, loss = MSE on masked patches only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.pos_embed import build_2d_sincos_pos_embed
from models.vit import Block


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
    return (per_patch * mask.float()).sum() / denom


def patchify(frame, patch_size):
    """(B, C, H, W) -> (B, N, C*P*P) per-patch flattened pixels (row-major patches)."""
    B, C, H, W = frame.shape
    P = patch_size
    x = frame.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // P) * (W // P), C * P * P)
    return x


class MAEModel(nn.Module):
    """Per-frame MAE over a shared SITS backbone.

    Trains `backbone.patch_embed` + `backbone.spatial_vit` (the same modules the linear probe
    reads via the spatial-only pathway), so MAE is a fair "reconstruction objective" comparison.
    A lightweight decoder reconstructs the masked patches' pixels (standard MAE random-shuffle
    masking). The decoder is discarded after pretraining.
    """

    def __init__(self, backbone, patch_size=16, in_chans=10, mask_ratio=0.75,
                 dec_dim=256, dec_depth=4, dec_heads=8):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        D = backbone.embed_dim
        N = backbone.num_patches
        self.dec_embed = nn.Linear(D, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("dec_pos", build_2d_sincos_pos_embed(backbone.grid_hw, dec_dim),
                             persistent=False)
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, in_chans * patch_size * patch_size)
        self.num_patches = N

    def forward(self, frame):
        """frame (B, C, H, W) -> scalar MAE loss on masked patches."""
        B = frame.shape[0]
        N = self.num_patches
        n_keep = int(round((1 - self.mask_ratio) * N))

        tok = self.backbone.patch_embed(frame) + self.backbone.spatial_pos.unsqueeze(0)  # (B,N,D)
        noise = torch.rand(B, N, device=frame.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]

        x_vis = torch.gather(tok, 1, ids_keep.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        enc_vis = self.backbone.spatial_vit(x_vis, pos_embed=None)                # (B, n_keep, D)

        # decoder: place encoded visible + mask tokens, unshuffle, add pos, decode
        x = self.dec_embed(enc_vis)                                               # (B, n_keep, dec)
        mask_tokens = self.mask_token.expand(B, N - n_keep, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1,
                              ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        x_full = x_full + self.dec_pos.unsqueeze(0)
        for blk in self.dec_blocks:
            x_full = blk(x_full)
        recon = self.dec_pred(self.dec_norm(x_full))                             # (B, N, C*P*P)

        target = patchify(frame, self.patch_size)
        mask = torch.ones(B, N, device=frame.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)                                        # True = masked
        return mae_loss(recon, target, mask)
