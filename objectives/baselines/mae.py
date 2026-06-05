"""MAE baseline (SatMAE-flavored) — masked PIXEL reconstruction.

Contrast with JEPA: MAE reconstructs pixels via a decoder; JEPA predicts latents. Implementing
MAE here gives the "reconstruction objective" comparison point (M4).

Spec: mask a high fraction of patch tokens, encode the visible ones, decode (lightweight
decoder) to reconstruct the masked patches' PIXELS, loss = MSE on masked patches only.

TODO (M4): encoder (reuse models/vit.py) + small decoder + masked-patch MSE.
"""
from __future__ import annotations


def mae_loss(reconstruction, target_pixels, mask):
    """MSE over MASKED patches only. TODO: implement (M4)."""
    raise NotImplementedError("M4")
