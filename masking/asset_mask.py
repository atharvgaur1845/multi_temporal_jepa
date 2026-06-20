"""Cross-sectional (asset) masking for Spatial-JEPA on a financial panel.

The finance analogue of masking/multiblock.py. PASTIS Spatial-JEPA masks *blocks of pixels* of one
frame and predicts them from a visible context block; here we mask a *subset of the day's assets*
and predict their latents from the visible assets. With only N≈9 assets there is no 2-D block
geometry, so we draw a random disjoint context/target split: this is the literal "predict masked
cross-section members from the visible ones" task — item-1 of the I-JEPA->Temporal contrast,
isolating the value of the temporal objective.
"""
from __future__ import annotations

import torch


def sample_asset_mask(num_assets, n_targets=None, generator=None):
    """Return (ctx_idx, tgt_idx): a random disjoint split of asset indices.

    n_targets defaults to floor(num_assets/2). ctx and tgt are disjoint and cover all assets, so
    context ∩ target = ∅ (no trivial copy), mirroring the disjointness invariant of multiblock.
    """
    if n_targets is None:
        n_targets = max(1, num_assets // 2)
    n_targets = min(n_targets, num_assets - 1)                 # keep >=1 context asset
    perm = torch.randperm(num_assets, generator=generator)
    tgt_idx = perm[:n_targets].sort().values
    ctx_idx = perm[n_targets:].sort().values
    return ctx_idx, tgt_idx
