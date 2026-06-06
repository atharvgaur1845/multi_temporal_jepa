"""I-JEPA multi-block masking sampler (spatial). Used by Spatial JEPA (Baseline 1) and,
optionally, within the future frame for Temporal JEPA.

Paper spec (Assran et al., CVPR 2023)
    target blocks : M = 4, scale (0.15, 0.20) of the grid, aspect ratio (0.75, 1.5)
    context block : 1 block, scale (0.85, 1.0), aspect ratio 1.0
    overlap removal: sample the 4 targets FIRST, then drop from the context every token that
                     overlaps ANY target -> context and target index sets are DISJOINT.

Disjointness is the whole point: if context and target overlap, the predictor can copy the
answer and the loss collapses to ~0 while learning nothing (Common Mistake #3).
tests/test_masking.py asserts disjointness.

Index convention: tokens are flattened row-major, flat = row * W' + col.
"""
from __future__ import annotations

import torch


def _u(rng, lo, hi):
    """Uniform scalar in [lo, hi) using a torch.Generator (or default RNG if None)."""
    r = torch.rand(1, generator=rng).item()
    return lo + r * (hi - lo)


def sample_block(grid_hw, scale_range, ar_range, rng):
    """Sample one rectangular block on an (H', W') token grid.

    Math/spec
        area  = U(scale_range) * H' * W'
        aspect= U(ar_range);  h = round(sqrt(area/aspect)); w = round(sqrt(area*aspect))
        clamp h,w to [1, grid]; pick a random valid top-left so the block fits.
    Returns: 1D LongTensor of flat token indices covered by the block.
    """
    H, W = grid_hw
    area = _u(rng, *scale_range) * H * W
    aspect = _u(rng, *ar_range)
    h = int(round((area / aspect) ** 0.5))
    w = int(round((area * aspect) ** 0.5))
    h = max(1, min(H, h))
    w = max(1, min(W, w))
    top = int(_u(rng, 0, H - h + 1))
    left = int(_u(rng, 0, W - w + 1))
    rows = torch.arange(top, top + h)
    cols = torch.arange(left, left + w)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return (rr * W + cc).flatten()


def sample_multiblock_mask(grid_hw, n_targets=4,
                           target_scale=(0.15, 0.20), target_ar=(0.75, 1.5),
                           context_scale=(0.85, 1.0), rng=None):
    """Return (context_idx, [target_idx_0, ..., target_idx_{M-1}]).

    Steps
        1. sample M target blocks (may overlap each other — allowed).
        2. sample 1 context block (square-ish, large).
        3. context_idx <- context_block_indices MINUS union(all target indices).

    Invariants (unit-tested in tests/test_masking.py)
        - context_idx is disjoint from every target_idx.
        - each target_idx is non-empty (resampled if a target came back empty).
    """
    targets = []
    target_union = set()
    for _ in range(n_targets):
        # resample defensively so no target is empty (can only happen at tiny grids)
        for _try in range(10):
            blk = sample_block(grid_hw, target_scale, target_ar, rng)
            if blk.numel() >= 1:
                break
        targets.append(blk)
        target_union.update(blk.tolist())

    context = sample_block(grid_hw, context_scale, (1.0, 1.0), rng)
    keep = [i for i in context.tolist() if i not in target_union]
    context_idx = torch.tensor(sorted(keep), dtype=torch.long)
    return context_idx, targets
