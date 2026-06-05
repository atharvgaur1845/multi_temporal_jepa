"""I-JEPA multi-block masking sampler (spatial). Used by Spatial JEPA (Baseline 1) and,
optionally, within the future frame for Temporal JEPA.

Paper spec (Assran et al., CVPR 2023)
    target blocks : M = 4, scale (0.15, 0.20) of the grid, aspect ratio (0.75, 1.5)
    context block : 1 block, scale (0.85, 1.0), aspect ratio 1.0
    overlap removal: sample the 4 targets FIRST, then drop from the context every token that
                     overlaps ANY target -> context and target index sets are DISJOINT.

Disjointness is the whole point: if context and target overlap, the predictor can copy the
answer and the loss collapses to ~0 while learning nothing (Common Mistake #3). tests/test_masking.py
asserts disjointness.
"""
from __future__ import annotations


def sample_block(grid_hw, scale_range, ar_range, rng):
    """Sample one rectangular block on an (H', W') token grid.

    Math/spec
        area = U(scale_range) * H' * W'
        aspect = U(ar_range);  h = round(sqrt(area/aspect)); w = round(sqrt(area*aspect))
        clamp h,w to the grid; pick a random top-left so the block fits.
    Returns: set/tensor of flat token indices covered by the block.
    TODO: implement.
    """
    raise NotImplementedError("M1")


def sample_multiblock_mask(grid_hw, n_targets=4,
                           target_scale=(0.15, 0.20), target_ar=(0.75, 1.5),
                           context_scale=(0.85, 1.0), rng=None):
    """Return (context_idx, [target_idx_0, ..., target_idx_{M-1}]).

    Steps
        1. sample M target blocks (may overlap each other — that's allowed).
        2. sample 1 context block.
        3. context_idx <- context_block_indices MINUS union(all target indices).

    Invariants (unit-tested)
        - context_idx is disjoint from every target_idx.
        - each target_idx is non-empty.
    TODO: implement using sample_block.
    """
    raise NotImplementedError("M1")
