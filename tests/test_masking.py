"""Spatial multi-block masking invariants. THE property that matters: context ∩ target = ∅.

These fail with NotImplementedError until you implement masking/multiblock.py — that's TDD.
"""
import torch

from masking.multiblock import sample_multiblock_mask


def _as_set(idx):
    return set(torch.as_tensor(idx).flatten().tolist())


def test_context_target_disjoint(grid_hw):
    """No token may be in both context and any target (else trivial-copy task)."""
    ctx, targets = sample_multiblock_mask(grid_hw)
    ctx_set = _as_set(ctx)
    for t in targets:
        assert ctx_set.isdisjoint(_as_set(t)), "context overlaps a target block — copy leak!"


def test_four_nonempty_targets(grid_hw):
    ctx, targets = sample_multiblock_mask(grid_hw, n_targets=4)
    assert len(targets) == 4
    for t in targets:
        assert len(_as_set(t)) >= 1, "a target block is empty"


def test_indices_within_grid(grid_hw):
    n = grid_hw[0] * grid_hw[1]
    ctx, targets = sample_multiblock_mask(grid_hw)
    all_idx = _as_set(ctx).union(*(_as_set(t) for t in targets))
    assert all(0 <= i < n for i in all_idx), "token index out of grid range"
