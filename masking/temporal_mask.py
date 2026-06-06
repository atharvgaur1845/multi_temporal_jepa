"""Causal past->future split for Temporal JEPA. THIS encodes your novel objective.

Unlike V-JEPA (which masks bidirectionally within a clip), here the context is strictly the
PAST and the target is strictly the FUTURE, separated by a horizon Δ. The model must predict
the latent of a future acquisition from past acquisitions only — a causal, world-model objective.

The single most dangerous bug here is FUTURE LEAKAGE (Common Mistake #5): an off-by-one at the
split boundary that lets a target frame sneak into the context inflates results and is invisible
unless you test for it (tests/test_temporal_mask.py).
"""
from __future__ import annotations

import torch


def split_past_future(dates, pad_mask, horizon, min_context=4, rng=None):
    """Split a (real) acquisition sequence into context (past) and target (future) indices.

    Math/spec
        Let R = the time indices with pad_mask True, in chronological order (dates already
        sorted ascending). We choose a split rank s (0-based, into R) such that the first
        s+1 real frames form the context, and the target is the real frame `horizon` steps
        after s, i.e. R[s + horizon].

        Constraints:
            s + 1 >= min_context          (enough past context)
            s + horizon <= len(R) - 1     (target exists and is real)
        Valid split ranks: s in [min_context - 1, len(R) - 1 - horizon].
        If that range is empty (sequence too short for this horizon) -> ValueError.

        s is drawn uniformly from the valid range (random for training augmentation; pass a
        seeded generator via `rng` for determinism). The target is a SINGLE future frame.

    Returns: (context_t_idx, target_t_idx) as 1D LongTensors of ORIGINAL time indices.

    Causality guarantee: context indices are R[:s+1] and the target is R[s+horizon] with
    horizon >= 1, so every context date < target date. Both are real frames by construction.
    """
    real = torch.nonzero(pad_mask, as_tuple=False).flatten()  # original indices of real frames
    n = real.numel()
    s_lo = min_context - 1
    s_hi = n - 1 - horizon
    if s_hi < s_lo:
        raise ValueError(
            f"sequence too short: {n} real frames, min_context={min_context}, horizon={horizon}"
        )
    if rng is not None:
        s = int(torch.randint(s_lo, s_hi + 1, (1,), generator=rng).item())
    else:
        s = int(torch.randint(s_lo, s_hi + 1, (1,)).item())

    context_t_idx = real[: s + 1]
    target_t_idx = real[s + horizon].unsqueeze(0)
    return context_t_idx.long(), target_t_idx.long()
