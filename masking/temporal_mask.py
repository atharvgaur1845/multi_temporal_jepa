"""Causal past->future split for Temporal JEPA. THIS encodes your novel objective.

Unlike V-JEPA (which masks bidirectionally within a clip), here the context is strictly the
PAST and the target is strictly the FUTURE, separated by a horizon Δ. The model must predict
the latent of a future acquisition from past acquisitions only — a causal, world-model objective.

The single most dangerous bug here is FUTURE LEAKAGE (Common Mistake #5): an off-by-one at the
split boundary that lets a target frame sneak into the context inflates results and is invisible
unless you test for it (tests/test_temporal_mask.py).
"""
from __future__ import annotations


def split_past_future(dates, pad_mask, horizon, min_context=4, rng=None):
    """Split a (real) acquisition sequence into context (past) and target (future) indices.

    Math/spec
        Consider only REAL frames (pad_mask True), in chronological order by `dates`.
        Choose a split position s such that there are >= min_context frames at indices <= s.
        context_t_idx = real frames with position <= s
        target_t_idx  = the frame(s) `horizon` acquisition-steps after s
                        (Δ is in acquisition steps; you may also expose the DOY gap).
        STRICT: no target index may appear in context (no leakage), and every target must be
        a real (non-pad) future frame; otherwise resample or skip the sample.

    Returns: (context_t_idx, target_t_idx) as 1D LongTensors of time indices.

    TODO: implement; make the causality property explicit and testable.
    """
    raise NotImplementedError("M2")
