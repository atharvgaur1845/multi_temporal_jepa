"""Causal past->future split invariants. THE property that matters: no future leakage."""
import torch

from masking.temporal_mask import split_past_future


def _dates(T=20):
    # strictly increasing day-of-year, irregular gaps
    return torch.tensor(sorted(torch.randint(1, 366, (T,)).tolist()))


def test_no_future_leakage():
    dates = _dates()
    pad = torch.ones(len(dates), dtype=torch.bool)
    ctx, tgt = split_past_future(dates, pad, horizon=1, min_context=4)
    # every context date must be strictly earlier than every target date
    assert dates[ctx].max() < dates[tgt].min(), "a target frame leaked into the context"


def test_min_context_respected():
    dates = _dates()
    pad = torch.ones(len(dates), dtype=torch.bool)
    ctx, _ = split_past_future(dates, pad, horizon=1, min_context=4)
    assert len(ctx) >= 4


def test_targets_are_real_frames():
    dates = _dates(20)
    pad = torch.ones(20, dtype=torch.bool)
    pad[15:] = False  # last 5 are padding
    ctx, tgt = split_past_future(dates, pad, horizon=1, min_context=4)
    assert pad[tgt].all(), "target index points at a padded (non-real) frame"
    assert pad[ctx].all(), "context index points at a padded frame"
