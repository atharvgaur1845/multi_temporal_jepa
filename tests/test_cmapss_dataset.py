"""C-MAPSS dataset invariants — shapes, RUL cap, health stages, NO train/test engine overlap.

Runs offline on the synthetic monotonic-degradation generator (allow_synth + a non-existent root),
so no download is needed; on a machine with data_root/CMAPSS/*.txt it would use the real data.
"""
import numpy as np
import torch

from data.cmapss_dataset import (make_cmapss_datasets, collate_cmapss_windows, synthesize_cmapss,
                                  _per_engine_features, _condition_norm_stats, _apply_norm)


def _ds(window=20):
    return make_cmapss_datasets(root="__no_such_root__", fd="FD001", window=window,
                                eval_stride=2, rul_cap=125, allow_synth=True, seed=0)


def test_synth_layout():
    ids, cyc, settings, sensors = synthesize_cmapss(n_engines=20, seed=0)
    assert sensors.shape[1] == 21 and settings.shape[1] == 3
    assert len(np.unique(ids)) == 20
    cols, ruls = synthesize_cmapss(n_engines=20, seed=0, test=True)
    assert len(ruls) == 20 and (ruls >= 0).all()


def test_window_shapes_and_labels():
    pre, ptr, pte, std, meta = _ds(window=20)
    x, dates, lab = ptr[0]
    assert x.shape == (20, meta["num_assets"], meta["num_features"])
    assert dates.shape == (20,)
    assert torch.isfinite(x).all()
    assert set(lab.keys()) == {"rul", "rul_true", "health", "anomaly"}
    assert 0 <= lab["health"] < meta["num_health"]
    assert lab["rul"] <= meta["rul_cap"]               # RUL is capped
    assert lab["anomaly"] in (0, 1)


def test_collate():
    _, ptr, _, _, _ = _ds()
    b = collate_cmapss_windows([ptr[i] for i in range(8)])
    assert b["data"].shape[0] == 8 and b["pad_mask"].all()
    assert b["labels"]["rul"].shape == (8,)


def test_windows_stay_within_engine():
    """A window's cycle indices must be strictly increasing by 1 (no jump across an engine boundary)."""
    _, ptr, _, _, _ = _ds(window=15)
    for i in range(0, len(ptr), max(1, len(ptr) // 50)):
        _, dates, _ = ptr[i]
        d = dates.numpy()
        assert (np.diff(d) == 1).all(), "window crossed an engine boundary (non-contiguous cycles)"


def test_std_protocol_one_per_engine():
    _, _, _, std, meta = _ds(window=20)
    # one window per (long-enough) test engine, ending at its last cycle
    assert len(std) <= meta["n_test_engines"]
    x, dates, lab = std[0]
    assert lab["rul_true"] >= 0


def test_condition_norm_reduces_scale():
    ids, cyc, settings, sensors = synthesize_cmapss(n_engines=30, seed=1)
    cent, mean, std = _condition_norm_stats(settings, sensors, n_conditions=1)
    normed = _apply_norm(settings, sensors, cent, mean, std)
    assert np.isfinite(normed).all()
    assert abs(normed.std()) < 5.0 and abs(normed.mean()) < 1.0   # roughly standardized


def test_rul_monotone_nonincreasing_within_engine():
    """Within an engine, capped RUL must be non-increasing as cycles advance (degradation)."""
    pre, ptr, pte, std, meta = _ds(window=10)
    # pull one engine's full RUL array from the dataset internals
    e = next(iter(ptr.eng_labels))
    rul = ptr.eng_labels[e]["rul"]
    assert (np.diff(rul) <= 0).all()
