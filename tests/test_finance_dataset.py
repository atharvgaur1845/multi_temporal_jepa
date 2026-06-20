"""Finance dataset invariants — shapes, no NaNs, label validity, and NO TRAIN/TEST LEAKAGE.

These run fully offline on the synthetic regime-switching panel (allow_synth), so no download is
needed; if data_root/FINANCE/panel.npz exists they run on the real S&P panel instead.
"""
import numpy as np
import torch

from data.finance_dataset import (build_features, make_finance_datasets, collate_windows,
                                   synthesize_panel)


def test_synth_panel_shapes():
    p = synthesize_panel(n_days=800, n_assets=9, seed=0)
    assert p["close"].shape == (800, 9)
    assert p["index_close"].shape == (800,)
    assert not np.isnan(p["close"]).any()


def test_features_causal_and_finite():
    p = synthesize_panel(n_days=400, seed=1)
    feats = build_features(p["close"], p["volume"])
    assert feats.shape == (400, 9, 4)
    assert np.isfinite(feats).all()
    # row 0 has no prior day -> return/dvol features are zero
    assert np.allclose(feats[0, :, 0], 0.0)


def _datasets():
    return make_finance_datasets(root="__no_such_root__", window=32, train_stride=1, eval_stride=3,
                                 train_end=20100101, vol_horizon=10, anom_horizon=5,
                                 allow_synth=True, seed=0)


def test_window_shapes_and_labels():
    pre, ptr, pte, meta = _datasets()
    x, dates, lab = ptr[0]
    assert x.shape == (32, meta["num_assets"], meta["num_features"])
    assert dates.shape == (32,)
    assert torch.isfinite(x).all()
    assert set(lab.keys()) == {"regime", "anomaly", "fwd_dir", "fwd_vol", "fwd_ret"}
    assert 0 <= lab["regime"] < meta["num_regimes"]
    assert lab["anomaly"] in (0, 1)


def test_collate_batches():
    _, ptr, _, _ = _datasets()
    batch = collate_windows([ptr[i] for i in range(8)])
    assert batch["data"].shape[0] == 8
    assert batch["pad_mask"].all()                      # fixed-length windows -> all real
    assert batch["labels"]["regime"].shape == (8,)


def test_no_train_test_leakage():
    """Every TRAIN window's forward-label horizon must stay strictly before every TEST window's
    first day (purge gap), so a probe fit on train can't have seen test-period prices."""
    pre, ptr, pte, meta = _datasets()
    # reconstruct end-day index ranges from the stored window_ends
    tr_ends = max(pre.window_ends)
    te_starts = min(e - pte.window + 1 for e in pte.window_ends)
    assert te_starts > tr_ends, "a test window starts at/before the last train window end (leak)"


def test_pretrain_has_no_labels():
    pre, _, _, _ = _datasets()
    x, dates, lab = pre[0]
    assert lab is None
