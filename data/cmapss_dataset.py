"""NASA C-MAPSS turbofan dataset — the industrial analogue of data/finance_dataset.py.

Mapping (see report_cmapss.md): an engine's run-to-failure record is a multivariate sensor time
series. We treat the **21 sensors as the cross-section** (tokens), **each operating cycle as a
frame**, and a **window of W cycles** as one sample — so the existing PanelEncoder / FinanceJEPA
stack applies unchanged (sensors ↔ assets ↔ pixels; cycles ↔ trading days ↔ acquisitions).

    data  : (W, N, F)   N sensors (after dropping constants), F per-sensor causal features
    dates : (W,)        operating-cycle index (MONOTONIC — needs a non-wrapping temporal_period,
                        unlike finance/PASTIS day-of-year; the encoder is told via cfg.temporal.period)
    labels: dict(rul, rul_true, health, anomaly) keyed on the window's LAST cycle  (eval only)

Why cycles, not day-of-year: degradation is monotonic, not periodic. A periodic period=366 encoding
would wrap engines that run >366 cycles (FD004 reaches 543); configs set period=1024.

Downstream labels (degradation is a smooth latent trajectory, so all are LEARNABLE — unlike the
finance "predict the unpredictable" trap):
    rul      : piecewise-linear Remaining Useful Life, capped at `rul_cap` (=125; early life is
               "healthy/flat", so RUL is clipped there — the standard C-MAPSS convention).
    rul_true : UNCAPPED RUL (for the standard last-cycle benchmark vs RUL_FDxxx.txt).
    health   : 4-stage {0 healthy, 1 early-degr, 2 late-degr, 3 critical} from RUL thresholds.
    anomaly  : 1 if RUL <= `anom_rul` (near-failure / stress) — for unsupervised AUROC/AP.

Condition normalization: FD002/FD004 mix 6 operating conditions that swamp the degradation signal;
we KMeans the 3 operating settings into `n_conditions` regimes (TRAIN only) and z-score each sensor
WITHIN its regime. FD001/FD003 (single condition) reduce to a global z-score. Constant/uninformative
sensors (≈0 variance after normalization on TRAIN) are dropped.

Splits: C-MAPSS ships separate train (run-to-failure) and test (truncated) engines, so there is no
train/test leakage by construction. We pretrain + fit probes on TRAIN-engine windows and score on
TEST-engine windows; a separate `std_protocol` set holds one window at each TEST engine's last cycle
(target = RUL_FDxxx.txt) for the field-standard RMSE + PHM08-score benchmark.

Offline fallback: if data_root/CMAPSS/train_<FD>.txt is absent, `synthesize_cmapss` generates
monotonic-degradation engines with exact health labels so the pipeline + tests run reproducibly.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = os.environ.get("CMAPSS_ROOT", "data_root/CMAPSS")
N_SENSORS = 21
SETTING_COLS = slice(2, 5)
SENSOR_COLS = slice(5, 26)
HEALTH_NAMES = ("healthy", "early_degr", "late_degr", "critical")


# --------------------------------------------------------------------------- IO / synth
def _read_fd(root, fd, split):
    """Return (engine ids (R,), cycles (R,), settings (R,3), sensors (R,21)) for one FD/split."""
    a = np.loadtxt(os.path.join(root, f"{split}_{fd}.txt"))
    return (a[:, 0].astype(int), a[:, 1].astype(int), a[:, SETTING_COLS], a[:, SENSOR_COLS])


def synthesize_cmapss(n_engines=100, n_sensors=N_SENSORS, seed=0, test=False):
    """Monotonic-degradation engines: each sensor drifts along a per-sensor degradation curve that
    accelerates near failure, + noise; one operating condition. Returns the raw-column layout that
    _read_fd would produce (ids, cycles, settings, sensors), plus per-engine true RUL for test."""
    rng = np.random.default_rng(seed + (1000 if test else 0))
    sens_dir = rng.standard_normal(n_sensors)                      # each sensor's degradation sign
    sens_dir[np.abs(sens_dir) < 0.3] = 0.0                         # a few ~constant sensors
    base = rng.uniform(400, 1600, n_sensors)
    ids, cycles, settings, sensors, ruls = [], [], [], [], []
    for e in range(1, n_engines + 1):
        life = int(rng.integers(140, 340))
        trunc = int(rng.integers(30, life - 20)) if test else life  # test engines are cut short
        t = np.arange(1, trunc + 1)
        frac = (np.arange(1, life + 1) / life)[:trunc]            # 0..1 progress
        degr = (frac ** 2)[:, None]                               # accelerating wear
        sig = base[None, :] + degr * sens_dir[None, :] * 80.0
        sig = sig + rng.standard_normal(sig.shape) * 6.0
        ids.append(np.full(trunc, e)); cycles.append(t)
        settings.append(np.zeros((trunc, 3))); sensors.append(sig)
        ruls.append(life - trunc)                                 # true RUL at last observed cycle
    out = (np.concatenate(ids), np.concatenate(cycles),
           np.concatenate(settings), np.concatenate(sensors))
    return (out, np.array(ruls)) if test else out


def _load_split(root, fd, split, allow_synth, seed):
    path = os.path.join(root, f"{split}_{fd}.txt")
    if os.path.isfile(path):
        return _read_fd(root, fd, split), None
    if not allow_synth:
        raise FileNotFoundError(f"{path} not found. Run scripts/download_cmapss.py (mirror or --zip).")
    if split == "train":
        return synthesize_cmapss(seed=seed, test=False), None
    cols, ruls = synthesize_cmapss(seed=seed, test=True)
    return cols, ruls


# --------------------------------------------------------------------------- normalization / features
def _condition_norm_stats(settings, sensors, n_conditions):
    """Per-(condition, sensor) mean/std from TRAIN. n_conditions=1 -> global. Returns
    (centroids (K,3) or None, mean (K,S), std (K,S))."""
    S = sensors.shape[1]
    if n_conditions <= 1:
        return None, sensors.mean(0, keepdims=True), sensors.std(0, keepdims=True)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_conditions, n_init=10, random_state=0).fit(settings)
    cent = km.cluster_centers_
    mean = np.zeros((n_conditions, S)); std = np.ones((n_conditions, S))
    for k in range(n_conditions):
        m = km.labels_ == k
        if m.sum() > 1:
            mean[k] = sensors[m].mean(0); std[k] = sensors[m].std(0)
    return cent, mean, std


def _apply_norm(settings, sensors, cent, mean, std):
    if cent is None:
        return (sensors - mean) / np.clip(std, 1e-6, None)
    d = ((settings[:, None, :] - cent[None, :, :]) ** 2).sum(-1)   # (R, K)
    lab = d.argmin(1)
    return (sensors - mean[lab]) / np.clip(std[lab], 1e-6, None)


def _per_engine_features(normed, ids, rolling=5):
    """Per-engine causal features: [value, delta, rolling-mean]. Returns dict engine->(Li,S,F)."""
    feats = {}
    for e in np.unique(ids):
        x = normed[ids == e]                                       # (Li, S)
        L, S = x.shape
        delta = np.zeros_like(x); delta[1:] = x[1:] - x[:-1]
        roll = np.zeros_like(x)
        for t in range(L):
            roll[t] = x[max(0, t - rolling + 1):t + 1].mean(0)
        feats[e] = np.stack([x, delta, roll], axis=-1).astype(np.float32)  # (Li, S, F=3)
    return feats


# --------------------------------------------------------------------------- labels
def _rul_arrays(ids, cycles, rul_cap, test_true_rul=None):
    """Per-engine capped + uncapped RUL. For train, RUL = (max_cycle - cycle). For test, add the
    engine's true remaining RUL from RUL_FDxxx.txt so the last cycle has RUL=test_true_rul."""
    capped, true = {}, {}
    uniq = np.unique(ids)
    for i, e in enumerate(uniq):
        c = cycles[ids == e]
        last = c.max()
        extra = 0 if test_true_rul is None else int(test_true_rul[i])
        r = (last - c) + extra                                     # uncapped RUL per cycle
        true[e] = r.astype(np.float32)
        capped[e] = np.clip(r, 0, rul_cap).astype(np.float32)
    return capped, true


def _health_stage(rul_capped, thr=(100, 50, 20)):
    """4 stages from capped RUL: >thr0 healthy, >thr1 early, >thr2 late, else critical."""
    out = {}
    for e, r in rul_capped.items():
        s = np.full(len(r), 3, dtype=np.int64)
        s[r > thr[2]] = 2; s[r > thr[1]] = 1; s[r > thr[0]] = 0
        out[e] = s
    return out


# --------------------------------------------------------------------------- dataset
class CMAPSSWindows(Dataset):
    """Sliding windows over per-engine feature arrays (windows never cross an engine boundary)."""

    def __init__(self, eng_feats, eng_cycles, eng_labels, window, windows, return_labels=True):
        self.eng_feats = eng_feats            # dict engine -> (Li, N, F)
        self.eng_cycles = eng_cycles          # dict engine -> (Li,)
        self.eng_labels = eng_labels          # dict engine -> dict(key -> (Li,))
        self.window = window
        self.windows = windows                # list of (engine, end_local_idx)
        self.return_labels = return_labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        e, end = self.windows[i]
        s = end - self.window + 1
        x = torch.from_numpy(self.eng_feats[e][s:end + 1]).float()        # (W,N,F)
        dates = torch.from_numpy(self.eng_cycles[e][s:end + 1].astype(np.int64))
        if not self.return_labels:
            return x, dates, None
        lab = {k: self.eng_labels[e][k][end] for k in ("rul", "rul_true", "health", "anomaly")}
        return x, dates, lab


def collate_cmapss_windows(batch):
    xs, dates, labs = zip(*batch)
    data = torch.stack(xs, 0)                                          # (B,W,N,F)
    dts = torch.stack(dates, 0)                                        # (B,W)
    out = {"data": data, "dates": dts, "pad_mask": torch.ones(dts.shape, dtype=torch.bool)}
    if labs[0] is None:
        out["labels"] = None
    else:
        out["labels"] = {
            "rul": torch.tensor([float(l["rul"]) for l in labs]),
            "rul_true": torch.tensor([float(l["rul_true"]) for l in labs]),
            "health": torch.tensor([int(l["health"]) for l in labs], dtype=torch.long),
            "anomaly": torch.tensor([int(l["anomaly"]) for l in labs], dtype=torch.long),
        }
    return out


def _window_ends(eng_cycles, window, stride, last_only=False):
    """List of (engine, end_local_idx) windows that fit within each engine."""
    out = []
    for e, c in eng_cycles.items():
        L = len(c)
        if L < window:
            continue
        if last_only:
            out.append((e, L - 1))
        else:
            out.extend((e, end) for end in range(window - 1, L, stride))
    return out


def make_cmapss_datasets(root=DEFAULT_ROOT, fd="FD001", window=40, train_stride=1, eval_stride=3,
                         n_conditions=None, rul_cap=125, anom_rul=20, health_thr=(100, 50, 20),
                         allow_synth=True, seed=0):
    """Build (pretrain_ds, probe_train_ds, probe_test_ds, std_protocol_ds, meta) for one FD subset.

    n_conditions defaults to 6 for FD002/FD004 (multi-condition), 1 for FD001/FD003.
    """
    if n_conditions is None:
        n_conditions = 6 if fd in ("FD002", "FD004") else 1
    (tr_ids, tr_cyc, tr_set, tr_sen), _ = _load_split(root, fd, "train", allow_synth, seed)
    (te_ids, te_cyc, te_set, te_sen), te_rul_file = _load_split(root, fd, "test", allow_synth, seed)
    if te_rul_file is None:                                            # real data: read RUL file
        te_rul_file = np.loadtxt(os.path.join(root, f"RUL_{fd}.txt")).astype(int).reshape(-1)

    # condition normalization (stats from TRAIN only)
    cent, mean, std = _condition_norm_stats(tr_set, tr_sen, n_conditions)
    tr_norm = _apply_norm(tr_set, tr_sen, cent, mean, std)
    te_norm = _apply_norm(te_set, te_sen, cent, mean, std)

    # drop near-constant sensors (decided on TRAIN); keep indices consistent across splits
    keep = np.where(tr_norm.std(0) > 1e-3)[0]
    tr_norm, te_norm = tr_norm[:, keep], te_norm[:, keep]

    tr_feats = _per_engine_features(tr_norm, tr_ids)
    te_feats = _per_engine_features(te_norm, te_ids)
    tr_cycles = {e: tr_cyc[tr_ids == e] for e in np.unique(tr_ids)}
    te_cycles = {e: te_cyc[te_ids == e] for e in np.unique(te_ids)}

    tr_rul, tr_rul_true = _rul_arrays(tr_ids, tr_cyc, rul_cap)
    te_rul, te_rul_true = _rul_arrays(te_ids, te_cyc, rul_cap, test_true_rul=te_rul_file)

    def _labels(rul_capped, rul_true):
        health = _health_stage(rul_capped, health_thr)
        return {e: {"rul": rul_capped[e], "rul_true": rul_true[e], "health": health[e],
                    "anomaly": (rul_capped[e] <= anom_rul).astype(np.int64)} for e in rul_capped}
    tr_labels, te_labels = _labels(tr_rul, tr_rul_true), _labels(te_rul, te_rul_true)

    pre = CMAPSSWindows(tr_feats, tr_cycles, tr_labels, window,
                        _window_ends(tr_cycles, window, train_stride), return_labels=False)
    ptr = CMAPSSWindows(tr_feats, tr_cycles, tr_labels, window,
                        _window_ends(tr_cycles, window, eval_stride), return_labels=True)
    pte = CMAPSSWindows(te_feats, te_cycles, te_labels, window,
                        _window_ends(te_cycles, window, eval_stride), return_labels=True)
    std_protocol = CMAPSSWindows(te_feats, te_cycles, te_labels, window,
                                 _window_ends(te_cycles, window, 1, last_only=True), return_labels=True)

    meta = {
        "fd": fd, "num_assets": int(len(keep)), "num_features": int(next(iter(tr_feats.values())).shape[-1]),
        "window": window, "num_health": len(HEALTH_NAMES), "health_names": list(HEALTH_NAMES),
        "n_conditions": n_conditions, "kept_sensors": [int(k) for k in keep], "rul_cap": rul_cap,
        "source": "synthetic" if not os.path.isfile(os.path.join(root, f"train_{fd}.txt")) else "nasa",
        "n_train_engines": len(tr_feats), "n_test_engines": len(te_feats),
        "n_pretrain_windows": len(pre), "n_probe_test_windows": len(pte),
    }
    return pre, ptr, pte, std_protocol, meta
