"""Financial cross-sectional panel dataset — the finance analogue of data/pastis_dataset.py.

Mapping (why the satellite pipeline transfers to markets, see report.md §F)
--------------------------------------------------------------------------
    PASTIS satellite series          ->   S&P sector panel
    frame at time t (H x W pixels)   ->   one trading day's cross-section of N sector ETFs
    spatial token (a pixel patch)    ->   one asset's feature vector that day
    DOY temporal position            ->   trading-day day-of-year (annual seasonality + "when")
    "predict the future frame"       ->   "predict tomorrow's market cross-section latent"
    "mask spatial blocks of a frame" ->   "mask a subset of the day's assets" (Spatial JEPA)

So a *sample* is a fixed-length WINDOW of W consecutive trading days:
    data  : (W, N, F)   F per-asset features (returns / abs-return / dvolume / vol-z), per day
    dates : (W,)        day-of-year in [1, 366] of each trading day
    labels: dict of downstream targets keyed on the window's LAST day (eval only; see below)

Windows are fixed length (no padding) so pad_mask is all-True; we keep it for uniformity with the
JEPA temporal forward (which is written for variable-length SITS and reused unchanged).

Downstream task labels (computed from the index ^GSPC, which the ENCODER NEVER SEES)
    regime    : 4-way {up-calm, up-volatile, down-calm, down-volatile} from the window's trailing
                index return sign x (realized vol above/below the TRAIN median) — a *contemporaneous*
                decode probe (like decoding the crop class from frozen PASTIS features).
    fwd_vol   : realized vol of the index over the NEXT `vol_horizon` days   (forward regression)
    anomaly   : 1 if the largest |index move| in the next `anom_horizon` days exceeds the TRAIN
                99th pct of daily |returns| (crash / stress windows)         (forward, for AUROC)
    fwd_ret   : index log-return on day t+1                                  (forward regression / IC)
    fwd_dir   : sign(fwd_ret) in {0,1}                                       (forward direction)

All forward labels use information strictly AFTER the window, so any model that only sees the window
is doing genuine prediction. Thresholds (vol median, anomaly q99) are computed on the TRAIN period
only and reused for TEST — no leakage. Train/test split is by calendar date with a purge gap so a
train window's forward-label horizon never reaches into the test period.

Offline fallback: if data_root/FINANCE/panel.npz is absent (e.g. no network for the downloader),
`synthesize_panel` generates a documented regime-switching multi-sector market so the whole
pipeline (train + eval + tests) runs reproducibly offline, with EXACT ground-truth regime labels.
"""
from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = os.environ.get("FINANCE_ROOT", "data_root/FINANCE")
FEATURE_NAMES = ("logret", "abs_logret", "dlog_volume", "vol_zscore")
REGIME_NAMES = ("up_calm", "up_volatile", "down_calm", "down_volatile")


# --------------------------------------------------------------------------- panel IO / synth
def _yyyymmdd_to_doy(d: int) -> int:
    d = int(d)
    return _dt.date(d // 10000, (d // 100) % 100, d % 100).timetuple().tm_yday


def synthesize_panel(n_days=6000, n_assets=9, seed=0):
    """Regime-switching multi-sector market with a market factor, sector betas, vol clustering and
    injected crashes. Returns the same dict layout as the Yahoo downloader's panel.npz.

    Dynamics: a latent regime in {bull-calm, bull-volatile, bear, crisis} switches with a sticky
    transition matrix; each regime sets the market drift and base vol. Daily vol follows a GARCH-ish
    process so stress clusters. Sector returns = beta * market + idiosyncratic. This gives EXACT
    regime ground truth and realistic cross-asset correlation, useful as an offline testbed.
    """
    rng = np.random.default_rng(seed)
    # regime -> (annual drift, annual base vol)
    spec = {0: (0.12, 0.10), 1: (0.08, 0.20), 2: (-0.10, 0.22), 3: (-0.35, 0.55)}
    P = np.array([[0.985, 0.010, 0.004, 0.001],
                  [0.020, 0.960, 0.015, 0.005],
                  [0.010, 0.020, 0.955, 0.015],
                  [0.030, 0.040, 0.080, 0.850]])
    betas = np.linspace(0.7, 1.3, n_assets)
    reg = 0
    h = (0.10 / np.sqrt(252)) ** 2                       # latent daily variance (GARCH state)
    mkt = np.zeros(n_days)
    sret = np.zeros((n_days, n_assets))
    for t in range(n_days):
        reg = rng.choice(4, p=P[reg])
        mu, base = spec[reg]
        target_var = (base / np.sqrt(252)) ** 2
        h = 0.92 * h + 0.08 * target_var                 # mean-revert variance toward regime base
        shock = rng.standard_t(5) * np.sqrt(h)           # fat-tailed daily market shock
        mkt[t] = mu / 252 + shock
        idio = rng.standard_normal(n_assets) * (base / np.sqrt(252)) * 0.6
        sret[t] = betas * mkt[t] + idio
    # prices from returns
    close = 100.0 * np.exp(np.cumsum(sret, axis=0))
    index_close = 1000.0 * np.exp(np.cumsum(mkt))
    volume = np.exp(rng.standard_normal((n_days, n_assets)) * 0.3 + 15.0) \
        * (1.0 + 5.0 * np.abs(sret))                     # volume spikes with |return|
    vix = np.clip(np.sqrt(np.maximum(1e-8, _rolling_var(mkt, 20))) * np.sqrt(252) * 100, 9, 90)
    start = _dt.date(1999, 1, 4)
    dates = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:                              # weekdays only (approx trading calendar)
            dates.append(d.year * 10000 + d.month * 100 + d.day)
        d += _dt.timedelta(days=1)
    return {
        "symbols": np.array([f"SEC{i}" for i in range(n_assets)]),
        "dates": np.array(dates, np.int64),
        "close": close.astype(np.float32), "volume": volume.astype(np.float32),
        "index_close": index_close.astype(np.float32), "vix": vix.astype(np.float32),
        "source": np.array("synthetic"),
    }


def load_or_synthesize_panel(root=DEFAULT_ROOT, allow_synth=True, seed=0):
    path = os.path.join(root, "panel.npz")
    if os.path.isfile(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    if not allow_synth:
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_finance.py` (needs network), or pass "
            "allow_synth=True to use the offline synthetic market.")
    print(f"[finance_dataset] {path} not found -> synthetic regime-switching panel (offline).")
    return synthesize_panel(seed=seed)


# --------------------------------------------------------------------------- feature / label math
def _rolling_var(x, w):
    """Causal rolling variance of a 1-D array (window w, min 2 obs); same length as x."""
    out = np.zeros_like(x, dtype=np.float64)
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        seg = x[lo:t + 1]
        out[t] = seg.var() if len(seg) > 1 else 0.0
    return out


def build_features(close, volume):
    """(T,N) close/volume -> (T,N,F) per-asset daily features. All causal (use only past/current).

    F = [log-return, |log-return|, dlog-volume, vol-standardized return]. Row 0 is zero-padded
    (no prior day). vol_zscore divides the return by its trailing-20d std (shifted, so causal).
    """
    close = close.astype(np.float64)
    volume = volume.astype(np.float64)
    T, N = close.shape
    r = np.zeros((T, N))
    r[1:] = np.log(close[1:] / np.clip(close[:-1], 1e-8, None))
    lv = np.log(volume + 1.0)
    dvol = np.zeros((T, N))
    dvol[1:] = lv[1:] - lv[:-1]
    # trailing 20d std of returns per asset, shifted by 1 so day t uses only days < t
    rvol = np.zeros((T, N))
    for n in range(N):
        rvol[:, n] = np.sqrt(np.clip(_rolling_var(r[:, n], 20), 1e-10, None))
    rvol_shift = np.vstack([rvol[:1], rvol[:-1]])
    rz = r / np.clip(rvol_shift, 1e-4, None)
    feats = np.stack([r, np.abs(r), dvol, rz], axis=-1)            # (T,N,F)
    return feats.astype(np.float32)


def _index_logret(index_close):
    ic = index_close.astype(np.float64)
    r = np.zeros(len(ic))
    r[1:] = np.log(ic[1:] / np.clip(ic[:-1], 1e-8, None))
    return r


def build_day_labels(index_close, window, vol_horizon, anom_horizon, train_end_idx):
    """Per-day label arrays keyed on a window's LAST day index e.

    Returns dict of float/int arrays length T (NaN / -1 where a forward horizon runs off the end or
    the trailing window is incomplete). Thresholds (vol median, anomaly q99) come from TRAIN days
    (index <= train_end_idx) only.
    """
    r = _index_logret(index_close)
    T = len(r)
    ann = np.sqrt(252.0)

    trail_ret = np.full(T, np.nan)
    trail_vol = np.full(T, np.nan)
    for e in range(window - 1, T):
        seg = r[e - window + 1:e + 1]
        trail_ret[e] = seg.sum()
        trail_vol[e] = seg.std() * ann

    fwd_vol = np.full(T, np.nan)
    for e in range(T):
        if e + vol_horizon < T:
            fwd_vol[e] = r[e + 1:e + 1 + vol_horizon].std() * ann

    fwd_ret = np.full(T, np.nan)
    fwd_ret[:-1] = r[1:]
    fwd_dir = np.where(np.isnan(fwd_ret), -1, (fwd_ret > 0).astype(np.int64))

    # anomaly threshold from TRAIN daily |returns|
    train_abs = np.abs(r[1:train_end_idx + 1])
    q99 = np.quantile(train_abs, 0.99) if len(train_abs) else np.inf
    anomaly = np.full(T, -1, dtype=np.int64)
    for e in range(T):
        if e + anom_horizon < T:
            anomaly[e] = int(np.abs(r[e + 1:e + 1 + anom_horizon]).max() > q99)

    # regime: vol median from TRAIN ends only
    tv_train = trail_vol[window - 1:train_end_idx + 1]
    vol_med = np.nanmedian(tv_train) if np.isfinite(tv_train).any() else np.nan
    regime = np.full(T, -1, dtype=np.int64)
    for e in range(T):
        if not (np.isfinite(trail_ret[e]) and np.isfinite(trail_vol[e])):
            continue
        up = trail_ret[e] > 0
        volatile = trail_vol[e] > vol_med
        regime[e] = (0 if up else 2) + (1 if volatile else 0)
    return {
        "regime": regime, "fwd_vol": fwd_vol.astype(np.float32),
        "anomaly": anomaly, "fwd_ret": fwd_ret.astype(np.float32), "fwd_dir": fwd_dir,
        "_meta": {"vol_med": float(vol_med) if np.isfinite(vol_med) else None, "anom_q99": float(q99)},
    }


# --------------------------------------------------------------------------- dataset
class FinancePanel(Dataset):
    """Sliding-window view over a precomputed (T,N,F) feature panel + per-day label arrays.

    Constructed via `make_finance_datasets` (which shares train-derived stats/thresholds across the
    train and test splits). `window_ends` is the list of valid last-day indices for this split.
    """

    def __init__(self, feats, doy, labels, window, window_ends, norm_mean, norm_std,
                 return_labels=True):
        self.feats = feats                                       # (T,N,F) float32
        self.doy = doy                                           # (T,) int
        self.labels = labels
        self.window = window
        self.window_ends = list(window_ends)
        self.norm_mean = norm_mean                               # (F,) or None
        self.norm_std = norm_std
        self.return_labels = return_labels

    def __len__(self):
        return len(self.window_ends)

    def __getitem__(self, i):
        e = self.window_ends[i]
        s = e - self.window + 1
        x = torch.from_numpy(self.feats[s:e + 1]).float()        # (W,N,F)
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std.clamp_min(1e-6)
        dates = torch.from_numpy(self.doy[s:e + 1].astype(np.int64))  # (W,)
        if not self.return_labels:
            return x, dates, None
        lab = {
            "regime": int(self.labels["regime"][e]),
            "anomaly": int(self.labels["anomaly"][e]),
            "fwd_dir": int(self.labels["fwd_dir"][e]),
            "fwd_vol": float(self.labels["fwd_vol"][e]),
            "fwd_ret": float(self.labels["fwd_ret"][e]),
        }
        return x, dates, lab


def collate_windows(batch):
    """Stack fixed-length windows -> dict(data,(B,W,N,F); dates,(B,W); pad_mask all-True; labels)."""
    xs, dates, labs = zip(*batch)
    data = torch.stack(xs, 0)                                    # (B,W,N,F)
    dts = torch.stack(dates, 0)                                  # (B,W)
    pad_mask = torch.ones(dts.shape, dtype=torch.bool)
    out = {"data": data, "dates": dts, "pad_mask": pad_mask}
    if labs[0] is None:
        out["labels"] = None
    else:
        out["labels"] = {
            "regime": torch.tensor([l["regime"] for l in labs], dtype=torch.long),
            "anomaly": torch.tensor([l["anomaly"] for l in labs], dtype=torch.long),
            "fwd_dir": torch.tensor([l["fwd_dir"] for l in labs], dtype=torch.long),
            "fwd_vol": torch.tensor([l["fwd_vol"] for l in labs], dtype=torch.float),
            "fwd_ret": torch.tensor([l["fwd_ret"] for l in labs], dtype=torch.float),
        }
    return out


def _date_to_idx(dates_yyyymmdd, boundary_yyyymmdd):
    """Largest day index whose date <= boundary."""
    idx = np.searchsorted(dates_yyyymmdd, boundary_yyyymmdd, side="right") - 1
    return int(np.clip(idx, 0, len(dates_yyyymmdd) - 1))


def make_finance_datasets(root=DEFAULT_ROOT, window=64, train_stride=1, eval_stride=5,
                          train_end=20171231, vol_horizon=20, anom_horizon=5,
                          allow_synth=True, seed=0):
    """Build (pretrain_ds, probe_train_ds, probe_test_ds, meta) sharing train-derived stats.

    * pretrain_ds  : TRAIN period, dense stride, labels off (SSL pretraining).
    * probe_train_ds: TRAIN period, eval stride, labels on (fit downstream heads / fit anomaly model).
    * probe_test_ds : TEST period, eval stride, labels on (report all metrics here).
    Train/test are split by `train_end` (YYYYMMDD) with a purge gap of `window + max_horizon` days so
    no train window's forward label reaches into the test period.
    """
    panel = load_or_synthesize_panel(root, allow_synth=allow_synth, seed=seed)
    close, volume = panel["close"], panel["volume"]
    dates = panel["dates"].astype(np.int64)
    T, N = close.shape
    feats = build_features(close, volume)                        # (T,N,F)
    doy = np.array([_yyyymmdd_to_doy(d) for d in dates], np.int64)

    train_end_idx = _date_to_idx(dates, train_end)
    labels = build_day_labels(panel["index_close"], window, vol_horizon, anom_horizon, train_end_idx)
    max_h = max(vol_horizon, anom_horizon, 1)

    def valid(e):                                                # all labels present at end-day e
        return (labels["regime"][e] >= 0 and labels["anomaly"][e] >= 0
                and labels["fwd_dir"][e] >= 0 and np.isfinite(labels["fwd_vol"][e]))

    # TRAIN: window fully + its forward horizon inside the train region. TEST: window starts after
    # train_end (purge) and its forward horizon fits within the series.
    train_ends = [e for e in range(window - 1, T)
                  if e + max_h <= train_end_idx and valid(e)]
    test_ends = [e for e in range(window - 1, T)
                 if (e - window + 1) > train_end_idx and e + max_h < T and valid(e)]

    # feature normalization from TRAIN windows only (per-feature mean/std over all train days used)
    train_days = np.unique(np.concatenate([np.arange(e - window + 1, e + 1) for e in train_ends])) \
        if train_ends else np.arange(0, train_end_idx + 1)
    flat = feats[train_days].reshape(-1, feats.shape[-1])
    norm_mean = torch.from_numpy(flat.mean(0)).float()
    norm_std = torch.from_numpy(flat.std(0)).float()

    def ds(ends, stride, return_labels):
        return FinancePanel(feats, doy, labels, window, ends[::stride], norm_mean, norm_std,
                            return_labels=return_labels)

    meta = {
        "num_assets": int(N), "num_features": int(feats.shape[-1]), "window": window,
        "feature_names": list(FEATURE_NAMES), "regime_names": list(REGIME_NAMES),
        "num_regimes": len(REGIME_NAMES), "source": str(panel["source"]),
        "train_end_idx": train_end_idx, "n_days": int(T),
        "n_train_windows": len(train_ends), "n_test_windows": len(test_ends),
        "label_meta": labels["_meta"],
    }
    return (ds(train_ends, train_stride, False),
            ds(train_ends, eval_stride, True),
            ds(test_ends, eval_stride, True),
            meta)
