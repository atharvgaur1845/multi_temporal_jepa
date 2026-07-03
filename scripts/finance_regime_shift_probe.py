"""Phase 5 — is the finance failure NON-STATIONARITY or UNPREDICTABILITY?

Phase 2/4 showed no SSL beats the raw-feature floor on the OUT-OF-TIME split (probe fit on 1999-2017,
tested on 2018-2026), and the distributional rescue failed. Two hypotheses remain for *why*:
  (A) distribution shift — the regime/vol relationship the probe learns on 1999-2017 is stale by 2018;
  (B) unpredictability — the tasks are just hard and no representation beats the engineered features.

This script isolates them CHEAPLY by reusing the ALREADY-TRAINED encoders (runs/finance/*.pt) and only
changing the EVAL protocol: instead of fitting the probe on 1999-2017, we fit and test it WITHIN the
recent 2018-2026 period (a temporal split inside the test era, still out-of-sample, no look-ahead). If
SSL features beat the raw-feature floor IN-PERIOD but not OUT-OF-TIME, hypothesis (A) is confirmed —
the failure is the shift, and a walk-forward/rolling protocol is the fix. If SSL still ties/loses to
raw IN-PERIOD, hypothesis (B) dominates — the tasks are unpredictable from any representation.

    python scripts/finance_regime_shift_probe.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.finance_dataset import make_finance_datasets, collate_windows  # noqa: E402
from eval.finance_tasks import (anomaly_detection, clustering, regime_classification,  # noqa: E402
                                volatility_prediction, extract_window_embeddings)


def raw_feature_embeddings(loader):
    """Mean-pool the raw input features per window -> (M, N*F) 'no-encoder' floor + labels."""
    keys = ("regime", "anomaly", "fwd_dir", "fwd_vol", "fwd_ret")
    feats, labs = [], {kk: [] for kk in keys}
    for b in loader:
        data, pad = b["data"], b["pad_mask"]
        m = pad.float()[:, :, None, None]
        pooled = (data * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)      # (B, N, F) mean over time
        feats.append(pooled.reshape(data.shape[0], -1).float())          # (B, N*F)
        for kk in keys:
            labs[kk].append(b["labels"][kk])
    X = torch.cat(feats, 0).numpy()
    return X, {kk: torch.cat(labs[kk], 0).numpy() for kk in keys}
from models.finance_encoder import PanelEncoder  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.device import resolve_device  # noqa: E402


def _load_encoder(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, meta = ck["cfg"], ck["meta"]
    enc = ck["encoder"]
    ec = cfg["encoder"]
    model = PanelEncoder(num_assets=meta["num_assets"], num_features=meta["num_features"],
                         embed_dim=ec["embed_dim"], depth=ec["depth"], num_heads=ec["num_heads"],
                         temporal_depth=ec.get("temporal_depth", 4),
                         temporal_period=cfg.get("temporal", {}).get("period", 366)).to(device)
    model.load_state_dict(enc)
    return model, ck.get("use_temporal", True)


def _tasks(Xtr, Ltr, Xte, Lte, num_regimes):
    r = {}
    r.update(regime_classification(Xtr, Ltr["regime"], Xte, Lte["regime"]))
    r.update(volatility_prediction(Xtr, Ltr["fwd_vol"], Xte, Lte["fwd_vol"]))
    r.update(anomaly_detection(Xtr, Xte, Lte["anomaly"]))
    r.update(clustering(Xte, Lte["regime"], num_regimes))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-dir", default="runs/finance")
    ap.add_argument("--cells", nargs="*",
                    default=["tjepa_h1", "tjepa_dist", "mae", "byol", "random"])
    ap.add_argument("--fit-frac", type=float, default=0.6, help="in-period probe-fit fraction (temporal)")
    args = ap.parse_args()
    device = resolve_device(args.device)
    data_cfg = load_yaml("configs/data/finance.yaml")
    _, _, pte, meta = make_finance_datasets(root=data_cfg["root"], window=data_cfg["window"],
                                            eval_stride=data_cfg.get("eval_stride", 5),
                                            train_end=data_cfg.get("train_end", 20171231))
    loader = DataLoader(pte, batch_size=128, collate_fn=collate_windows)   # chronological (no shuffle)
    nreg = meta["num_regimes"]
    k = int(len(pte) * args.fit_frac)
    print(f"[shift-probe] test-period windows={len(pte)}  in-period temporal split: fit first {k} "
          f"(~2018-2023), test last {len(pte)-k} (~2023-2026)\n")
    hdr = f"{'method':12s} {'regime':>8s} {'volR2':>8s} {'anom':>8s} {'NMI':>8s}"
    print("IN-PERIOD (probe fit+test both inside 2018-2026):")
    print(hdr); print("-" * len(hdr))

    def split_run(X, L):
        Xtr, Xte = X[:k], X[k:]
        Ltr = {kk: v[:k] for kk, v in L.items()}
        Lte = {kk: v[k:] for kk, v in L.items()}
        return _tasks(Xtr, Ltr, Xte, Lte, nreg)

    # raw-feature floor, in-period
    Xr, Lr = raw_feature_embeddings(loader)
    r = split_run(Xr, Lr)
    print(f"{'raw_features':12s} {r['regime_acc']:>8.3f} {r['vol_r2']:>8.3f} {r['anom_auroc']:>8.3f} {r['clust_nmi']:>8.3f}")
    for cell in args.cells:
        p = os.path.join(args.ckpt_dir, f"{cell}.pt")
        if not os.path.exists(p):
            print(f"{cell:12s}  (no ckpt)"); continue
        enc, use_temporal = _load_encoder(p, device)
        X, L = extract_window_embeddings(enc, loader, device, use_temporal)
        r = split_run(X, L)
        print(f"{cell:12s} {r['regime_acc']:>8.3f} {r['vol_r2']:>8.3f} {r['anom_auroc']:>8.3f} {r['clust_nmi']:>8.3f}")
    print("\nCompare to OUT-OF-TIME (runs/finance_results.csv): regime ~0.80 for raw/random, 0.61 tjepa.")
    print("If SSL now BEATS raw in-period -> non-stationarity was the killer (walk-forward is the fix).")
    print("If SSL still ~ raw in-period    -> unpredictability dominates (no representation helps).")


if __name__ == "__main__":
    main()
