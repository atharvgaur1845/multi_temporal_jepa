"""Mechanistic probe (H-mech-2): do the frozen SPATIAL features encode acquisition time?

If the temporal objective made the encoder phenology-aware, a linear probe on its per-frame
spatial features should recover the acquisition DATE far better than spatial-JEPA's / a baseline's.

Crucial: we probe `encode_full` (the per-frame spatial ViT only) — NOT the temporal pathway,
which adds an explicit DOY positional encoding and would make the test circular. So this asks:
did training shape the *spatial* representation to carry seasonal/phenological structure?

For each encoder we extract per-frame pooled features (X) with their day-of-year (y), fit two
probes on the train folds and evaluate on val/test:
  - month classification (12 classes; chance = 8.3%) -> accuracy (higher = more time info),
  - circular DOY regression (predict sin/cos) -> mean absolute angular error in DAYS (lower better).

Usage (compare temporal vs spatial side by side):
    python scripts/mechanistic.py --encoder-ckpt runs/matrix/tjepa_h1.pt runs/matrix/spatial_jepa.pt \
        --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pastis_dataset import PASTIS, collate_variable_length  # noqa: E402
from data.transforms import compute_band_stats  # noqa: E402
from models.jepa import SITSEncoder  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.device import resolve_device  # noqa: E402


def _load_encoder(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    ec = blob["cfg"]["encoder"]
    enc = SITSEncoder(patch_size=ec["patch_size"], embed_dim=ec["embed_dim"], depth=ec["depth"],
                      num_heads=ec["num_heads"], temporal_depth=ec.get("temporal_depth", 4)).to(device)
    enc.load_state_dict(blob["encoder"])
    enc.eval()
    return enc


@torch.no_grad()
def per_frame_features(encoder, loader, device, max_frames=20000):
    """Per-frame SPATIAL features (encode_full -> mean over tokens) + DOY. Returns (X[N,D], y[N])."""
    X, Y = [], []
    n = 0
    for batch in loader:
        data, dates, pad = batch["data"], batch["dates"], batch["pad_mask"]
        B, T = data.shape[:2]
        for b in range(B):
            t_idx = torch.nonzero(pad[b], as_tuple=False).flatten()
            frames = data[b, t_idx].to(device)                      # (k, C, H, W)
            feat = encoder.encode_full(frames).mean(dim=1)          # (k, D)  spatial-only pathway
            X.append(feat.cpu().float()); Y.append(dates[b, t_idx].clone())
            n += len(t_idx)
        if n >= max_frames:
            break
    return torch.cat(X).numpy(), torch.cat(Y).numpy()


def evaluate_encoder(name, enc, tr_loader, ev_loader, device):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = per_frame_features(enc, tr_loader, device)
    Xev, yev = per_frame_features(enc, ev_loader, device)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xev = sc.transform(Xtr), sc.transform(Xev)

    # month classification (12 classes); chance = 1/12
    mtr = np.clip((ytr - 1) * 12 // 366, 0, 11)
    mev = np.clip((yev - 1) * 12 // 366, 0, 11)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, mtr)  # multinomial by default
    month_acc = (clf.predict(Xev) == mev).mean()

    # circular DOY regression: predict (sin, cos), report mean angular error in days
    def sc_targets(y):
        a = 2 * np.pi * y / 366.0
        return np.stack([np.sin(a), np.cos(a)], axis=1)
    reg = Ridge(alpha=1.0).fit(Xtr, sc_targets(ytr))
    pred = reg.predict(Xev)
    pa = np.arctan2(pred[:, 0], pred[:, 1])
    ta = 2 * np.pi * yev / 366.0
    d = np.abs(np.angle(np.exp(1j * (pa - ta))))            # circular abs diff in radians
    days_err = (d * 366.0 / (2 * np.pi)).mean()
    print(f"  {name:<22} month-acc {month_acc*100:5.1f}%   DOY circular MAE {days_err:5.1f} days   "
          f"(n_train={len(ytr)}, n_eval={len(yev)})")
    return month_acc, days_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", nargs="+", required=True, help="one or more runs/matrix/<cell>.pt")
    ap.add_argument("--config", default="configs/model/tjepa_8gb.yaml")
    ap.add_argument("--data", default="configs/data/pastis.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--test", action="store_true", help="evaluate on test folds (default: val)")
    ap.add_argument("--max-patches", type=int, default=150, help="patches per split (caps runtime)")
    args = ap.parse_args()

    cfg = load_yaml(args.config); data_cfg = load_yaml(args.data)
    device = resolve_device(args.device or cfg.get("device"))
    msl = data_cfg.get("max_seq_len")
    ev_folds = data_cfg["test_folds"] if args.test else data_cfg["val_folds"]

    tr_unlab = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False, max_seq_len=msl)
    mean, std = compute_band_stats(tr_unlab, max_samples=200)
    from torch.utils.data import Subset
    tr = Subset(PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False,
                       norm_mean=mean, norm_std=std, max_seq_len=msl), range(args.max_patches))
    ev_full = PASTIS(data_cfg["root"], folds=ev_folds, return_label=False,
                     norm_mean=mean, norm_std=std, max_seq_len=msl)
    ev = Subset(ev_full, range(min(args.max_patches, len(ev_full))))
    trl = DataLoader(tr, batch_size=4, collate_fn=collate_variable_length)
    evl = DataLoader(ev, batch_size=4, collate_fn=collate_variable_length)

    print(f"[mechanistic] H-mech-2: decode acquisition time from frozen SPATIAL features "
          f"({'test' if args.test else 'val'} fold). Higher month-acc / lower DOY-MAE = more time info.")
    print(f"[mechanistic] chance month-acc = {100/12:.1f}%\n")
    for path in args.encoder_ckpt:
        name = os.path.basename(path).replace(".pt", "")
        evaluate_encoder(name, _load_encoder(path, device), trl, evl, device)


if __name__ == "__main__":
    main()
