"""Part 6 #2 — predictability-conditioned objective weighting, tested on mixed-predictability data.

Formalizes the predictability hypothesis into a TRAINING METHOD: weight each window's latent-prediction
loss by its measured spectral predictability Ω^γ, so the encoder spends capacity on the windows whose
dynamics are actually learnable and does not waste it fitting noise.

Clean falsifiable test: build a dataset that MIXES predictable (AR φ=0.95) and unpredictable (white)
windows 50/50. Train two otherwise-identical temporal JEPAs — uniform loss vs predictability-weighted
loss — and compare latent-recovery on the PREDICTABLE subset. This is a within-JEPA comparison, so it
is immune to the raw-baseline confound of report_predictability.md §3.

    Hypothesis: weighted ≥ uniform on the predictable subset (the curriculum focuses capacity).
    Falsified if weighted ≤ uniform (down-weighting noise windows does not help).

    python scripts/predictability_curriculum.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.synthetic_dynamics import generate, DynamicsWindows, collate  # noqa: E402
from engine.train_finance import train_finance_jepa  # noqa: E402
from scripts.predictability_sweep import _cfg, _extract, _probe_r2  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def _mixed(T, noise_frac, seed):
    """Concatenate predictable (AR φ=0.95) and white windows; return data, latent, predictable-mask."""
    gp = generate(regime="ar1", phi=0.95, T=T, obs_dim=8, W=32, snr=1.0, seed=seed)
    gw = generate(regime="white", phi=0.0, T=int(T * noise_frac / (1 - noise_frac)),
                  obs_dim=8, W=32, snr=1.0, seed=seed + 1)
    data = np.concatenate([gp["data"], gw["data"]], 0)
    latent = np.concatenate([gp["latent"], gw["latent"]], 0)
    mask = np.concatenate([np.ones(len(gp["data"]), bool), np.zeros(len(gw["data"]), bool)])
    return data.astype(np.float32), latent.astype(np.float32), mask, gp["meta"]


def _run(data, weighted, epochs, meta, device):
    seed_everything(0)
    loader = DataLoader(DynamicsWindows(data), batch_size=256, shuffle=True,
                        collate_fn=collate, drop_last=True)
    cfg = _cfg(epochs)
    if weighted:
        cfg["loss"]["predictability_weighted"] = True
        cfg["loss"]["predictability_gamma"] = 2.0
    enc, _ = train_finance_jepa(loader, cfg, meta, device, logger=lambda *a: None)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--noise-frac", type=float, default=0.5)
    args = ap.parse_args()
    device = resolve_device(args.device)

    data, latent, mask, meta = _mixed(args.T, args.noise_frac, seed=0)
    print(f"[curriculum] mixed windows: {mask.sum()} predictable + {(~mask).sum()} noise "
          f"({100*args.noise_frac:.0f}% noise); eval on the PREDICTABLE subset")

    def eval_on_predictable(enc, tag):
        X = _extract(enc, data, True, device)
        Xp, Yp = X[mask], latent[mask]                       # predictable subset only
        r2 = _probe_r2(Xp, Yp)
        print(f"  {tag:26s} latent-recovery R² (predictable subset) = {r2:.4f}")
        return r2

    r_uni = eval_on_predictable(_run(data, False, args.epochs, meta, device), "uniform JEPA")
    r_wt = eval_on_predictable(_run(data, True, args.epochs, meta, device),
                               "predictability-weighted JEPA")
    print(f"\n[verdict] Δ (weighted − uniform) = {r_wt - r_uni:+.4f}")
    print("  > 0 -> the curriculum HELPS (focuses capacity on learnable windows)")
    print("  ≤ 0 -> no benefit from down-weighting noise windows")


if __name__ == "__main__":
    main()
