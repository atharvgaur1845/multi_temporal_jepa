"""Part 6 #7 — hierarchical / multi-timescale temporal JEPA.

Does predicting SEVERAL future horizons jointly (e.g. Δ∈{1,5,20}) — forcing the encoder to model both
fast and slow dynamics — beat a single-horizon (Δ=1) objective? Tested on synthetic dynamics that carry
multiple timescales (periodic = several frequencies; AR(0.9) = long memory). Downstream = latent
recovery R² (frozen encoder), a within-JEPA comparison (immune to the raw-baseline confound).

    python scripts/hierarchical_bench.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.synthetic_dynamics import generate, DynamicsWindows, collate  # noqa: E402
from engine.train_finance import train_finance_jepa  # noqa: E402
from scripts.predictability_sweep import _cfg, _extract, _probe_r2  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def _run(data, meta, horizons, epochs, device):
    seed_everything(0)
    loader = DataLoader(DynamicsWindows(data), batch_size=256, shuffle=True,
                        collate_fn=collate, drop_last=True)
    cfg = _cfg(epochs)
    cfg["temporal"]["min_context"] = 8
    if len(horizons) > 1:
        cfg["temporal"]["horizons"] = horizons
    else:
        cfg["temporal"]["horizon"] = horizons[0]
    enc, _ = train_finance_jepa(loader, cfg, meta, device, logger=lambda *a: None)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--T", type=int, default=6000)
    args = ap.parse_args()
    device = resolve_device(args.device)
    schemes = {"single Δ=1": [1], "hier Δ=[1,5]": [1, 5], "hier Δ=[1,5,20]": [1, 5, 20]}

    print(f"{'regime':11s} | " + "".join(f"{k:>16s}" for k in schemes))
    for regime, phi in [("periodic", 0.0), ("ar1", 0.9)]:
        g = generate(regime=regime, phi=phi, T=args.T, obs_dim=8, W=40, snr=2.0, seed=0)
        meta, data, Y = g["meta"], g["data"], g["latent"]
        row = {}
        for name, hz in schemes.items():
            enc = _run(data, meta, hz, args.epochs, device)
            row[name] = _probe_r2(_extract(enc, data, True, device), Y)
        tag = f"{regime}:{phi:.1f}" if regime == "ar1" else regime
        print(f"{tag:11s} | " + "".join(f"{row[k]:>16.3f}" for k in schemes))
    print("\n(higher latent-recovery R² = better; does multi-timescale prediction help?)")


if __name__ == "__main__":
    main()
