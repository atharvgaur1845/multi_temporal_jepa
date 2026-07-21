"""The predictability sweep — the falsifiable test of the project's central hypothesis (Part 7).

For a grid of synthetic dynamics spanning the predictability axis (periodic -> AR(1) φ-sweep ->
Lorenz -> white noise) we: (1) MEASURE predictability on the ground-truth latent (eval/predictability),
(2) train an IDENTICAL small temporal JEPA and a reconstruction (MAE) baseline on the rendered
observations, (3) probe how well each frozen representation RECOVERS the clean latent z_t (ridge ->
R²), and (4) plot the JEPA advantage vs measured predictability.

Hypothesis (falsifiable): JEPA's advantage over the raw-feature floor is a MONOTONE INCREASING
function of measured predictability, and VANISHES (advantage ≤ 0) at low predictability. The
hypothesis is FALSIFIED if the advantage is flat in predictability, or if JEPA never beats raw.

    python scripts/predictability_sweep.py --device cuda:0            # full sweep -> CSV + PNG
    python scripts/predictability_sweep.py --device cuda:0 --quick    # fast smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from data.synthetic_dynamics import generate, DynamicsWindows, collate  # noqa: E402
from eval.predictability import predictability_report, PRED_KEYS  # noqa: E402
from engine.train_finance import train_finance_jepa, FIN_TRAINERS  # noqa: E402
from models.finance_encoder import PanelEncoder  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def _cfg(epochs):
    return {
        "objective": "temporal_jepa",
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4, "temporal_depth": 2,
                    "grad_checkpoint": False},
        "predictor": {"embed_dim": 32, "depth": 2, "num_heads": 4},
        "ema": {"base_momentum": 0.996, "final_momentum": 1.0},
        "temporal": {"horizon": 1, "min_context": 8, "period": 4096},
        "loss": {"type": "l2", "target_layernorm": True, "var_coeff": 1.0, "cov_coeff": 0.04},
        "optim": {"name": "adamw", "lr": 5e-4, "warmup_epochs": 2, "min_lr": 1e-6,
                  "weight_decay_start": 0.04, "weight_decay_end": 0.4, "epochs": epochs,
                  "batch_size": 256, "grad_accum": 1, "amp": True, "augment": True, "jitter": 0.02},
        "log": {"diagnostics_every": 10000, "seed": 0},
    }


@torch.no_grad()
def _extract(encoder, data, use_temporal, device):
    """Representation of the window's LAST step (we recover the current latent z_e). For JEPA the
    last temporal token has INTEGRATED the past (temporal denoising); the MAE baseline encodes only
    the last frame (no temporal context). The gap = the value of temporal integration."""
    encoder.eval()
    loader = DataLoader(DynamicsWindows(data), batch_size=256, collate_fn=collate)
    out = []
    for b in loader:
        d, dt, pad = b["data"].to(device), b["dates"].to(device), b["pad_mask"].to(device)
        if use_temporal:
            tok = encoder.encode_temporal(d, dt, pad)[:, -1]      # (B,N,D) last timestep, past-aware
        else:
            tok = encoder.encode_full(d[:, -1])                   # (B,N,D) last frame only
        out.append(tok.mean(dim=1).cpu().float().numpy())         # pool over the N cross-section tokens
    return np.concatenate(out)


def _probe_r2(X, Y, train_frac=0.7):
    """Ridge X->Y (latent recovery), temporal split, average R² over latent dims."""
    k = int(len(X) * train_frac)
    sc = StandardScaler().fit(X[:k])
    reg = Ridge(alpha=1.0).fit(sc.transform(X[:k]), Y[:k])
    return float(r2_score(Y[k:], reg.predict(sc.transform(X[k:]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/predictability_sweep.csv")
    ap.add_argument("--fig", default="runs/figures/predictability_sweep.png")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.device)

    grid = [("periodic", 0.0), ("ar1", 0.98), ("ar1", 0.9), ("ar1", 0.8), ("ar1", 0.6),
            ("ar1", 0.4), ("ar1", 0.2), ("lorenz", 0.0), ("white", 0.0)]
    epochs, T = (4, 3000) if args.quick else (args.epochs, args.T)
    if args.quick:
        grid = [("periodic", 0.0), ("ar1", 0.9), ("ar1", 0.4), ("white", 0.0)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"{'regime':12s}{'Ω':>7s}{'MI':>7s}{'ar_r2':>7s} | "
          f"{'JEPA':>7s}{'MAE':>7s}{'raw':>7s} | {'adv_raw':>8s}{'adv_mae':>8s}")
    for regime, phi in grid:
        seed_everything(0)
        g = generate(regime=regime, phi=phi, T=T, obs_dim=8, W=32, snr=2.0, seed=0)
        meta, data, Y = g["meta"], g["data"], g["latent"]
        pr = predictability_report(g["z_full"])
        loader = DataLoader(DynamicsWindows(data), batch_size=256, shuffle=True,
                            collate_fn=collate, drop_last=True)
        cfg = _cfg(epochs)
        enc_j, _ = train_finance_jepa(loader, cfg, meta, device, logger=lambda *a: None)
        enc_m, _ = FIN_TRAINERS["mae"](loader, cfg, meta, device, logger=lambda *a: None)
        Xj = _extract(enc_j, data, True, device)
        Xm = _extract(enc_m, data, False, device)
        Xr = data[:, -1].reshape(len(data), -1)                            # raw floor: last obs frame
        r2j, r2m, r2r = _probe_r2(Xj, Y), _probe_r2(Xm, Y), _probe_r2(Xr, Y)
        row = {"regime": regime, "phi": phi, **pr, "r2_jepa": r2j, "r2_mae": r2m, "r2_raw": r2r,
               "adv_jepa_raw": r2j - r2r, "adv_jepa_mae": r2j - r2m}
        rows.append(row)
        tag = f"{regime}:{phi:.2f}" if regime == "ar1" else regime
        print(f"{tag:12s}{pr['spectral_omega']:>7.3f}{pr['past_future_mi']:>7.2f}{pr['ar_r2']:>7.3f} | "
              f"{r2j:>7.3f}{r2m:>7.3f}{r2r:>7.3f} | {r2j - r2r:>8.3f}{r2j - r2m:>8.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[sweep] wrote {args.out}")

    # verdict
    omega = np.array([r["spectral_omega"] for r in rows])
    adv = np.array([r["adv_jepa_raw"] for r in rows])
    corr = np.corrcoef(omega, adv)[0, 1] if len(rows) > 2 else float("nan")
    print(f"[verdict] corr(predictability Ω, JEPA-advantage-over-raw) = {corr:+.3f}")
    print("  >0 and rising -> HYPOTHESIS SUPPORTED (advantage grows with predictability)")
    print("  ~0 / flat      -> HYPOTHESIS FALSIFIED")

    try:
        _plot(rows, args.fig)
        print(f"[sweep] wrote {args.fig}")
    except Exception as e:
        print(f"[sweep] plot skipped: {e}")


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    omega = [r["spectral_omega"] for r in rows]
    order = np.argsort(omega)
    om = np.array(omega)[order]
    labels = [(f"AR φ={rows[i]['phi']:.2f}" if rows[i]["regime"] == "ar1" else rows[i]["regime"])
              for i in order]
    j = np.array([rows[i]["r2_jepa"] for i in order])
    m = np.array([rows[i]["r2_mae"] for i in order])
    r = np.array([rows[i]["r2_raw"] for i in order])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(om, j, "o-", label="Temporal JEPA", lw=2)
    ax[0].plot(om, m, "s--", label="MAE (reconstruction)", lw=1.5)
    ax[0].plot(om, r, "^:", label="raw features (floor)", lw=1.5)
    ax[0].set_xlabel("measured predictability  (spectral Ω)")
    ax[0].set_ylabel("latent-recovery R²  (test)")
    ax[0].set_title("Downstream quality vs predictability")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].plot(om, j - r, "o-", color="crimson", lw=2, label="JEPA − raw")
    ax[1].plot(om, j - m, "s--", color="darkorange", lw=1.5, label="JEPA − MAE")
    for x, y, lab in zip(om, j - r, labels):
        ax[1].annotate(lab, (x, y), fontsize=7, xytext=(2, 3), textcoords="offset points")
    ax[1].set_xlabel("measured predictability  (spectral Ω)")
    ax[1].set_ylabel("JEPA advantage (ΔR²)")
    ax[1].set_title("Advantage grows with predictability?")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
