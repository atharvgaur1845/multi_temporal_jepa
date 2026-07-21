"""Benchmark the structured latent-dynamics predictors (Part 6 #1 LKF, #3 Koopman, #4 Neural-ODE).

Two experiments on the synthetic dynamics (where predictability is dialed + measured):

  A) PREDICTOR SWAP — train temporal JEPA with the free-form transformer vs Koopman vs Neural-ODE
     predictor; report downstream latent-recovery R² and the Koopman spectral radius ρ(K). Structured
     predictors impose a dynamics prior that should fit smooth/predictable latents better and expose
     ρ(K) (|λ|≈1 predictable, |λ|>1 expansive/chaotic).

  B) LKF — fit a linear process operator A to the (true) latent, add measurement noise, and compare
     RMSE of the raw measurements vs an open-loop rollout vs the Kalman-filtered estimate. The filter
     gain should scale with predictability (a better process model corrects more).

    python scripts/structured_predictor_bench.py --device cuda:0
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
from eval.predictability import spectral_predictability, past_future_mi  # noqa: E402
from engine.train_finance import train_finance_jepa  # noqa: E402
from models.finance_jepa import build_finance_model  # noqa: E402
from models.latent_filter import lkf_report  # noqa: E402
from scripts.predictability_sweep import _cfg, _extract, _probe_r2  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def _fit_operator(z):
    """Least-squares one-step linear operator A: z_{t+1} ≈ A z_t (true-dynamics Koopman)."""
    Zt, Zt1 = z[:-1], z[1:]
    At, *_ = np.linalg.lstsq(Zt, Zt1, rcond=None)          # solves Zt @ At = Zt1 -> A = At.T
    A = At.T
    fit_r2 = 1.0 - ((Zt1 - Zt @ At) ** 2).sum() / (((Zt1 - Zt1.mean(0)) ** 2).sum() + 1e-9)
    return A, float(fit_r2), float(np.abs(np.linalg.eigvals(A)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--T", type=int, default=6000)
    args = ap.parse_args()
    device = resolve_device(args.device)
    grid = [("periodic", 0.0), ("ar1", 0.9), ("ar1", 0.5), ("lorenz", 0.0), ("white", 0.0)]

    print("=== A) PREDICTOR SWAP — downstream latent-recovery R² (higher better) ===")
    print(f"{'regime':11s}{'Ω':>6s} | {'freeform':>9s}{'koopman':>9s}{'ode':>7s} | {'ρ(K)learned':>12s}")
    for regime, phi in grid:
        seed_everything(0)
        g = generate(regime=regime, phi=phi, T=args.T, obs_dim=8, W=32, snr=2.0, seed=0)
        meta, data, Y = g["meta"], g["data"], g["latent"]
        om = spectral_predictability(g["z_full"])
        res, rho = {}, float("nan")
        for ptype in ("transformer", "koopman", "ode"):
            seed_everything(0)
            loader = DataLoader(DynamicsWindows(data), batch_size=256, shuffle=True,
                                collate_fn=collate, drop_last=True)
            cfg = _cfg(args.epochs); cfg["predictor"]["type"] = ptype
            model, _ = train_finance_jepa(loader, cfg, meta, device, logger=lambda *a: None,
                                          return_model=True)
            enc = model.target_encoder
            res[ptype] = _probe_r2(_extract(enc, data, True, device), Y)
            if ptype == "koopman":
                rho = model.predictor.spectral_radius()
        tag = f"{regime}:{phi:.1f}" if regime == "ar1" else regime
        print(f"{tag:11s}{om:>6.2f} | {res['transformer']:>9.3f}{res['koopman']:>9.3f}"
              f"{res['ode']:>7.3f} | {rho:>12.3f}")

    print("\n=== B) LKF — RMSE vs clean latent (lower better). dyn_gain = static-filter − dynamic-filter,"
          " isolating what the PROCESS MODEL adds; should grow with one-step predictability, ~0 at white ===")
    print(f"{'regime':11s}{'Ω':>6s}{'A_fit_r2':>9s}{'ρ(A)':>6s} | "
          f"{'measure':>8s}{'static':>7s}{'filtered':>9s}{'dyn_gain':>9s}")
    for regime, phi in grid:
        g = generate(regime=regime, phi=phi, T=4000, seed=0)
        z = g["z_full"]
        A, fit_r2, rho = _fit_operator(z)
        r = lkf_report(z, A, meas_noise=0.6, seed=0)
        tag = f"{regime}:{phi:.1f}" if regime == "ar1" else regime
        print(f"{tag:11s}{spectral_predictability(z):>6.2f}{fit_r2:>9.3f}{rho:>6.2f} | "
              f"{r['rmse_measurement']:>8.3f}{r['rmse_static']:>7.3f}{r['rmse_filtered']:>9.3f}"
              f"{r['dynamics_gain']:>9.3f}")


if __name__ == "__main__":
    main()
