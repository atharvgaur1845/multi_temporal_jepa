"""ALIGNMENT falsification — does benefit track PREDICTABILITY (H1) or ALIGNMENT (H2)?

The three real domains confound these (PASTIS/C-MAPSS: predictable component IS task-relevant;
finance: neither). This sweep breaks the confound by construction.

    data/synthetic_dynamics.generate_aligned holds the OBSERVATION fixed — it always contains a
    predictable AR(0.95) block and an unpredictable white block — and moves only the LABEL, from
    "reads the predictable block" (alpha=1) to "reads the unpredictable block" (alpha=0).

So measured input predictability (Omega, past->future MI) is invariant to alpha BY CONSTRUCTION, and
we VERIFY that empirically (if it drifts, the design is broken and the run is void).

    H1 (predictability only): JEPA advantage is FLAT in alpha.
    H2 (alignment):           advantage FALLS as alpha->0; temporal JEPA should LOSE to MAE at
                              alpha=0 despite high measured predictability.

H1 is the project's current published thesis, so this experiment can falsify our own claim — which
is the point. Also reports `alignment_index` (eval/predictability.py), the proposed label-aware
replacement for raw predictability, to test whether it predicts the advantage better than Omega does.

    python scripts/alignment_bench.py --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.synthetic_dynamics import generate_aligned, DynamicsWindows, collate  # noqa: E402
from engine.train_finance import train_finance_jepa, FIN_TRAINERS  # noqa: E402
from eval.predictability import (alignment_index, past_future_mi,  # noqa: E402
                                 spectral_predictability)
from scripts.predictability_sweep import _cfg, _extract, _probe_r2  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/alignment_bench.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3, help="repeats per alpha (error bars)")
    ap.add_argument("--snr", type=float, default=2.0,
                    help="observation SNR. LOW snr makes temporal denoising valuable, so the "
                         "JEPA-vs-MAE contrast is sensitive; at high snr the last frame already "
                         "contains the label and no encoder can beat the raw floor (insensitive).")
    ap.add_argument("--hard-render", action="store_true",
                    help="deep tanh-MLP observation map (not linearly invertible). Pre-stated "
                         "engineering fix for the V1 resolving-power failure: with the shallow map "
                         "the raw floor beat every encoder in 30/30 cells.")
    ap.add_argument("--embed-dim", type=int, default=None,
                    help="encoder width override (default 64). The 64d/2-layer/15-epoch model was "
                         "the second named underpowering reason.")
    ap.add_argument("--depth", type=int, default=None, help="encoder depth override (default 2)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = resolve_device(args.device)

    alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
    epochs, T, n_seeds = args.epochs, args.T, args.seeds
    if args.quick:
        alphas, epochs, T, n_seeds = [1.0, 0.5, 0.0], 4, 3000, 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"alignment sweep: alphas={alphas} seeds={n_seeds} epochs={epochs} T={T} "
          f"snr={args.snr} hard_render={args.hard_render} embed_dim={args.embed_dim or 64} "
          f"device={device}")
    print("(Omega/MI are properties of the INPUT and must stay ~constant across alpha)\n")
    print(f"{'alpha':>6s}{'seed':>5s}{'Omega':>7s}{'MI':>6s}{'align':>7s} | "
          f"{'JEPA':>7s}{'MAE':>7s}{'raw':>7s} | {'J-MAE':>7s}{'J-raw':>7s}")

    for alpha in alphas:
        for seed in range(n_seeds):
            seed_everything(seed)
            g = generate_aligned(alpha=alpha, T=T, obs_dim=8, W=32, snr=args.snr, seed=seed,
                                 hard_render=args.hard_render)
            meta, data, Y = g["meta"], g["data"], g["label"]

            # input-side measurements (label-agnostic) — must be invariant to alpha
            om = spectral_predictability(g["x_full"])
            mi = past_future_mi(g["x_full"])
            ai = alignment_index(data, Y)                      # label-aware, should TRACK alpha

            loader = DataLoader(DynamicsWindows(data), batch_size=256, shuffle=True,
                                collate_fn=collate, drop_last=True)
            cfg = _cfg(epochs)
            cfg["log"]["seed"] = seed
            if args.embed_dim:
                cfg["encoder"]["embed_dim"] = args.embed_dim
                cfg["predictor"]["embed_dim"] = max(16, args.embed_dim // 2)   # keep the bottleneck
            if args.depth:
                cfg["encoder"]["depth"] = args.depth
            enc_j, _ = train_finance_jepa(loader, cfg, meta, device, logger=lambda *a: None)
            enc_m, _ = FIN_TRAINERS["mae"](loader, cfg, meta, device, logger=lambda *a: None)

            r2j = _probe_r2(_extract(enc_j, data, True, device), Y)
            r2m = _probe_r2(_extract(enc_m, data, False, device), Y)
            r2r = _probe_r2(data[:, -1].reshape(len(data), -1), Y)     # raw floor: last frame

            rows.append({"alpha": alpha, "seed": seed, "snr": args.snr,
                         "hard_render": int(args.hard_render), "embed_dim": cfg["encoder"]["embed_dim"], "omega": om, "mi": mi, "align_index": ai,
                         "r2_jepa": r2j, "r2_mae": r2m, "r2_raw": r2r,
                         "adv_jepa_mae": r2j - r2m, "adv_jepa_raw": r2j - r2r})
            print(f"{alpha:>6.2f}{seed:>5d}{om:>7.3f}{mi:>6.2f}{ai:>7.3f} | "
                  f"{r2j:>7.3f}{r2m:>7.3f}{r2r:>7.3f} | {r2j - r2m:>7.3f}{r2j - r2r:>7.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[alignment] wrote {args.out}")

    # ---- verdict -------------------------------------------------------------------------------
    def agg(a, key):
        v = [r[key] for r in rows if r["alpha"] == a]
        return float(np.mean(v)), float(np.std(v))

    print(f"\n{'alpha':>6s}{'Omega':>8s}{'align':>8s}{'JEPA-MAE (mean+-sd)':>24s}")
    for a in alphas:
        om_m, _ = agg(a, "omega"); ai_m, _ = agg(a, "align_index")
        d_m, d_s = agg(a, "adv_jepa_mae")
        print(f"{a:>6.2f}{om_m:>8.3f}{ai_m:>8.3f}{d_m:>17.3f} +-{d_s:>5.3f}")

    om_all = np.array([r["omega"] for r in rows])
    om_spread = float(om_all.max() - om_all.min())
    adv = np.array([r["adv_jepa_mae"] for r in rows])
    al = np.array([r["alpha"] for r in rows])
    ai_all = np.array([r["align_index"] for r in rows])
    c_alpha = float(np.corrcoef(al, adv)[0, 1])
    c_omega = float(np.corrcoef(om_all, adv)[0, 1]) if om_spread > 1e-6 else float("nan")
    c_index = float(np.corrcoef(ai_all, adv)[0, 1])

    def _corr_p(r, n):
        """Two-sided p for a Pearson r: t = r*sqrt(n-2)/sqrt(1-r^2), normal approximation on t."""
        if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
            return float("nan")
        t = abs(r) * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r * r))
        norm_cdf = 0.5 * (1.0 + np.tanh(0.7988 * t * (1 + 0.04417 * t * t)))
        return float(2 * (1 - norm_cdf))

    n = len(rows)
    p_alpha, p_index = _corr_p(c_alpha, n), _corr_p(c_index, n)
    d1, s1 = agg(alphas[0], "adv_jepa_mae")
    d0, s0 = agg(alphas[-1], "adv_jepa_mae")
    jepa_beats_raw = float(np.mean([r["adv_jepa_raw"] > 0 for r in rows]))

    print(f"\n[design check] Omega spread across alpha = {om_spread:.4f} "
          f"({'OK — input predictability held fixed' if om_spread < 0.05 else 'BROKEN — Omega moved with alpha; result VOID'})")
    print(f"[sensitivity] fraction of cells where JEPA beats the RAW floor = {jepa_beats_raw:.2f}")
    print(f"[verdict] corr(alpha, JEPA-vs-MAE advantage) = {c_alpha:+.3f}  (p={p_alpha:.3f}, n={n})")
    print(f"[verdict] corr(alignment_index, advantage)   = {c_index:+.3f}  (p={p_index:.3f})")
    print(f"[verdict] advantage at alpha={alphas[0]:.2f}: {d1:+.3f}+-{s1:.3f}   "
          f"at alpha={alphas[-1]:.2f}: {d0:+.3f}+-{s0:.3f}")

    # A trend is only interpretable if the instrument can resolve anything at all: if the RAW floor
    # beats every learned encoder, JEPA-vs-MAE is a contrast between two encoders that both failed,
    # and any correlation across alpha is noise. Gate on sensitivity BEFORE reading the trend.
    if jepa_beats_raw < 0.25:
        print("  => INCONCLUSIVE (instrument insensitive): the raw-feature floor beats the learned"
              "\n     encoders in >=75% of cells, so no encoder-vs-encoder trend is interpretable."
              "\n     Fix the testbed (harder observation map / more capacity+epochs) before ruling"
              "\n     on H1 vs H2. Reporting a verdict from these numbers would be unsound.")
    elif abs(d1 - d0) < max(s1, s0):
        print("  => INCONCLUSIVE: the alpha=1 vs alpha=0 gap is smaller than its own seed variance.")
    elif c_alpha > 0.3 and p_alpha < 0.05 and d0 < d1:
        print("  => H2 SUPPORTED: benefit tracks ALIGNMENT at fixed predictability."
              "\n     'Predictability' alone is INSUFFICIENT — the current thesis is incomplete.")
        if d0 < 0:
            print("     Temporal JEPA is HARMFUL (loses to MAE) when the label needs the "
                  "unpredictable component, despite high measured predictability.")
    else:
        print("  => H2 NOT supported at this sensitivity: no significant alignment trend"
              f" (p={p_alpha:.3f}). H1 survives THIS test; absence of evidence only.")


if __name__ == "__main__":
    main()
