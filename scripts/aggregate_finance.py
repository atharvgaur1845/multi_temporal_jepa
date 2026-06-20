"""Aggregate finance matrix CSVs into the headline comparison table + a per-task verdict.

Reads runs/finance_results*.csv (globs multi-seed runs), prints, for each downstream task, every
method's score (mean ± std over seeds) and marks the winner, then summarizes whether Temporal JEPA
beats Spatial JEPA and the MAE/BYOL/SimCLR baselines — the markets analogue of scripts/aggregate.py.

Higher-is-better for every reported metric except none (vol/forecast use R^2 / IC / accuracy, all
higher-better; anomaly uses AUROC/AP higher-better).
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict

TASKS = [
    ("Regime classif.", "regime_acc", "accuracy"),
    ("Regime classif.", "regime_f1", "macro-F1"),
    ("Volatility pred.", "vol_r2", "R^2"),
    ("Volatility pred.", "vol_ic", "rank-IC"),
    ("Anomaly detect.", "anom_auroc", "AUROC"),
    ("Anomaly detect.", "anom_ap", "avg-prec"),
    ("Clustering", "clust_nmi", "NMI"),
    ("Clustering", "clust_ari", "ARI"),
    ("Forecasting", "fcast_dir_acc", "dir-acc"),
    ("Forecasting", "fcast_ret_ic", "ret-IC"),
]
ORDER = ["tjepa_h1", "spatial_jepa", "mae", "byol", "simclr", "random",
         "tjepa_h5", "tjepa_h20", "tjepa_noreg"]


def _fmt(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "  n/a  ", float("-inf")
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return f"{mean:.3f}±{sd:.2f}", mean
    return f"{mean:.3f} ", mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/finance_results*.csv")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"no CSVs match {args.glob}")
        return
    # rows[cell][metric] = list of values across seeds/files
    rows = defaultdict(lambda: defaultdict(list))
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                for _, key, _ in TASKS:
                    try:
                        rows[r["cell"]][key].append(float(r[key]))
                    except (KeyError, ValueError):
                        pass
    cells = [c for c in ORDER if c in rows] + [c for c in rows if c not in ORDER]
    print(f"Aggregated {len(files)} file(s): {', '.join(files)}\n")

    header = f"{'task / metric':24s} " + " ".join(f"{c[:11]:>12s}" for c in cells)
    print(header)
    print("-" * len(header))
    win = defaultdict(int)
    competitors = [c for c in cells if c not in ("random",)]
    for task, key, unit in TASKS:
        means = {}
        cellstr = {}
        for c in cells:
            s, m = _fmt(rows[c].get(key, []))
            cellstr[c], means[c] = s, m
        best = max((c for c in competitors if c in means), key=lambda c: means[c], default=None)
        label = f"{task} {unit}"
        line = f"{label:24s} "
        for c in cells:
            mark = "*" if c == best else " "
            line += f"{cellstr[c]:>11s}{mark}"
        print(line)
        if best:
            win[best] += 1
    print("-" * len(header))
    print("'*' = best among trained methods (random control excluded).  wins/task:",
          ", ".join(f"{c}={win[c]}" for c in cells if win[c]))

    # verdict: temporal vs each peer, averaged over the metrics it should win
    print("\nVerdict (Temporal JEPA tjepa_h1 vs peers), per-task winner count above.")
    t = rows.get("tjepa_h1", {})
    for peer in ("spatial_jepa", "mae", "byol", "simclr"):
        if peer not in rows:
            continue
        beats = sum(1 for _, key, _ in TASKS
                    if _fmt(t.get(key, []))[1] > _fmt(rows[peer].get(key, []))[1])
        print(f"  tjepa_h1 beats {peer:13s} on {beats}/{len(TASKS)} metrics")


if __name__ == "__main__":
    main()
