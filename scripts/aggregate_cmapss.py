"""Aggregate C-MAPSS matrix CSVs into per-FD comparison tables + a per-task verdict.

Like scripts/aggregate_finance.py but (a) grouped by FD subset and (b) DIRECTION-AWARE: RUL RMSE and
the PHM08 score are lower-better, everything else higher-better. Marks the best TRAINED method per
metric (excluding the random / raw-feature floors) and counts how many tasks Temporal JEPA wins vs
each peer and vs the two floors.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict

# (label, csv_key, unit, higher_is_better)
TASKS = [
    ("RUL", "rul_r2", "R^2", True),
    ("RUL", "rul_rmse", "RMSE(win)", False),
    ("RUL", "rul_ic", "rank-IC", True),
    ("RUL", "rul_std_rmse", "RMSE(std)", False),
    ("RUL", "rul_phm08", "PHM08", False),
    ("Health", "health_acc", "accuracy", True),
    ("Health", "health_f1", "macro-F1", True),
    ("Anomaly", "anom_auroc", "AUROC", True),
    ("Anomaly", "anom_ap", "avg-prec", True),
    ("Cluster", "clust_nmi", "NMI", True),
    ("Cluster", "clust_ari", "ARI", True),
    ("Retrieval", "retr_health_prec", "health-p@k", True),
    ("Retrieval", "retr_rul_ic", "RUL-IC", True),
]
ORDER = ["tjepa_h1", "spatial_jepa", "mae", "byol", "simclr", "random", "raw_features",
         "tjepa_h5", "tjepa_h20", "tjepa_noreg"]
FLOORS = {"random", "raw_features"}


def _agg(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "  n/a ", None
    m = sum(vals) / len(vals)
    if len(vals) > 1:
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        return f"{m:.3f}±{sd:.2f}", m
    return f"{m:.3f}", m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/cmapss_results*.csv")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"no CSVs match {args.glob}")
        return
    # data[fd][cell][key] = [values across seeds]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                for _, key, _, _ in TASKS:
                    try:
                        data[r["fd"]][r["cell"]][key].append(float(r[key]))
                    except (KeyError, ValueError):
                        pass
    print(f"Aggregated {len(files)} file(s): {', '.join(files)}")

    overall_beats = defaultdict(lambda: defaultdict(int))
    overall_n = defaultdict(int)
    for fd in sorted(data):
        rows = data[fd]
        cells = [c for c in ORDER if c in rows] + [c for c in rows if c not in ORDER]
        print(f"\n===== {fd} =====")
        header = f"{'task / metric':22s} " + " ".join(f"{c[:11]:>12s}" for c in cells)
        print(header); print("-" * len(header))
        win = defaultdict(int)
        competitors = [c for c in cells if c not in FLOORS]
        for label, key, unit, hib in TASKS:
            means, cellstr = {}, {}
            for c in cells:
                s, m = _agg(rows[c].get(key, []))
                cellstr[c], means[c] = s, m
            valid = {c: means[c] for c in competitors if means.get(c) is not None}
            best = (max(valid, key=valid.get) if hib else min(valid, key=valid.get)) if valid else None
            line = f"{label+' '+unit:22s} "
            for c in cells:
                mark = "*" if c == best else " "
                line += f"{cellstr[c]:>11s}{mark}"
            print(line)
            if best:
                win[best] += 1
        print("-" * len(header))
        print("'*' = best TRAINED method (floors excluded).  wins/task:",
              ", ".join(f"{c}={win[c]}" for c in cells if win[c]))

        # verdict for this FD: tjepa_h1 vs each peer/floor, direction-aware
        t = rows.get("tjepa_h1", {})
        for peer in ("spatial_jepa", "mae", "byol", "simclr", "random", "raw_features"):
            if peer not in rows:
                continue
            beats = 0
            for _, key, _, hib in TASKS:
                _, tm = _agg(t.get(key, []))
                _, pm = _agg(rows[peer].get(key, []))
                if tm is None or pm is None:
                    continue
                if (tm > pm) if hib else (tm < pm):
                    beats += 1
                overall_n[peer]  # touch
            overall_beats["tjepa_h1"][peer] += beats
            print(f"  tjepa_h1 beats {peer:13s} on {beats}/{len(TASKS)} metrics")

    if len(data) > 1:
        print("\n===== OVERALL (summed across FD subsets) =====")
        for peer, c in overall_beats["tjepa_h1"].items():
            print(f"  tjepa_h1 beats {peer:13s} on {c}/{len(TASKS)*len(data)} metric-subsets")


if __name__ == "__main__":
    main()
