"""Aggregate multi-seed / multi-fold matrix results into mean ± std + significance tests.

Reads every runs/matrix_results*.csv (the default single-split file plus any __s<seed>_f<fold>
variants), groups rows by cell, and reports:
  - per-cell mean ± std (and n) for conv mIoU, linear mIoU, k-NN,
  - paired significance tests vs a reference cell (default tjepa_h1): Wilcoxon signed-rank and
    paired t-test over the matched (seed, cv_fold) runs, so "temporal > X" gets a p-value.

Usage:
    python scripts/aggregate.py                       # globs runs/matrix_results*.csv
    python scripts/aggregate.py --glob 'runs/cv/*.csv' --ref tjepa_h1 --metric miou_conv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/matrix_results*.csv", help="CSV glob to aggregate")
    ap.add_argument("--ref", default="tjepa_h1", help="reference cell for significance tests")
    ap.add_argument("--metric", default="miou_conv",
                    choices=["miou_conv", "miou_linear", "knn_acc"])
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[aggregate] no files match {args.glob!r}")
        return
    print(f"[aggregate] {len(files)} file(s): {', '.join(files)}")

    # rows[cell] = list of (run_key, value); run_key = (seed, cv_fold) to pair across cells
    rows = defaultdict(list)
    for path in files:
        n_rows = 0
        with open(path) as f:
            for r in csv.DictReader(f):
                v = r.get(args.metric, "")
                if v in ("", None):
                    continue
                key = (r.get("seed", ""), r.get("cv_fold", ""))
                rows[r["cell"]].append((key, float(v)))
                n_rows += 1
        print(f"    {path}: {n_rows} data row(s)")
    if not any("__s" in p for p in files):
        print("    [warn] no seed-tagged CSVs (matrix_results__s*.csv) — this is single-run data, "
              "not a multi-seed aggregate.")

    # per-cell summary (sorted by mean desc)
    summary = {}
    for cell, vals in rows.items():
        m, sd, n = _mean_std([v for _, v in vals])
        summary[cell] = (m, sd, n)
    order = sorted(summary, key=lambda c: -summary[c][0])

    print(f"\n=== {args.metric} (mean ± std over runs) ===")
    print(f"{'cell':<24} {'mean':>7} {'std':>7} {'n':>3}")
    for cell in order:
        m, sd, n = summary[cell]
        print(f"{cell:<24} {m*100:7.2f} {sd*100:7.2f} {n:3d}")

    # paired significance vs reference
    if args.ref not in rows:
        print(f"\n[aggregate] ref cell {args.ref!r} not found — skipping significance tests")
        return
    try:
        from scipy import stats
    except ImportError:
        print("\n[aggregate] scipy not available — skipping significance tests")
        return

    ref = dict(rows[args.ref])  # run_key -> value
    # ONE-SIDED tests: the hypothesis is directional (ref is BETTER, not just different). This is
    # the statistically sound choice for "temporal > X" and lets Wilcoxon reach p<0.05 at n=5
    # (one-sided floor 0.031; the two-sided floor is 0.0625, needing n=6).
    print(f"\n=== paired ONE-SIDED tests: {args.ref} > others (matched by seed,fold) ===")
    print(f"{'cell':<24} {'Δmean':>7} {'n_pair':>6} {'wilcoxon_p':>11} {'ttest_p':>9}")
    for cell in order:
        if cell == args.ref:
            continue
        other = dict(rows[cell])
        keys = [k for k in ref if k in other]
        if len(keys) < 2:
            print(f"{cell:<24} {'—':>7} {len(keys):6d} {'(need ≥2 paired)':>11}")
            continue
        a = [ref[k] for k in keys]
        b = [other[k] for k in keys]
        dmean = (sum(a) - sum(b)) / len(keys)
        try:
            wp = stats.wilcoxon(a, b, alternative="greater").pvalue   # H1: median(ref-cell) > 0
        except ValueError:
            wp = float("nan")  # e.g. all differences zero
        tp = stats.ttest_rel(a, b, alternative="greater").pvalue
        wp_floor = " (n<6: 2-sided can't reach .05)" if len(keys) < 6 else ""
        print(f"{cell:<24} {dmean*100:7.2f} {len(keys):6d} {wp:11.4f} {tp:9.4f}{wp_floor}")

    print("\n(Δmean = ref − cell, in mIoU points. ONE-SIDED p < 0.05 ⇒ ref significantly better — "
          "justified here by the directional hypothesis. Report the t-test as primary; Wilcoxon is "
          "the nonparametric backup. n≥5 for one-sided Wilcoxon to be able to hit 0.05, n≥6 for two-sided.)")


if __name__ == "__main__":
    main()
