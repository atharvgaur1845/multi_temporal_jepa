#!/usr/bin/env python3
"""Consolidate every PASTIS result into two CSVs the paper can be checked against.

  REO-2/evidence/pastis_runs_long.csv  — one row per (cell, run). Every raw number.
  REO-2/evidence/pastis_summary.csv    — one row per (cell, config). mean, sd, n.

Provenance is explicit because the runs are NOT all comparable: the multi-seed cells
used per-step batch 12, the single-seed baselines used 32, and the VICReg term is
computed per micro-batch — so effective batch 192 does NOT make them equivalent.
The two groups are tagged and never pooled. Floors never train, so they are exempt.
"""
from __future__ import annotations
import csv, glob, os, statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
FLOORS = {"random", "raw_features"}

def rows():
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "matrix_results*.csv"))):
        if f.endswith(".bak"):
            continue
        base = os.path.basename(f)
        multi = "__s" in base
        with open(f) as fh:
            head = fh.readline()
            if not head.startswith("cell,"):
                print(f"[warn] {base}: no header, skipped")   # the seed-1 failure mode
                continue
            fh.seek(0)
            for r in csv.DictReader(fh):
                r["source_file"] = base
                # Floors have no trained weights, so batch size cannot affect them.
                # tjepa_noreg was produced by REO-2/run_local.sh at batch 12, like the
                # multi-seed cells, even though it landed in the unsuffixed CSV (no --seed
                # was passed). Classify it by how it was RUN, not by which file it is in.
                laptop = multi or r["cell"] == "tjepa_noreg"
                r["batch_config"] = ("floor (untrained)" if r["cell"] in FLOORS
                                     else ("b12xa16" if laptop else "b32xa6"))
                r["comparable_group"] = ("floor" if r["cell"] in FLOORS
                                         else ("multiseed" if multi else
                                               ("ablation_n1" if r["cell"] == "tjepa_noreg"
                                                else "server_n1")))
                yield r

def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = list(rows())
    # De-duplicate by (cell, group, seed). The floors were probed once from the
    # unsuffixed file and again in the __s0 pass -- SAME SEED, so counting both is
    # pseudo-replication and inflates n from 3 to 4. Prefer the __s* rows, which are
    # the config the multi-seed cells use. (This is the same mistake that produced the
    # first "n=3" aggregate, where seed 0 appeared twice for tjepa_h1.)
    best = {}
    for r in all_rows:
        k = (r["cell"], r["comparable_group"], r["seed"])
        prefer = "__s" in r["source_file"]
        if k not in best or (prefer and "__s" not in best[k]["source_file"]):
            best[k] = r
    keep = list(best.values())
    dropped = len(all_rows) - len(keep)
    if dropped:
        print(f"[csv] dropped {dropped} duplicate (cell, group, seed) row(s)")

    cols = ["cell", "objective", "seed", "cv_fold", "eval_split", "miou_linear", "miou_conv",
            "knn_acc", "gpu_hours", "peak_mem_gb", "batch_config", "comparable_group",
            "source_file"]
    long_p = os.path.join(OUT, "pastis_runs_long.csv")
    with open(long_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(keep, key=lambda r: (r["cell"], r["comparable_group"], r["seed"])):
            w.writerow(r)
    print(f"[csv] {long_p}: {len(keep)} rows")

    # summary, grouped so incomparable configs never merge
    groups = {}
    for r in keep:
        groups.setdefault((r["cell"], r["comparable_group"], r["batch_config"]), []).append(r)
    sum_p = os.path.join(OUT, "pastis_summary.csv")
    METRICS = [("miou_conv", 100), ("miou_linear", 100), ("knn_acc", 100), ("gpu_hours", 1)]
    with open(sum_p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "comparable_group", "batch_config", "n"] +
                   [f"{m}_{s}" for m, _ in METRICS for s in ("mean", "sd")])
        for (cell, grp, bc), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            out = [cell, grp, bc, len(rs)]
            for m, scale in METRICS:
                # A cell may be missing a metric entirely (tjepa_noreg was run without
                # --knn, so it has no k-NN value). Emit blanks rather than a zero, which
                # would silently read as "scored 0" in the table.
                v = [float(r[m]) * scale for r in rs if r[m] not in ("", None)]
                if not v:
                    out += ["", ""]
                else:
                    out += [round(st.mean(v), 4), round(st.stdev(v), 4) if len(v) > 1 else ""]
            w.writerow(out)
    print(f"[csv] {sum_p}: {len(groups)} groups")

if __name__ == "__main__":
    main()
