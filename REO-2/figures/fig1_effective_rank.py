#!/usr/bin/env python3
"""Fig 1 — effective rank of the context embedding during pretraining, VICReg on vs off.

The designated money figure. It reads the `effrank` traces that `engine/train_jepa` already prints:

    step 0 loss 1.9986 lr 0.00e+00 wd 0.040 std 0.343 effrank 15.2 varratio 0.055

A log holds many cells back to back. Cells are segmented on `step 0` and named from the
`[run_matrix] <cell>: linear=... conv=...` summary line that follows the segment.

THIS SCRIPT DOES NOT INVENT THE MISSING SERIES. If the VICReg-off run is absent it says so and
exits non-zero, because a two-curve figure drawn from one curve is a fabricated result.
See ../protocol/P0_floors.md (addendum) for the run that produces it.

Usage:
    python fig1_effective_rank.py --logs ../evidence/logs/run_s0.log ../../run.log
    python fig1_effective_rank.py --logs ... --list          # just show which cells are present
"""
from __future__ import annotations

import argparse
import os
import re
import sys

STEP_RE = re.compile(r"^\s*step\s+(\d+)\s+loss\s+\S+.*?\beffrank\s+([0-9.]+)")
CELL_RE = re.compile(r"\[run_matrix\]\s+([A-Za-z0-9_.]+):\s+linear=")

ON_CELL = "tjepa_h1"        # VICReg on  (var_coeff 1.0, cov_coeff 0.04)
OFF_CELL = "tjepa_noreg"    # VICReg off (var_coeff 0.0, cov_coeff 0.0) -- pure I-JEPA


def parse_log(path):
    """-> {cell_name: [(step, effrank), ...]}. Unnamed trailing segments keep a positional name."""
    segments, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                step, erank = int(m.group(1)), float(m.group(2))
                if step == 0 or cur is None:
                    cur = {"name": None, "pts": []}
                    segments.append(cur)
                cur["pts"].append((step, erank))
                continue
            c = CELL_RE.search(line)
            if c and cur is not None and cur["name"] is None:
                cur["name"] = c.group(1)
                cur = None  # this segment is closed; the next `step 0` opens a new one

    out = {}
    for i, seg in enumerate(segments):
        if not seg["pts"]:
            continue
        name = seg["name"] or f"<unnamed#{i}>"
        # keep the longest trace when a cell appears more than once across logs
        if name not in out or len(seg["pts"]) > len(out[name]):
            out[name] = seg["pts"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="training logs to parse")
    ap.add_argument("--on", default=ON_CELL, help=f"VICReg-on cell name (default {ON_CELL})")
    ap.add_argument("--off", default=OFF_CELL, help=f"VICReg-off cell name (default {OFF_CELL})")
    ap.add_argument("--out", default="../paper/figures/fig1_effective_rank.pdf")
    ap.add_argument("--list", action="store_true", help="list parsed cells and exit")
    args = ap.parse_args()

    cells = {}
    for p in args.logs:
        if not os.path.exists(p):
            print(f"[fig1] missing log: {p}", file=sys.stderr)
            continue
        for k, v in parse_log(p).items():
            if k not in cells or len(v) > len(cells[k]):
                cells[k] = v

    if args.list or not cells:
        for k, v in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            steps = [s for s, _ in v]
            print(f"  {k:<28} {len(v):>5d} pts  step {min(steps)}..{max(steps)}  "
                  f"effrank {v[0][1]:.1f} -> {v[-1][1]:.1f}")
        if not cells:
            print("[fig1] no effrank traces found in the given logs", file=sys.stderr)
            return 2
        return 0

    missing = [n for n in (args.on, args.off) if n not in cells]
    if missing:
        print(f"[fig1] CANNOT DRAW — no effrank trace for: {', '.join(missing)}", file=sys.stderr)
        if args.off in missing:
            print("[fig1] `tjepa_noreg` was logged SKIPPED (budget) in every committed PASTIS pass.\n"
                  "[fig1] The VICReg-off collapse curve does not exist. Run it "
                  "(../protocol/P0_floors.md, addendum) or drop Fig 1.\n"
                  "[fig1] Refusing to draw a two-curve figure from one curve.", file=sys.stderr)
        print(f"[fig1] cells present: {sorted(cells)}", file=sys.stderr)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for name, label, style in ((args.on, "VICReg on", "-"), (args.off, "VICReg off", "--")):
        xs = [s for s, _ in cells[name]]
        ys = [e for _, e in cells[name]]
        ax.plot(xs, ys, style, label=label, linewidth=1.6)
    ax.set_xlabel("pretraining step")
    ax.set_ylabel("effective rank")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[fig1] wrote {args.out}")
    print(f"[fig1] {args.on}: {cells[args.on][0][1]:.1f} -> {cells[args.on][-1][1]:.1f}  "
          f"({len(cells[args.on])} pts)")
    print(f"[fig1] {args.off}: {cells[args.off][0][1]:.1f} -> {cells[args.off][-1][1]:.1f}  "
          f"({len(cells[args.off])} pts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
