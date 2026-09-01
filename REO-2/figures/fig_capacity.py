"""Figure: the binding floor, and the margin over it, as functions of probe receptive field.

Left  : mIoU vs RF for the two floors and temporal JEPA. The raw-band floor is not a constant.
Right : paired per-seed margin (temporal JEPA minus raw floor) with a +-1 sd band and a zero line.

RF 1 and RF 3 are the published linear and conv columns (the probe heads are structurally
identical, see eval/linear_probe._build_head); RF 5 and RF 9 come from the capacity sweep.
"""
from __future__ import annotations

import csv, os, statistics as st, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWEEP = os.path.join(ROOT, "REO-2/evidence/probe_capacity.csv")
LONG = os.path.join(ROOT, "REO-2/evidence/pastis_runs_long.csv")
OUT = os.path.join(ROOT, "REO-2/paper/figures/fig_capacity.pdf")

CELLS = {"raw_features": ("raw features (floor)", "#c44e52", "o"),
         "random": ("random init (floor)", "#8172b2", "s"),
         "tjepa_h1": ("temporal JEPA", "#4c72b0", "D")}


def load():
    d = {}
    for r in csv.DictReader(open(SWEEP)):
        d.setdefault(r["cell"], {}).setdefault(int(r["receptive_field"]), {})[int(r["seed"])] = \
            float(r["miou"]) * 100
    for r in csv.DictReader(open(LONG)):
        if r["comparable_group"] not in ("multiseed", "floor"):
            continue
        c, s = r["cell"], int(r["seed"])
        d.setdefault(c, {}).setdefault(1, {})[s] = float(r["miou_linear"]) * 100
        d.setdefault(c, {}).setdefault(3, {})[s] = float(r["miou_conv"]) * 100
    return d


def main():
    d = load()
    rfs = [1, 3, 5, 9]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 1.72))

    for cell, (label, colour, mk) in CELLS.items():
        xs, ys, es = [], [], []
        for rf in rfs:
            v = d.get(cell, {}).get(rf, {})
            if len(v) < 2:
                continue
            xs.append(rf); ys.append(st.mean(v.values() if False else list(v.values())))
            es.append(st.stdev(list(v.values())))
        ax.errorbar(xs, ys, yerr=es, marker=mk, color=colour, label=label,
                    capsize=2.5, lw=1.5, ms=4)
    ax.set_xscale("log"); ax.set_xticks(rfs); ax.set_xticklabels([str(r) for r in rfs])
    ax.set_xlabel("probe receptive field"); ax.set_ylabel("mIoU")
    ax.set_title("Floors move with probe capacity", fontsize=9)
    # headroom so the legend does not sit on the RF 1 markers
    ax.set_ylim(top=ax.get_ylim()[1] + 7.5)
    ax.legend(fontsize=6.5, loc="upper left", frameon=True, framealpha=0.85,
              edgecolor="none", borderpad=0.3)
    ax.grid(alpha=0.25, lw=0.5)

    # paired margin, temporal JEPA minus the raw-band floor, per seed
    xs, gm, gs = [], [], []
    for rf in rfs:
        t, r = d["tjepa_h1"].get(rf, {}), d["raw_features"].get(rf, {})
        ss = sorted(set(t) & set(r))
        if len(ss) < 2:
            continue
        g = [t[s] - r[s] for s in ss]
        xs.append(rf); gm.append(st.mean(g)); gs.append(st.stdev(g))
    ax2.axhline(0, color="k", lw=0.9, ls="--")
    ax2.errorbar(xs, gm, yerr=gs, marker="D", color="#4c72b0", capsize=2.5, lw=1.5, ms=4)
    ax2.fill_between(xs, [a - b for a, b in zip(gm, gs)], [a + b for a, b in zip(gm, gs)],
                     color="#4c72b0", alpha=0.15)
    ax2.set_xscale("log"); ax2.set_xticks(rfs); ax2.set_xticklabels([str(r) for r in rfs])
    ax2.set_xlabel("probe receptive field")
    # a long rotated label is clipped at this panel height; the caption defines the quantity
    ax2.set_ylabel("margin (mIoU)")
    ax2.set_title("The margin decays to zero", fontsize=9)
    ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)
    for rf, m, s in zip(xs, gm, gs):
        print(f"  RF{rf}: margin {m:+.2f} +- {s:.2f}  ({m/s if s else float('inf'):+.2f} sd)")


if __name__ == "__main__":
    main()
