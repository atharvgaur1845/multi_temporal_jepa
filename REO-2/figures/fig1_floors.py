#!/usr/bin/env python3
"""Figure 1 — which floor binds depends on the readout.

Three panels, one per probe. Each shows the five objectives against the two trivial
floors. The BINDING floor (the larger of the two) is drawn solid; the other dashed.
Bars are dark where the objective clears the binding floor, light where it does not.

All values are read from the committed CSVs. Nothing is hard-coded.
    python REO-2/figures/fig1_floors.py
"""
from __future__ import annotations
import csv, glob, os, statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")

READOUTS = [("miou_conv", "conv mIoU", 100), ("miou_linear", "linear mIoU", 100),
            ("knn_acc", "parcel $k$-NN accuracy", 100)]
ORDER = ["tjepa_h1", "spatial_jepa", "byol", "simclr", "mae"]
LABEL = {"tjepa_h1": "Temporal JEPA", "spatial_jepa": "Spatial JEPA",
         "byol": "BYOL", "simclr": "SimCLR", "mae": "MAE"}


def load():
    multi, single = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "matrix_results__s[0-9].csv"))):
        for r in csv.DictReader(open(f)):
            multi.setdefault(r["cell"], []).append(r)
    for r in csv.DictReader(open(os.path.join(ROOT, "runs", "matrix_results.csv"))):
        if r["cell"] in ("mae", "byol", "simclr"):
            single[r["cell"]] = r
    return multi, single


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    multi, single = load()
    stat = lambda c, k, s: [float(r[k]) * s for r in multi[c]]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85))
    CLEAR, BELOW = "#1f4e79", "#c8cdd4"
    F_BIND, F_OTHER = "#b03a2e", "#9aa4ae"

    for ax, (key, title, sc) in zip(axes, READOUTS):
        raw, rnd = st.mean(stat("raw_features", key, sc)), st.mean(stat("random", key, sc))
        bind = max(raw, rnd)
        bind_name = "raw features" if raw > rnd else "random init"
        other, other_name = min(raw, rnd), ("random init" if raw > rnd else "raw features")

        vals, errs = [], []
        for c in ORDER:
            if c in multi:
                x = stat(c, key, sc); vals.append(st.mean(x)); errs.append(st.stdev(x))
            else:
                vals.append(float(single[c][key]) * sc); errs.append(0.0)

        y = list(range(len(ORDER)))
        # Dot plot, not bars: the interesting action sits within a few points of the
        # floors, and a dot plot carries no zero-baseline obligation, so each panel can
        # be scaled to its own range without misleading anyone.
        lo, hi_v = min(vals + [raw, rnd]), max(vals + [raw, rnd])
        pad = (hi_v - lo) * 0.16
        for yi, (vv, ee) in enumerate(zip(vals, errs)):
            col = CLEAR if vv > bind else BELOW
            ax.plot([lo - pad, vv], [yi, yi], color="#dfe3e8", lw=0.8, zorder=2)
            if ee:
                ax.errorbar(vv, yi, xerr=ee, fmt="none", ecolor="#5a6470",
                            elinewidth=0.9, capsize=2.2, zorder=4)
            ax.plot(vv, yi, "o", ms=5.2, color=col, mec="#2b2b2b", mew=0.5, zorder=5)
        ax.axvline(bind, color=F_BIND, lw=1.4, zorder=3)
        ax.axvline(other, color=F_OTHER, lw=1.0, ls=(0, (4, 2)), zorder=3)
        ax.set_xlim(lo - pad, hi_v + pad * 1.15)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8, pad=12)
        ax.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#eef0f2", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        # Offset each label away from its own line so the rule does not strike the text.
        span = (hi_v + pad * 1.15) - (lo - pad)
        dx = span * 0.02
        b_ha, b_dx = ("left", dx) if bind < other else ("right", -dx)
        o_ha, o_dx = ("right", -dx) if bind < other else ("left", dx)
        ax.text(bind + b_dx, -1.05, f"{bind_name} {bind:.2f}", color=F_BIND, fontsize=6.3,
                ha=b_ha, va="bottom", fontweight="bold")
        ax.text(other + o_dx, -0.5, f"{other_name} {other:.2f}", color="#6b7480",
                fontsize=6.0, ha=o_ha, va="bottom")
        ax.set_ylim(len(ORDER) - 0.45, -1.35)

    for ax, lab in zip(axes, ("mIoU", "mIoU", "accuracy (%)")):
        ax.set_xlabel(lab, fontsize=7)
    fig.tight_layout(pad=0.5, w_pad=1.4)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "fig1_floors.pdf")
    fig.savefig(p, bbox_inches="tight")
    print(f"[fig1] wrote {p}")
    for ax, (key, title, sc) in zip(axes, READOUTS):
        raw, rnd = st.mean(stat("raw_features", key, sc)), st.mean(stat("random", key, sc))
        bind = max(raw, rnd)
        cl = [LABEL[c] for c in ORDER
              if (st.mean(stat(c, key, sc)) if c in multi else float(single[c][key]) * sc) > bind]
        print(f"[fig1] {title:<18} binding floor {bind:6.2f} -> clears: {', '.join(cl)}")


if __name__ == "__main__":
    main()
