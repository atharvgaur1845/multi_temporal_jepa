#!/usr/bin/env python3
"""After C2 lands: does the paper's headline still hold?

Recomputes the three-way binding-floor table from whatever is currently in runs/,
using ONLY config-matched cells, and reports whether each claim in the paper
survives. Run this before touching main.tex.

    python REO-2/check_headline.py
"""
from __future__ import annotations
import csv, glob, os, statistics as st, math, sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
METRICS = [("miou_conv", "conv mIoU"), ("miou_linear", "linear mIoU"), ("knn_acc", "parcel k-NN")]
FLOORS = ("raw_features", "random")


def load():
    """cell -> metric -> [values], from the seed-tagged CSVs only (matched config)."""
    v = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "matrix_results__s[0-9].csv"))):
        with open(f) as fh:
            if not fh.readline().startswith("cell,"):
                print(f"  !! {os.path.basename(f)} has no header, SKIPPED", file=sys.stderr); continue
            fh.seek(0)
            for r in csv.DictReader(fh):
                for m, _ in METRICS:
                    if r[m] not in ("", None):
                        v.setdefault(r["cell"], {}).setdefault(m, []).append(float(r[m]) * 100)
    return v


def main():
    v = load()
    cells = sorted(v)
    print(f"cells with matched-config seeds: {cells}\n")
    for m, label in METRICS:
        have = {c: v[c][m] for c in cells if m in v[c]}
        if not all(f in have for f in FLOORS):
            print(f"{label}: floors missing, skipped"); continue
        fl = {f: st.mean(have[f]) for f in FLOORS}
        binder = max(fl, key=fl.get); bind = fl[binder]
        # is the floor ordering actually separated?
        a, b = have["random"], have["raw_features"]
        d = [x - y for x, y in zip(a, b)]
        md, sdd = st.mean(d), (st.stdev(d) if len(d) > 1 else float("nan"))
        sep = "separated" if abs(md) > sdd else "NOT separated"
        print(f"{label}: binding floor = {binder} {bind:.2f}   "
              f"(raw {fl['raw_features']:.2f} / random {fl['random']:.2f}; {sep})")
        clears, fails = [], []
        for c in sorted(have):
            if c in FLOORS: continue
            mu = st.mean(have[c]); n = len(have[c])
            (clears if mu > bind else fails).append(f"{c} {mu:.2f} (n={n})")
        print(f"    clears : {', '.join(clears) or 'none'}")
        print(f"    below  : {', '.join(fails) or 'none'}\n")

    # the paper's specific claims
    print("=" * 62)
    conv = {c: st.mean(v[c]["miou_conv"]) for c in cells if "miou_conv" in v[c]}
    raw = conv["raw_features"]
    learned = [c for c in conv if c not in FLOORS]
    failing = [c for c in learned if conv[c] < raw]
    print(f"CLAIM 'four of five fail on at least one readout':")
    print(f"  learned objectives with matched seeds: {len(learned)} -> {sorted(learned)}")
    print(f"  below the conv raw floor ({raw:.2f}): {len(failing)} -> {sorted(failing)}")
    t = conv.get("tjepa_h1")
    if t is not None:
        sd = st.stdev(v["tjepa_h1"]["miou_conv"])
        hi = max(raw, conv["random"])
        ok = t > conv["random"] and t > raw and (t - hi) > sd
        print(f"\nPRE-REGISTERED RULE P1: {'STILL PASSES' if ok else '*** NO LONGER PASSES ***'}")
        print(f"  tjepa {t:.2f} vs random {conv['random']:.2f} / raw {raw:.2f}; "
              f"margin {t-hi:.2f} vs sd {sd:.2f}")


if __name__ == "__main__":
    main()
