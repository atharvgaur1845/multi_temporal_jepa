#!/usr/bin/env python3
"""Fig 2 — label efficiency, temporal vs spatial JEPA at 1 / 5 / 10 / 100% of labels.

`scripts/evaluate.py --fewshot` PRINTS its results and writes no file (verified at
scripts/evaluate.py:138-150), so the few-shot numbers quoted in the parent README have no artifact
behind them in this checkout. Regenerate them and tee the output:

    python scripts/evaluate.py --encoder-ckpt runs/matrix/tjepa_h1.pt --head conv --fewshot \
        --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml 2>&1 | tee fs_temporal.log
    python scripts/evaluate.py --encoder-ckpt runs/matrix/spatial_jepa.pt --head conv --fewshot \
        --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml 2>&1 | tee fs_spatial.log

then

    python fig2_label_efficiency.py --temporal fs_temporal.log --spatial fs_spatial.log \
        --full-temporal <conv mIoU from the matrix CSV> --full-spatial <same>

The 100% point is the ordinary matrix result, not a few-shot run — pass it explicitly rather than
letting the script guess, so the figure cannot silently mix protocols.

Parsed line format (scripts/evaluate.py:150):
    [evaluate]     1% labels (n=37) -> mIoU 9.20
"""
from __future__ import annotations

import argparse
import os
import re
import sys

FS_RE = re.compile(r"\[evaluate\]\s+(\d+)%\s+labels\s+\(n=(\d+)\)\s*->\s*mIoU\s+([0-9.]+)")


def parse_fewshot(path):
    """-> {percent: miou}. Later occurrences win (a re-run in the same log supersedes)."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = FS_RE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(3))
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", required=True, help="log from evaluate.py --fewshot, temporal JEPA")
    ap.add_argument("--spatial", required=True, help="same, spatial JEPA")
    ap.add_argument("--full-temporal", type=float, default=None,
                    help="100%%-label conv mIoU for temporal, from the matrix CSV")
    ap.add_argument("--full-spatial", type=float, default=None,
                    help="100%%-label conv mIoU for spatial, from the matrix CSV")
    ap.add_argument("--out", default="../paper/figures/fig2_label_efficiency.pdf")
    args = ap.parse_args()

    t, s = parse_fewshot(args.temporal), parse_fewshot(args.spatial)
    missing = [n for n, v in (("temporal", t), ("spatial", s)) if v is None]
    if missing:
        print(f"[fig2] CANNOT DRAW — no few-shot lines parsed for: {', '.join(missing)}",
              file=sys.stderr)
        print("[fig2] The few-shot numbers in the parent README are stdout-only and were never\n"
              "[fig2] written to a file. Re-run scripts/evaluate.py --fewshot and tee the output;\n"
              "[fig2] see this script's docstring. Refusing to plot numbers with no artifact.",
              file=sys.stderr)
        return 1

    if args.full_temporal is not None:
        t[100] = args.full_temporal
    if args.full_spatial is not None:
        s[100] = args.full_spatial

    xs = sorted(set(t) & set(s))
    if not xs:
        print("[fig2] CANNOT DRAW — no label fraction present in BOTH runs", file=sys.stderr)
        return 1
    dropped = sorted((set(t) | set(s)) - set(xs))
    if dropped:
        print(f"[fig2] warning: fractions present in only one run, dropped: {dropped}",
              file=sys.stderr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(xs, [t[x] for x in xs], "o-", label="temporal JEPA", linewidth=1.6, markersize=4)
    ax.plot(xs, [s[x] for x in xs], "s--", label="spatial JEPA", linewidth=1.6, markersize=4)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.set_xlabel("labelled fraction of training folds")
    ax.set_ylabel("conv mIoU")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[fig2] wrote {args.out}")
    for x in xs:
        rel = (t[x] - s[x]) / s[x] * 100 if s[x] else float("nan")
        print(f"[fig2]  {x:>4}%  temporal {t[x]:6.2f}  spatial {s[x]:6.2f}  "
              f"gap {t[x]-s[x]:+6.2f} ({rel:+.0f}% rel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
