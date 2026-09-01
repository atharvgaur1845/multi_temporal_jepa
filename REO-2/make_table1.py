#!/usr/bin/env python3
"""Emit the LaTeX body of Table 1 from the committed summary CSV.

Run after new cells land so the main table is regenerated rather than hand-edited.
Marks each cell that falls below its column's binding floor, and prints n per row.

    python REO-2/make_table1.py           # prints rows to paste into main.tex
"""
from __future__ import annotations
import csv, glob, os, statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
FLOORS = ("raw_features", "random")
NICE = {"tjepa_h1": r"\textbf{Temporal JEPA} ($\Delta{=}1$)", "spatial_jepa": "Spatial JEPA",
        "simclr": "SimCLR", "byol": "BYOL", "mae": "MAE",
        "raw_features": r"\emph{raw features} (floor)", "random": r"\emph{random init} (floor)"}
ORDER = ["tjepa_h1", None, "raw_features", "random", None,
         "spatial_jepa", "simclr", "byol", "mae"]


def load():
    v = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "matrix_results__s[0-9].csv"))):
        with open(f) as fh:
            if not fh.readline().startswith("cell,"): continue
            fh.seek(0)
            for r in csv.DictReader(fh):
                for m in ("miou_conv", "miou_linear", "knn_acc"):
                    if r[m] not in ("", None):
                        v.setdefault(r["cell"], {}).setdefault(m, []).append(float(r[m]) * 100)
    return v


def main():
    v = load()
    binding = {}
    for m in ("miou_conv", "miou_linear", "knn_acc"):
        binding[m] = max(st.mean(v[f][m]) for f in FLOORS if m in v.get(f, {}))
    print("% binding floors: " + ", ".join(f"{m}={binding[m]:.2f}" for m in binding))
    for cell in ORDER:
        if cell is None:
            print(r"    \midrule"); continue
        if cell not in v:
            print(f"    % {cell}: NO DATA"); continue
        cols = []
        for m in ("miou_conv", "miou_linear", "knn_acc"):
            if m not in v[cell]:
                cols.append(""); continue
            x = v[cell][m]; mu = st.mean(x)
            body = f"${mu:.2f} \\pm {st.stdev(x):.2f}$" if len(x) > 1 else f"${mu:.2f}$"
            if cell not in FLOORS and mu < binding[m]:
                body += r"\,$\downarrow$"
            if cell in FLOORS:
                body = body.replace("$", "").join(["$\\emph{", "}$"]) if False else \
                       f"\\emph{{{mu:.2f} $\\pm$ {st.stdev(x):.2f}}}"
            cols.append(body)
        n = max(len(v[cell][m]) for m in v[cell])
        print(f"    {NICE.get(cell, cell)} & " + " & ".join(cols) + f" & {n} & \\\\")


if __name__ == "__main__":
    main()
