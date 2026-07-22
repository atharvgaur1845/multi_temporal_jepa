"""Apply the validity criteria (eval/validity.py) to THIS PROJECT'S OWN published claims.

Reads the committed result CSVs under runs/ and asks, for each headline claim, whether the
experiment that produced it was ENTITLED to it — independent of whether the claim is true.

This is the paper's central move: the criteria are only credible if they are turned inward first,
and are allowed to downgrade our own results. They do.

    python scripts/audit_claims.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from eval.validity import (NA, Audit, Check, comparison_hygiene,  # noqa: E402
                           effect_vs_noise, manipulation_isolation, resolving_power)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def _read(name):
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def audit_pastis():
    rows = _read("matrix_results.csv")
    if not rows:
        return None
    by = {r["cell"]: r for r in rows}
    a = Audit("PASTIS: temporal JEPA (22.06 conv mIoU) beats spatial JEPA and MAE/BYOL/SimCLR")
    # V1: PASTIS matrix has NO random-init / raw-feature cell at all -> the floor is UNMEASURED.
    learned = {c: _f(by[c], "miou_conv") for c in by}
    floor = _f(by["random"], "miou_conv") if "random" in by else None
    a.add(resolving_power(learned, floor, focal="tjepa_h1"))
    # V2: seed-tagged CSVs (matrix_results__s*.csv) come from the SERVER rigor pass and are not in
    # this checkout. report.md documents 3 seeds + paired t-tests; we mark this UNVERIFIABLE here
    # rather than assert a single-seed failure we cannot substantiate from data present.
    seeded = [f for f in os.listdir(RUNS) if f.startswith("matrix_results__s")]
    if seeded:
        a.add(effect_vs_noise(_f(by["tjepa_h1"], "miou_conv") - _f(by["spatial_jepa"], "miou_conv"),
                              None, n_seeds=len(seeded)))
    else:
        a.add(Check("V2 effect vs noise", NA,
                    "seed-tagged CSVs absent from this checkout (server rigor pass); report.md "
                    "documents 3 seeds, paired t-test p=0.041 — not verifiable from local data"))
    a.add(comparison_hygiene({c: by[c]["eval_split"] for c in by}))
    return a


def audit_finance_harm():
    rows = _read("finance_results.csv")
    if not rows:
        return None
    by = {r["cell"]: r for r in rows}
    a = Audit("FINANCE: temporal JEPA is HARMFUL — it scores below its own random-init encoder")
    # This is a WITHIN-ARCHITECTURE comparison: trained vs untrained, same net. The floor is the
    # comparator itself, so V1 is about whether the probe can read anything at all.
    # The comparator IS the random-init net, so the question is whether the probe reads real
    # structure at all: random-init must clear chance, else "below random" is meaningless.
    a.add(resolving_power({"random-init (comparator)": _f(by["random"], "regime_acc")},
                          floor_score=0.5, focal="random-init (comparator)"))
    eff = _f(by["tjepa_h1"], "regime_acc") - _f(by["random"], "regime_acc")
    a.add(effect_vs_noise(eff, None, n_seeds=len({r["seed"] for r in rows})))
    a.add(comparison_hygiene({c: "test-2018-2026" for c in by}))
    return a


def audit_finance_ssl_ranking():
    rows = _read("finance_results.csv")
    if not rows:
        return None
    by = {r["cell"]: r for r in rows}
    a = Audit("FINANCE: MAE/BYOL are the strongest SSL encoders (i.e. the SSL RANKING is meaningful)")
    learned = {c: _f(by[c], "regime_acc") for c in ("tjepa_h1", "spatial_jepa", "mae", "byol", "simclr")
               if c in by}
    a.add(resolving_power(learned, _f(by["random"], "regime_acc")))   # random-init floor
    a.add(effect_vs_noise(_f(by.get("mae", {}), "regime_acc") - _f(by.get("byol", {}), "regime_acc"),
                          None, n_seeds=len({r["seed"] for r in rows})))
    return a


def audit_cmapss():
    rows = _read("cmapss_results.csv")
    if not rows:
        return None
    fd = [r for r in rows if r["fd"] == "FD001"]
    by = {r["cell"]: r for r in fd}
    a = Audit("C-MAPSS FD001: temporal JEPA beats the baselines AND both floors on RUL R2")
    learned = {c: _f(by[c], "rul_r2") for c in ("tjepa_h1", "spatial_jepa", "mae", "byol", "simclr")
               if c in by}
    floor = max(_f(by[c], "rul_r2") for c in ("random", "raw_features") if c in by)
    a.add(resolving_power(learned, floor, focal="tjepa_h1"))
    a.add(effect_vs_noise(_f(by["tjepa_h1"], "rul_r2") - floor, None,
                          n_seeds=len({r["seed"] for r in fd})))
    a.add(comparison_hygiene({c: "held-out-test-engines" for c in by}))
    return a


def audit_koopman():
    rows = _read("cmapss_results.csv")
    if not rows:
        return None
    by = {r["cell"]: r for r in rows if r["fd"] == "FD001"}
    if "tjepa_koopman" not in by:
        return None
    a = Audit("STRUCTURED PREDICTORS: Koopman/Neural-ODE beat the free-form transformer (real C-MAPSS)")
    floor = max(_f(by[c], "rul_r2") for c in ("random", "raw_features") if c in by)
    a.add(resolving_power({c: _f(by[c], "rul_r2") for c in ("tjepa_koopman", "tjepa_ode", "tjepa_h1")},
                          floor, focal="tjepa_koopman"))
    a.add(effect_vs_noise(_f(by["tjepa_koopman"], "rul_r2") - _f(by["tjepa_h1"], "rul_r2"),
                          None, n_seeds=len({r["seed"] for r in by.values()} or {0})))
    return a


def audit_pred_sweep():
    rows = _read("predictability_sweep.csv")
    if not rows:
        return None
    a = Audit("PREDICTABILITY SWEEP: JEPA advantage grows with measured predictability (synthetic)")
    frac = float(np.mean([_f(r, "r2_jepa") > _f(r, "r2_raw") for r in rows]))
    a.add(resolving_power({f"regime[{i}]": _f(r, "r2_jepa") for i, r in enumerate(rows)},
                          floor_score=float(np.mean([_f(r, "r2_raw") for r in rows]))))
    a.add(effect_vs_noise(float(np.mean([_f(r, "adv_jepa_raw") for r in rows])), None, n_seeds=1))
    return a


def audit_alignment():
    r2 = _read("alignment_bench_snr2.csv")
    r05 = _read("alignment_bench_snr05.csv")
    if not r2:
        return None
    a = Audit("ALIGNMENT (H2): benefit tracks predictable/task-relevant OVERLAP at fixed predictability")
    learned = {f"a={r['alpha']},s={r['seed']}": _f(r, "r2_jepa") for r in r2}
    a.add(resolving_power(learned, floor_score=float(np.mean([_f(r, "r2_raw") for r in r2]))))
    # seed variance IS measured here (3 seeds/condition) — the effect is the alpha=1 vs alpha=0 gap
    g1 = [_f(r, "adv_jepa_mae") for r in r2 if float(r["alpha"]) == 1.0]
    g0 = [_f(r, "adv_jepa_mae") for r in r2 if float(r["alpha"]) == 0.0]
    a.add(effect_vs_noise(np.mean(g1) - np.mean(g0), float(np.std(g1 + g0)), n_seeds=len(g1)))
    a.add(manipulation_isolation([_f(r, "omega") for r in r2], tol=0.05, label="spectral Omega"))
    if r05:
        s2 = np.corrcoef([float(r["alpha"]) for r in r2], [_f(r, "adv_jepa_mae") for r in r2])[0, 1]
        s05 = np.corrcoef([float(r["alpha"]) for r in r05], [_f(r, "adv_jepa_mae") for r in r05])[0, 1]
        from eval.validity import Check, FAIL, PASS
        agree = np.sign(s2) == np.sign(s05)
        a.add(Check("replication across SNR", PASS if agree else FAIL,
                    f"corr(alpha,adv) = {s2:+.3f} (snr2.0) vs {s05:+.3f} (snr0.5) — "
                    f"{'consistent' if agree else 'OPPOSITE SIGNS: not replicable'}"))
    return a


def main():
    audits = [f() for f in (audit_pastis, audit_finance_harm, audit_finance_ssl_ranking,
                            audit_cmapss, audit_koopman, audit_pred_sweep, audit_alignment)]
    audits = [a for a in audits if a is not None]
    print("=" * 100)
    print("VALIDITY AUDIT OF THIS PROJECT'S OWN CLAIMS".center(100))
    print("Does the experiment ENTITLE us to the claim? (separate from whether the claim is true)".center(100))
    print("=" * 100 + "\n")
    for a in audits:
        print(a.report())
    n_ok = sum(a.entitled for a in audits)
    print("=" * 100)
    print(f"SUMMARY: {n_ok}/{len(audits)} claims are entitled by their own experiment.")
    for a in audits:
        print(f"  {'OK  ' if a.entitled else 'FAIL'}  {a.claim[:88]}")
    print("\nA FAIL is not a refutation — it means the experiment cannot settle the question and the")
    print("claim must be stated more weakly, or the benchmark repaired. Both are honest outcomes.")


if __name__ == "__main__":
    main()
