# Pre-registration — decision rules fixed BEFORE the confirmatory runs

**Written: 2026-07-23, before any of the runs below were executed.**
Commit this file *before* launching. Its purpose is V5 (`eval/validity.py`): a decision rule chosen
after seeing results converts noise into a finding. Our own alignment script did exactly that once
(declared "H2 SUPPORTED" at p=0.245 on a post-hoc `corr > 0.3` threshold); this document exists so
that cannot recur.

Every rule below is evaluated by `scripts/audit_claims.py` against the committed CSVs. **We commit
in advance to reporting each outcome — PASS, FAIL, or INCONCLUSIVE — in the paper, including any
that contradict our prior claims.**

---

## Scope of the confirmatory runs

Three repairs, each with a stated reason that is independent of which way it moves any result:

| # | repair | reason (structural, pre-stated) |
|---|---|---|
| R1 | add `random` + `raw_features` floor cells to the PASTIS matrix | the audit found PASTIS has **no floor at all**; V1 is unassessable without one |
| R2 | 3 seeds for finance, C-MAPSS, and the PASTIS main cells | V2 cannot be evaluated at n=1; every real-domain claim currently fails on this alone |
| R3 | harder observation map + wider encoder in the alignment testbed | the shallow `tanh(2z)` map is ridge-invertible, so the raw floor beat every encoder in **30/30** cells (V1 FAIL); 64d/2-layer/15-epoch is the second named underpowering reason |

R3 is a repair of the *instrument*, not of the hypothesis. It raises the achievable ceiling for
**both** learned encoders symmetrically. A smoke test confirmed it restores resolving power
(JEPA clears the raw floor in 33% of cells, up from 0%) **before** any α-trend was examined.

---

## Rules

### P1 — PASTIS: "temporal JEPA's win reflects learning, not architecture"
**Entitled iff** `tjepa_h1` conv mIoU exceeds **both** `random` and `raw_features`,
**and** the margin over the larger floor exceeds the across-seed standard deviation (n=3).

- If `tjepa_h1` ≤ `random`: the win is attributable to the temporal-attention **architecture**, not
  the objective. We will say so, and the headline claim is retracted to "the architecture helps."
- If `tjepa_h1` ≤ `raw_features`: no representation learning is demonstrated on PASTIS at all.
- *Prior expectation:* passes comfortably. Stated so the reader can judge whether we were surprised.

### P2 — Finance: "temporal JEPA is actively harmful"
**Entitled iff** `random` regime-accuracy minus `tjepa_h1` regime-accuracy > across-seed sd (n=3),
with the sign in that direction.

- If the gap is within noise, the claim weakens to "temporal JEPA does not help on finance," and the
  stronger "training actively destroys structure" claim is withdrawn.

### P3 — C-MAPSS: "temporal JEPA beats both floors"
**Entitled iff** `tjepa_h1` RUL R² > max(`random`, `raw_features`) **and** the margin > across-seed
sd (n=3), on FD001. Reported per-subset for FD002–FD004 without a separate rule.

### P4 — Structured predictors: "Koopman/Neural-ODE beat the free-form transformer"
**Entitled iff** `tjepa_koopman` RUL R² − `tjepa_h1` RUL R² > across-seed sd (n=3), FD001.

- The observed single-seed effect is **+0.028**. If the seed sd exceeds that, this claim — one of
  only two method-level positives in the project — is withdrawn to "within noise."

### P5 — Alignment (H1 vs H2), the confirmatory test
**H2 is supported iff ALL FOUR hold:**

1. **V1 gate:** JEPA beats the raw floor in **≥ 25%** of cells (else INCONCLUSIVE, not a verdict).
2. **Trend:** `corr(α, JEPA−MAE advantage) > 0.3` with **p < 0.05** (n = 5 α × 3 seeds = 15).
3. **Effect vs noise:** the α=1 versus α=0 gap exceeds the larger of the two group standard deviations.
4. **Replication:** the sign of the trend agrees across **both** SNR settings (2.0 and 0.5).

**H1 is retained** if the V1 gate passes and the trend is flat (|corr| < 0.3, or p ≥ 0.05).
**Neither** is concluded if the V1 gate fails — that is a statement about the benchmark, not the world.

*We note in advance:* the first (underpowered) run gave corr = **+0.307 / −0.049** across the two
SNRs — failing rule 4 outright. We are not permitted to cite the +0.307 in isolation.

---

## Committed reporting

- All CSVs land under `runs/` and are committed regardless of outcome.
- `scripts/audit_claims.py` is re-run after the reruns; its output is pasted into PAPER.md §4
  verbatim, whatever it says.
- If P1 fails, the paper's empirical section leads with that.
- No rule in this file is edited after the runs begin. Changes, if any, are appended below with a
  timestamp and reason, leaving the original text intact.

## Amendments
*(none)*
