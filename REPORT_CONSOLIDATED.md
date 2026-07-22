# Multi-Temporal JEPA — Consolidated Research Record

**One document, chronological.** This traces the project as it actually developed: what we believed at
each stage, what the data said, what broke, and how the claim changed in response. Per-phase detail
remains in the appendix reports; the paper-shaped argument is in [PAPER.md](PAPER.md).

Every number is reproduced from a committed CSV under `runs/`. None is estimated. Negative results
appear at the same prominence as positive ones — several supersede earlier claims of ours.

| | |
|---|---|
| **Original question** | Does causal future-latent-prediction (temporal JEPA) beat MAE/BYOL/SimCLR? |
| **Where it ended** | It wins on 2 of 3 domains — but **we cannot yet say why**, and by our own criteria only 1 of 7 headline claims is entitled by its experiment. |
| **Current contribution** | A methodology: five executable validity criteria + a self-audit + a confounder-isolating testbed. |
| **Status** | Confirmatory runs pre-registered ([PREREGISTRATION.md](PREREGISTRATION.md)); awaiting server execution. |

---

## Timeline at a glance

| stage | question | outcome |
|---|---|---|
| **1** Satellite (PASTIS) | does temporal beat spatial JEPA? | **WIN** +6.0 mIoU, p=0.041 |
| **2** Finance (S&P-500) | does it transfer? | **LOSS** — worse than its own untrained init |
| **3** Industrial (C-MAPSS) | is the loss modality-specific? | **WIN** 51/52 metrics vs SimCLR |
| **4** Distributional rescue | can β-NLL fix finance? | **REJECTED** |
| **5** Shift vs unpredictability | is it distribution shift? | **No** — unpredictability |
| **6** Eight method systems | which algorithmic priors help? | 2 wins, 4 negatives |
| **7** Predictability measurement | can we quantify the mechanism? | indices separate the domains |
| **8** Graph JEPA on PASTIS | does a local prior beat attention? | **NEGATIVE** −1.8 mIoU |
| **9** H1-vs-H2 confounder | is it predictability, or *alignment*? | **UNRESOLVED** — benchmark underpowered |
| **10** Validity audit + pivot | are we entitled to any of this? | **1 of 7 claims** |
| **11** Repairs (pre-registered) | fix floors, seeds, sensitivity | **pending server run** |

---

## Stage 1 — PASTIS satellite image time series: the original win

Factorized space→time encoder (per-frame spatial ViT → temporal transformer), causal past→future
split, EMA target, stop-gradient, narrow predictor, VICReg anti-collapse. Frozen-probe evaluation at
matched effective batch 192 for every method.

**3 seeds, paired t-test vs temporal, conv mIoU on the val fold:**

| method | conv mIoU | Δ vs temporal | p | test (1 seed) |
|---|---|---|---|---|
| **Temporal JEPA (Δ=1)** | **22.3 ± 1.8** | — | — | **22.1** |
| Spatial JEPA | 16.2 ± 0.4 | +6.0 | **0.041** | 16.1 |
| Spatial JEPA — compute-matched (3.5× epochs) | 15.8 ± 1.2 | +6.5 | **0.036** | 17.1 |
| SimCLR | 7.3 ± 0.8 | +15.0 | **0.009** | 7.1 |
| BYOL | 7.1 ± 0.9 | +15.2 | **0.001** | 4.9 |
| MAE | 6.5 ± 1.1 | +15.8 | **0.009** | 3.6 |
| _supervised U-TAE (ceiling, not a peer)_ | — | — | — | _63.1_ |

- **Not compute:** spatial JEPA given 3.5× the epochs gains nothing (15.8), still 6.5 behind.
- **Grows as labels shrink:** few-shot at 1/5/10% is 9.2/13.1/15.9 vs spatial 4.6/6.9/9.5 — a
  **+100% relative gap at 1%**.
- **Horizon-insensitive:** Δ=1/2/4/8 → 22.3/20.8/21.8/22.6, all within noise.
- **Anti-collapse is load-bearing:** without VICReg the objective is solved by emitting a constant
  (per-dim std → 0.04). This recurs in every later domain.

> **Superseded in Stage 10.** This result has **no floor** — no random-init or raw-feature cell was
> ever run. We therefore cannot show the win is a fact about *learning* rather than about the
> temporal-attention *architecture*. Repair R1 is pre-registered.

## Stage 2 — Finance: the transfer fails, and fails downward

Same stack, panel abstraction `(B, W, N_assets, F)`. Out-of-time TEST 2018–2026.

| task | **Temporal JEPA** | Spatial | MAE | BYOL | SimCLR | _random_ | _raw floor_ |
|---|---|---|---|---|---|---|---|
| Regime accuracy | 0.609 | 0.758 | **0.797** | 0.787 | 0.790 | _0.802_ | _0.804_ |
| Volatility R² | −0.228 | −0.435 | 0.157 | **0.181** | 0.099 | _0.169_ | _0.112_ |
| Anomaly AUROC | 0.745 | 0.553 | **0.837** | 0.738 | 0.521 | _0.726_ | _0.837_ |
| Clustering NMI | 0.157 | 0.132 | 0.333 | **0.367** | 0.252 | _0.141_ | _0.329_ |
| Forecast dir-acc | 0.523 | 0.479 | 0.499 | 0.511 | **0.523** | _0.496_ | _0.533_ |

Three findings, escalating in importance:

1. Temporal JEPA loses to every baseline (it beats only spatial JEPA — the satellite *ordering*
   replicates even as the *result* inverts).
2. **No SSL method beats the raw-feature floor.** SSL buys ≈ nothing here.
3. **Temporal JEPA is uniquely harmful.** Untrained 0.80 → trained 0.61. Longer horizons erase more
   (0.609 → 0.516 → 0.494 for Δ=1/5/20; return-IC 0.085 → 0.009 → **−0.086**).

Direction accuracy sits at ~50% for all methods, exactly as efficient-market theory predicts — a
sanity check that the harness isn't leaking labels.

## Stage 3 — C-MAPSS: the win returns on a different modality

All four subsets, held-out test engines, 13 metrics each (52 metric-subsets).

| Temporal JEPA beats… | FD001 | FD002 | FD003 | FD004 | **total** |
|---|---|---|---|---|---|
| SimCLR | 13 | 13 | 13 | 12 | **51 / 52** |
| MAE | 12 | 12 | 11 | 11 | **46 / 52** |
| raw features _(floor)_ | 11 | 13 | 10 | 11 | **45 / 52** |
| random-init _(floor)_ | 8 | 12 | 9 | 11 | **40 / 52** |

Standard NASA RUL benchmark (frozen probe) — best of all seven cells in every subset:
RMSE 16.4 / 26.2 / 14.8 / 27.0 and PHM08 471 / 6465 / 425 / 5128 for FD001–FD004.

**The most informative slice:** the margin over the *random-init* floor is difficulty-dependent —
8/13 on the easy single-condition FD001 but **12/13 on FD002**. On easy subsets a random temporal
projection already captures the monotone degradation signal; **learning matters where the problem is
hard.** This is why floors are reported everywhere (and why their absence on PASTIS matters).

## Stage 4 — Distributional rescue of finance: rejected

If next-day returns are near-martingale, the *point* target is noise. So predict a **distribution**
(μ, σ²) over the future latent, trained with β-NLL (Seitzer et al. 2022), letting the model assign
high variance where the future is unlearnable.

**Rejected.** The variance head does learn volatility structure, but no downstream win follows. It is
also a mild net negative on the *predictable* domain (C-MAPSS FD001 RUL R² 0.658 vs 0.677, PHM08 578
vs 471) — on a highly predictable signal the variance head is unhelpful overhead.

## Stage 5 — Is it non-stationarity or unpredictability?

Re-pretrain on ≤2019 and evaluate fully in-period on 2020–2026, so **no distribution shift exists**.

| protocol | raw features | MAE | temporal JEPA |
|---|---|---|---|
| re-pretrain recent, in-period | **0.831** | 0.685 | **0.460** |

Even with the shift entirely removed, raw features beat every SSL method and temporal JEPA is worst.
**The failure is unpredictability, not non-stationarity.** The finance negative now survives both an
algorithmic and a protocol rescue — which is what makes it robust rather than a tuning artifact.

## Stage 6 — Eight method systems

| # | system | outcome |
|---|---|---|
| 1 | LKF (Kalman over latents) | ✓ dynamics gain scales with one-step predictability; **exactly 0** on white noise |
| 2 | predictability-weighted loss | ✗ neutral to harmful |
| 3 | **Koopman predictor** | ✅ **WIN** |
| 4 | **Neural-ODE predictor** | ✅ **WIN** |
| 5 | distributional β-NLL | ✗ (Stage 4) |
| 6 | info-theoretic (past→future MI) | ✓ diagnostic (Stage 7) |
| 7 | hierarchical multi-horizon | ✗ flat (0.591/0.590/0.587) |
| 8 | graph temporal JEPA | ✗ (Stage 8) |

**The wins.** Replacing the free-form transformer predictor with an explicit dynamics model, on real
C-MAPSS FD001:

| predictor | RUL R² ↑ | PHM08 ↓ | health ↑ |
|---|---|---|---|
| free-form transformer | 0.677 | 471 | 0.744 |
| **Koopman** | **0.706** | **343** | **0.782** |
| **Neural-ODE** | **0.707** | 349 | 0.778 |
| _random-init floor_ | _0.651_ | _457_ | _0.776_ |

≈27% better PHM08, and both pull *further ahead of the floor* than the transformer. On synthetic data
the margin is largest exactly where predictability is highest and negligible on white noise.

> **Superseded in Stage 10.** Single-seed. The +0.028 R² effect has no measured noise floor; rule P4
> may withdraw it.

## Stage 7 — Quantifying the mechanism

Seven predictability indices measured on each domain's **real observed dynamics**:

| observed series | Ω | past→future MI | reads as |
|---|---|---|---|
| **finance — daily returns** | **0.053** | **0.01** | **≡ white noise** |
| **C-MAPSS — engine sensors** | 0.359 | **25.9** | structured |
| _synthetic white (ref)_ | _0.054_ | _0.02_ | — |
| _synthetic Lorenz (ref)_ | _0.349_ | _13.5_ | — |

Finance returns are indistinguishable from white noise; C-MAPSS sensors carry more predictive
information than the Lorenz attractor. The domains land exactly where the win/loss pattern predicts.

*Methodological aside worth keeping:* the finance index **level** measures as highly predictable
(Ω 0.83) — a spurious unit-root artifact. The *returns* are the learnable object. Failing to
difference would have produced a confident, wrong conclusion.

## Stage 8 — Graph temporal JEPA on real PASTIS: negative

A grid-GNN spatial backbone (local message passing) versus the global-attention ViT. Identical
config, same split, same probe budget — backbone the only variable.

| backbone | linear | **conv mIoU** | k-NN | eff. rank |
|---|---|---|---|---|
| ViT / global attention | 17.91 | **22.46** | 67.63 | 252.8 |
| grid-GNN / local passing | 16.79 | **20.65** | 68.05 | 249.4 |

Not collapse — the diagnostics are near-identical. At patch-size 8 a parcel spans few of the 16×16
nodes, so **locality was never the bottleneck**. The pre-registered hypothesis ("a modest gain") is
**rejected**.

**Two protocol bugs caught here**, both of which could have produced a false result: the baselines
were probed on **val** and the new cell initially on **test** (split confound), and
`evaluate.py --encoder-ckpt` hardcoded `SITSEncoder` so a graph checkpoint could not load at all.
These became criteria V4 and part of the audit.

## Stage 9 — The confounder we had missed

All three domains **confound predictability with task-relevance**:

| domain | predictable? | does the label depend on the predictable part? |
|---|---|---|
| PASTIS | yes | yes — phenology determines crop class |
| C-MAPSS | yes | yes — degradation trend is what RUL reads |
| finance | no | no |

So the evidence is equally consistent with **H1** (benefit depends on predictability — our claim) and
**H2** (benefit depends on the *overlap* between predictable and task-relevant subspaces).

We built a testbed that separates them: two latent blocks, `z_slow` (AR φ=0.95) and `z_fast` (white),
**both always in the observation**, with only the label moving via α. Input predictability is
invariant to α **by construction, asserted to 1e-9 in tests**.

**Result: inconclusive.** The two SNR settings disagreed in sign — corr(α, advantage) = **+0.307
(p=0.245)** versus **−0.049 (p=0.860)** — because JEPA beat the raw floor in **0 of 30 cells**. The
contrast was between two encoders that both failed.

**One positive:** `alignment_index` tracks α at **r ≈ 0.95** while Ω is held flat — a validated
label-aware instrument whose predictive value remains untested.

## Stage 10 — The audit, and the pivot

We distilled five **validity criteria** from failures that actually occurred here — V1 resolving
power, V2 effect-vs-noise, V3 manipulation isolation, V4 comparison hygiene, V5 pre-registered
decision rule — implemented them executably (`eval/validity.py`, 15 tests), and turned them on our
own seven headline claims.

**1 of 7 is entitled by its own experiment.**

| claim | V1 | V2 | entitled |
|---|---|---|---|
| PASTIS: temporal beats spatial + baselines | N/A¹ | N/A² | yes (2 unassessed) |
| Finance: temporal JEPA is harmful | PASS | FAIL | no |
| Finance: MAE/BYOL are strongest SSL | **FAIL** | FAIL | no |
| C-MAPSS: beats baselines and both floors | PASS | FAIL | no |
| Koopman/ODE beat free-form predictor | PASS | FAIL | no |
| Predictability sweep | **FAIL** | FAIL | no |
| Alignment (H2) | **FAIL** | FAIL | no |

¹ **PASTIS has no floor cell** — unnoticed for the entire project.
² Seed-tagged CSVs live on the server, not in this checkout.

Most failures are V2 (single seed) — real, already-known, and cheaply repairable. One is substantive:
the finance **SSL ranking** fails V1 because no method beat the raw floor, so "MAE is best" *orders
failures*. That is sharper than what we had written.

**The audit also caught itself.** Its first run reported 0/7; two of those were bugs in the audit
(V1 used a population fraction, penalising the C-MAPSS win for being the *only* method above the
floor; and it treated absent server data as single-seed). The instrument needed validating exactly
like the benchmarks it judges.

**Consequence — the claim is restated in its defensible form:**

> ~~Temporal JEPA helps when the latent process is predictable.~~
> **Temporal JEPA helped on the domains where the predictable component was also the task-relevant
> one. Whether predictability alone is sufficient is open.**

## Stage 11 — Repairs, pre-registered (pending)

Three repairs, each with a structural reason stated *before* execution and independent of which way
it moves any result — see [PREREGISTRATION.md](PREREGISTRATION.md) for the binding decision rules.

| repair | reason |
|---|---|
| R1 PASTIS `random` + `raw_features` floor cells | V1 unassessable without a floor |
| R2 3 seeds for finance / C-MAPSS / PASTIS | V2 cannot be evaluated at n=1 |
| R3 harder observation map + wider encoder | `tanh(2z)` is ridge-invertible → raw floor dominated 30/30 |

R3 is a repair of the **instrument**, not the hypothesis: it raises the ceiling for both learned
encoders symmetrically. A smoke test confirmed it restores resolving power (JEPA clears the raw floor
in 33% of cells, up from 0%) **before** any α-trend was examined.

---

## What is established, and what is not

**Established.**
1. Temporal prediction beats reconstruction/contrastive SSL on satellite time series by a large,
   compute-controlled margin that widens in the low-label regime.
2. It fails on near-efficient financial panels, robustly — surviving both an algorithmic and a
   protocol rescue.
3. Anti-collapse regularization is necessary in every temporal domain tested.
4. Structural priors matching the dynamics (Koopman/ODE) outperform generic tweaks — *pending P4*.

**Not established.**
- **Why** it works. The predictability criterion is consistent with n=3 domains but confounded with
  task-relevance; the experiment built to separate them was underpowered.
- **That the PASTIS win reflects learning** rather than the temporal architecture — no floor yet.
- **Any effect's size**, outside PASTIS — everything else is single-seed.
- **Novelty of the objective.** Causal/temporal JEPA is prior art; distributional JEPA appears as
  VJEPA (arXiv 2601.14354) and Var-JEPA (arXiv 2603.20111). Nothing here is compared to a large
  pretrained world model.

## Scoreboard

| | count |
|---|---|
| Domain wins | 2 (PASTIS, C-MAPSS) |
| Domain losses | 1 (finance — below every floor) |
| Method wins | 2 (Koopman, Neural-ODE) — *pending P4* |
| Method negatives | 4 (weighting, distributional, hierarchical, graph) |
| Claims entitled by their own experiment | **1 of 7** |

**Six of eleven headline results are negative, and all are reported.** The negatives carry the
argument: the finance failure plus its two failed rescues is what converts "temporal JEPA is good"
into a falsifiable claim, and the self-audit is what keeps the remaining claims honest.

## Reproducibility

```bash
pytest -q                          # offline, no data required
python scripts/audit_claims.py     # regenerates the Stage-10 audit from committed CSVs
```

## Appendix — detailed per-stage reports

| document | stages |
|---|---|
| [PAPER.md](PAPER.md) | the submission write-up (methodology framing) |
| [PREREGISTRATION.md](PREREGISTRATION.md) | binding decision rules for Stage 11 |
| [report.md](report.md) | 1 — PASTIS, architecture, ablations |
| [report_finance.md](report_finance.md) | 2, 4, 5 — finance and both rescues |
| [report_cmapss.md](report_cmapss.md) | 3 — C-MAPSS, five probes, NASA benchmark |
| [report_predictability.md](report_predictability.md) | 7, 9 — indices, alignment testbed |
| [report_structured.md](report_structured.md) | 6, 8 — Koopman/ODE/LKF/hierarchical/graph |
| [report_full.md](report_full.md) | full monograph (background, derivations, related work) |
