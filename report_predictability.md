# Quantifying the Predictability Hypothesis — Measurement + Synthetic Falsification (Part 6/7)

**Status:** the measurement framework and the domain-level quantitative test are complete and
**positive**; the within-testbed synthetic advantage-curve is **confounded** (an honest refinement).
This phase turns the project's qualitative "predictability spectrum" (report_full.md §14/§18) into
*measured numbers*, and tests the falsifiable claim that a causal future-latent-prediction (JEPA)
objective helps to the degree the latent trajectory is predictable.

> **Hypothesis (H-pred).** Downstream benefit of temporal JEPA is a monotone function of the
> *measured* predictability of the (latent) dynamics; at zero predictability, JEPA ≤ a trivial floor.

---

## 1. What was built

- **`eval/predictability.py`** — seven standard predictability indices, numpy-only:
  spectral predictability Ω (1−normalized spectral entropy), permutation-entropy predictability,
  linear AR(p) forecast R², 1/e autocorrelation time, Rosenstein largest-Lyapunov (approximate),
  **past→future mutual information** (Gaussian estimate of the *predictive information* /
  excess entropy — Bialek-Nemenman-Tishby 2001), and intrinsic dimension (participation ratio).
- **`data/synthetic_dynamics.py`** — a latent-dynamics generator spanning the predictability axis
  (periodic → AR(1)-φ-sweep → Lorenz → white), rendered through a (nonlinear) observation map at a
  fixed SNR into the panel format the JEPA stack consumes, with the clean latent as the recovery target.
- **`scripts/predictability_sweep.py`** — the falsification experiment: dial + measure predictability,
  train identical JEPA vs MAE vs raw, probe latent recovery, plot advantage vs predictability.
- **`tests/test_predictability.py`** — 9 tests; the indices provably order the regimes
  (periodic ≫ white; AR(1) monotone in φ; Lorenz low intrinsic-dim). Full suite: 61 pass / 3 skip.

---

## 2. The positive result — the predictability spectrum is now QUANTITATIVE

Measuring the indices on the **real observed dynamics** of the three domains places them exactly where
the win/loss pattern predicts:

| domain (observed series) | Ω | AR-R² | past→future MI | autocorr | reads as |
|---|---|---|---|---|---|
| **finance — index daily returns** | **0.053** | **−0.003** | **0.01** | 1.0 | **≡ white noise** |
| **C-MAPSS — engine sensor bank** | 0.359 | — | **25.9** | 25.1 | ≡ structured / Lorenz-like |
| _synthetic white (ref)_ | _0.054_ | _−0.004_ | _0.02_ | _1.0_ | — |
| _synthetic Lorenz (ref)_ | _0.349_ | _1.000_ | _13.5_ | _9.3_ | — |
| _synthetic periodic (ref)_ | _0.802_ | _1.000_ | _25.3_ | _2.7_ | — |

**This is the mechanism, measured.** Finance *returns* carry **predictive information ≈ 0** (MI 0.01,
Ω 0.05) — statistically indistinguishable from white noise — which is *why* temporal JEPA (Phases 2/4/5)
failed and no algorithmic or protocol fix rescued it: there is no learnable future to predict.
C-MAPSS sensors carry **more predictive information than the Lorenz attractor** (MI 25.9) — which is
*why* temporal JEPA won (Phase 3). The falsifiable hypothesis holds at the domain level, now with
numbers rather than a hand-wave. (Aside: the finance index *level* measures as highly "predictable"
— Ω 0.83, autocorr 200 — but that is a spurious unit-root/trend artifact; the *returns* are the
learnable object, and they are noise. This is exactly why one must difference.)

---

## 3. The honest negative — the synthetic advantage-curve is confounded

`scripts/predictability_sweep.py` (latent recovery, last-step readout; `runs/predictability_sweep.csv`,
`runs/figures/predictability_sweep.png`):

| regime | Ω | R² JEPA | R² MAE | R² raw |
|---|---|---|---|---|
| periodic | 0.745 | 0.558 | 0.631 | **0.827** |
| AR φ=0.9 | 0.260 | 0.467 | 0.571 | **0.731** |
| AR φ=0.2 | 0.058 | 0.471 | 0.567 | **0.725** |
| Lorenz | 0.341 | 0.753 | 0.777 | **0.856** |
| white | 0.053 | 0.499 | 0.572 | **0.822** |

corr(Ω, JEPA−raw) = +0.17 — **flat/negative: the advantage-∝-predictability curve is NOT confirmed
on this testbed.** But the reason is instructive, not a failure of the hypothesis: **the raw/linear
baseline dominates everywhere**, because a *predictable* system is (by construction) a *simple,
linearly-accessible* one — so a plain ridge on the raw observation (or a linear delay-embedding)
recovers the latent at least as well as an undertrained, frozen SSL encoder. We verified the same
under a nonlinear observation map and under partial observability: linear baselines stay on top.

This **refines** H-pred rather than refuting it: *predictability is necessary but not sufficient for a
learned-representation advantage — the task must also be **non-trivially complex** (nonlinear /
not single-frame-observable / high-SNR-limited) for representation learning to beat a strong linear
baseline.* It is the synthetic echo of the whole project's most robust empirical lesson: **raw and
random-init baselines are hard to beat**, and were it not for those floors (finance §12, C-MAPSS §4.4)
we would have over-claimed. The clean "advantage curve" would require decoupling predictability from
task-linearity (e.g. chaotic systems at low SNR with genuinely nonlinear, partially-observed readouts,
and a properly-scaled encoder) — a worthwhile follow-on.

---

## 4. Where this leaves the research agenda (Parts 6–7)

**Delivered:** the *measurement* half of Part 7 (the predictability indices) and the synthetic
falsification testbed — the reusable core the rest of Part 6 builds on. The single most valuable
output is §2: the predictability indices **explain and predict** the cross-domain JEPA win/loss,
converting the spectrum thesis into a measured, falsifiable claim.

**Part 6 #2 — predictability-conditioned objective weighting (IMPLEMENTED; honest negative).**
We turned the hypothesis into a *training method*: weight each window's latent-loss by its measured
spectral predictability Ω^γ (a GPU `batch_spectral_omega`, threaded through `jepa_latent_loss`'s new
`sample_weight`), so the encoder spends capacity on learnable windows. Test on 50/50 and 80/20
predictable/white mixtures (`scripts/predictability_curriculum.py`), evaluating latent recovery on the
predictable subset (within-JEPA, immune to the raw confound):

| noise fraction | uniform JEPA | weighted JEPA | Δ |
|---|---|---|---|
| 50% | 0.283 | 0.286 | **+0.003** (negligible) |
| 80% | 0.187 | 0.169 | **−0.018** (hurts) |

The curriculum does **not** help — neutral at moderate noise, harmful at high noise. Mechanism: JEPA
already *self-regulates* on unpredictable windows (their low-gradient "predict-the-mean" loss), so
explicit down-weighting mostly removes useful regularization and shrinks the effective batch size.
Another honest negative, consistent with the pattern below.

**Other natural next (build on this module):**
- **Information-theoretic falsification (Part 6 #6):** the past→future-MI estimator here is the
  quantity to bound downstream linear-probe accuracy against.
- **Larger, structured predictors (Koopman/SSM, Neural-ODE, Kalman-filtered rollouts — Part 6 #1,3,4):**
  each is its own build; the predictability metrics are how you'd *measure* whether they help.

**Honest framing.** This is a *measurement + controlled-testbed* contribution, not a new algorithm.
Its value is that it makes the project's central claim quantitative and falsifiable — and, in keeping
with the rest of the project, it reports the confound honestly rather than forcing a clean curve.

## 5. The hidden confounder — predictability vs *alignment* (H1 vs H2). **UNRESOLVED.**

### 5.1 The problem with §2

§2 places three domains on a predictability axis and shows the win/loss pattern follows it. But the
three domains **confound predictability with task-relevance**:

| domain | predictable? | does the LABEL depend on the predictable part? |
|---|---|---|
| PASTIS | yes | yes (phenological cycle *is* what determines crop class) |
| C-MAPSS | yes | yes (degradation trend *is* what RUL reads) |
| finance | no | no |

Those three points are equally consistent with two different hypotheses:

> **H1** — benefit depends on the **predictability** of the process. *(the project's current thesis)*
> **H2** — benefit depends on the **overlap** between the predictable subspace and the task-relevant
> subspace; predictability is *necessary but not sufficient*.

**Nothing in §2, or in any of the three real domains, distinguishes them.** H2 is strictly stronger:
it makes a prediction H1 does not — that a process can be highly predictable and temporal JEPA still
fail (or hurt), if the label reads the *unpredictable* component.

### 5.2 The testbed (`data/synthetic_dynamics.generate_aligned`, `scripts/alignment_bench.py`)

The latent is two independent blocks — `z_slow` (AR φ=0.95, predictable) and `z_fast` (white,
unpredictable) — and **both are always rendered into the observation**. Only the label moves:

$$y \;=\; \alpha\cdot\mathrm{std}(z_{\text{slow}}\!\cdot\! w_s)\;+\;(1-\alpha)\cdot\mathrm{std}(z_{\text{fast}}\!\cdot\! w_f)$$

So **input predictability is invariant to α by construction** — asserted as a first-class property in
`tests/test_alignment.py` (Ω and past→future MI identical across α to 1e-9; observations bitwise
identical while labels decorrelate). Empirically the Ω spread across the sweep was 0.019 (SNR 2.0)
and 0.006 (SNR 0.5). **The design is sound; the experiment run on it was not decisive.**

### 5.3 Result — **INCONCLUSIVE**, at both sensitivities

5 α-values × 3 seeds × 2 SNR configurations = 30 pretrainings. JEPA-minus-MAE advantage:

| α | align. index | J−MAE @ SNR 2.0 | J−MAE @ SNR 0.5 |
|---|---|---|---|
| 1.00 | 0.97 | −0.053 ± 0.091 | −0.033 ± 0.049 |
| 0.75 | 0.94 | −0.031 ± 0.066 | −0.004 ± 0.040 |
| 0.50 | 0.53 | −0.032 ± 0.035 | +0.017 ± 0.036 |
| 0.25 | 0.03 | −0.070 ± 0.046 | +0.000 ± 0.048 |
| 0.00 | 0.00 | −0.103 ± 0.032 | −0.027 ± 0.046 |
| | **corr(α, adv)** | **+0.307 (p=0.245)** | **−0.049 (p=0.860)** |

**The two configurations disagree in sign and neither is significant.** Neither trend is monotone,
and the per-α standard deviations exceed the α=1→α=0 differences. No conclusion about H1 vs H2 can
be drawn.

**Why — the instrument is insensitive.** In **0 of 30 cells** did temporal JEPA beat the
raw-feature floor (J−raw ranged −0.16 to −0.32). The JEPA-vs-MAE contrast is therefore a comparison
between *two encoders that both failed to reach the floor*, and any correlation across α is noise.
This is the **same confound documented in §3** — on this synthetic testbed, small undertrained
encoders lose to ridge-on-raw-features, so encoder-vs-encoder deltas are uninterpretable.

> **A verdict-logic bug is worth recording.** The first version of `alignment_bench.py` declared
> "H2 SUPPORTED" at SNR 2.0 on `corr = +0.307` alone — an arbitrary `corr > 0.3` threshold with no
> significance test and no sensitivity gate. It was a **false positive** on non-significant,
> non-monotone, noise-dominated data. The script now gates on (i) the raw floor being beaten in ≥25%
> of cells, (ii) the α=1-vs-α=0 gap exceeding its own seed variance, and (iii) p < 0.05 — and it now
> correctly reports INCONCLUSIVE. Had this not been caught, the project would have published a
> stronger claim than the data supports.

### 5.4 The one solid positive — the alignment index is a validated instrument

`eval/predictability.alignment_index` (ridge past→present, then
`R²(predictable part → y) / R²(full present → y)`) **tracks α at r = +0.957 (SNR 2.0) and +0.950
(SNR 0.5) while Ω is held flat.** It is cheap, linear, label-aware, needs no pretraining, and
measures exactly the quantity H2 says should matter. Whether it *predicts downstream benefit* is
the open question — this testbed could not test that.

### 5.5 What a decisive version needs

1. **Make the learned encoders beat the raw floor** — the blocking issue. Either a genuinely
   probe-hostile observation map (the current `tanh` is near-linear in range, so ridge inverts it),
   or materially more capacity/epochs than the 64-dim / 2-layer / 15-epoch models used here.
2. Only then sweep α; the design and the index are already in place and tested.
3. Confirm on a *real* domain: C-MAPSS with a relabeled target that depends on the high-frequency
   sensor residual rather than the degradation trend would be the natural real-data analogue.

**Status: the confounder identified is real, and remains unresolved.** §2's domain-level
correspondence stands as evidence, but it cannot separate H1 from H2, and this report should not be
read as having done so.

## 6. Reproducibility
`pytest tests/test_predictability.py` (9) + `tests/test_alignment.py` (7), offline. Sweep:
`python scripts/predictability_sweep.py --device cuda:0` → `runs/predictability_sweep.csv` +
`runs/figures/predictability_sweep.png`. Alignment: `python scripts/alignment_bench.py --device
cuda:0 --snr {2.0,0.5} --seeds 3` → `runs/alignment_bench_snr*.csv`, log `runs/alignment.log`.
Real-domain indices: `eval.predictability.predictability_report(series)` on any (T,) or (T,D) array.
