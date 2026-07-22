# When Is a Self-Supervised Benchmark Entitled to Its Conclusion?

### Validity criteria for mechanistic hypotheses in SSL, derived from — and applied to — a failed investigation

---

## Abstract

Self-supervised learning research increasingly advances *mechanistic* claims: not merely "objective O
scores higher," but "O helps **because** of mechanism M." Such claims are typically supported by a
benchmark comparison across conditions. We argue that these comparisons are frequently **not entitled
to their conclusions**, for reasons that are detectable *before* interpreting any result, and that the
field lacks executable standards for detecting them.

We arrive at this position empirically. We set out to test a specific mechanistic hypothesis — that a
causal future-latent-prediction (temporal JEPA) objective helps in proportion to the *predictability*
of the latent process — across three unrelated modalities (satellite image time series, financial
panels, turbofan degradation). The domain-level pattern was striking and initially appeared to confirm
the hypothesis. It does not: the three domains **confound predictability with task-relevance**, and a
synthetic benchmark built specifically to separate them proved **insufficiently powered to resolve
either way**.

From the concrete failures encountered — five of which produced, or nearly produced, a false published
result — we distill five **validity criteria** (V1–V5) and implement them as executable checks. We
then apply them to our own seven headline claims. **One of seven survives.** We report the audit in
full, restate the surviving claims in their weaker defensible form, and release the criteria, the
alignment testbed, and a validated measurement instrument.

Our contribution is not a new objective or a new theory. It is a **methodology, an audit protocol, and
a worked demonstration that the protocol has teeth — including against the authors' own results.**

---

## 1. Introduction

### 1.1 The shape of the problem

A mechanistic claim in SSL has the form: *objective O outperforms baselines B because property P of the
data makes O's pretext task informative.* Establishing it requires more than a table of numbers. It
requires that the experiment **could have come out otherwise** — that the benchmark had the resolving
power to detect the absence of the effect, that the manipulated variable was isolated, and that the
decision rule was fixed in advance.

These are old ideas in experimental science. They are not, in our experience, routinely enforced in SSL
benchmarking, and they are rarely *executable*. The result is that an underpowered benchmark and a
genuine effect can produce indistinguishable-looking tables.

### 1.2 How we got here

This paper began as a conventional empirical study. The intended contribution was the predictability
criterion of §2. Two things happened:

1. A reviewer-style critique identified that our three domains **cannot distinguish** the hypothesis we
   were advancing (H1) from a strictly stronger one (H2). Our evidence was consistent with both.
2. The synthetic benchmark we built to separate them **failed to resolve the question** — and our own
   analysis script initially reported a false positive from it.

Rather than repair the benchmark until the hypothesis won — a temptation we name explicitly in §6 — we
report the investigation as it happened and extract the transferable result: **the criteria that would
have flagged each failure at the time.**

### 1.3 Contributions

- **C1.** Five validity criteria for mechanistic SSL benchmarks (§3), each derived from a documented
  failure, implemented as executable checks (`eval/validity.py`) rather than prose guidance.
- **C2.** A **self-audit** of our own seven headline claims (§4). One is entitled. We report all seven.
- **C3.** An **alignment testbed** (§5) that holds measured predictability fixed to 1e-9 while varying
  task-relevance — a design that isolates a confounder present in most cross-domain SSL comparisons.
- **C4.** `alignment_index`, a cheap label-aware measurement that tracks the alignment knob at
  r ≈ 0.95 while spectral predictability is held constant (§5.3).
- **C5.** The underlying three-domain empirical study (§2), reported with its claims correctly weakened.

---

## 2. The investigation that motivated this

### 2.1 Setup

One architecture, one objective, three modalities. Temporal JEPA predicts the future latent of a
sequence from its past (EMA target, stop-gradient, narrow predictor, VICReg anti-collapse), compared
against spatial JEPA, MAE, BYOL and SimCLR under matched effective batch, with frozen-probe evaluation.

### 2.2 The pattern

| domain | result | headline |
|---|---|---|
| PASTIS Sentinel-2 crop series | win | +6.0 conv mIoU over spatial JEPA (p=0.041, 3 seeds), +15–16 over MAE/BYOL/SimCLR |
| NASA C-MAPSS turbofan | win | beats SimCLR on 51/52 metrics; best of 7 cells on the standard NASA RUL benchmark |
| S&P-500 sector panel | **loss** | regime accuracy **0.61 trained vs 0.80 untrained** — training is actively harmful |

The finance failure survived two independent rescue attempts: an algorithmic one (distributional
β-NLL prediction over the future latent) and a protocol one (re-pretraining on recent data so that no
train→test distribution shift exists). Neither moved it.

### 2.3 The apparent mechanism

Measuring seven standard predictability indices on each domain's observed dynamics placed them exactly
where the win/loss pattern would predict:

| observed series | spectral Ω | past→future MI | reads as |
|---|---|---|---|
| finance — index daily returns | 0.053 | **0.01** | **≡ white noise** |
| C-MAPSS — engine sensors | 0.359 | **25.9** | structured |
| _synthetic white (ref)_ | _0.054_ | _0.02_ | — |
| _synthetic Lorenz (ref)_ | _0.349_ | _13.5_ | — |

This looked like a mechanism, measured. **It is not established.** §5 explains why.

---

## 3. Five validity criteria

Each criterion below is stated with the failure that produced it. All are implemented in
`eval/validity.py` and unit-tested in `tests/test_validity.py` (15 tests).

### V1 — Resolving power
> A trivial floor (raw features, random initialization) must not dominate the learned methods.

If every learned method loses to ridge-on-raw-features, a method-vs-method delta is a **contrast
between two failures**, and any trend across conditions is noise regardless of its magnitude or
correlation with the manipulated variable.

*Caught:* our predictability sweep (0/9 conditions beat raw) and our alignment benchmark (**0/30**).
*Subtlety, learned the hard way:* the criterion must be **claim-shaped**. A claim "method X helps"
requires only that **X** clear the floor — other methods failing to is a finding about them, not a
defect. A **ranking** claim requires the population to clear it. Our first implementation conflated
these and marked a genuine C-MAPSS win INCONCLUSIVE; the regression is now a test.

### V2 — Effect versus noise
> The reported effect must exceed run-to-run variance, and single-seed effects have no measured noise floor.

*Caught:* our alignment benchmark reported an effect of 0.050 against a seed standard deviation of
0.073. Also flags, correctly, that most of our own real-domain results are single-seed.

### V3 — Manipulation isolation
> When claiming "we varied X holding Y fixed," Y must be *verified* constant, not assumed.

*Designed in:* our alignment generator holds spectral Ω and past→future MI invariant across the
manipulated variable **to 1e-9**, asserted as a first-class unit test. Had this drifted, the entire
experiment would have been void — and silently so.

### V4 — Comparison hygiene
> Compared cells must share evaluation split, probe budget, and protocol.

*Caught:* we probed a new architecture on the **test** folds while every baseline had been probed on
**val**. The resulting −1.8 mIoU gap conflated backbone with split. Caught before interpretation;
would have been invisible in the results table.

### V5 — Pre-registered decision rule
> The decision rule must be fixed before the data is seen.

*Caught, embarrassingly, in our own analysis code:* the alignment script declared **"H2 SUPPORTED"**
on `corr > 0.3` — a threshold with no significance test and no sensitivity gate — from data with
p = 0.245, a non-monotone trend, and standard deviations exceeding the effect. The script now gates on
resolving power, effect-versus-noise, and p < 0.05, and correctly returns INCONCLUSIVE.

---

## 4. Self-audit: applying V1–V5 to our own claims

`scripts/audit_claims.py` reads the committed result CSVs and evaluates each headline claim. The
criteria are only credible if permitted to downgrade the authors' results. They do.

| # | claim | V1 | V2 | V3 | V4 | entitled? |
|---|---|---|---|---|---|---|
| 1 | PASTIS: temporal JEPA beats spatial + all baselines | N/A¹ | N/A² | — | PASS | **yes**(with unassessed) |
| 2 | Finance: temporal JEPA is *harmful* (below random init) | PASS | FAIL | — | PASS | no |
| 3 | Finance: MAE/BYOL are the strongest SSL encoders | **FAIL** | FAIL | — | — | no |
| 4 | C-MAPSS: temporal JEPA beats baselines and both floors | PASS | FAIL | — | PASS | no |
| 5 | Koopman/Neural-ODE beat the free-form predictor | PASS | FAIL | — | — | no |
| 6 | Predictability sweep: advantage grows with Ω | **FAIL** | FAIL | — | — | no |
| 7 | Alignment (H2): benefit tracks predictable/relevant overlap | **FAIL** | FAIL | PASS | ³ | no |

¹ **PASTIS has no random-init or raw-feature floor cell at all** — an omission the audit surfaced that
we had not noticed across the entire project. Its floor is *unmeasured*, so V1 is unassessable.
² Seed-tagged CSVs come from a server rigor pass absent from this checkout; the 3-seed t-test is
documented but not locally verifiable. Marked N/A rather than asserting a failure we cannot
substantiate.
³ Additionally fails a **replication** check: corr(α, advantage) = **+0.307** at SNR 2.0 versus
**−0.049** at SNR 0.5 — opposite signs.

**One of seven claims is entitled by its own experiment, and that one only with two criteria
unassessed.**

### 4.1 What the audit does and does not mean

A FAIL is **not** a refutation. It means the experiment cannot settle the question. Claims 2, 4 and 5
fail on V2 alone (single seed) — they are directionally supported and plausibly correct, and would
likely clear the bar under a multi-seed rerun, which is a concrete and affordable repair. Claim 3
fails on V1 for a substantive reason: **no SSL method beat the raw-feature floor on finance**, so the
*ranking among them* orders failures. That is a sharper and more useful statement than "MAE was best,"
and we had not made it.

The audit also demonstrates its own limits. Our first run reported **0/7** — two of those failures were
bugs in the audit (a miscalibrated V1, and treating absent server data as a single-seed failure). The
instrument required validation before its verdicts could be trusted, exactly as the benchmarks it
judges do. We consider this instructive rather than embarrassing, and it motivates the tests in
`tests/test_validity.py`.

---

## 5. The alignment testbed: a confounder, and a benchmark that could not resolve it

### 5.1 The confounder

Our three domains confound *predictability* with *task-relevance*:

| domain | predictable? | does the label depend on the predictable part? |
|---|---|---|
| PASTIS | yes | yes — phenological cycle determines crop class |
| C-MAPSS | yes | yes — degradation trend is what RUL reads |
| finance | no | no |

These three points are equally consistent with:

> **H1** benefit depends on the **predictability** of the process *(our original claim)*
> **H2** benefit depends on the **overlap** between the predictable and task-relevant subspaces —
> predictability being **necessary but not sufficient**

H2 is strictly stronger: it predicts a process can be highly predictable and temporal JEPA still fail,
if the label reads the *unpredictable* component. **No real domain we have distinguishes them.**

### 5.2 A design that separates them

The latent comprises two independent blocks — `z_slow` (AR φ=0.95, predictable) and `z_fast` (white,
unpredictable) — and **both are always rendered into the observation**. Only the label moves:

$$y \;=\; \alpha\cdot\mathrm{std}(z_{\text{slow}}\!\cdot\! w_s)\;+\;(1-\alpha)\cdot\mathrm{std}(z_{\text{fast}}\!\cdot\! w_f)$$

Input predictability is therefore **invariant to α by construction** (V3), asserted to 1e-9 in tests.

### 5.3 Result: inconclusive, at both sensitivities

| α | alignment index | JEPA−MAE @ SNR 2.0 | JEPA−MAE @ SNR 0.5 |
|---|---|---|---|
| 1.00 | 0.97 | −0.053 ± 0.091 | −0.033 ± 0.049 |
| 0.50 | 0.53 | −0.032 ± 0.035 | +0.017 ± 0.036 |
| 0.00 | 0.00 | −0.103 ± 0.032 | −0.027 ± 0.046 |
| | **corr(α, adv)** | **+0.307 (p=0.245)** | **−0.049 (p=0.860)** |

The two configurations **disagree in sign** and neither is significant. The cause is diagnosable:
temporal JEPA beat the raw floor in **0 of 30 cells** (V1). The contrast is between two failures.

**The one positive.** `alignment_index` — ridge past→present, then
`R²(predictable part → y) / R²(full present → y)` — tracks α at **r = +0.957 / +0.950** while Ω is held
flat. It is cheap, linear, label-aware, requires no pretraining, and measures precisely the quantity
H2 identifies. Whether it *predicts downstream benefit* is the open question this benchmark could not
reach.

### 5.4 Why we did not repair the benchmark

We can name concrete engineering reasons for the underpowering: the observation map `tanh(2z)` is
near-linear in range and therefore ridge-invertible; the encoders are 64-dimensional, 2-layer, trained
15 epochs. Repair is feasible.

We did not, because **the motivation would have been to make the hypothesis win.** The distinction we
draw — and recommend — is between repairing a benchmark for a stated engineering reason identified
*before* seeing which way it moves the result, and repairing it until the desired outcome appears.
Both produce the same code. Only the first produces evidence.

---

## 6. Discussion

### 6.1 What we claim

- The five criteria are **general** (nothing in them is specific to JEPA) and **executable**.
- They have **teeth**: applied to seven of our own claims, one survives.
- Two failure modes they catch — floor dominance and post-hoc thresholds — are, in our judgment,
  common in SSL benchmarking and largely invisible in published result tables.
- The alignment confounder (§5.1) plausibly affects **any** cross-domain SSL comparison in which the
  domains were selected because the method works on some and not others.

### 6.2 What we do not claim

- **No new objective, architecture, or theory.** Causal/temporal JEPA variants are prior art;
  distributional JEPA appears as VJEPA (arXiv 2601.14354) and Var-JEPA (arXiv 2603.20111).
- **No frontier-model comparison.** All numbers are frozen linear/kNN probes against
  same-architecture SSL baselines and floors.
- **No resolution of H1 versus H2.** It remains open. Our headline empirical claim is stated only in
  its weaker form: *temporal JEPA helped on the domains where the predictable component was also the
  task-relevant one.*
- **The criteria are not proven complete.** They are five that we needed. Others surely exist.

### 6.3 Threats to validity

The criteria were derived from a single project, so they are shaped by its failure modes. V1's
threshold (25% for ranking claims) and V2's (ratio > 1) are conventions, not derived quantities. The
self-audit is not adversarial — we chose which claims to audit. The alignment testbed is synthetic,
and its real-domain analogue (relabeling C-MAPSS so the target depends on the high-frequency sensor
residual rather than the degradation trend) is untested.

---

## 7. Reproducibility

```bash
pytest -q                          # 102 passed, 3 skipped — offline, no data required
python scripts/audit_claims.py     # regenerates the §4 audit from the committed CSVs
python scripts/alignment_bench.py --device cuda:0 --snr 2.0 --seeds 3   # §5
```

All result CSVs are committed under `runs/`. Every number in this paper is traceable to one of them;
none is estimated. Seeds are explicit and logged. Detailed per-phase reports:
[REPORT_CONSOLIDATED.md](REPORT_CONSOLIDATED.md), [report.md](report.md),
[report_finance.md](report_finance.md), [report_cmapss.md](report_cmapss.md),
[report_predictability.md](report_predictability.md), [report_structured.md](report_structured.md).

## 8. Concrete next steps

1. **Multi-seed rerun** of finance and C-MAPSS — the cheapest repair; would move claims 2, 4, 5 from
   FAIL to a genuine test on V2.
2. **Add a random-init floor cell to the PASTIS matrix** — the audit's most surprising finding is that
   our strongest result has an unmeasured floor.
3. **Power the alignment benchmark** (probe-hostile observation map, larger encoders), then resolve
   H1 versus H2 — with the decision rule pre-registered *first*.
4. **Adversarial audit** — have someone else choose the claims.
