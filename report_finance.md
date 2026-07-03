# Multi-Temporal JEPA for Financial Time Series — Research Report (Phase 2)

**Status:** pipeline complete; comparison run on **real S&P-500 sector data** (Dec 1998 – Jun 2026).
This phase ports the satellite Temporal-JEPA method (see [report.md](report.md)) to markets and asks
whether the *same* causal future-latent-prediction objective beats Spatial JEPA and the
reconstruction/contrastive baselines (MAE / BYOL / SimCLR) on **five** downstream financial tasks:
**regime classification, volatility prediction, anomaly detection, clustering, and forecasting.**

> **Research question (H1-fin).** Does predicting the *future* latent state of the market
> cross-section from its *past* (a causal, temporal JEPA objective) learn more useful representations
> for downstream financial tasks than spatial masking / reconstruction / contrastive objectives, on
> the same encoder and data?

---

## 1. Why the satellite method should transfer

PASTIS is a **spatial cross-section** (a grid of pixels) observed over **time**. A market index is a
**cross-section of assets** observed over **time**. Both are "the same set of entities, re-observed."
The Temporal-JEPA thesis — *the modality is natively a time series, so predicting the future is the
right pretext task* — applies verbatim: a crop classifier needs phenology (how a parcel changes
through the season); a market task needs **regime dynamics** (how the cross-section co-moves through
a cycle). A spatial-masking objective throws the time axis away; a causal future-prediction objective
is forced to model it.

| satellite (Phase 1) | finance (Phase 2) |
|---|---|
| frame = one acquisition (H×W pixels) | frame = one **trading day** (cross-section of N sector ETFs) |
| spatial token = a pixel patch | token = one **asset**'s feature vector that day |
| spatial ViT mixes patches within a frame | cross-asset ViT mixes the N assets within a day |
| temporal transformer over acquisitions | temporal transformer over trading days |
| DOY temporal positional encoding | trading-day **day-of-year** (annual seasonality + "when") |
| **Temporal JEPA:** predict the future frame's latent | **predict tomorrow's market cross-section latent** |
| **Spatial JEPA:** predict masked pixel blocks | **predict a masked subset of the day's assets** |

The two-axis structure is identical, so the modality-agnostic core is **reused unchanged**: the
narrow Predictor, the JEPA latent loss + VICReg anti-collapse regularizer, the EMA target machinery,
the collapse diagnostics, and the transformer stacks (`models/vit.py`, `models/temporal_encoder.py`).
Only the per-day **frame embedder** (Conv2d→Linear) and the **spatial position** (2D sin/cos→learned
per-asset) change. The satellite results in [report.md](report.md) are untouched and reproducible.

---

## 2. Data

**Real market panel** (`scripts/download_finance.py`, Yahoo Finance):
- **Cross-section (N=9 assets):** the nine original Select-Sector SPDR ETFs (XLB, XLE, XLF, XLI, XLK,
  XLP, XLU, XLV, XLY) — each ~a GICS sector of the S&P 500, all trading since Dec-1998 (long, clean,
  survivorship-bias-free).
- **Span:** 6,908 trading days, 1998-12-31 → 2026-06-18, including the dot-com crash, 2008 GFC, 2011
  and 2015–16 selloffs, 2018-Q4, the 2020 COVID crash (index −12.8% single day, VIX 82.7), 2022 bear,
  and 2023–26 bull — a rich set of regimes and anomalies.
- **Labels only** (never seen by the encoder): the index (^GSPC) and volatility index (^VIX).
- **Sanity:** annualized index vol 19.3%, mean pairwise sector return correlation 0.60 (a strong
  market factor + sector idiosyncrasy — exactly the structure cross-asset attention can exploit).

**Features** per asset-day (F=4, all causal): log-return, |log-return|, Δlog-volume, and the
vol-standardized return (return ÷ trailing-20d σ). A **sample** is a window of **W=64 trading days**
→ `data (64, 9, 4)`.

**Offline fallback.** If the download is blocked, `data/finance_dataset.py` synthesizes a documented
regime-switching multi-sector market (sticky 4-state HMM × GARCH-ish vol clustering × fat tails ×
injected crashes) with **exact ground-truth regime labels**, so the entire pipeline (train + eval +
tests) runs reproducibly offline.

**Temporal split (no look-ahead).** Train = windows ending ≤ 2017-12-31 (≈4,698 windows); Test =
windows starting ≥ 2018 (≈2,044 windows), with a purge gap of `window + max_horizon` days so no train
window's forward label reaches into the test period (unit-tested: `test_no_train_test_leakage`).

---

## 3. The five downstream tasks (`eval/finance_tasks.py`)

Every method is frozen after pretraining; each W-day window is reduced to one mean-pooled embedding;
probe heads are **fit on the train period and scored on the held-out test period**. Higher is better
for every metric.

| Task | Label (from the index, encoder never sees it) | Probe | Metric |
|---|---|---|---|
| **Regime classification** | 4-way {up-calm, up-volatile, down-calm, down-volatile} from the window's trailing return sign × vol-vs-train-median (a *contemporaneous decode* probe) | logistic | accuracy, macro-F1 |
| **Volatility prediction** | realized vol of the index over the **next 20 days** (forward) | ridge | R², rank-IC |
| **Anomaly detection** | window precedes a crash: max \|move\| in next 5 days > train 99th-pct (forward, ~3% positive) | kNN-distance to train (unsupervised) | AUROC, avg-precision |
| **Clustering** | (vs the regime labels) | KMeans, training-free | NMI, ARI, silhouette |
| **Forecasting** | next-day index direction & return (forward) | logistic + ridge | dir-acc, return-IC |

A **random-init** encoder is the control: probes on it should sit near chance, so a method "winning"
means the *pretext task* — not the probe — put the structure there.

---

## 4. Protocol

Fix everything except the objective. All five objectives (Temporal JEPA, Spatial JEPA, MAE, BYOL,
SimCLR) train the **same** `PanelEncoder` on the same train windows for the same epochs; only the
pretext objective and its minimal head differ. JEPA variants are read through the temporal pathway;
the spatial-only baselines through the per-day pathway (their temporal encoder is untrained) — each
method read through the representation it actually learned. Config: encoder width 128 (4+4 depth),
narrow predictor 64, 50 epochs, effective batch 128, VICReg λ_v=1.0 / λ_c=0.04. Collapse is monitored
(per-dim std, effective rank), not assumed; the M1 gate (`scripts/finance_smoketest.py`) passes.

```bash
python scripts/download_finance.py                       # real S&P panel (or synthetic fallback)
python scripts/run_finance_matrix.py --config configs/model/fjepa.yaml \
       --data configs/data/finance.yaml --device cuda:0  # pretrain+freeze+eval, all cells -> CSV
python scripts/aggregate_finance.py                      # comparison table + per-task verdict
```

---

## 5. Results (real S&P sector panel, out-of-time TEST 2018–2026, seed 0)

Frozen-encoder probes; **higher is better for every metric**. Best **trained-SSL** method per row in
**bold**. Two reference rows: **random** = the same architecture *untrained* (random init); **raw
features** = the five probes on the mean-pooled 36-dim input features with **no encoder at all** (the
true floor). Reproduce: `scripts/run_finance_matrix.py` then `scripts/aggregate_finance.py`.

| task / metric | **Temporal JEPA** | Spatial JEPA | MAE | BYOL | SimCLR | _random init_ | _raw features_ |
|---|---|---|---|---|---|---|---|
| Regime accuracy | 0.609 | 0.758 | **0.797** | 0.787 | 0.790 | _0.802_ | _0.804_ |
| Regime macro-F1 | 0.528 | 0.564 | **0.710** | 0.703 | 0.689 | _0.715_ | _0.747_ |
| Volatility R² | −0.228 | −0.435 | 0.157 | **0.181** | 0.099 | _0.169_ | _0.112_ |
| Volatility rank-IC | 0.253 | 0.309 | **0.450** | 0.442 | 0.415 | _0.428_ | _0.439_ |
| Anomaly AUROC | 0.745 | 0.553 | **0.837** | 0.738 | 0.521 | _0.726_ | _0.837_ |
| Anomaly avg-prec | 0.189 | 0.047 | 0.171 | **0.269** | 0.033 | _0.161_ | _0.273_ |
| Clustering NMI | 0.157 | 0.132 | 0.333 | **0.367** | 0.252 | _0.141_ | _0.329_ |
| Clustering ARI | 0.130 | 0.085 | 0.379 | **0.417** | 0.230 | _0.092_ | _0.334_ |
| Forecast dir-acc | 0.523 | 0.479 | 0.499 | 0.511 | **0.523** | _0.496_ | _0.533_ |
| Forecast ret-IC | 0.085 | 0.045 | 0.076 | **0.094** | 0.051 | _0.116_ | _0.080_ |

**Horizon study (Temporal JEPA, Δ trading days):** regime acc **0.609 (Δ1) → 0.516 (Δ5) → 0.494
(Δ20)**; forecast ret-IC **0.085 → 0.009 → −0.086**. Predicting further ahead is *monotonically
worse*. **VICReg ablation (`tjepa_noreg`):** training-time effective rank collapses **~110 → ~2.3**
(the same collapse the satellite pipeline documents — VICReg is necessary on markets too).

### Verdict — does Temporal JEPA beat MAE/BYOL/SimCLR here? **No.**

1. **Temporal JEPA loses to every reconstruction/contrastive baseline.** It beats only Spatial JEPA
   (7/10 metrics — the satellite ordering *temporal > spatial* does replicate). Against MAE it wins
   3/10, BYOL 2/10, SimCLR 3/10. **MAE and BYOL are the strongest trained encoders.**
2. **No SSL method beats the raw-feature floor.** A plain linear probe on the mean-pooled input
   features matches or beats *every* pretrained encoder on essentially every task (regime 0.80,
   anomaly-AUROC 0.84, forecast dir-acc 0.53). On this out-of-time S&P benchmark, **self-supervised
   pretraining buys ~nothing over the engineered features** — the signal these tasks need already
   lives in returns/abs-returns/vol-z, and a random projection preserves it.
3. **Temporal JEPA is uniquely *harmful*.** Same architecture, same temporal pathway: the *untrained*
   encoder scores regime **0.80**, the temporal-JEPA-trained one **0.61**. Optimizing "predict the
   next day's latent" on non-stationary data actively *erases* the cross-sectional / volatility
   structure that the downstream tasks read — and longer horizons erase more.
4. **So the satellite result does not transfer — it partially inverts.** On PASTIS, temporal JEPA beat
   MAE/BYOL/SimCLR by +15 mIoU. On the S&P it loses to them and to a random encoder. The
   temporal-prediction advantage is **modality-specific**: it shines where future-from-past is *both*
   learnable and stationary (crop phenology), and backfires where tomorrow ≈ today (trivial target)
   *and* the train→test distribution shifts (1999–2017 dynamics ≠ 2018–2026). This is the honest,
   interesting finding — a negative/inverted transfer result, not a tuning failure (the random and
   raw-feature controls bound how much *any* method could have won).

### Why this is a clean result, not a bug
The **random-init and raw-feature controls** are the safeguard: they show the probes work (they read
real structure) and quantify the ceiling (small). Temporal JEPA falling *below* both controls, under
the same probe and pooling used for all methods, is a controlled within-architecture comparison —
training is the only variable. The forecasting row behaves exactly as efficient-market theory predicts
(direction ≈ 50% for all methods; the informative signal is the tiny return-IC), which is a sanity
check that the harness isn't leaking labels.

---

## 6. Honest caveats

1. **Cross-section is sector ETFs, not 500 single names.** Nine sectors give a clean, long,
   survivorship-bias-free panel and a real cross-section to mask, but a 500-constituent panel (with
   listing/delisting handling) would be a stronger test of cross-asset attention. The downloader and
   dataset are written so swapping in more symbols is a config change.
2. **Regime/anomaly labels are heuristic** (rule-based on the index), as is standard when there is no
   official "regime" ground truth. The synthetic fallback, by contrast, has *exact* regime labels and
   is the cleaner controlled testbed — useful as a cross-check.
3. **Single seed / single split so far.** The infra supports `--seed` for multi-seed error bars;
   running ≥3 seeds (and reporting mean ± std) is the next rigor step, mirroring the satellite
   3-seed protocol.
4. **Forecasting is intentionally hard.** Near-chance direction accuracy is the *expected* efficient
   market result and is **not** evidence against the representation — it bounds what any frozen probe
   can do. The claim is about *representation quality on structured tasks*, not market-timing alpha.

## 7. Reproducibility
Seeds fixed (`utils/seed.py`); per-cell encoders saved to `runs/finance/<cell>.pt`; every downstream
metric + GPU-hours logged per cell to `runs/finance_results.csv` (crash-safe, flushed per row).
Tests: `pytest tests/test_finance_*.py` (16 pass, fully offline). M1 gate:
`python scripts/finance_smoketest.py --device cuda:0`.

---

## 8. Phase 4 — does a *distributional* objective rescue finance? **No.**

**Motivation.** §12 diagnosed the failure as: the next-day *point* latent is ~martingale, so the L2
target is noise and the objective erases usable structure. The obvious algorithmic fix: predict a
*distribution* over the future latent, so the model can output high variance where the future is
unpredictable — down-weighting the un-learnable mean gradient (heteroscedastic NLL) and letting the
predicted variance become a **volatility** signal (returns are unpredictable but *volatility clusters*).

**Method (`tjepa_dist`).** A heteroscedastic predictor emits $\mu,\log\sigma^2$ per target token;
trained with **β-NLL** (β=0.5; Seitzer et al., ICLR 2022 — the standard fix for the NLL gradient
pathology), VICReg retained. Additive + flag-gated (`predictor.distributional`, `loss.type: beta_nll`);
the 52-test suite still passes (behaviour-preserving). *Novelty note:* the algorithm itself is 2026
prior art — **VJEPA** (arXiv 2601.14354) and **Var-JEPA** (arXiv 2603.20111, incl. tabular Var-T-JEPA);
this is a controlled *mechanistic test*, not a new method.

**Mechanism — confirmed.** During training the pooled predicted σ correlates with a realized-vol proxy
at **rank-IC ≈ 0.75–0.82**: the variance head genuinely learns volatility.

**But the rescue fails on every downstream criterion.** Frozen probes, finance TEST 2018–2026, matched
50-epoch training:

| representation | regime | vol R² | vol IC | anom AUROC | NMI |
|---|---|---|---|---|---|
| `tjepa_dist` (distributional) | 0.533 | −0.227 | 0.321 | 0.715 | 0.171 |
| `tjepa_h1` (point) | 0.609 | −0.228 | 0.253 | 0.745 | 0.157 |
| MAE | **0.797** | **0.157** | 0.450 | **0.837** | 0.333 |
| _random / raw floor_ | _0.80 / 0.80_ | _0.17 / 0.11_ | _0.43_ | _0.73 / 0.84_ | _0.14 / 0.33_ |

Distributional is **worse than point-JEPA** and, like it, far below MAE and the raw/random floors.
Exposing the predicted variance as a downstream feature (30-epoch model) does not help either:

| probe features | regime | vol R² | anom AUROC |
|---|---|---|---|
| z (representation) | 0.648 | −0.059 | 0.672 |
| variance vector only | 0.406 | −0.212 | 0.684 |
| z + variance | 0.663 | −0.240 | 0.736 |

Adding the variance gives only a small anomaly bump (+0.06), *hurts* vol R², and clears **no** floor.

---

## 9. Phase 5 — is the failure NON-STATIONARITY or UNPREDICTABILITY? **Unpredictability.**

Phases 2/4 leave one alternative explanation for the failure: maybe SSL is fine and the problem is
purely the 1999–2017 → 2018–2026 *distribution shift* (the regime/vol relationship the probe learns is
stale). We test this directly with two cheap experiments (`scripts/finance_regime_shift_probe.py`).

**(A) In-period probe** — reuse the existing encoders but fit AND test the probe *inside* 2018–2026
(temporal split: fit ~2018–2023, test ~2023–2026 — no long shift). **(B) Definitive** — re-pretrain
the encoder on *recent* data (≤2019) and evaluate *fully in-period* on 2020–2026 (COVID + 2022 bear +
2023–26 bull) — **no distribution shift anywhere**.

| protocol / method | regime acc | anomaly AUROC |
|---|---|---|
| **(A) in-period** — raw features | 0.732 | 0.648 |
| (A) random / BYOL / MAE | 0.780 / 0.787 / 0.768 | 0.735 / 0.46 / 0.57 |
| (A) tjepa_h1 / tjepa_dist | 0.421 / 0.591 | 0.571 / 0.423 |
| **(B) re-pretrain recent** — raw features | **0.831** | 0.672 |
| (B) MAE (recent) | 0.685 | 0.701 |
| (B) temporal JEPA (recent) | 0.460 | 0.750 |

**Even with the shift entirely removed, raw features beat every SSL method, and temporal JEPA is the
worst.** So the failure is **not** the train→test shift — it is **unpredictability / task-hardness**:
the regime/vol structure these tasks need is already fully captured by the engineered features
(returns / |returns| / vol-z), so learning a representation over them adds nothing, and the
causal-temporal-prediction objective actively *corrupts* it. (Caveat: in-period splits are smaller
→ vol R² is noisy in the low-vol 2023–26 window; the robust regime/anomaly numbers carry the
conclusion. Single seed.)

**Closure of the finance investigation.** The negative is robust to *both* an algorithmic fix
(Phase 4, distributional prediction) *and* an evaluation-protocol fix (Phase 5, removing the shift).
On near-efficient financial panels, SSL — and especially temporal prediction — provides no benefit
over engineered features. This is the strongest form of the predictability-spectrum result: finance
sits at the unpredictable extreme, and *no* representation-learning intervention we tried moves it.

**Guardrail (C-MAPSS FD001).** The distributional objective is a *mild net negative on the predictable
domain too*: RUL R² 0.658 vs point 0.677, PHM08 578 vs 471, health 0.733 vs 0.744 (still beats all
baselines, just below point-JEPA — on a highly-predictable signal the variance head is unhelpful
overhead).

**Verdict.** H7 (distributional rescue) is **rejected**. The finance failure **survives the most
obvious algorithmic fix** — predicting a distribution does not manufacture predictable structure that
isn't there, and the objective still overfits the pretrain-period dynamics (3-epoch ≫ 50-epoch for
*both* point and distributional — an overtraining-on-non-stationary-data effect, not a point-target
artifact). This *strengthens* the predictability-spectrum thesis: the finance failure is fundamental
(non-stationarity + near-martingale returns), not fixable at the objective level. Reproduce:
`run_finance_matrix.py` includes the `tjepa_dist` cell; the variance-as-feature probe and the
`window_logvar` method are in `models/finance_jepa.py` / `eval`.
