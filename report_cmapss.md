# Multi-Temporal JEPA for Industrial Degradation (NASA C-MAPSS) — Research Report (Phase 3)

**Status:** pipeline complete; comparison running on the **real NASA C-MAPSS turbofan** dataset
(FD001–FD004). This phase is the third modality in a study of **when** a causal temporal-prediction
(JEPA) objective beats reconstructive/contrastive SSL, and it is deliberately chosen to sit at the
*predictable* end of the spectrum.

> **Research question (H1-ind).** Does predicting a *future cycle's* sensor latent from past cycles
> (a causal, temporal JEPA objective) learn more useful frozen representations for engine-degradation
> tasks than spatial sensor-masking / reconstruction / contrastive objectives — and, unlike finance,
> does it **beat the raw-feature floor**?

---

## 1. The three-point thesis (why this phase exists)

The project is mapping the *predictability spectrum* of a modality against whether Temporal JEPA wins:

| Phase | Modality | Temporal structure | Temporal JEPA result |
|---|---|---|---|
| 1 | PASTIS satellite | periodic / seasonal (phenology) — **predictable** | **wins** (beats spatial + MAE/BYOL/SimCLR) |
| 2 | S&P finance | stochastic / near-random-walk — **unpredictable** | **loses** (beats spatial only; below MAE & raw floor) |
| 3 | **C-MAPSS engines** | **monotonic degradation — highly predictable** | *hypothesis: wins, and beats the raw floor* |

C-MAPSS is **not a stress test** (finance was). Engine wear is a smooth latent trajectory
(healthy → wear → failure) — exactly what latent future-prediction should model. It is the
**confirmation case** the thesis needs after the finance loss: if Temporal JEPA wins here *and* beats
the raw-feature floor (which it never did on finance), the spectrum hypothesis is supported on three
independent modalities.

**Finance lessons carried forward:** (1) the bar is the **raw-feature floor + a random-init encoder**,
not just the SSL baselines — both are run as cells; (2) we drop unpredictable-target forecasting;
RUL is forecasting-flavored but *learnable*, so it stays as a regression probe.

---

## 2. Method — reuse, not reimplementation

C-MAPSS is another "panel of N entities × F features over T steps," so the entire generic stack from
Phase 2 is reused: `PanelEncoder`, `FinanceJEPA` (+ build), `engine/train_finance.py` (JEPA +
MAE/BYOL/SimCLR), `masking/asset_mask.py` (now sensor-masking), the JEPA loss + VICReg, EMA, and the
collapse diagnostics. Mapping: **21 sensors = the cross-section (tokens)**, **each operating cycle =
a frame**, **a window of W=40 cycles = one sample**.

**The one adaptation** vs finance: the temporal positional encoding. Cycles are *monotonic*, not
periodic, so a `period=366` day-of-year phase would wrap engines that run >366 cycles (FD004 reaches
543). We threaded a `temporal_period` argument (default 366, behaviour-preserving for satellite +
finance — verified by re-running the full 34-test suite) and set `period=1024` for C-MAPSS so cycle
phases stay monotonic and distinct.

Architecturally: a day's cross-section of sensors is encoded by cross-sensor attention (the "spatial"
ViT), then each sensor's token is integrated across cycles by the temporal transformer. Temporal
JEPA predicts a future cycle's sensor latent; Spatial JEPA predicts a masked subset of the cycle's
sensors from the visible ones.

---

## 3. Data (`data/cmapss_dataset.py`)

- **Source:** NASA C-MAPSS (`scripts/download_cmapss.py` — accepts a local `CMAPSSData.zip` or pulls
  the plain-text files from a public mirror; synthetic monotonic-degradation fallback for offline runs).
- **Subsets:** FD001 (1 condition, 1 fault, 100 engines), FD002 (6 conditions, 260), FD003 (1 cond,
  2 faults, 100), FD004 (6 cond, 2 faults, 249). Engines run 128–543 cycles.
- **Features per sensor (F=3, causal):** condition-normalized value + 1-step delta + 5-cycle rolling
  mean. Constant/uninformative sensors (≈0 variance on TRAIN) are dropped (15 kept in FD001).
- **Condition normalization:** FD002/FD004 mix 6 operating conditions; we KMeans the 3 operating
  settings into 6 regimes (TRAIN only) and z-score each sensor within its regime, so degradation —
  not operating point — drives the features. FD001/FD003 reduce to a global z-score.
- **Labels:** RUL piecewise-linear capped at 125; 4-stage health {healthy/early/late/critical};
  anomaly = RUL ≤ 20 (near-failure). All derived from the standard C-MAPSS RUL.
- **Split:** C-MAPSS ships separate train (run-to-failure) and test (truncated) **engines** — no
  leakage by construction. Pretrain + fit probes on TRAIN-engine windows; score on TEST-engine
  windows. A `std_protocol` set holds one window at each test engine's last cycle (vs RUL_FDxxx.txt).

---

## 4. The five downstream tasks (`eval/cmapss_tasks.py`)

Freeze the encoder; mean-pool each window to one embedding; fit light probes on TRAIN, score on TEST.

| Task | Probe | Metric |
|---|---|---|
| **RUL regression** | ridge | R², RMSE, rank-IC (windowed) + **standard last-cycle RMSE + PHM08 score** vs RUL.txt |
| **Health-stage classification** | logistic | accuracy, macro-F1 (4 stages) |
| **Anomaly detection** | kNN-distance to train (unsupervised) | AUROC, avg-precision |
| **Clustering** | KMeans vs health stage (training-free) | NMI, ARI, silhouette |
| **NN retrieval** | cosine kNN in train-embedding space | health precision@k, neighbour-RUL rank-IC |

Controls: **random** (untrained encoder) and **raw_features** (the five probes on pooled raw sensors,
no encoder) — the floors that finance failed to clear.

---

## 5. Protocol

Fix everything except the objective. Temporal/Spatial JEPA + MAE/BYOL/SimCLR train the same
`PanelEncoder` for the same epochs on the same TRAIN-engine windows; JEPA read through the temporal
pathway, baselines through the per-cycle pathway. Config: encoder width 128 (4+4 depth), narrow
predictor 64, `temporal.period 1024`, VICReg λ_v=1.0/λ_c=0.04, 20 epochs, batch 256. Horizon sweep
Δ∈{1,5,20} (degradation is slow, so Δ=1 may be near-trivial — longer Δ tests whether the model must
learn the degradation *rate*). Collapse monitored; M1 gate (`scripts/cmapss_smoketest.py`) passes.

```bash
python scripts/download_cmapss.py            # NASA mirror, or --zip CMAPSSData.zip
python scripts/run_cmapss_matrix.py --config configs/model/cjepa.yaml \
       --data configs/data/cmapss.yaml --device cuda:0   # all FDs, all cells -> runs/cmapss_results.csv
python scripts/aggregate_cmapss.py           # per-FD comparison tables + per-task verdict
```

---

## 6. Results (real C-MAPSS, TEST engines, seed 0)

Frozen-encoder probes on the held-out TEST engines of all four subsets. Reproduce with
`scripts/run_cmapss_matrix.py` then `scripts/aggregate_cmapss.py`.

### 6.1 Headline — Temporal JEPA vs the field

Across all four subsets × 13 metrics = **52 metric-subsets**, how often Temporal JEPA (Δ=1) wins:

| Temporal JEPA beats… | metric-subsets won |
|---|---|
| SimCLR | **51 / 52** |
| MAE | **46 / 52** |
| raw features *(floor)* | **45 / 52** |
| Spatial JEPA | **43 / 52** |
| BYOL | **43 / 52** |
| random-init *(floor)* | **40 / 52** |

**Temporal JEPA is the best SSL objective on C-MAPSS** (beats Spatial JEPA, MAE, BYOL, SimCLR), and
— unlike finance — **it clears the raw-feature floor** (45/52).

### 6.2 RUL prediction (the canonical task), per subset

| RUL metric | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **Temporal JEPA — R²** | **0.677** | **0.634** | **0.806** | **0.667** |
| best SSL baseline — R² | 0.577 (mae) | 0.551 (mae) | 0.662 (spatial) | 0.566 (mae) |
| _random init — R²_ | _0.651_ | _0.600_ | _0.714_ | _0.596_ |
| _raw features — R²_ | _0.344_ | _0.291_ | _0.344_ | _0.183_ |
| **Temporal JEPA — std-RMSE** ↓ | 16.4 | 26.2 | 14.8 | 27.0 |
| **Temporal JEPA — PHM08** ↓ | 471 | 6 465 | 425 | 5 128 |

The standard last-cycle RMSE (16.4 on FD001) is in the expected band for a **frozen linear probe**
(supervised end-to-end nets reach ~12–16 but fine-tune the whole encoder; the ordering across
objectives is the result, not the absolute number).

### 6.3 Representation-quality tasks (Temporal JEPA / best baseline / random / raw)

| task (metric ↑) | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Health acc | **0.74**/0.74/_0.78_/_0.66_ | **0.73**/0.69/_0.71_/_0.61_ | **0.86**/0.81/_0.83_/_0.73_ | **0.81**/0.77/_0.79_/_0.69_ |
| Anomaly AUROC | **0.99**/0.98/_0.98_/_0.93_ | **0.98**/0.97/_0.96_/_0.89_ | 0.98/**0.99**/_0.99_/_0.92_ | **0.97**/0.97/_0.95_/_0.81_ |
| Retrieval p@k | **0.66**/0.65/_0.63_/_0.58_ | **0.67**/0.61/_0.61_/_0.52_ | **0.77**/0.74/_0.72_/_0.70_ | **0.78**/0.72/_0.70_/_0.65_ |

Clustering (NMI/ARI) is weak for **every** method (~0.1) and is the one task where the floors
sometimes edge ahead — health stages form a continuum, not separated clusters, so KMeans purity is
a poor probe here (reported in full in the CSV; not a JEPA-specific failure).

### 6.4 The decisive nuance — the random-init control

Temporal JEPA beats the **untrained** encoder on **40/52** metric-subsets, but the margin is
**difficulty-dependent**:

| Temporal JEPA beats random on… | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| metrics (of 13) | 8 | **12** | 9 | **11** |

On the **easiest** subset (FD001: 1 condition, 1 fault) a random network nearly ties Temporal JEPA
(R² 0.651 vs 0.677; random even wins PHM08/health) — the degradation signal is so strong that a
random temporal-attention projection already captures it, and a linear probe extracts it. On the
**harder multi-condition** subsets (FD002/FD004: 6 conditions), learning matters: Temporal JEPA
pulls clearly ahead of random (12/13, 11/13). So **the value of the pretext task grows with task
difficulty** — exactly what you'd hope.

### Findings (the verdict)
- **H1-ind supported.** On predictable industrial degradation, Temporal JEPA is the **best SSL
  objective** (beats Spatial JEPA + MAE/BYOL/SimCLR on 43–51 / 52 metric-subsets) and **beats the
  raw-feature floor** (45/52) — the bar finance never cleared. This is the clean win that completes
  the three-point thesis (PASTIS win, finance loss, **C-MAPSS win**).
- **Temporal > Spatial replicates a third time** (43/52) — the core satellite ordering holds across
  all three modalities.
- **Honest caveat (why the controls matter).** On the easiest subset a *random-init* encoder is
  competitive; the learned advantage is real but modest there and only becomes decisive as the
  signal gets messier (multi-condition). Reported prominently because the Phase-2 lesson is that the
  floors — not the SSL baselines — are the real bar.

### 6.5 Ablations (FD001): horizon & VICReg

| variant | RUL R² ↑ | std-RMSE ↓ | PHM08 ↓ | health ↑ | retrieval ↑ | emb-std |
|---|---|---|---|---|---|---|
| Temporal Δ=1 | 0.677 | 16.4 | 471 | 0.744 | **0.664** | 0.381 |
| Temporal Δ=5 | 0.671 | 16.5 | 466 | 0.733 | 0.659 | 0.375 |
| Temporal Δ=20 | 0.655 | 17.4 | 547 | 0.725 | 0.657 | 0.375 |
| Temporal Δ=1, **VICReg-off** | 0.703 | 16.1 | 438 | 0.789 | 0.636 | **0.262** |

- **Horizon is nearly flat** (R² 0.677→0.671→0.655 across Δ=1→20; all far above every baseline) — a
  *predictable* trajectory is learnable whether you predict 1 or 20 cycles ahead. Sharp contrast with
  finance, where longer horizons monotonically destroyed the result (the unpredictable-target trap).
- **VICReg is less critical here than on satellite/finance.** Turning it off drops embedding variance
  (std 0.38→0.26, retrieval 0.66→0.64) — heading toward collapse — yet the RUL/health probes actually
  tick *up* slightly: the degradation signal is dominant enough to survive reduced regularization. On
  PASTIS/finance, VICReg-off was catastrophic (eff-rank → ~2). Another marker that C-MAPSS is the
  "easy / strong-signal" end of the spectrum — anti-collapse matters less when one signal dominates.

---

## 7. Honest caveats
1. **Frozen-probe, not fine-tuned.** Absolute RUL RMSE/PHM08 will be above end-to-end supervised
   C-MAPSS leaders (which fine-tune the whole net); the *relative ordering across objectives* is the
   result, exactly as in Phases 1–2.
2. **Short test engines** (< window=40 cycles) are excluded from the standard last-cycle benchmark;
   the count kept is logged per FD.
3. **Health/anomaly labels are derived from RUL** (heuristic thresholds), as is standard when no
   official stage labels exist; the synthetic fallback has exact health ground truth as a cross-check.
4. **Single seed / single split so far** (C-MAPSS supplies the split); multi-seed error bars via
   `--seed` are the next rigor step.

## 8. Reproducibility
Seeds fixed; per-(fd,cell) encoders saved to `runs/cmapss/`; every metric + GPU-hours logged per
cell to `runs/cmapss_results.csv` (crash-safe). Tests: `pytest tests/test_cmapss_*.py` (12, offline).
M1 gate: `python scripts/cmapss_smoketest.py --device cuda:0`.
