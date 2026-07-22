# Multi-Temporal JEPA — when does predicting the future help?

Self-supervised representation learning by **causal latent prediction**: predict the *future* latent
state from *past* observations — a world-model-flavored JEPA objective — and test it against spatial
JEPA (I-JEPA-style), MAE, BYOL and SimCLR across **three unrelated modalities**, with honest floors
(random-init and raw-feature) everywhere.

> **Research question.** Does a causal temporal-prediction objective learn better representations
> than reconstruction/contrastive SSL — and *can you predict in advance where it will work?*

**Answer: only where the latent trajectory carries predictive information — and yes, you can measure
that beforehand.** Temporal JEPA wins big on satellite crop series and turbofan degradation, and
fails on financial panels so thoroughly it scores below its own untrained initialization.

📄 **Start here: [PAPER.md](PAPER.md)** — *When Is a Self-Supervised Benchmark Entitled to Its
Conclusion?* The submission-track write-up: five executable **validity criteria** for mechanistic SSL
claims, derived from this project's own failures, and a **self-audit in which 1 of our 7 headline
claims survives**. [REPORT_CONSOLIDATED.md](REPORT_CONSOLIDATED.md) is the full empirical record;
this README is the engineering reference.

```bash
python scripts/audit_claims.py    # audit this project's own claims against V1-V5
```

---

## Headline results

| domain | outcome | key number |
|---|---|---|
| **PASTIS** Sentinel-2 crop series | ✅ **WIN** | +6.0 mIoU over spatial JEPA (p=0.041), +15–16 over MAE/BYOL/SimCLR |
| **NASA C-MAPSS** turbofan | ✅ **WIN** | beats SimCLR 51/52 metrics; best of 7 cells on the standard NASA RUL benchmark |
| **S&P-500** sector panel | ❌ **LOSS** | regime acc **0.61 trained vs 0.80 untrained** — training actively hurts |

### PASTIS — conv mIoU, val fold, mean ± std over 3 seeds, paired t-test vs temporal

| Method | conv mIoU | Δ vs temporal | p |
|---|---|---|---|
| **Temporal JEPA (Δ=1)** | **22.3 ± 1.8** | — | — |
| Spatial JEPA | 16.2 ± 0.4 | +6.0 | **0.041** |
| Spatial JEPA — compute-matched (3.5× epochs) | 15.8 ± 1.2 | +6.5 | **0.036** |
| SimCLR | 7.3 ± 0.8 | +15.0 | **0.009** |
| BYOL | 7.1 ± 0.9 | +15.2 | **0.001** |
| MAE | 6.5 ± 1.1 | +15.8 | **0.009** |

The win **is the objective, not compute** (compute-matched spatial gains nothing), is
**horizon-insensitive** (Δ=1–8 all ≈22), and **grows as labels shrink** — the temporal-vs-spatial gap
goes from +37% at full labels to **+100% at 1%**. Supervised U-TAE (63.1) is a ceiling, not a peer.

### Why finance fails — the mechanism, measured

| observed series | Ω (spectral) | **past→future MI** | reads as |
|---|---|---|---|
| finance — index daily returns | 0.053 | **0.01** | **≡ white noise** |
| C-MAPSS — engine sensors | 0.359 | **25.9** | structured |

**There is no learnable future in returns.** The failure survives both an algorithmic rescue
(distributional/β-NLL prediction, Phase 4) and a protocol rescue (removing distribution shift
entirely, Phase 5) — so it is unpredictability, not non-stationarity and not a bug.

---

## Anti-collapse note (important for temporal data)

Standard JEPA (EMA target + stop-grad + narrow predictor) **collapses on real PASTIS**: consecutive
acquisitions of a field are nearly identical, so "predict the future latent" is solvable by emitting
a constant (loss→0, per-dim std→0.04). The fix — part of the architecture, **on by default** — is a
**VICReg-style variance–covariance regularizer** on the trainable context embedding
(`objectives/jepa_loss.py: variance_covariance_reg`, weights `loss.var_coeff` / `loss.cov_coeff`;
set both to 0 to recover pure I-JEPA and reproduce the collapse). It is necessary on markets too:
effective rank collapses ~110 → 2.3 without it.

---

## 1. The idea

```
PASTIS series X[T,10,128,128] + DOY dates d[T]
   │  causal split by date  →  context = past frames | target = future frame (gap = horizon Δ)
   ▼
per-frame PatchEmbed (Conv 10→D, P=8 → 16×16=256 tokens/frame) + 2D spatial pos + DOY temporal pos
   ├─► CONTEXT path:  spatial ViT (per frame) → temporal transformer (past frames) → z_ctx
   │                                                              │
   │                  PREDICTOR (narrow, 384 < encoder 512): z_ctx + mask tokens(future pos/DOY)
   │                                                              ▼  ẑ_future
   └─► TARGET path (EMA encoder, stop-grad): encode FUTURE frame → LayerNorm → z_future
                                                                  │
        Loss = ‖ ẑ_future − sg(LayerNorm(z_future)) ‖²  +  var/cov reg on z_ctx  ◄┘
```

Anti-collapse: **EMA target** + **stop-gradient** + **narrow predictor** + **VICReg var/cov**.

**The panel abstraction is what makes this multi-domain.** Finance and C-MAPSS reuse the same stack
via `(B, W, N, F)` = batch × window × entities × features, where entities are *assets* (finance) or
*sensors* (C-MAPSS) instead of image patches. One encoder, one objective, three modalities.

---

## 2. Repository map

```
configs/      yaml configs (data + model/training); see §4
data/         PASTIS / finance / C-MAPSS datasets, synthetic dynamics, collate, splits
masking/      spatial multi-block sampler, causal past→future split, asset masking
models/       patch embed, pos enc, ViT, temporal enc, predictor, JEPA assembly,
              graph backbone, structured (Koopman/ODE) predictors, latent Kalman filter
objectives/   JEPA latent loss (L2 + β-NLL) + var/cov reg + MAE/BYOL/SimCLR losses
engine/       JEPA + finance training loops, baseline drivers, EMA, collapse diagnostics
eval/         probes (mIoU / k-NN / few-shot), finance & C-MAPSS task suites,
              predictability indices, feature-space analysis
utils/        seeding, config, checkpointing, GPU-hour metering, device knob
scripts/      download, smoketests, experiment matrices, aggregation, benchmarks
tests/        17 test modules; the synthetic ones run the full wiring without any data
```

---

## 3. Module-by-module logic

### `data/`
- **`pastis_dataset.py`** — reads `metadata.geojson`, indexes patches by fold, loads
  `DATA_S2/S2_<id>.npy` `(T,10,128,128)`, converts dates to **calendar DOY** in [1,366], normalizes
  per band. Real frames are **front-packed** by `collate_variable_length`, which builds the boolean
  `pad_mask` the temporal split relies on.
- **`finance_dataset.py`** — S&P sector panel → `(B, W, N_assets, F)` windows; features are
  returns / |returns| / vol-z; strict out-of-time splits with a purge gap.
- **`cmapss_dataset.py`** — NASA turbofan → `(B, W, N_sensors, F)` windows, per-engine splits so no
  engine appears in both train and test.
- **`synthetic_dynamics.py`** — latent-dynamics generator spanning the predictability axis
  (periodic → AR(1)-φ-sweep → Lorenz → white), rendered nonlinearly at fixed SNR, with the clean
  latent kept as the recovery target.
- **`transforms.py`** — band stats over train folds only, `two_view_augment` (BYOL/SimCLR).
  **`splits.py`** — official 5-fold, `cv_split`, stratified few-shot subsets.

### `masking/`
- **`multiblock.py`** — I-JEPA spatial sampler: 4 target blocks, then a context block with
  overlapping tokens removed → context ∩ target = ∅ (no trivial copy).
- **`temporal_mask.py`** — `split_past_future`: causal split with horizon Δ; context date < target
  date guaranteed (no future leakage; unit-tested).

### `models/`
- **`patch_embed.py`** / **`pos_embed.py`** — `Conv2d(10→D, k=stride=P)`; 2D sin/cos spatial and
  **DOY** sin/cos temporal (phase `d/366·2π`, periodic over a year; padded frames zeroed).
- **`vit.py`** / **`temporal_encoder.py`** — fused-SDPA attention with key-padding mask, pre-norm
  blocks; factorized space→time (folds N into batch, attends over time). Optional `grad_checkpoint`.
- **`predictor.py`** — **narrow** transformer (`pred_dim` < encoder dim; 384 vs 512). Optional
  `predict_variance` head returns `(mu, logvar)` for the distributional objective.
- **`jepa.py`** — `SITSEncoder` + `JEPA` (context encoder, frozen EMA target, predictor);
  `build_model(cfg)` asserts/clamps the predictor bottleneck. `encoder.spatial_backbone: vit|graph`.
- **`graph_layers.py`** / **`graph_encoder.py`** — dependency-light GNN (no torch_geometric):
  `grid_edge_index`, `scatter_mean`, GraphSAGE-mean blocks, `GridGraphEncoder` (drop-in for
  `ViTEncoder`), and `GraphSITSEncoder` swapping the spatial ViT for local message passing.
- **`structured_predictors.py`** — `KoopmanPredictor` (linear operator in a learned observable
  space, exposes `spectral_radius()`) and `NeuralODEPredictor` (RK4 latent flow, identity-init).
- **`latent_filter.py`** — Kalman filter over latents: encoder gives measurements, Koopman operator
  is the process model; `lkf_report` isolates `dynamics_gain` against a static A=0 baseline.
- **`finance_jepa.py`** / **`finance_encoder.py`** — `PanelEncoder` + `FinanceJEPA` (shared by
  finance and C-MAPSS), with `distributional`, `predictor_type`, and multi-`horizons` flags.

### `objectives/`
- **`jepa_loss.py`** — `jepa_latent_loss` (detach target, LayerNorm, L2/L1, optional
  `sample_weight`); `jepa_beta_nll_loss` (β-NLL, Seitzer et al. 2022); **`variance_covariance_reg`**
  (the anti-collapse term).
- **`baselines/`** — `mae.py`, `byol.py`, `simclr.py`.

### `engine/`
- **`ema.py`** (momentum 0.996→1.0), **`diagnostics.py`** (`per_dim_std`, `effective_rank`,
  `offdiag_covariance` — *a falling loss alone is not success*), **`train_jepa.py`** (AMP,
  grad-accum, warmup→cosine, EMA-after-step), **`train_baselines.py`**, **`train_finance.py`**
  (β-NLL branch, GPU spectral-Ω, variance↔volatility correlation).

### `eval/`
- **`linear_probe.py`** — dense features (temporal pathway for JEPA, spatial-only for baselines) →
  `linear_probe_segmentation(head='linear'|'conv')` → **mIoU**.
- **`knn.py`**, **`fewshot.py`**, **`feature_analysis.py`** (t-SNE/UMAP, purity, silhouette).
- **`finance_tasks.py`** / **`cmapss_tasks.py`** — the five downstream probes per domain, each with
  raw-feature and random-init floors; C-MAPSS adds the PHM08 asymmetric score and healthy-reference
  anomaly detection.
- **`predictability.py`** — seven indices: spectral Ω, permutation entropy, AR(p) R², autocorrelation
  time, Rosenstein Lyapunov, **past→future mutual information**, intrinsic dimension.

### `utils/`
`seed.py`, `config.py`, `checkpoint.py` (RNG state for resume), `gpu_hours.py` (wall-clock + peak
memory), **`device.py`** (`resolve_device` — the single GPU knob).

---

## 4. Configs — which to use

| Config | Use |
|---|---|
| `configs/model/tjepa_server.yaml` | big card (~48 GB): batch-48 × accum-4, **eff 192** |
| `configs/model/tjepa_8gb.yaml` | **same quality, fits 8 GB** (batch-16 × accum-12, eff 192) |
| `configs/model/tjepa_laptop.yaml` | fast pilot (embed-256, 50 epochs) |
| `configs/model/tjepa_graph.yaml` | grid-GNN spatial backbone (Part 6 #8) |
| `configs/model/fjepa.yaml` / `cjepa.yaml` | finance / C-MAPSS panel models |
| `configs/data/{pastis,finance,cmapss}.yaml` | per-domain data configs |

Key knobs: `objective` (temporal_jepa | spatial_jepa | mae | byol | simclr), `device`,
`encoder.{patch_size,embed_dim,grad_checkpoint,spatial_backbone}`, `predictor.{embed_dim,type,
distributional}` (`type: transformer|koopman|ode`), `loss.{type,var_coeff,cov_coeff}`,
`optim.{batch_size,grad_accum}`, `temporal.{horizon,horizons,period}`.
PASTIS classes: **num_classes 20, ignore_index 19** (void).

**Memory levers** (impact order): `grad_checkpoint` → `max_seq_len` → `batch_size`/`grad_accum`
(grad-accum raises the *effective* batch for free). All laptop cells stay **under 6.5 GB**.
Don't guess — `python scripts/fit_batch.py --config <cfg> --device cuda:0`.

---

## 5. Running the experiments

```bash
pip install -r requirements.txt
pytest -q                            # 80 passed, 3 skipped — offline, no data needed
```

### Satellite (PASTIS)
```bash
bash scripts/download_pastis.sh ./data_root          # ~29 GB, resumable, md5-verified
python scripts/overfit8_smoketest.py --pastis --objective temporal_jepa --device cuda:0   # M1 gate

python scripts/run_matrix.py --config configs/model/tjepa_server.yaml \
    --data configs/data/pastis.yaml --device cuda:0 --max-cells 5 --knn --resume
#   --max-cells 5   = the main objective cells (temporal h1 vs spatial vs MAE/BYOL/SimCLR)
#   --max-cells 9   = + compute-matched spatial + horizon study (Δ=2,4,8)
#   omit            = + ablations (VICReg, predictor width/depth, embed dim, graph)
#   --only <cell>   = run ONE named cell against this base config (add a new cell without
#                     re-running or contaminating the existing baselines)
#   --resume        = skip cells already done   --test = probe on test folds instead of val

# write-up numbers for a cell — NO retrain (reuses the saved encoder)
python scripts/evaluate.py --encoder-ckpt runs/matrix/tjepa_h1.pt \
    --config configs/model/tjepa_server.yaml --data configs/data/pastis.yaml \
    --head both --knn --fewshot --test

# RIGOR PASS: multi-seed + 5-fold CV (tags outputs), then error bars + significance
for s in 0 1 2; do python scripts/run_matrix.py --seed $s --max-cells 9 --knn --resume \
    --config configs/model/tjepa_server.yaml --data configs/data/pastis.yaml --device cuda:0; done
python scripts/aggregate.py          # mean ± std + paired Wilcoxon / t-test vs temporal
```

> ⚠️ **Match the eval split when comparing cells.** The baselines above are probed on **val**;
> adding `--test` to only one cell confounds backbone/objective with split.

### Finance & industrial
```bash
python scripts/download_finance.py && python scripts/run_finance_matrix.py --device cuda:0
python scripts/aggregate_finance.py

python scripts/download_cmapss.py  && python scripts/run_cmapss_matrix.py  --device cuda:0
python scripts/aggregate_cmapss.py

python scripts/finance_regime_shift_probe.py     # Phase 5: shift vs unpredictability
```

### Measurement & method systems
```bash
python scripts/predictability_sweep.py       --device cuda:0   # indices + falsification sweep
python scripts/structured_predictor_bench.py --device cuda:0   # Koopman / Neural-ODE / LKF
python scripts/hierarchical_bench.py         --device cuda:0   # multi-horizon ablation
```

### Where results are saved
- `runs/matrix_results[__s<seed>_f<fold>].csv` — one row/cell: `cell, objective, seed, cv_fold,
  eval_split, miou_linear, miou_conv, knn_acc, gpu_hours, peak_mem_gb` (flushed per cell,
  crash-safe). `runs/{finance,cmapss}_results.csv` likewise.
- `runs/matrix/<cell>[__s<seed>_f<fold>].pt` — per-cell encoder; reuse via `evaluate.py
  --encoder-ckpt` with no retrain. The checkpoint stores its own config, so the right architecture
  (including a graph backbone) is rebuilt automatically.
- Long runs: `nohup … > run.log 2>&1 &` or tmux.
- **Schema note:** if a CSV predates the `seed`/`cv_fold` columns, `--resume` appends 10-col rows
  under an 8-col header and `csv.DictReader` misaligns. Repair with
  `python scripts/migrate_matrix_csv.py runs/matrix_results.csv` (writes a `.bak`, idempotent).

---

## 6. Tests & the M1 gate

```bash
pytest -q                                   # 80 passed, 3 skipped
PASTIS_ROOT=/path/to/PASTIS pytest -q       # also runs test_dataset + test_probe_sanity
```
Coverage: masking disjointness, **no future leakage**, EMA, stop-grad, collapse diagnostics, full
forward grad-routing on synthetic tensors, finance/C-MAPSS datasets and task suites, β-NLL
correctness, predictability indices, Koopman/ODE/LKF, graph layers and encoder-checkpoint round-trip.

**M1 gate** (`scripts/overfit8_smoketest.py`): loss ↓ **while** std / effective-rank stay high —
the loss will *not* reach 0 with the variance regularizer, and that is the point.

---

## 7. Scope & honesty

- **Not an algorithmic-novelty claim.** Causal/temporal JEPA variants exist in prior art;
  distributional JEPA appears as VJEPA (arXiv 2601.14354) and Var-JEPA (arXiv 2603.20111). The
  contribution is **empirical and mechanistic**: a controlled cross-modality study with floors, a
  measurable predictability criterion, and reported negatives.
- **Not a frontier-model comparison.** Every number is a frozen linear/kNN probe against
  same-architecture SSL baselines plus two floors. No large pretrained world model is involved.
- **Seeds.** PASTIS main comparison is 3 seeds with significance tests. Finance, C-MAPSS and all
  Part-6 benchmarks are **single-seed**.
- **6 of 11 headline results are negative, and all are reported** — see
  [REPORT_CONSOLIDATED.md](REPORT_CONSOLIDATED.md) §5.

## 8. Reports

| document | contents |
|---|---|
| **[REPORT_CONSOLIDATED.md](REPORT_CONSOLIDATED.md)** | **all phases in one document — start here** |
| [report.md](report.md) | Phase 1 — PASTIS satellite, architecture, ablations |
| [report_finance.md](report_finance.md) | Phases 2/4/5 — finance loss and two failed rescues |
| [report_cmapss.md](report_cmapss.md) | Phase 3 — C-MAPSS, five probes, NASA RUL benchmark |
| [report_predictability.md](report_predictability.md) | Part 6 — the seven indices, domain measurement |
| [report_structured.md](report_structured.md) | Part 6 — Koopman / ODE / LKF / hierarchical / graph |
| [report_full.md](report_full.md) | Full graduate-level monograph |
