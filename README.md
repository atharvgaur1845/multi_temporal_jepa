# Temporal JEPA for Satellite Image Time Series

Self-supervised representation learning for satellite image time series (SITS) by **latent
prediction**: predict the *future* latent state of a location from its *past* observations — a
causal, world-model-flavored JEPA objective — and compare it against *spatial* JEPA (I-JEPA-style
masked-region prediction) and reconstruction/contrastive baselines (MAE, BYOL, SimCLR) on
**PASTIS** (Sentinel-2 crop time series).

> **Research question.** Can a *temporal* JEPA objective learn more useful representations for
> remote sensing than *spatial* JEPA, under equal epochs?

**For the full write-up** (hypothesis, method, what differs from I-JEPA, protocol, results,
caveats) see **[report.md](report.md)**. This README is the engineering reference.

## Headline result (PASTIS held-out TEST fold, P8 / embed-512 / 100 epochs)

conv mIoU, val fold, **mean ± std over 3 seeds**, paired t-test vs temporal:

| Method | conv mIoU (3 seeds) | Δ vs temporal | t-test p |
|---|---|---|---|
| **Temporal JEPA (Δ=1)** | **22.3 ± 1.8** | — | — |
| Spatial JEPA | 16.2 ± 0.4 | +6.0 | **0.041** |
| Spatial JEPA — compute-matched (3.5× epochs) | 15.8 ± 1.2 | +6.5 | **0.036** |
| SimCLR | 7.3 ± 0.8 | +15.0 | **0.009** |
| BYOL | 7.1 ± 0.9 | +15.2 | **0.001** |
| MAE | 6.5 ± 1.1 | +15.8 | **0.009** |

**Temporal JEPA significantly outperforms spatial JEPA (+6.0 mIoU, p=0.041) and every baseline
(p<0.01)** across 3 seeds. The win survives a **compute-matched** spatial run (3.5× epochs → no
gain). **Horizon-insensitive** (Δ=1–8 all ≈22 ± noise, all beat spatial). On the held-out **test
fold**, the temporal-vs-spatial gap widens from +37% (full labels) to **+100% at 1% labels**
(few-shot). Full analysis + caveats: **[report.md](report.md)**. Consistent across three
independent probes (dense mIoU, k-NN, few-shot), and it generalizes from val to test. (*k-NN shown
from the val fold. Supervised U-TAE = 63.1 is the end-to-end ceiling, not a frozen-probe peer.
MAE/BYOL/SimCLR train at effective batch 16 vs JEPA's 192 — see report.md §8.) Full analysis in
**[report.md](report.md)**.

---

## Anti-collapse note (important for temporal SITS)

Standard JEPA (EMA target + stop-grad + narrow predictor) **collapses on real PASTIS**: consecutive
acquisitions of the same field are nearly identical, so "predict the future latent" is solvable by
emitting a constant (loss→0, per-dim std→0.04). The fix — part of the architecture, **on by default**
— is a **VICReg-style variance–covariance regularizer** on the trainable context embedding
(`objectives/jepa_loss.py: variance_covariance_reg`, weights `loss.var_coeff` / `loss.cov_coeff`;
set both to 0 to recover pure I-JEPA and reproduce the collapse).

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

---

## 2. Repository map

```
configs/      yaml configs (data + model/training); see §4 for which to use
data/         PASTIS Dataset, variable-length collate, normalization, splits
masking/      spatial multi-block sampler + causal past→future temporal split
models/       patch embed, positional encodings, ViT, temporal encoder, predictor, JEPA assembly
objectives/   JEPA latent loss + var/cov reg + MAE/BYOL/SimCLR baseline losses & heads
engine/       JEPA training loop, baseline training drivers, EMA, collapse diagnostics
eval/         linear/conv probe (mIoU), k-NN, few-shot, feature-space analysis
utils/        seeding, config, checkpointing, GPU-hour metering, device knob
scripts/      download, overfit-8 smoketest (M1 gate), experiment matrix, evaluate
tests/        unit tests (TDD); test_model_synthetic runs the full wiring without data
```

---

## 3. Module-by-module logic

### `data/`
- **`pastis_dataset.py`** — `PASTIS(Dataset)` reads `metadata.geojson`, indexes patches by fold,
  loads `DATA_S2/S2_<id>.npy` `(T,10,128,128)`, converts dates to **calendar DOY** in [1,366],
  applies per-band normalization, returns `(data, dates, label)` (channel-0 semantic map, 0=bg,
  1–18 crops, 19=void). `max_seq_len`/`subsample_train` cap frames/series to bound memory. Real
  frames are **front-packed** by `collate_variable_length`, which also builds the boolean
  `pad_mask` (the temporal split relies on this).
- **`transforms.py`** — `compute_band_stats` (per-band mean/std over train folds only),
  `normalize_bands`, `temporal_subsample`, `two_view_augment` (BYOL/SimCLR).
- **`splits.py`** — `fold_indices` (official 5-fold), `fewshot_subset` (stratified 1/5/10%).

### `masking/`
- **`multiblock.py`** — I-JEPA spatial sampler: 4 target blocks first, then a context block with
  overlapping tokens removed → context ∩ target = ∅ (no trivial copy).
- **`temporal_mask.py`** — `split_past_future`: causal past→future split with horizon Δ; context
  date < target date guaranteed (no future leakage; unit-tested). The JEPA forward inlines a
  vectorized **per-sample** version.

### `models/`
- **`patch_embed.py`** — `Conv2d(10→D, kernel=stride=P)`; `P=8` → 256 tokens/frame (dense mIoU),
  `P=16` → 64 (cheaper, coarser).
- **`pos_embed.py`** — 2D sin/cos spatial; **DOY** sin/cos temporal (phase `d/366·2π`, periodic
  over a year; padded frames zeroed).
- **`vit.py`** — `Attention` (fused SDPA, key-padding mask), pre-norm `Block`, `ViTEncoder`
  (block stack + final LayerNorm; optional `grad_checkpoint`).
- **`temporal_encoder.py`** — factorized space→time: adds DOY along T, folds N into batch, attends
  over time with pad-mask; optional `grad_checkpoint`.
- **`predictor.py`** — **narrow** transformer (`pred_dim` < encoder dim; default 384 vs 512).
  Context → mask tokens (shared learnable + target pos/DOY) → blocks → read mask slots.
- **`jepa.py`** — `SITSEncoder` (shared encoder: `encode_full` / `encode_subset` / `encode_temporal`);
  `JEPA` (context encoder + frozen EMA target + predictor; `forward` returns
  `(pred, target, context_repr)`); `build_model(cfg)` (constructs from config, asserts/clamps the
  predictor bottleneck, threads `grad_checkpoint`). `_forward_temporal` does the per-sample causal
  split + masked-mean context + future-DOY query.

### `objectives/`
- **`jepa_loss.py`** — `jepa_latent_loss` (detach target, LayerNorm, L2/L1); **`variance_covariance_reg`**
  (VICReg variance hinge + covariance penalty — the SITS anti-collapse term).
- **`baselines/`** — `mae.py` (`MAEModel` decoder + masked-patch MSE), `byol.py` (`mlp_head`,
  `byol_loss`), `simclr.py` (`projector`, `nt_xent_loss`).

### `engine/`
- **`ema.py`** — `momentum_schedule` (0.996→1.0 linear), `ema_update` (in-place, no-grad).
- **`diagnostics.py`** — `per_dim_std`, `effective_rank`, `variance_ratio`, `offdiag_covariance`,
  `collapse_metrics`. A falling loss alone is not success — these are logged every N steps.
- **`train_jepa.py`** — JEPA pretraining loop: AMP, grad-accum, warmup→cosine LR, cosine wd ramp,
  flip augment, EMA-after-step, **var/cov reg added to the loss**, collapse diagnostics on the
  trainable embedding, checkpointing. `main(config, data, device)` is the entry point.
- **`train_baselines.py`** — `TRAINERS` = {mae, byol, simclr}; each trains the same `SITSEncoder`
  backbone (chunked frame pooling + gradient checkpointing to bound memory) and returns it.

### `eval/`
- **`linear_probe.py`** — `extract_dense_features(use_temporal=…)` (temporal pathway for JEPA,
  spatial-only for baselines); `linear_probe_segmentation(head='linear'|'conv')` → dense **mIoU**;
  `_sanitize_labels` (out-of-range/void → ignore).
- **`knn.py`** — parcel-mean features → cosine k-NN. **`fewshot.py`** — probe on 1/5/10% labels.
  **`feature_analysis.py`** — t-SNE/UMAP, cluster purity, silhouette.

### `utils/`
- `seed.py`, `config.py`, `checkpoint.py` (RNG state for resume), `gpu_hours.py` (device-aware
  wall-clock + peak memory), **`device.py`** (`resolve_device` — the single GPU knob).

### `scripts/`
- `download_pastis.sh` (resumable, md5-verified), `overfit8_smoketest.py` (**M1 gate**),
  `run_matrix.py` (the comparison driver, `--seed`/`--cv-fold` for the rigor pass — see §5),
  `evaluate.py` (load a checkpoint → probe/few-shot/test), `aggregate.py` (multi-seed/fold
  mean ± std + paired Wilcoxon/t-test), `feature_figure.py` (t-SNE/UMAP qualitative panel),
  `mechanistic.py` (H-mech-2: decode acquisition time from frozen spatial features), `fit_batch.py`
  (max batch that fits a GPU).

---

## 4. Configs — which to use

| Config | Use |
|---|---|
| `configs/model/tjepa.yaml` | server default (P8, embed-512, batch-32) |
| `configs/model/tjepa_8gb.yaml` | **same quality, fits 8 GB** (batch-16 × grad-accum-12, eff 192) — the one used for the results above |
| `configs/model/tjepa_laptop.yaml` | fast pilot (embed-256, 50 epochs) |
| `configs/model/tjepa_p16.yaml` | P16 — only to evaluate the old coarse pilot checkpoint |
| `configs/data/pastis.yaml` | full data (max_seq_len 32); `pastis_laptop.yaml` = 24 frames |

Key knobs: `objective` (temporal_jepa | spatial_jepa | mae | byol | simclr), `device`
(cuda | cuda:1 | cpu), `encoder.{patch_size,embed_dim,grad_checkpoint}`, `predictor.embed_dim`
(must be < encoder; auto-clamped), `loss.{var_coeff,cov_coeff}`, `optim.{batch_size,grad_accum}`,
`data.max_seq_len`. PASTIS classes: **num_classes 20, ignore_index 19** (void).

**Memory levers** (in impact order): `grad_checkpoint` → `max_seq_len` → `batch_size`/`grad_accum`
(grad-accum raises the *effective* batch for free). The 8 GB config peaks ~6 GB.

---

## 5. Running the comparison

```bash
pip install -r requirements.txt
export PASTIS_ROOT=$(pwd)/data_root/PASTIS    # after scripts/download_pastis.sh

# correctness first (no data needed, then on 8 real samples)
pytest -q
python scripts/overfit8_smoketest.py --pastis --objective temporal_jepa --device cuda:0

# the matrix = pretrain + freeze + probe, per cell, in ONE run (resumable; saves CSV + encoders)
python scripts/run_matrix.py \
    --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml \
    --device cuda:0 --max-cells 5 --knn --resume
#   --max-cells 5  = the 5 main objective cells (temporal h1 vs spatial vs MAE/BYOL/SimCLR)
#   --max-cells 9  = + compute-matched spatial + horizon study (Δ=2,4,8)
#   omit / 22      = + ablations (VICReg, predictor width/depth, embed dim)
#   --resume       = skip cells already done (continue after a crash/OOM)
#   --test         = probe on test folds instead of val

# final write-up numbers for a cell — NO retrain (reuses the saved encoder)
python scripts/evaluate.py --encoder-ckpt runs/matrix/tjepa_h1.pt \
    --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml \
    --device cuda:0 --head both --knn --fewshot --test

# RIGOR PASS: multi-seed + 5-fold CV (tags outputs), then error bars + significance tests
for s in 0 1 2; do python scripts/run_matrix.py --seed $s --max-cells 9 --knn --resume \
    --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml --device cuda:0; done
python scripts/run_matrix.py --cv-fold 1 ...   # repeat F=1..5 for 5-fold CV
python scripts/aggregate.py                    # mean ± std + paired Wilcoxon / t-test vs temporal

# qualitative figure: t-SNE of parcel embeddings + cluster purity/silhouette
python scripts/feature_figure.py --encoder-ckpt runs/matrix/tjepa_h1.pt \
    --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml \
    --device cuda:0 --method tsne --out runs/figures/tjepa_h1_tsne.png
```

### Where results are saved
- `runs/matrix_results[__s<seed>_f<fold>].csv` — one row/cell: `cell, objective, seed, cv_fold,
  eval_split, miou_linear, miou_conv, knn_acc, gpu_hours, peak_mem_gb` (flushed per cell, crash-safe).
- `runs/matrix/<cell>[__s<seed>_f<fold>].pt` — per-cell encoder (reuse for `evaluate.py
  --encoder-ckpt`, no retrain). `scripts/aggregate.py` globs all the CSVs.
- Redirect console to a log with `> run.log 2>&1` (use `nohup`/`tmux` for long runs).

Single-objective pretrain (instead of the matrix): `python -m engine.train_jepa --config
configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml --device cuda:0` → `runs/tjepa/last.ckpt`,
then `evaluate.py --ckpt runs/tjepa/last.ckpt ...`.

---

## 6. Tests & the M1 gate

```bash
pytest -q                                  # 18 pass, 3 skip (data tests skip without PASTIS_ROOT)
PASTIS_ROOT=/path/to/PASTIS pytest -q       # also runs test_dataset + test_probe_sanity
```
- `test_masking` (disjointness), `test_temporal_mask` (no future leakage), `test_ema`,
  `test_loss` (stop-grad), `test_diagnostics`, `test_model_synthetic` (full forward grad-routing).
- **M1 gate** (`scripts/overfit8_smoketest.py`): loss↓ **while** std/effective-rank stay high
  (loss won't reach 0 with the variance regularizer — expected). Passes on real PASTIS.

---

## 7. Phase 2 — Financial time series (S&P-500)

A second modality that tests whether the **same** temporal-JEPA objective wins beyond satellites.
A market is a cross-section of assets observed over time — structurally identical to PASTIS (a
cross-section of pixels observed over time) — so the method ports almost 1:1. Full write-up:
**[report_finance.md](report_finance.md)**.

```
S&P sector panel X[W,N,F]  (W=64 trading days, N=9 sector ETFs, F=4 features) + DOY dates
   │  causal split by trading day → context = past days | target = future day (horizon Δ)
   ▼
per-day FrameEmbed (Linear F→D) + LEARNED per-asset pos + DOY temporal pos
   ├─► cross-asset ViT (within a day) → temporal transformer (across days) → z_ctx
   │        PREDICTOR (narrow) : z_ctx + mask tokens(future asset-pos + DOY) → ẑ_future
   └─► TARGET (EMA, stop-grad): encode FUTURE day → z_future ;  Loss = ‖ẑ−sg(LN(z))‖² + var/cov reg
```

**Reused unchanged** (modality-agnostic): Predictor, JEPA latent loss + VICReg, EMA, diagnostics,
ViT/temporal transformer stacks. **Swapped**: Conv2d patch-embed → `Linear(F→D)` per asset-day;
2D sin/cos spatial pos → learned per-asset pos (`models/finance_encoder.py`, `models/finance_jepa.py`).
The satellite code in §2 is untouched.

**Five downstream tasks** (frozen encoder, train-period probe → test-period score; `eval/finance_tasks.py`):
regime classification (acc/F1), volatility prediction (R²/IC), anomaly detection (AUROC/AP),
clustering (NMI/ARI), next-day forecasting (dir-acc/IC). Question: does Temporal JEPA beat Spatial
JEPA / MAE / BYOL / SimCLR **here too**? A random-init encoder is the floor control.

> **Headline (real S&P sectors, out-of-time test 2018–2026): NO — and it inverts.** Temporal JEPA
> beats *Spatial* JEPA (7/10 metrics, as on satellites) but **loses to MAE/BYOL/SimCLR**, and — the
> kicker — **no SSL method beats a plain linear probe on the raw features**, while Temporal JEPA falls
> *below* its own random initialization (regime acc 0.61 trained vs 0.80 untrained; worse at longer
> horizons). The temporal-prediction advantage is **modality-specific**: it shines on stationary crop
> phenology, not on non-stationary near-efficient markets. An honest negative/inverted-transfer result
> — controls (random-init, raw-features) confirm it's real, not under-tuning. Full table:
> **[report_finance.md](report_finance.md) §5**.

```bash
python scripts/download_finance.py                       # real S&P sector panel via Yahoo (cookie+crumb);
                                                         # auto-falls back to a synthetic regime market offline
python scripts/run_finance_matrix.py --config configs/model/fjepa.yaml \
       --data configs/data/finance.yaml --device cuda:0  # pretrain+freeze+eval all 9 cells -> runs/finance_results.csv
python scripts/aggregate_finance.py                      # comparison table + per-task verdict (Temporal vs peers)
python scripts/finance_smoketest.py --device cuda:0      # M1 gate: loss↓ while std/eff-rank stay healthy
pytest tests/test_finance_*.py -q                        # 16 tests, fully offline (synthetic panel)
```

Finance modules: `data/finance_dataset.py` (panel windows, features, labels, temporal split,
synthetic fallback) · `models/finance_encoder.py` + `models/finance_jepa.py` · `masking/asset_mask.py`
(cross-sectional masking for Spatial JEPA) · `engine/train_finance.py` (JEPA + MAE/BYOL/SimCLR) ·
`eval/finance_tasks.py` · `scripts/{download_finance,run_finance_matrix,aggregate_finance,finance_smoketest}.py`
· `configs/model/fjepa.yaml`, `configs/data/finance.yaml`.

---

## 8. Reference

I-JEPA (Assran 2023, 2301.08243) · V-JEPA (Bardes 2024, 2404.08471) · PASTIS/U-TAE (ICCV 2021,
2107.07933; Zenodo 10.5281/zenodo.5012942) · MAE (2111.06377) · BYOL (2006.07733) ·
SimCLR (2002.05709) · VICReg (2105.04906). Full design/results: **[report.md](report.md)**.
Build plan: `~/.claude/plans/read-the-i-jepa-paper-purring-barto.md`.
