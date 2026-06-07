# Temporal JEPA for Satellite Image Time Series

Self-supervised representation learning for satellite image time series (SITS) by **latent
prediction**: predict the *future* latent state of a location from its *past* observations — a
causal, world-model-flavored JEPA objective — and compare it against *spatial* JEPA
(I-JEPA-style masked-region prediction) and reconstruction/contrastive baselines (MAE, BYOL,
SimCLR) on **PASTIS** (Sentinel-2 crop time series).

> **Research question.** Can a *temporal* JEPA objective learn more useful representations for
> remote sensing than *spatial* JEPA, under equal compute?

**Status:** the full pipeline — temporal/spatial JEPA **and** all three baselines (MAE, BYOL,
SimCLR) — is implemented and verified. `pytest` is green (21 passed with `PASTIS_ROOT` set;
18 + 3 skipped without). A single `run_matrix.py` run fills the entire comparison table; nothing
is left unwired.

**Architecture note — anti-collapse on temporal SITS (important).** Standard JEPA (EMA target +
stop-grad + narrow predictor) **collapses on real PASTIS**: consecutive acquisitions of the same
field are nearly identical, so "predict the future latent" is solvable by emitting a constant
(the M1 gate caught this — loss→0 while per-dim std→0.04, effective rank→2.4). The fix, now part
of the architecture and **on by default in the `main()` training path**, is a **VICReg-style
variance–covariance regularizer** on the trainable context embedding (`objectives/jepa_loss.py:
variance_covariance_reg`, weights `loss.var_coeff` / `loss.cov_coeff`). With it, std stays
healthy; set both weights to 0 to recover pure I-JEPA (and to ablate that the collapse is real).

---

## 0. TL;DR — run order

```bash
pip install -r requirements.txt
pytest -q                                   # logic tests (no data needed)
python scripts/overfit8_smoketest.py        # M1 gate on synthetic data (no download)
# --- on the server, with a GPU + ~30 GB free disk ---
# pick your GPU ONCE: set `device:` in configs/model/tjepa.yaml (cuda / cuda:1 / cuda:2 / cpu)
bash scripts/download_pastis.sh ./data_root # ~28.76 GB
export PASTIS_ROOT=$(pwd)/data_root/PASTIS
python -m engine.train_jepa --config configs/model/tjepa.yaml --data configs/data/pastis.yaml
python scripts/run_matrix.py                # full matrix: JEPA + all baselines (GPU-weeks)
```

Full server command reference is in **§7**.

---

## 1. The idea in one diagram

```
PASTIS series X[T,10,128,128] + DOY dates d[T]
   │  causal split by date  →  context = past frames | target = future frame (gap = horizon Δ)
   ▼
per-frame PatchEmbed (Conv 10→D, P=8 → 16×16=256 tokens/frame) + 2D spatial pos + DOY temporal pos
   ├─► CONTEXT path:  spatial ViT (per frame) → temporal transformer (past frames) → z_ctx
   │                                                              │
   │                         PREDICTOR (narrow, 384 < encoder 512): z_ctx + mask tokens(future pos/DOY)
   │                                                              ▼  ẑ_future
   └─► TARGET path (EMA encoder, stop-grad): encode FUTURE frame → LayerNorm → z_future
                                                                  │
                                  Loss = mean ‖ ẑ_future − sg(LayerNorm(z_future)) ‖²  ◄┘
```

Anti-collapse mechanisms working together: **EMA target** (lagging teacher), **stop-gradient**
on the target, the **narrow predictor** bottleneck (predictor width < encoder width), and —
because temporal SITS makes *future ≈ present* and thus collapse very easy — a **VICReg-style
variance–covariance regularizer** on the trainable context embedding (`loss.var_coeff` /
`loss.cov_coeff`; set both to 0 for pure I-JEPA). Without the regularizer the M1 gate collapses
on real PASTIS (loss→0, std→0.04); with it, std stays healthy.

---

## 2. Repository map

```
configs/      yaml configs (data + model/training)
data/         PASTIS Dataset, variable-length collate, normalization, splits
masking/      spatial multi-block sampler + causal past→future temporal split
models/       patch embed, positional encodings, ViT, temporal encoder, predictor, JEPA assembly
objectives/   JEPA latent loss + MAE/BYOL/SimCLR baseline losses & heads (incl. MAEModel)
engine/       JEPA training loop, baseline training drivers, EMA, collapse diagnostics
eval/         linear probe (mIoU), k-NN, few-shot, feature-space analysis
utils/        seeding, config loading, checkpointing, GPU-hour metering, device knob
scripts/      download, overfit-8 smoketest (M1 gate), experiment-matrix driver
tests/        unit tests (TDD); test_model_synthetic runs the full wiring without data
```

---

## 3. Module-by-module logic (every function)

### `data/`
- **`pastis_dataset.py`**
  - `PASTIS(Dataset)` — reads `metadata.geojson` once, indexes the `ID_PATCH`es in the requested
    folds, and stores each patch's acquisition dates. `__getitem__` loads
    `DATA_S2/S2_<id>.npy` `(T,10,128,128)`, converts each `YYYYMMDD` date to **calendar
    day-of-year** in `[1,366]`, optionally applies per-band normalization, and (if
    `return_label`) loads channel 0 of `ANNOTATIONS/TARGET_<id>.npy` as the `(128,128)`
    semantic map. DOY (not frame index) is the temporal coordinate because PASTIS cadence is
    irregular.
  - `collate_variable_length(batch)` — pads variable-length series to `T_max`, builds a boolean
    `pad_mask` (True = real frame), pads data with 0 and **dates with DOY 0** (never a real
    date). Real frames are **front-packed**, which the temporal split relies on.
  - helpers `_yyyymmdd_to_doy`, `_parse_dates_field` (dates-S2 may be a dict or a JSON string).
- **`transforms.py`**
  - `compute_band_stats(dataset)` — streaming per-band mean/std over **train folds only**
    (accumulates sum & sum-of-squares; std from `E[x²]−E[x]²`). Train-only avoids val/test leakage.
  - `normalize_bands` — per-band `(x−μ)/σ`, broadcast over T,H,W, with a `σ` floor.
  - `temporal_subsample` — truncate long series: evenly-spaced for eval (deterministic), random
    sorted subset for train augmentation; keeps data/dates aligned.
  - `two_view_augment` — two views for BYOL/SimCLR (temporal crop, spatial flips, band jitter).
- **`splits.py`**
  - `fold_indices(root, folds)` — patch ids per the **official** 5-fold table (don't invent splits).
  - `fewshot_subset(dataset, fraction, seed)` — stratified by each patch's dominant non-background
    class, reproducible from `seed`; used for 1/5/10% few-shot eval.

### `masking/`
- **`multiblock.py`** — I-JEPA spatial sampler. `sample_block` draws one rectangle on the token
  grid from area/aspect ranges. `sample_multiblock_mask` draws **4 target blocks first**, then
  one large context block, then **removes from the context every token overlapping any target**
  so context ∩ target = ∅ (prevents the trivial copy task). Tokens are flat row-major
  (`flat = row*W' + col`).
- **`temporal_mask.py`** — `split_past_future(dates, pad_mask, horizon, min_context)`: over the
  real frames (chronological), picks a split rank `s` with `s+1 ≥ min_context` and `s+horizon ≤
  last`, returns `context = R[:s+1]` and `target = R[s+horizon]`. Because `horizon ≥ 1`, every
  context date < target date — **no future leakage** (the #1 silent bug; `test_temporal_mask`
  guards it).

### `models/`
- **`patch_embed.py`** — `Conv2d(10→D, kernel=stride=P)` patchify+project in one op; exposes
  `grid_hw` and `num_patches`. Input is 10-channel (not 3). At `P=8` → 16×16=256 tokens/frame
  (default, for dense mIoU); `P=16` → 8×8=64 (cheaper, coarser).
- **`pos_embed.py`** — `build_2d_sincos_pos_embed` (fixed 2D sin/cos over the token grid);
  `doy_sincos_pos_embed` (sin/cos with phase `DOY/366·2π`, **periodic over a year**; zeros out
  padded frames). DOY encoding is what makes "Δ acquisitions" carry real elapsed time.
- **`vit.py`** — `Attention` (multi-head, fused `scaled_dot_product_attention`, optional
  key-padding mask), pre-norm `Block`, `ViTEncoder` (block stack + final LayerNorm — the final
  norm is the representation the loss is computed against).
- **`temporal_encoder.py`** — factorized **space→time**: adds DOY pos along T, folds the N spatial
  tokens into the batch so attention runs over the time axis with the per-frame `pad_mask`, returns
  time-aware tokens `(B,T,N,D)`. Factorization avoids O((T·N)²) full 3-D attention.
- **`predictor.py`** — **narrow** transformer (`pred_dim` must be < encoder `embed_dim`; default
  384 vs 512). Projects context to `pred_dim`, appends one shared **learnable mask token +
  projected target positional embedding** per target slot, runs blocks, reads out the mask-token
  outputs, projects back to encoder dim. The narrow width is half the anti-collapse mechanism
  (I-JEPA's literal 384 is relative to a much wider ViT-H encoder; the invariant is
  predictor < encoder). `JEPA.__init__` asserts it and `build_model` clamps a mis-set config.
- **`jepa.py`**
  - `SITSEncoder` — the shared encoder (patch embed + spatial ViT + temporal encoder) with
    `encode_full` (whole frame), `encode_subset` (visible context tokens only, I-JEPA style),
    and `encode_temporal` (sequence → time-aware tokens).
  - `JEPA` — holds `context_encoder`, a **deep-copied frozen** `target_encoder`
    (`requires_grad=False`), and the `predictor`. `_forward_spatial` samples a multi-block mask on
    one frame, encodes the visible context, predicts the target-block latents, and reads targets
    from the **full-frame** EMA encoding. `_forward_temporal` does a **per-sample** causal split
    (each sample draws its own split rank; a context-only mask blocks future leakage), pools the
    past with a **masked-mean over context frames**, predicts the future frame's tokens using its
    DOY as the query, and reads the target from the EMA encoder on the future frame. Returns
    `(pred, target, context_repr)`; the loss applies LayerNorm + stop-grad, and `context_repr`
    (the trainable branch, not the lagging EMA target) is what the collapse diagnostics watch.
  - `build_model(cfg)` — construct a `JEPA` from a `tjepa.yaml`-style dict.

### `objectives/`
- **`jepa_loss.py`** — `jepa_latent_loss(pred, target, norm_target, loss_type)`: **detaches**
  the target (stop-grad — the #1 collapse bug if forgotten), optionally LayerNorms it over the
  feature dim, then L2 (I-JEPA) or L1 (V-JEPA ablation) mean error. Plus
  **`variance_covariance_reg(z)`** — the VICReg anti-collapse term: a variance hinge
  (`relu(1 − std_d)` per dim, keep batch-std ≥ 1) + a covariance penalty (decorrelate dims).
  Applied to the trainable context embedding in the training loops; the EMA target is not
  regularized. This is what stops temporal-SITS collapse (see the architecture note up top).
- **`baselines/`** — `mae.py` (`random_patch_mask`, `patchify`, `mae_loss` = MSE on masked
  patches only, and **`MAEModel`** = shared backbone + lightweight decoder doing standard
  random-shuffle masked reconstruction); `byol.py` (`mlp_head`, `byol_loss` = `2−2·cos`, target
  detached); `simclr.py` (`projector`, `nt_xent_loss` over a `(2B,D)` view-stacked batch,
  positives at offset B). The *training drivers* that use these live in `engine/train_baselines.py`.

### `engine/`
- **`ema.py`** — `momentum_schedule` (linear `0.996→1.0`, clamped) and `ema_update`
  (`teacher ← m·teacher + (1−m)·student` for params and float buffers, under `no_grad`;
  m=0 copies student, m=1 freezes teacher).
- **`diagnostics.py`** — collapse early-warning: `per_dim_std`, `effective_rank` (spectral
  entropy of the covariance), `variance_ratio` (pred vs target variance), `offdiag_covariance`
  (VICReg-style), bundled by `collapse_metrics`. **A falling loss alone is not success** — a
  collapsed model also has ~0 loss, so these are logged every N steps.
- **`train_jepa.py`** — the JEPA pretraining loop: AMP autocast, gradient accumulation,
  linear-warmup → cosine LR, **cosine weight-decay ramp** (`weight_decay_start→end`), optional
  flip augmentation, **EMA step after the optimizer step**, the **VICReg variance–covariance
  anti-collapse term** added to the latent loss (`var_coeff`/`cov_coeff`), diagnostics on the
  trainable context embedding paired with loss, checkpointing. `main(config, data, device)` —
  the entry point used for real pretraining — builds everything and runs with the regularizer on.
- **`train_baselines.py`** — training drivers for MAE / BYOL / SimCLR (`TRAINERS` dict). All
  three train the **same `SITSEncoder` spatial backbone** so the probe reads them through one
  uniform pathway (`use_temporal=False`); each returns the trained backbone. BYOL/SimCLR use a
  global masked-mean pool over two augmented views; MAE uses `MAEModel` on a sampled frame. The
  temporal encoder is deliberately *not* trained by the baselines — that is the JEPA contribution.

### `eval/`
- **`linear_probe.py`** — `extract_dense_features(encoder, batch, use_temporal=…)` (frozen
  encoder → masked-mean over time → bilinear upsample tokens to pixels). `use_temporal=True`
  uses the temporal encoder (JEPA cells); `use_temporal=False` is spatial-only per frame (the
  baselines, whose temporal encoder is untrained — fair eval). Then trains a single `1×1` conv
  head and reports **mIoU** (dense, per-pixel — not global top-1). `miou_from_confusion` handles
  the ignore class. `head=` selects `'linear'` (strict 1×1 probe) or `'conv'` (a small 2-layer
  conv decoder — fairer dense readout from coarse upsampled tokens); `scripts/evaluate.py --head
  both` reports each. `_sanitize_labels` maps out-of-range/void labels to `ignore_index`.
- **`knn.py`** — `parcel_embeddings` (masked-mean feature + dominant label per patch) and
  `knn_accuracy` (cosine k-NN, majority vote). Training-free probe.
- **`fewshot.py`** — `fewshot_eval` runs the linear probe on stratified 1/5/10% label subsets.
- **`feature_analysis.py`** — `project_2d` (t-SNE/UMAP), `cluster_purity` (KMeans), `silhouette`.

### `utils/`
- `seed.py` (`seed_everything`), `config.py` (`load_yaml`/`load_config`),
  `checkpoint.py` (`save_checkpoint`/`load_checkpoint` with RNG state for reproducible resume),
  `gpu_hours.py` (`GpuHourMeter` — wall-clock + peak memory, reported per experiment so
  comparisons are honestly "under equal compute"), `device.py` (`resolve_device` — the single
  GPU knob; see §4).

### `scripts/`
- `download_pastis.sh` — downloads PASTIS.zip (resumable `wget`), verifies size + md5, extracts.
- `overfit8_smoketest.py` — **M1 hard gate**: overfit 8 samples; PASS requires loss drop **and**
  healthy std/effective-rank. Runs on synthetic data by default (`--pastis` for real samples).
- `run_matrix.py` — enumerates the experiment matrix (horizon study + ablations + baselines),
  trains+probes each cell, writes a CSV with mIoU and GPU-hours; `--max-cells` caps the run and
  **logs every skipped cell** (no silent truncation); `--dry-run` lists cells.

---

## 4. Configuration

- `configs/data/pastis.yaml` — dataset root, folds (official 5-fold), band count, **num_classes
  20** (PASTIS labels: 0=background, 1–18 crops, 19=void) with **ignore_index 19** (void excluded
  from loss + mIoU; background kept, U-TAE-style), DOY encoding, few-shot fractions. Set `root`
  after download (or use the `PASTIS_ROOT` env var).
- `configs/model/tjepa.yaml` — objective switch (`temporal_jepa | spatial_jepa | mae | byol |
  simclr`); **`device:`** — the single GPU knob (`cuda` / `cuda:1` / `cuda:2` / `cpu`), overridable
  per-run with `--device`; encoder dims (default 512); **predictor 384** (must stay < encoder —
  `build_model` auto-clamps otherwise); EMA schedule; horizon Δ + min_context; loss (L2 + target
  LayerNorm + stop-grad, **`var_coeff`/`cov_coeff`** VICReg anti-collapse weights — default on,
  set to 0 for pure I-JEPA); optimizer (AdamW, warmup→cosine LR, cosine wd ramp, flip `augment`,
  AMP, grad-accum).

**GPU / memory (tuned for a 15–16 GB card, at patch_size 8).** Defaults: `patch_size 8`
(256 tokens/frame — for good DENSE mIoU), `embed_dim 512`, `max_seq_len 32`, `batch_size 32`,
`grad_accum 6`, `grad_checkpoint: true` → **est. peak ≈ 10.8 GB** (calibrated model
`mem ≈ 0.0096·B·T + 0.95 GB`; batch 40 ≈ 13 GB). The levers, in order of impact:
- **`encoder.patch_size`** — 8 = 256 tokens (4× finer, 4× heavier; the dense-mIoU lever); 16 = 64
  tokens (cheap/coarse — fine for capacity ablations where resolution doesn't matter).
- **`encoder.grad_checkpoint: true`** — recompute activations in backward (~3× less memory).
  Required at P8. Costs ~25% wall-clock.
- **`data.max_seq_len`** (32) — `B×T` frames flow through the spatial ViT; raise toward 48/61 only
  if you drop `patch_size` or `batch_size`.
- **`optim.batch_size` / `optim.grad_accum`** — per-step `batch_size` sets memory; `grad_accum`
  raises the *effective* batch for **free**. Default 32×6 = effective 192. `run_matrix` auto-halves
  the batch (doubling accum) for the heavy `embed_dim 768` ablation cell so the sweep can't OOM.

**Compute reality (read before launching the full matrix).** P8 is ~4× the compute of P16. One
P16 cell took ~4 GPU-h (100 epochs); a P8 cell is **roughly a day**. The 16-cell matrix at P8 is
GPU-*weeks*. Scope deliberately: run the 5 main objective cells + horizon study at P8, and either
keep the capacity ablations (embed_dim / predictor-depth) at `patch_size 16` or cut epochs — the
ablations show *trends*, which resolution doesn't change. `--max-cells` caps and logs the rest.

---

## 5. Method ↔ baseline switch (one codebase)

| Objective | Target | Predictor query | Where |
|---|---|---|---|
| **Temporal JEPA** (method) | future frame's EMA latent | spatial pos + **future DOY** | `JEPA._forward_temporal` |
| **Spatial JEPA** (baseline) | masked target blocks of one frame | spatial pos | `JEPA._forward_spatial` |
| MAE / BYOL / SimCLR | pixels / EMA view / contrastive | — | `engine/train_baselines.py` (+ `objectives/baselines/`) |

All five are selectable via `objective:` in the config and all run end-to-end in `run_matrix.py`.
JEPA cells probe with the temporal encoder; baseline cells probe spatial-only (`use_temporal=False`).

---

## 6. Tests & the M1 gate

```bash
pytest -q                       # 18 pass, 3 skip (data tests skip until PASTIS_ROOT is set)
PASTIS_ROOT=/path/to/PASTIS pytest -q   # also runs test_dataset + test_probe_sanity
```

- `test_masking` — context ∩ target = ∅; `test_temporal_mask` — no future leakage.
- `test_ema` — teacher frozen; m=0 copies, m=1 freezes; schedule ramps.
- `test_loss` — target detached (no grad reaches teacher); loss=0 when pred==target.
- `test_diagnostics` — std/effective-rank distinguish collapsed vs healthy.
- `test_model_synthetic` — full JEPA forward (spatial + temporal), grad reaches predictor but
  **not** the target encoder, EMA moves the target. (Runs without any download.)

The **M1 correctness gate** is `scripts/overfit8_smoketest.py`: it must show loss dropping
**while** per-dim std and effective rank stay high (loss won't reach 0 with the variance
regularizer on — that is expected and healthy). It catches real collapse: on PASTIS, *without*
the VICReg term it fails (loss→0 but std→0.04, effrank→2.4); *with* it (default), std stays
healthy. Pass `--var-coeff 0 --cov-coeff 0` to reproduce the collapsing baseline.

---

## 7. Server commands — download data & run experiments

Run these on the GPU server (needs CUDA, ~30 GB free disk, the venv from `requirements.txt`).

```bash
# 0. environment
cd multi_temporal_jepa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0b. PICK YOUR GPU ONCE — edit configs/model/tjepa.yaml:  device: cuda:1
#     (or override any command below with --device cuda:1)

# 1. sanity before touching data (fast)
pytest -q
python scripts/overfit8_smoketest.py --objective temporal_jepa   # add --device cuda:1 if needed
python scripts/overfit8_smoketest.py --objective spatial_jepa

# 2. download PASTIS (~28.76 GB; resumable, verifies md5, extracts)
bash scripts/download_pastis.sh ./data_root
export PASTIS_ROOT=$(pwd)/data_root/PASTIS
#   then set configs/data/pastis.yaml: root: ./data_root/PASTIS

# 3. re-run the data-dependent tests now that PASTIS exists
PASTIS_ROOT=$PASTIS_ROOT pytest -q
#   and the M1 gate on 8 REAL samples
python scripts/overfit8_smoketest.py --pastis --objective temporal_jepa

# 4. pretrain Temporal JEPA (edit configs/model/tjepa.yaml for epochs/batch/horizon/device)
python -m engine.train_jepa --config configs/model/tjepa.yaml --data configs/data/pastis.yaml
#   checkpoints land in runs/tjepa/last.ckpt   (override GPU: --device cuda:2)

# 5. pretrain Spatial JEPA baseline (same loop, different objective)
#   set `objective: spatial_jepa` in the config (or copy it) and re-run step 4.

# 6. full experiment matrix — ALL cells (JEPA horizon study + ablations + MAE/BYOL/SimCLR)
python scripts/run_matrix.py --dry-run                 # preview the 16 cells
python scripts/run_matrix.py --max-cells 4             # budgeted run; logs skipped cells
python scripts/run_matrix.py --device cuda:1           # everything → runs/matrix_results.csv

# tip: run long jobs under tmux/nohup so they survive disconnects
#   nohup python -m engine.train_jepa --config ... --data ... > train.log 2>&1 &
```

**Notes & caveats**
- `run_matrix.py` runs **all** cells end-to-end — temporal/spatial JEPA *and* MAE/BYOL/SimCLR;
  `--max-cells N` caps the run and explicitly logs every skipped cell (no silent truncation).
- Confirm the Sentinel-2 band order against the official loader before trusting band indices
  (it's `[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]` per utae-paps; recorded in `download_pastis.sh`).
- Compare downstream mIoU against supervised **U-TAE = 63.1** as the reference ceiling.
- If you OOM at 512-dim/batch-64, lower `optim.batch_size` + raise `grad_accum`, or drop
  `encoder.embed_dim` to 256 (see §4).

---

## 8. Reference

I-JEPA (Assran 2023, 2301.08243) · V-JEPA (Bardes 2024, 2404.08471) · PASTIS/U-TAE (ICCV 2021,
2107.07933; Zenodo 10.5281/zenodo.5012942) · MAE (2111.06377) · BYOL (2006.07733) ·
SimCLR (2002.05709) · VICReg (2105.04906). Design doc / build plan:
`~/.claude/plans/read-the-i-jepa-paper-purring-barto.md`.
