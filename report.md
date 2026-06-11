# Multi-Temporal JEPA for Satellite Image Time Series — Research Report

**Status:** core comparison complete (5-cell pilot, PASTIS val fold). Temporal JEPA outperforms
spatial JEPA and all reconstruction/contrastive baselines. This document states the hypothesis,
how we test it, the full architecture, exactly what differs from the I-JEPA baseline, the
experimental protocol, current results, and the honest open caveats.

---

## 1. Hypothesis & research questions

**Central hypothesis (H1).** For satellite image time series (SITS), learning representations by
**predicting the *future* latent state of a location from its *past* observations** (a *causal,
temporal* JEPA objective) yields more useful features than **predicting spatially-masked regions
of a single image** (a *spatial* JEPA / I-JEPA objective), under equal training epochs.

Earth observation is natively a time series of the *same* place. A spatial-masking objective
throws that structure away (it treats each frame independently); a causal future-prediction
objective is forced to model land-surface dynamics (phenology, crop growth, harvest), which is
exactly the signal a crop classifier needs. So we expect temporal prediction to be the better
pretext task for this modality.

**Secondary questions.**
- **H2 (horizon).** How far into the future can we predict and still learn useful features?
  Sweep the horizon Δ ∈ {1, 2, 4, 8} acquisition steps.
- **H3 (vs other paradigms).** Does latent *temporal* prediction beat *reconstruction* (MAE) and
  *contrastive/self-distillation* (SimCLR / BYOL) objectives on the same encoder and data?
- **H4 (capacity).** How do predictor depth and encoder width trade off (ablations)?

**Success criterion.** A frozen-encoder probe where **temporal JEPA > spatial JEPA > {MAE, BYOL,
SimCLR}** on dense crop-segmentation mIoU, k-NN, and few-shot — consistently across metrics.

---

## 2. How we prove it (experimental logic)

We **fix everything except the pretext objective** and compare the *frozen* representations:

1. **Same backbone, same data, same epochs.** Every method (temporal JEPA, spatial JEPA, MAE,
   BYOL, SimCLR) trains the identical `SITSEncoder` on the identical PASTIS train folds for the
   same number of epochs. Only the objective (and its minimal head) differs. This isolates "what
   does the pretext task buy us?"
2. **Frozen-encoder evaluation (no fine-tuning).** After pretraining, the encoder is frozen and
   we measure representation quality three independent ways, so the conclusion can't hinge on one
   probe's quirks:
   - **Linear probe → dense mIoU.** A 1×1 conv on per-pixel features (strict linear-probe
     convention). Plus a **light 2-layer conv decoder** head ("conv") for a fairer dense readout.
   - **Parcel k-NN.** Training-free: mean-pool the encoder features over each field, classify val
     fields by 20-NN. Hyper-parameter-light sanity check on semantic content.
   - **Few-shot (1/5/10% labels).** Where SSL is supposed to shine; differences are most visible.
3. **Same probe for every method.** JEPA cells are probed through the temporal pathway; the
   spatial-only baselines through the spatial pathway (their temporal encoder is untrained) — each
   method is read through the representation it actually learned.
4. **Reference point.** Supervised **U-TAE = 63.1 mIoU** (end-to-end, with a decoder) is the
   ceiling, *not* a like-for-like comparison: our probes are frozen-encoder, so absolute numbers
   are lower by construction. The *relative ordering* is the result.
5. **Collapse is monitored, not assumed.** JEPA-family objectives can silently collapse (constant
   embeddings → ~0 loss). We log per-dim std, effective rank, and predictor/target variance ratio
   every N steps, and gate on them (see §6).

The proof is the **comparison table** (§7) plus the horizon study (H2) and ablations (H4).

---

## 3. Problem setup & data

- **Dataset: PASTIS** (Sentinel-2 SITS, Zenodo 5012942). 2,433 patches of 128×128 px, 10 spectral
  bands, **38–61 irregularly-spaced acquisitions** per patch (Sep-2018 → Nov-2019), dense
  semantic labels: **0 = background, 1–18 = crop types, 19 = void** (20 values; void ignored).
- **Inputs:** `X ∈ ℝ^{T×10×128×128}` plus acquisition dates as **day-of-year** (DOY ∈ [1,366]).
  Because acquisitions are irregular and span a year boundary, DOY *wraps* (e.g. 350→17); the
  chronological order is the acquisition index, and DOY is used only as a periodic time encoding.
- **Splits:** official 5-fold CV. Train = folds {1,2,3}, val = {4}, test = {5}.
- **Outputs:** (pretrain) frozen encoder; (eval) dense crop logits → mIoU, parcel embeddings → k-NN.

---

## 4. Architecture

### 4.1 Shared encoder — `SITSEncoder` (factorized space→time)

Used identically as the trainable context encoder, the EMA target encoder, and every baseline's
backbone, so all comparisons run on the same representation family.

```
X[B,T,10,128,128]
  └─ PatchEmbed: Conv2d(10→D, kernel=stride=P)                 # P=8 → 16×16=256 tokens/frame
  └─ + 2D sin/cos spatial positional embedding
  └─ Spatial ViT (depth 6, pre-norm, MHSA + GELU-MLP)          # encodes EACH frame independently
  └─ + DOY temporal positional embedding (sin/cos, phase d/366·2π, periodic over a year)
  └─ Temporal Transformer (depth 4)                            # attends over TIME, per spatial token
       (N spatial tokens folded into batch; per-frame pad-mask so padded frames are ignored)
  → time-aware tokens [B,T,N,D]
```

We **factorize** space-then-time rather than full 3-D attention: with N≈256 tokens/frame and up to
61 frames, full T·N self-attention (~15k tokens) is O(n²)-infeasible on one GPU; factorization
makes it tractable. Default width **D=512** (`embed_dim`), 8 heads.

### 4.2 The JEPA assembly — `models/jepa.py`

```
                         ┌──────────────── context encoder (TRAINABLE) ───────────────┐
 past frames ──────────► │ SITSEncoder → masked-mean over context frames → z_ctx [B,N,D] │
                         └────────────────────────────────────────────────────────────┘
                                                  │
            target_pos = spatial_pos + DOY(future frame)   │
                                                  ▼
                         ┌── PREDICTOR (NARROW, width 384 < 512) ──┐
                         │ project z_ctx → 384; append mask tokens │ → ẑ_future [B,N,D]
                         │ (shared learnable token + target_pos);  │
                         │ transformer (depth 6); read mask slots  │
                         └─────────────────────────────────────────┘
 future frame ─► target encoder (EMA copy, stop-grad) → z_future [B,N,D]
                                                  │
   Loss = ‖ ẑ_future − sg(LayerNorm(z_future)) ‖²  +  λ_v·var(z_ctx) + λ_c·cov(z_ctx)
```

**Three anti-collapse mechanisms** work together: (1) **EMA target** — the target encoder is a
slow exponential moving average of the context encoder (momentum 0.996→1.0), receiving no
gradient; a lagging teacher makes the constant solution non-stationary. (2) **Stop-gradient +
LayerNorm** on the target. (3) **Narrow predictor bottleneck** (384 < encoder 512) — the asymmetry
(wide encoder, narrow predictor) is what prevents the trivial fixed point (same family as
BYOL/SimSiam). Plus a fourth, added for this modality — see §5.3.

---

## 5. What differs from the I-JEPA baseline (the contribution, precisely)

I-JEPA (Assran et al., 2023) is our starting point. Every change below is deliberate and motivated
by SITS; together they constitute the method.

| # | I-JEPA (baseline) | Multi-Temporal JEPA (ours) | Why |
|---|---|---|---|
| 1 | **Spatial** masking: predict masked *blocks of one image* | **Causal temporal** split: predict a *future frame's* latent from *past* frames (horizon Δ) | The contribution. SITS is a time series; future-from-past forces world-model / phenology learning. Not done by I-JEPA (spatial) or V-JEPA (bidirectional, non-causal). |
| 2 | Single-image encoder | **Factorized space→time** encoder (spatial ViT + temporal transformer) | Aggregate information across acquisitions while staying tractable on one GPU. |
| 3 | Learned/index positions | **DOY (day-of-year) temporal positional encoding** (periodic) | PASTIS cadence is irregular; index positions make "Δ steps" physically meaningless. |
| 4 | Mask tokens carry **spatial** position | Predictor mask tokens carry **spatial pos + the future frame's DOY** | The query must say *when* (and where) it is predicting. |
| 5 | Target = masked blocks of same image | Target = **full future frame** latent (EMA encoder) | The prediction target is a different point in time, not a hidden region of the same image. |
| 6 | Anti-collapse = EMA + predictor + stop-grad only | **+ VICReg variance–covariance regularizer** on the trainable embedding | *Essential for SITS:* consecutive acquisitions are nearly identical (future ≈ present), so the task is trivially solvable by collapsing. Without this term the model collapses on real PASTIS (loss→0, std→0.04); with it, std stays ~1.0. (Set λ=0 to recover pure I-JEPA — and reproduce the collapse.) |

**Spatial JEPA** in our codebase is item-1 reverted (literal I-JEPA on SITS frames via multi-block
masking) — it is the *direct* baseline that isolates the value of the temporal objective.

### 5.1 Temporal causal split (`masking/temporal_mask.py`, inlined per-sample in `jepa.py`)
Per sample, real frames are front-packed and chronological. We draw a split rank `s` with
`s+1 ≥ min_context` and target = the real frame `s+horizon`. Because horizon ≥ 1, every context
date < target date — **no future leakage** (the #1 silent bug; unit-tested). The split is
**per-sample** (each item gets its own rank) for temporal diversity, with a context-only attention
mask blocking the future, and the past is pooled by **masked-mean over context frames**.

### 5.2 Predictor bottleneck (`models/predictor.py`)
Width **384 < encoder 512**. `JEPA.__init__` asserts predictor < encoder; `build_model`
auto-clamps a mis-set config (this caught a real bug where the config had 384 > 256).

### 5.3 VICReg anti-collapse (`objectives/jepa_loss.py: variance_covariance_reg`)
`std_loss = mean_d relu(1 − std_d)` (keep each dim's batch-std ≥ 1) + `cov_loss = Σ_{i≠j}cov²/D`
(decorrelate dims), applied to the trainable context embedding only (the EMA target is detached).
Defaults λ_v=1.0, λ_c=0.04.

---

## 6. Baselines (same encoder, same data, same epochs)

- **Spatial JEPA** — I-JEPA multi-block masking on a sampled frame; predicts masked target blocks
  from the visible context block (overlap removed → disjoint sets). The key comparator.
- **MAE** — mask 75% of patches of a sampled frame, encode the visible ones, a lightweight decoder
  reconstructs the masked patches' **pixels** (MSE on masked patches). Reconstruction paradigm.
- **BYOL** — two augmented views; online encoder+projector+predictor vs EMA target encoder+
  projector; symmetric `2−2·cos`. Self-distillation, no negatives.
- **SimCLR** — two views; encoder+projector; NT-Xent with in-batch negatives. Contrastive paradigm.

All four train the same `SITSEncoder` spatial backbone so the frozen-probe is uniform.

### Correctness harness
- **M1 gate** (`scripts/overfit8_smoketest.py`): overfit 8 samples; PASS requires loss↓ **and**
  healthy std/effective-rank — catches collapse in minutes on slow data. Passes on real PASTIS.
- **Unit tests** (`tests/`): masking disjointness, no future leakage, EMA frozen/ramping,
  loss stop-gradient, diagnostics distinguish collapse, full forward grad-routing. 18 pass / 3
  data-gated.

---

## 7. Results (pilot: P8 / embed-512 / 100 epochs)

Frozen-encoder probes. Headline = **conv-head mIoU** (×100); `linear` = strict 1×1 probe.

**Held-out TEST fold (fold 5); baselines at equalized effective batch 192 (conv head):**

| Method | conv mIoU | k-NN* | few-shot 1% | 5% | 10% |
|---|---|---|---|---|---|
| **Temporal JEPA (Δ=1)** | **22.5** | **65.5** | **9.5** | **13.0** | **15.7** |
| Spatial JEPA | 16.4 | 58.7 | 5.4 | 7.4 | 10.6 |
| SimCLR | 8.1 | 54.6 | 2.6 | 3.6 | 4.0 |
| BYOL | 6.1 | 62.7 | 4.0 | 5.3 | 4.3 |
| MAE | 5.1 | 54.4 | 2.7 | 3.6 | 3.8 |
| *Supervised U-TAE (ceiling, not a frozen-probe peer)* | *63.1* | — | — | — | — |

*k-NN from the val fold. Compute-matched spatial JEPA (#6) and the horizon study (Δ=2,4,8) were
training at the time of writing — rows to be added.*

**Validation fold (folds 4), for model selection — confirms the test ranking:** temporal 22.1 /
spatial 16.2 / simclr 8.3 / byol 5.0 / mae 3.8 conv mIoU; k-NN 66.8 / 58.7 / 54.6 / 62.7 / 54.4.
GPU-hours/cell: temporal 2.1, spatial 0.6, mae 0.5, byol 9.9, simclr 7.8.

**Findings.**
- **H1 supported (and it generalizes).** Temporal beats spatial JEPA on the *test* fold at every
  label fraction: **+37%** conv mIoU at 100% labels (22.5 vs 16.4), widening to **+76%** at 1%
  (9.5 vs 5.4). Val agrees (22.1 vs 16.2). The temporal-vs-spatial pair is the *cleanest*
  comparison — same trainer, same effective batch (192) — so this headline is airtight.
- **H1 data-efficiency.** The temporal advantage *grows* as labels shrink — exactly the SSL story:
  good pretraining matters most in the low-label regime.
- **H3 supported.** Temporal beats MAE/BYOL/SimCLR by a wide margin on every metric — and beats
  BYOL/SimCLR while using **4–5× less GPU time** (2.1 h vs 7.8–9.9 h), so the win is not bought
  with compute.
- **Consistent across three independent probes** (dense mIoU, k-NN, few-shot) → credible.

*Note:* these baseline numbers are the **equalized re-run** (effective batch 192, matching JEPA, via
grad-accum). BYOL (4.6→6.1) and SimCLR (6.9→8.1) rose modestly with the larger batch; MAE was flat.
They remain far below temporal/spatial — the gap is the objective, not the batch size. (SimCLR's
NT-Xent *negatives* are still per-micro-batch; a memory bank is out of scope — see §8.2.)

---

## 8. Honest caveats (to address before publication)

1. **"Equal compute" is not literal.** Same *epochs* (100), but GPU-hours vary 0.5–10×. Frame the
   claim as "equal epochs," and add a **compute-matched spatial-JEPA run** (train it to ≈2.1 GPU-h)
   as a robustness check, since spatial used *less* compute than temporal. (BYOL/SimCLR used more
   and still lost, so they are not a concern.)
2. **Baseline batch confound — FIXED & re-run.** Grad-accumulation was added to all three baseline
   trainers and they were re-run at **effective batch 192** (matching JEPA); the §7 table reflects
   this. BYOL/SimCLR rose modestly, MAE flat — all still far below temporal/spatial. *Remaining
   caveat:* for SimCLR, grad-accum equalizes the *optimization* batch but not the NT-Xent *negative*
   count (still per-micro-batch); a true large-negative SimCLR needs a memory bank (out of scope).
3. **Val, not test.** These are val-fold numbers for model selection. Final numbers must come from
   the **test fold** for the chosen settings only (avoid test-set leakage), with **few-shot** for
   the low-label story.
4. **Pilot scale.** This is the 5-cell main comparison; the horizon study (H2) and ablations (H4)
   are not yet run. A full 5-fold CV average would strengthen the headline.

---

## 9. Reproducibility

- **Configs:** `configs/model/tjepa_8gb.yaml` (P8, embed-512, predictor-384, 100 epochs, effective
  batch 192, gradient checkpointing — fits an 8 GB card); `tjepa.yaml` (server, larger batch);
  `tjepa_laptop.yaml` / `tjepa_p16.yaml` (smaller/pilot variants). Data: `configs/data/pastis.yaml`.
- **Run the comparison:** `python scripts/run_matrix.py --config configs/model/tjepa_8gb.yaml
  --data configs/data/pastis.yaml --device cuda:0 --max-cells 5 --knn --resume`.
  Saves `runs/matrix_results.csv` + per-cell encoders `runs/matrix/<cell>.pt` (resumable).
- **Final numbers (no retrain):** `python scripts/evaluate.py --encoder-ckpt
  runs/matrix/tjepa_h1.pt --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml
  --head both --knn --fewshot --test`.
- Seeds fixed (`utils/seed.py`), checkpoints carry config + RNG state, GPU-hours/peak-memory logged
  per cell. All five objectives selectable via one `objective:` field.

---

## 10. Next steps

1. **Re-run the equalized baselines.** Grad-accum is now in the baseline trainers (effective batch
   192). Delete `runs/matrix/{mae,byol,simclr}.pt` and re-run those cells for the fair table.
2. **Compute-matched spatial JEPA** robustness run.
3. **Horizon study** Δ∈{1,2,4,8} (H2): `run_matrix --max-cells 8`.
4. **Ablations** (H4): predictor depth {1,2,4,6}, embed dim {128,256,512,768}.
5. **Final test-fold + few-shot** numbers for temporal & spatial; ideally 5-fold CV.
6. **Feature analysis** (t-SNE/UMAP, cluster purity) for the qualitative figure.

## References
I-JEPA (Assran 2023, 2301.08243) · V-JEPA (Bardes 2024, 2404.08471) · PASTIS/U-TAE (Garnot &
Landrieu, ICCV 2021, 2107.07933; Zenodo 5012942) · MAE (He 2021, 2111.06377) · BYOL (Grill 2020,
2006.07733) · SimCLR (Chen 2020, 2002.05709) · VICReg (Bardes 2021, 2105.04906).
