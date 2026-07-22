# Structured Latent-Dynamics Predictors — Koopman, Neural-ODE, LKF (Part 6 #1/#3/#4)

**Status:** implemented, tested (8 tests; suite 69 pass / 3 skip), benchmarked on the synthetic
predictability testbed. **The first clean method-level *positives* of the whole Part 6/7 exploration:**
imposing a dynamical-systems prior on the JEPA predictor beats the free-form transformer, and the gain
scales with predictability.

The default JEPA predictor is a free-form transformer with no dynamical structure. These three systems
replace or wrap it with an explicit dynamics model — the right inductive bias when the latent
trajectory is smooth and low-dimensional, and a source of analytic diagnostics (spectral radius,
continuous-time flow, filtered rollouts) the transformer cannot provide. All are drop-in
(`predictor.type: koopman|ode` in the config; temporal objective, point prediction).

- **#3 Koopman** (`models/structured_predictors.py:KoopmanPredictor`): learn a linear operator K in a
  learned observable space, ẑ_{t+Δ} = dec(K^Δ·enc(z_t)) + b. ρ(K) = spectral radius (drift/stability).
- **#4 Neural-ODE** (`NeuralODEPredictor`): learn a latent vector field f, integrate dz/dt = f(z) over
  the real time gap Δ by fixed-step RK4 (identity-init) — continuous-time, native to irregular sampling.
- **#1 LKF** (`models/latent_filter.py`): the encoder gives noisy *measurements*, the Koopman operator
  is the linear *process model*; a Kalman filter fuses them, correcting an open-loop rollout online.

---

## A) Predictor swap — structured priors beat the free-form transformer

Temporal JEPA on synthetic dynamics, identical everything except the predictor; downstream
latent-recovery R² (frozen encoder → ridge → clean latent z). Higher is better.

| regime (Ω) | free-form | **Koopman** | **Neural-ODE** | Δ vs free-form | ρ(K) learned |
|---|---|---|---|---|---|
| periodic (0.75) | 0.593 | **0.632** | **0.633** | **+0.040** | 1.17 |
| AR φ=0.9 (0.26) | 0.479 | **0.515** | **0.518** | **+0.039** | 1.16 |
| AR φ=0.5 (0.09) | 0.522 | 0.536 | 0.539 | +0.017 | 1.15 |
| Lorenz (0.34) | 0.693 | 0.690 | 0.690 | −0.003 | 1.16 |
| white (0.05) | 0.574 | 0.577 | 0.583 | +0.009 | 1.16 |

**Koopman and Neural-ODE beat the free-form transformer**, and the margin is **largest on the smooth,
predictable regimes** (periodic / AR-0.9: +0.04) and negligible on the unpredictable ones — the
dynamics prior helps exactly where there are dynamics to model. (Lorenz is a near-tie: its chaotic
flow is not well captured by the small models, and the free-form predictor is already strong there.)
The learned Koopman spectral radius sits just above 1 (≈1.16) across regimes — near the unit circle, as
expected for slowly-evolving latents. This is the clean method-level win the earlier interventions
(advantage-curve §3, predictability-weighting §4 of report_predictability.md) did not produce, and it
makes sense: replacing a generic predictor with the *correct* dynamical form is a much stronger prior
than reweighting a generic one.

---

## B) LKF — the filter's dynamics gain scales with predictability

Fit a linear process operator A to the (true) latent, add measurement noise, and compare RMSE (vs the
clean latent) of the raw measurement, a **static** filter (A=0, optimal shrinkage — no dynamics), and
the **dynamic** Kalman filter (process model A, data-driven process noise). `dyn_gain` =
static−dynamic RMSE isolates *what the dynamics add beyond static denoising*.

| regime (Ω) | A one-step fit R² | measure | static (A=0) | **dynamic KF** | **dyn_gain** |
|---|---|---|---|---|---|
| periodic (0.80) | 0.735 | 0.600 | 0.509 | 0.487 | 0.022 |
| AR φ=0.9 (0.27) | 0.814 | 0.604 | 0.518 | 0.420 | **0.099** |
| AR φ=0.5 (0.09) | 0.229 | 0.599 | 0.515 | 0.500 | 0.014 |
| Lorenz (0.35) | 0.966 | 0.600 | 0.520 | 0.423 | **0.098** |
| white (0.05) | 0.000 | 0.599 | 0.176 | 0.176 | **−0.000** |

**The dynamic filter denoises everywhere** (filtered < measurement), but the part attributable to the
*process model* (`dyn_gain`) **scales with the one-step predictability** (A's fit R²): ≈0.10 where the
linear model is accurate (AR-0.9, Lorenz), and **exactly 0 for white noise** (A=0, no dynamics to
exploit → the filter reduces to static shrinkage). The one honest nuance: `dyn_gain` tracks the
accuracy of the *linear* one-step model, not spectral Ω per se — periodic has high Ω but a single
linear operator cannot represent its multiple frequencies (fit R² 0.735), so its dynamics gain is
modest. This is itself informative: the LKF's value is bounded by the *model class* of the process
model, motivating the richer Koopman/ODE process models of §A as the filter's forward model.

---

## C) Real-data validation — the dynamics prior helps on genuine C-MAPSS degradation

The synthetic wins transfer to real turbofan data. On C-MAPSS FD001 (frozen-probe, same protocol as
report_cmapss.md; `predictor.type: koopman|ode`, grad-checkpointed to stay < 6.5 GB):

| predictor | RUL R² ↑ | PHM08 ↓ | health acc ↑ | anomaly ↑ | retrieval ↑ |
|---|---|---|---|---|---|
| free-form transformer (`tjepa_h1`) | 0.677 | 471 | 0.744 | **0.992** | **0.664** |
| **Koopman** | **0.706** | **343** | **0.782** | 0.987 | 0.645 |
| **Neural-ODE** | **0.707** | 349 | 0.778 | 0.987 | 0.647 |
| _random-init floor_ | _0.651_ | _457_ | _0.776_ | _0.978_ | _0.627_ |

Both structured predictors **beat the free-form transformer on the canonical RUL task** (R² +0.03) and,
strikingly, on the asymmetric **PHM08 score (471 → ~345, ≈27 % lower)** and health accuracy (+0.04) —
and they pull the JEPA *further ahead of the random-init floor* than the transformer did (RUL R² 0.706
vs random 0.651; PHM08 343 vs 457). The transformer keeps a slight edge on anomaly/retrieval. Net: on
genuine monotone degradation — the predictable modality where the temporal objective already won — the
correct dynamical form (linear Koopman / continuous-time ODE) is a *better predictor* than a generic
transformer, a clean real-data confirmation of §A.

## D) Hierarchical / multi-timescale (Part 6 #7) — flat, no benefit (honest negative)

Predicting several horizons jointly (forcing fast + slow dynamics) does not help. Downstream
latent-recovery R² on synthetic multi-timescale data (`scripts/hierarchical_bench.py`):

| regime | single Δ=1 | hier Δ={1,5} | hier Δ={1,5,20} |
|---|---|---|---|
| periodic | 0.591 | 0.590 | 0.587 |
| AR φ=0.9 | 0.471 | 0.471 | 0.475 |

The multi-horizon objective is **flat within noise** — consistent with the horizon-insensitivity
already seen on real PASTIS/finance/C-MAPSS (report_full.md §16): once the encoder models the
one-step dynamics, adding longer-horizon targets forces the *same* representation and buys nothing.
So #7 is a clean negative — unlike the structured *form* of the predictor (§A/§C), the *number of
timescales* it predicts is not a useful lever here. (Implementation is additive/tested; a genuine
multi-*scale* architecture — separate encoders per timescale, à la H-JEPA — rather than multi-*horizon*
targets on one encoder, remains the untested stronger version.)

## E) Graph Temporal JEPA (Part 6 #8) — RUN on real PASTIS; local message passing LOSES to global attention (honest negative)

The satellite spatial ViT mixes patch tokens by *global* attention; but PASTIS patches live on a grid
with strong *local* structure (a parcel is a contiguous blob), so a GNN with local message passing over
the patch-grid graph is a better-matched spatial prior — and generalizes to an arbitrary *parcel
adjacency* graph. Implemented dependency-light (no torch_geometric): a graph is a precomputed
`edge_index`; a GraphSAGE-mean block aggregates neighbour features with a residual MLP (`models/
graph_layers.py`), and `GraphSITSEncoder` (`models/graph_encoder.py`) swaps the spatial ViT for a
GridGraphEncoder (8-connectivity + self-loops) while reusing the patch-embed, 2D positions and temporal
transformer unchanged. Wired into the satellite `JEPA` via a flag (`encoder.spatial_backbone: graph`,
default `vit` → **behaviour-preserving**, satellite ViT results untouched). Temporal-objective only
(graph message passing over the full grid can't produce the disjoint context/target sets I-JEPA spatial
masking needs — `encode_subset` raises).

**Result (real PASTIS, A6000 server).** Both cells share an *identical* config (`tjepa_server.yaml`,
patch 8, embed 512, 100 epochs, **effective batch 192**) and differ *only* in the spatial backbone;
both were probed by the **same script, on the same val folds, at the same probe budget** (15 epochs),
so the gap is attributable to the backbone alone.

| backbone | linear mIoU ↑ | conv mIoU ↑ | parcel k-NN ↑ | eff. rank | GPU-h |
|---|---|---|---|---|---|
| ViT / global attention (`tjepa_h1`) | **17.91** | **22.46** | 67.63 | 252.8 | 2.12 |
| grid-GNN / local message passing (`tjepa_graph`) | 16.79 | 20.65 | **68.05** | 249.4 | 1.95 |
| Δ (graph − ViT) | **−1.12** | **−1.81** | +0.42 | −3.4 | −0.17 |

**The local graph prior does not help — it costs ≈1.8 mIoU on segmentation** and is a wash on parcel
k-NN (+0.42, inside noise). The result is *not* a collapse or training artifact: the graph encoder is
representationally healthy and essentially indistinguishable from the ViT on every collapse diagnostic
(effective rank 249.4 vs 252.8 of 512; per-dim std 0.726 vs 0.714; off-diagonal covariance 0.0046 vs
0.0040). It simply learned a slightly *worse* representation. Probe-rerun noise is ≈0.4 mIoU (the same
`tjepa_h1` checkpoint scores conv 22.06 under `run_matrix` and 22.46 on re-probe), so the −1.81 gap is
≈4× noise and likely real — though this is a **single pretraining seed**, and pretrain-seed variance is
the larger unmeasured quantity.

**Why (the likely mechanism).** At patch_size 8 a 128 px tile becomes a 16×16 = 256-node grid, and a
PASTIS parcel spans only a handful of nodes — so *locality was never the bottleneck*. Global attention
already recovers local structure when it is useful **and** retains long-range context (field-to-field
context, whole-tile phenology); 6 GraphSAGE layers with mean aggregation give a comparable receptive
field on a 16×16 grid but through a strictly **lower-capacity, lower-selectivity** operator (a fixed
uniform mean over 8 neighbours vs learned content-dependent attention weights). The GNN therefore
trades away capacity for a locality prior that the ViT was not lacking. Per-class IoU is consistent:
the graph loses most on the well-populated crop classes (0.195→0.140, 0.378→0.335, 0.417→0.348) while
picking up marginal ground on a few rare ones — i.e. slightly noisier, not differently structured.

**Verified on synthetic tensors before the run** (no PASTIS): grid-graph construction, message passing,
the encoder interface, JEPA forward + collapse-safe grad routing, encoder-checkpoint round-trip, and
the ViT default — `tests/test_graph.py` (9 pass; full suite 80 pass / 3 skip). Reproduce:

```bash
# on the server, after scripts/download_pastis.sh.  Use the SAME base config as the existing
# baselines (tjepa_server.yaml) + --only so the graph cell inherits identical hyperparameters and
# ONLY the spatial backbone changes (clean apples-to-apples vs tjepa_h1). Do NOT pass
# tjepa_graph.yaml as the base to the full matrix — the non-graph cells would inherit graph too.
python scripts/run_matrix.py --config configs/model/tjepa_server.yaml --data configs/data/pastis.yaml \
    --only tjepa_graph --device cuda:0 --resume --knn

# head-to-head: probe BOTH checkpoints with the same script/split/budget (the table above)
for c in tjepa_graph tjepa_h1; do
  python scripts/evaluate.py --encoder-ckpt runs/matrix/$c.pt \
      --config configs/model/tjepa_server.yaml --data configs/data/pastis.yaml \
      --head both --knn --probe-epochs 15
done
```

*Two protocol traps this section had to fix, recorded so the comparison stays honest:* (i) the
baselines were probed on **val**, so probing the graph cell on `--test` would have confounded backbone
with split — both numbers above are val; (ii) `scripts/evaluate.py --encoder-ckpt` hardcoded
`SITSEncoder`, so a graph checkpoint could not load at all — it now dispatches on the checkpoint's
saved `encoder.spatial_backbone` (regression-tested both ways in `tests/test_graph.py`).

**Hypothesis rejected.** The pre-registered expectation was *"a modest gain — local spatial structure
is real in SITS."* It is not borne out: the gain is negative. Local spatial structure being *real* does
not make an architectural locality prior *useful*, because global attention already captures it without
giving up long-range context. This is the spatial-domain analogue of §D's negative — matching the
*form* of the dynamics helps (§A/§C), but constraining a backbone that was not the bottleneck does not.

## Verdict & scope
- **Positive:** structured dynamics priors (Koopman, Neural-ODE) improve the JEPA representation over a
  free-form predictor, scaling with predictability; the LKF's dynamics gain scales with (one-step)
  predictability and vanishes at zero — both confirm the thesis at the *method* level, complementing
  the *measurement*-level confirmation (report_predictability.md §2).
- **Negative (§E):** a local grid-GNN spatial backbone *loses* to global attention on real PASTIS
  (conv mIoU 20.65 vs 22.46), with no collapse to blame — a locality prior does not pay when the
  baseline was not locality-limited.
- **Honest limits:** margins are modest (+0.04 R²), single-seed, small synthetic models; the LKF is
  demonstrated on the true latent (a clean filter demo), not yet closed-loop through the JEPA encoder
  space (the natural next step). The Koopman is global-linear; a deep/locally-linear Koopman and a
  Lorenz-tuned ODE would likely widen the chaotic-regime gap. §E is one GNN (GraphSAGE-mean) on a
  *grid* graph at one patch size; the stronger untested version is a true **parcel-adjacency** graph
  from an over-segmentation, where nodes are semantic regions rather than fixed tiles — that changes
  the graph from "a coarser way to be local" into genuine structure the ViT lacks.
- **Part-6 agenda now fully implemented AND run** (all 8): #1 LKF (§B), #2 predictability-weighting
  (report_predictability.md §4, negative), #3 Koopman + #4 Neural-ODE (§A/§C, **positive**), #5
  distributional (Phase 4, report_finance.md §8, negative), #6 info-theoretic falsification
  (past→future-MI, report_predictability.md §2), #7 hierarchical (§D, flat), #8 graph temporal JEPA
  (§E, **negative** on real PASTIS). No open cells remain.
- **The through-line across all 8:** structural priors that *match the dynamics* (Koopman/ODE) help,
  on synthetic and on real C-MAPSS; generic tweaks to a free-form predictor (reweighting, extra
  horizons, distributions) and priors on a component that was not the bottleneck (the graph backbone)
  do not. **Scoreboard: 2 wins (#3, #4), 1 diagnostic confirmation (#1, #6), 4 clean negatives
  (#2, #5, #7, #8)** — the negatives are as informative as the wins, and all are reported.

## Reproducibility
`pytest tests/test_structured.py` (8, offline). Benchmark: `python scripts/structured_predictor_bench.py
--device cuda:0`. Configs: set `predictor.type: koopman` or `ode` on any fjepa/cjepa temporal cell.
