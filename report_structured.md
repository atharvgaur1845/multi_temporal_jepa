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

## E) Graph Temporal JEPA (Part 6 #8) — built + tested, PASTIS run pending (server)

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

**Verified on synthetic tensors** (no PASTIS): grid-graph construction, message passing, the encoder
interface, JEPA forward + collapse-safe grad routing, and the ViT default — `tests/test_graph.py`
(7 pass; full suite 78 pass / 3 skip). **The PASTIS pretrain is a server run** (GPU-hours/cell; can't
fit the 8 GB laptop):

```bash
# on the server, after scripts/download_pastis.sh.  Use the SAME base config as the existing
# baselines (tjepa_server.yaml) + --only so the graph cell inherits identical hyperparameters and
# ONLY the spatial backbone changes (clean apples-to-apples vs tjepa_h1). Do NOT pass
# tjepa_graph.yaml as the base to the full matrix — the non-graph cells would inherit graph too.
python scripts/run_matrix.py --config configs/model/tjepa_server.yaml --data configs/data/pastis.yaml \
    --only tjepa_graph --device cuda:0 --resume --knn --test
python scripts/evaluate.py --encoder-ckpt runs/matrix/tjepa_graph.pt \
    --config configs/model/tjepa_server.yaml --data configs/data/pastis.yaml --head both --knn --test
```

Compare the resulting conv-mIoU / kNN to the ViT temporal JEPA (report.md §7) to see whether the local
graph prior beats global attention on crop segmentation. *(Hypothesis: a modest gain — local spatial
structure is real in SITS — but untested until the server run.)*

## Verdict & scope
- **Positive:** structured dynamics priors (Koopman, Neural-ODE) improve the JEPA representation over a
  free-form predictor, scaling with predictability; the LKF's dynamics gain scales with (one-step)
  predictability and vanishes at zero — both confirm the thesis at the *method* level, complementing
  the *measurement*-level confirmation (report_predictability.md §2).
- **Honest limits:** margins are modest (+0.04 R²), single-seed, small synthetic models; the LKF is
  demonstrated on the true latent (a clean filter demo), not yet closed-loop through the JEPA encoder
  space (the natural next step). The Koopman is global-linear; a deep/locally-linear Koopman and a
  Lorenz-tuned ODE would likely widen the chaotic-regime gap.
- **Part-6 agenda now fully implemented** (all 8): #1 LKF (§B), #2 predictability-weighting
  (report_predictability.md §4, negative), #3 Koopman + #4 Neural-ODE (§A/§C, **positive**), #5
  distributional (Phase 4, report_finance.md §8, negative), #6 info-theoretic falsification
  (past→future-MI, report_predictability.md §2), #7 hierarchical (§D, flat), #8 graph temporal JEPA
  (§E, built + tested, PASTIS run pending on the server).
- **The through-line across all 8:** structural priors that *match the dynamics* (Koopman/ODE) help,
  on synthetic and on real C-MAPSS; generic tweaks to a free-form predictor (reweighting, extra
  horizons, distributions) do not. The remaining open question is #8 on real PASTIS (server).

## Reproducibility
`pytest tests/test_structured.py` (8, offline). Benchmark: `python scripts/structured_predictor_bench.py
--device cuda:0`. Configs: set `predictor.type: koopman` or `ode` on any fjepa/cjepa temporal cell.
