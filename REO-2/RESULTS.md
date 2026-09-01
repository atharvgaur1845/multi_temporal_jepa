# Results — live, as runs land

Every number here is read from a committed CSV. Nothing is carried over from drafting text.

---

## P0 — the floors. **LANDED 2026-08-26.**

`runs/matrix_results.csv`, PASTIS val fold, seed 0, conv head, probe budget 15 epochs, identical
backbone and effective batch 192 across every trained cell.

| cell | conv mIoU | parcel kNN | gpu-h | note |
|---|---|---|---|---|
| **tjepa_h1** | **22.06** | **0.6680** | 2.12 | |
| spatial_jepa | 16.19 | 0.5871 | 0.58 | kNN **below both floors** |
| _raw_features_ | _14.46_ | _0.6037_ | 0.00 | **FLOOR** — patch-mean bands, 0 parameters |
| _random_ | _11.18_ | _0.6058_ | 0.00 | **FLOOR** — tjepa_h1 architecture, never trained |
| simclr | 8.29 | 0.5456 | 7.81 | below both floors |
| byol | 4.99 | 0.6266 | 9.93 | below both floors on mIoU |
| mae | 3.78 | 0.5436 | 0.49 | below both floors |

`gpu_hours = 0.00` and `peak_mem_gb` of 0.31 / 0.01 confirm neither floor trained.

### Pre-registered rule P1 (`evidence/PREREGISTRATION.md`)

> Entitled iff `tjepa_h1` conv mIoU exceeds **both** `random` and `raw_features`, **and** the margin
> over the larger floor exceeds the across-seed standard deviation (n=3).

| condition | result |
|---|---|
| `tjepa_h1` 22.06 > `random` 11.18 | **PASS** |
| `tjepa_h1` 22.06 > `raw_features` 14.46 | **PASS** |
| margin over the larger floor | **+7.60 mIoU** |
| margin > across-seed sd | **PENDING** — needs the P1 seed re-run |

**The headline claim is not retracted.** The clause that would have forced "the architecture helps"
does not trigger: `tjepa_h1` clears the random-init control by +10.88. Full entitlement still waits on
the seed sd, which is the only remaining condition.

### The floors did not behave like floors

This is the substantive finding, and it was invisible for the entire project because the cells had
never been run.

1. **MAE, BYOL and SimCLR all fall below the raw-feature floor** on conv mIoU — 3.78 / 4.99 / 8.29
   against 14.46. Patch-mean raw bands, with **zero parameters**, beat all three.
2. **They fall below the random-init floor too** (11.18). An untrained encoder beats them.
3. **`raw_features` is a strong baseline, not a formality.** At 14.46 it sits only 1.73 below
   spatial JEPA. Most of what a linear/conv probe reads off these representations is available
   directly from the raw bands.
4. **On parcel kNN, spatial JEPA (0.5871) is below both floors** (0.6058 / 0.6037). Only `tjepa_h1`
   (0.6680) and BYOL (0.6266) clear. The kNN evidence for spatial JEPA is negative.

### What this obliges us to rewrite

The drafting template's abstract claimed temporal JEPA beats "reconstruction/contrastive baselines"
by +15–16 mIoU. Under the project's own V1 criterion that is **a contrast between two failures**:
those baselines do not clear a trivial floor, so their ranking orders failures rather than methods.

The defensible claims after P0:

- **Keep:** temporal JEPA clears both floors (+7.60 over the larger), pending the sd condition.
- **Keep:** temporal JEPA beats spatial JEPA (+5.87 conv mIoU). Both clear the raw floor, so this
  comparison is between two methods that demonstrably learn something. This is the paper's spine.
- **Restate:** *not* "temporal JEPA beats MAE/BYOL/SimCLR by +15–16", but "**MAE, BYOL and SimCLR
  fail to beat patch-mean raw bands on PASTIS under a matched frozen-probe protocol**". That is a
  sharper, more useful, and more honest statement — and it is a result an EO audience should hear,
  because these baselines are routinely reported without any floor at all.
- **Add to Limitations:** our SimCLR uses the negative count afforded by the per-step batch; below-floor
  performance for contrastive methods may partly reflect that. Say it; do not use it to explain the
  result away, since MAE is not contrastive and fails just as badly.

Table 1 should put the floors in the middle, with everything below them visibly marked.

---

## P1 — seeds. **LANDED 2026-08-28.** n=3, 686 min wall-clock.

`runs/matrix_results__s{0,1,2}.csv` — PASTIS val fold, conv head, batch 12 x accum 16.

### Pre-registered rule P1: **ENTITLED**

| condition | value | verdict |
|---|---|---|
| (a) `tjepa_h1` 19.26 > `random` 11.15 | +8.11 | **PASS** |
| (b) `tjepa_h1` 19.26 > `raw_features` 14.47 | +4.78 | **PASS** |
| (c) margin over larger floor (4.78) > across-seed sd (1.30) | 3.7x | **PASS** |

**All three conditions met. This is the first claim in the project entitled by its own protocol.**

| cell | per-seed conv mIoU | mean ± sd | kNN mean ± sd |
|---|---|---|---|
| **tjepa_h1** | 20.03 / 19.99 / 17.75 | **19.26 ± 1.30** | **68.67 ± 1.44** |
| _raw_features_ | 14.48 / 14.47 / 14.47 | _14.47 ± 0.01_ | _60.03 ± 0.52_ |
| spatial_jepa | 14.84 / 14.20 / 12.65 | 13.90 ± 1.13 | 58.92 ± 3.10 |
| _random_ | 11.35 / 11.15 / 10.95 | _11.15 ± 0.20_ | _60.58 ± 0.75_ |

Paired one-sided t-tests, matched by seed (n=3): vs `spatial_jepa` **p=0.0008**,
vs `raw_features` **p=0.0119**, vs `random` **p=0.0032**.
Wilcoxon is 0.125 for all three — **its floor at n=3**, not evidence of anything. Report the
t-test as primary and say why, or run seeds 3–4 (one-sided Wilcoxon can reach 0.03125 at n=5).

### The headline result, restated

**Temporal JEPA is the only method that clears either floor — on both metrics.**

`spatial_jepa` at 13.90 ± 1.13 is now **below** the raw-feature floor (14.47 ± 0.01) on conv mIoU,
and below both floors on kNN (58.92 vs 60.03 / 60.58). Combined with the n=1 baselines from P0
(SimCLR 8.29, BYOL 4.99, MAE 3.78, all far under 14.47):

> On PASTIS, under a matched frozen-probe protocol, **no standard SSL objective beats patch-mean
> raw bands — except temporal latent prediction.**

That is a sharper and more useful paper than "temporal beats spatial by +6", and it is what the
data supports. The floors carry it, and the floors had never been run.

### CAVEAT — do not pool these with the seed-0 server row

`runs/matrix_results.csv` (server, **batch 32** x accum 6) gives tjepa 22.06 / spatial 16.19.
The laptop runs (**batch 12** x accum 16) give 19.26 / 13.90. Same *effective* batch 192, and the
numbers still moved together by ~2.5 mIoU. The reason is structural:

`engine/train_jepa.py` computes `variance_covariance_reg(ctx)` **inside the micro-batch loop**,
before `loss/grad_accum`. So the VICReg variance hinge and covariance estimate see `batch_size`
samples — 12 or 32 — **not** the effective 192. **Gradient accumulation does not make the
anti-collapse term batch-size invariant.** "Matched effective batch" is therefore true of the
gradient signal and false of the regularizer.

Consequences:
- The three laptop seeds share one config, so every **between-cell** comparison above is valid.
  Rule P1 is unaffected.
- The server row is a **different configuration** and must not be pooled. The n=3 aggregate that
  mixed them (19.95 ± 2.16) double-counted seed 0 and is withdrawn.
- Report the per-step batch alongside the effective batch in the paper, and state this limitation.
  A reviewer who knows VICReg will ask.

### Two bugs found in this run

1. **`__s1.csv` had no header.** `--resume` appends when the output file *exists*; the earlier OOM
   left a zero-byte CSV, so the header was never written, and `csv.DictReader` then ate the first
   data row as the header. Seed 1 silently vanished from the aggregate with no error.
   Fixed in `run_matrix.py`: append only when the file is non-empty *and* starts with `cell,`.
2. **The default glob pools incompatible configs.** `aggregate.py --glob 'runs/matrix_results*.csv'`
   sweeps in the unsuffixed server file. Use `'runs/matrix_results__s[0-9].csv'`.


---

## Anti-collapse ablation (`tjepa_noreg`). **LANDED 2026-08-29.**

`runs/matrix_results.csv`, seed 0, batch 12 x 16 (same config as the multi-seed cells,
so it pairs directly with `tjepa_h1__s0.csv`).

| | VICReg on (`tjepa_h1`, seed 0) | VICReg off (`tjepa_noreg`) |
|---|---|---|
| per-dimension std | 0.9479 | **0.1721** |
| effective rank (of 512) | 312.41 | **1.49** |
| off-diagonal covariance | 0.00248 | **0.56422** |
| conv mIoU | 20.03 | **8.68** |
| linear mIoU | 16.03 | **6.11** |

Effective rank $1.49$ of $512$ is a representation with essentially one direction. At $8.68$ conv
mIoU the unregularized model sits **below both floors** (raw 14.47, random 11.15): on this data the
temporal objective without the penalty is worse than not training at all.

**Provenance caveat, stated in the paper.** The per-step `std`/`effrank` traces printed during
training were never written to a file (`run_local.sh` does not tee). These diagnostics were therefore
measured post-hoc from the saved encoders by `REO-2/figures/collapse_diagnostics.py`, on the
validation fold, through `encode_temporal` + pad-masked temporal pooling, which is the exact pathway
`eval/linear_probe.extract_dense_features` uses for JEPA cells. Both cells are single-seed.
Raw output: `REO-2/evidence/collapse_diagnostics.json`.

**Note on grouping.** `tjepa_noreg` landed in the unsuffixed CSV because no `--seed` was passed, but
it was run at batch 12 by `run_local.sh`, not at the server's 32. `make_results_csv.py` classifies it
by how it was run (`ablation_n1`, `b12xa16`), not by which file it sits in.

---

## Mechanism probe (H-mech-2). **LANDED 2026-08-31.**

Decoding acquisition time from the frozen **spatial** features only. `scripts/mechanistic.py`
probes `encode_full`, the per-frame spatial ViT, and never the temporal pathway: the temporal
encoder carries an explicit day-of-year encoding, so probing it would be circular. Val fold,
seed-0 checkpoints, 4800 train / 4800 eval frames.

| encoder | month accuracy (chance 8.3%) | DOY circular MAE |
|---|---|---|
| `tjepa_h1__s0` | **63.9%** | **29.9 days** |
| `spatial_jepa__s0` | 47.0% | 40.8 days |

A +16.9 point gap in month accuracy and 10.9 days less circular error, from the *spatial*
representation alone. Since PASTIS crop classes separate by phenological stage, this is direct
evidence that the future-prediction objective made the single-frame representation season-aware,
which is the mechanism behind the segmentation gap.

It also answers the alternative reading raised in Discussion. If the frozen-probe protocol were
simply insensitive to what the other objectives encode, there would be no reason for temporal
JEPA's *spatial* features to carry markedly more seasonal information than spatial JEPA's. They do.

These numbers supersede the 61.3% / 46.3% quoted in earlier drafts, which came from a server run
at a different per-step batch and were never traceable to a committed artifact. Raw output:
`REO-2/evidence/mechanistic_s0.log`. Single seed.

## C3 temporal-order pretext: ABORTED, not reported in the paper

Ran `objectives/baselines/temporal_order.py` (frame-order verification, DOY kept chronological so
day-of-year cannot leak the permutation) at the matched configuration: batch 12, accum 16, seed 0,
`.tjepa_order.yaml`. Aborted at epoch 20/100 (756 optimizer steps) on a pre-set learning gate.

Train accuracy over 20 epochs: 0.465-0.527, no trend. Loss pinned at ~0.693 = ln 2, i.e. an exactly
uninformative binary classifier. Log: `REO-2/evidence/temporal_order_s0.log`.

**Not reported as a negative result**, and the Limitations clause "we have no non-predictive temporal
pretext" is deliberately left standing. The run does not establish that the pretext fails: the
suspected cause is design, not objective. `pool_spatiotemporal` takes the mean over frames, and a
mean is permutation-invariant, so order information survives only through temporal-encoder attention
before pooling. Fixing that pooling is the first thing to try before this is claimed either way.

An earlier 2-epoch smoke (15 optimizer steps) also sat at 0.500; that was too short to be
diagnostic and only verified the code path, memory footprint and CSV/checkpoint writes.

## Reframe pass (2026-09-01): claim audit found three unsupported markers

Recomputed every Table 1 cell as a mean per-seed PAIRED difference from that readout's binding
floor, in units of its across-seed sd. Three published markers were not supported:

| cell | conv | linear | k-NN | was | now |
|---|---|---|---|---|---|
| spatial JEPA | -0.5 sd | +1.0 sd | -0.7 sd | marked below on conv and k-NN | indistinguishable on all three |
| MAE | -3.2 sd | -2.9 sd | **-0.1 sd** | marked below on k-NN | indistinguishable on k-NN |
| SimCLR | -24.2 | -14.7 | -3.5 | below | below (unchanged) |
| BYOL | -8.8 | -22.1 | -20.2 | below | below (unchanged) |
| temporal JEPA | +3.7 | +6.0 | +7.8 | clears | clears (unchanged) |

Consequence: **"four of five fail to clear the binding floor" overstated what the data establishes.**
Corrected claim: only temporal JEPA establishes a margin above any binding floor; three of the
remaining four are established below it and spatial JEPA is indistinguishable from a zero-parameter
baseline on every readout. Table 1 now carries a `conv gap/sd` column and distinguishes
$\downarrow$ (established below, >=2 sd) from $\approx$ (within 2 sd).

p-values removed from the body per the reframe; paired tests retained in Appendix B only.
Setup now states explicitly that cells are matched on epochs and batch but **not on compute**
(MAE 0.55 GPU-h/seed vs temporal JEPA 6.24, a factor of 11), since equal-epoch and equal-compute
are different questions and this paper answers only the first.

## Probe-capacity sweep (2026-09-01): the margin over the floor vanishes with probe capacity

Frozen encoders re-probed at receptive field 1/3/5/9 (n 3x3 convs then 1x1; RF=2n+1). `linear` and
`conv` are structurally identical to RF1 and RF3, so published columns are points on this curve.
Probe head-init seed pinned to 0 to match run_matrix, which never passes `seed=`.

Validation: raw_features s0 RF1 = 4.75 vs published linear 4.78; tjepa_h1 s0 RF1 = 16.04 vs
published 16.03. (Not bit-identical: the probe train loader is shuffle=True and built once, so
data order depends on cell ordering, worth ~0.03 mIoU.)

Seed 0, conv-family mIoU:

| RF | raw floor | random floor | temporal JEPA | tjepa - raw |
|----|-----------|--------------|---------------|-------------|
| 1  |  4.75     |  8.26 (pub)  | 16.04         | +11.29 |
| 3  | 14.52     | 11.35 (pub)  | 20.03 (pub)   |  +5.51 |
| 5  | 20.27     | 13.11        | 20.77         |  +0.50 |
| 9  | 21.61     | (pending)    | 21.21         |  -0.40 |

The raw-band floor is NOT a fixed quantity: it moves 4.75 -> 21.61, a factor of 4.5, purely as a
function of probe receptive field. Temporal JEPA's margin over it decays monotonically and is gone
by RF5. **n=1 so far; +0.50 and -0.40 are both well inside the ~1.3 across-seed sd, so no reversal
is established.** Seeds 1,2 at RF5/RF9 are running; the claim stands or falls on those.

If it holds at n=3, the headline "temporal JEPA clears the binding floor" must be scoped to RF<=3,
and the paper's thesis strengthens: objective rankings are only meaningful at a stated probe
capacity, because a zero-parameter baseline closes the entire gap when the probe can mix spatially.

## CONFIRMED at n=3 (2026-09-01 18:00): the margin is a function of probe capacity

| RF | raw floor | random floor | temporal JEPA | tjepa - raw | ratio | verdict |
|----|-----------|--------------|---------------|-------------|-------|---------|
| 1  |  4.76+-0.02 |  8.41+-0.15 | 15.32+-1.13 | +10.56+-1.12 | +9.39 | established above |
| 3  | 14.47+-0.01 | 11.15+-0.20 | 19.26+-1.30 |  +4.78+-1.30 | +3.67 | established above |
| 5  | 20.24+-0.02 | 12.31+-0.78 | 20.09+-1.29 |  -0.15+-1.28 | -0.12 | indistinguishable |
| 9  | 21.45+-0.14 |  8.62+-2.85 | 19.88+-1.39 |  -1.57+-1.28 | -1.23 | indistinguishable |

NOT a reversal at RF9: -1.23 sd does not establish the sign. What IS established is that the
positive margin exists at RF<=3 and is absent at RF>=5.

Paper rewritten around this. Title now "The Floor Moves: Probe Capacity Decides Whether
Self-Supervised Objectives Beat Trivial Baselines on Satellite Image Time Series". Fig 1 replaced
by the capacity figure (fig_capacity.pdf); fig1_floors.pdf retired. New Appendix B gives every
point. Headline explicitly scoped: "clears the binding floor on all three readouts AT RECEPTIVE
FIELD 3, and not above it."

Honest caveat recorded in Limitations: the sweep varies RF at a fixed 15-epoch probe budget, so
capacity and ease of optimisation are not separated.

## Shuffled-time control COMPLETE (2026-09-02, seed 0)

| seed 0 | conv | linear | k-NN | GPU-h |
|---|---|---|---|---|
| temporal JEPA (real future) | 20.03 | 16.03 | 67.84 | 6.19 |
| shuffled-time control | 18.38 | 14.48 | 65.56 | 3.80 |
| raw features (floor) | 14.48 | 4.78 | 59.54 | 0 |

Permuting frames (dates left chronological) costs only **1.65 conv mIoU**. The control still clears
the raw floor by 3.90 of temporal JEPA's 5.55, i.e. it keeps ~70% of the margin. 1.65 is 1.3
across-seed sd of the temporal cell, so at n=1 the cost of shuffling is NOT established as non-zero.
Effective rank rose to 411 during training, so the control did not collapse and the small gap is not
an artifact of a degenerate run.

Interpretation: what the temporal cells mostly buy is **use of the temporal axis**, not prediction of
the actual future. Written into Results as two lines plus Appendix E.

## Final review pass (2026-09-02): all five items applied

1. Pre-registration is capacity-contingent -- new paragraph. VERIFIED against the rule: it passes at
   RF 1 AND 3, fails at 5 and 9 (temporal JEPA does not exceed the raw floor there at all). Stated as
   a threshold, not as "RF3 was lucky".
2. Abstract now states the overtake explicitly (20.24 baseline vs 20.09 encoder) and surfaces
   "per-step and effective batch matched, compute deliberately not matched".
3. Contributions bullet 2 rewritten to what the sweep licenses; "one of five" left in Results.
4. Title -> "The Floor Moves With the Probe: Trivial Baselines for Self-Supervised Learning on
   Satellite Imagery". Two clean lines, no orphan. ("Satellite Image Time Series" would not fit two
   lines at title width; "Decides" dropped as overclaiming.)
5. n=2 sd clause added; Appendix A leads with verification before the CSV race; figure y-axis label
   WAS genuinely truncated ("raw flo") from the reduced panel height -- fixed to "margin (mIoU)",
   legend reframed with headroom.
