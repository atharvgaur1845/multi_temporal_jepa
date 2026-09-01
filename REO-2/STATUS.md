# STATUS — what this checkout can actually substantiate

> **2026-08-26: P0 HAS LANDED.** The floors exist now and the pre-registered rule's floor
> conditions both PASS. See **`RESULTS.md`** for the numbers and what they oblige us to rewrite.
> The audit below describes the state *before* that run; the "floors never run" row is now closed.

Audit date: 2026-08-25. Source of truth: files committed in this repo, nothing else.
Written before any REO-2 text, because the run priorities depend on it.

**Headline: the checkout supports one PASTIS run — val fold, seed 0, five cells, no floors.
Every other PASTIS number in `README.md` / `PAPER.md` comes from a server pass that is not here.**

---

## Substantiated locally

`runs/matrix_results.csv` — PASTIS, **val fold, single seed, no seed column**:

| cell | linear mIoU | conv mIoU | kNN | gpu-h |
|---|---|---|---|---|
| tjepa_h1 | 17.38 | **22.06** | 0.668 | 2.12 |
| spatial_jepa | 10.25 | 16.19 | 0.587 | 0.58 |
| mae | 3.96 | 3.78 | 0.544 | 0.49 |
| byol | 5.48 | 4.99 | 0.627 | 9.93 |
| simclr | 3.82 | 8.29 | 0.546 | 7.81 |

`run.log` (same pass, cells that never reached the CSV — see `evidence/logs/run_main_matrix_cells.txt`):

| cell | conv mIoU |
|---|---|
| spatial_jepa_matched (3.5x epochs) | 15.62 |
| tjepa_h2 / h4 / h8 | 19.07 / 18.98 / 21.85 |
| tjepa_preddepth 1/2/4/6 | 17.98 / 19.18 / 20.81 / 21.36 |
| tjepa_dim 128/256/512/768 | 21.71 / 24.01 / 21.20 / 6.76 |

Per-step `effrank` traces for VICReg-**on** temporal JEPA: `evidence/logs/run_main_effrank_steps.txt`
(2792 steps) and `run_s0.log`. This is the only half of Fig 1 that exists.

## NOT substantiated locally

| number quoted in README/PAPER | where it should live | actual state |
|---|---|---|
| `22.3 ± 1.8` and every other 3-seed mean/sd | seed-tagged `runs/matrix_results__s*.csv` | **absent** |
| `p = 0.041` vs spatial (and 0.036 / 0.009 / 0.001) | derived from the above | **cannot be recomputed** |
| `random` and `raw_features` floors | rows in `matrix_results.csv` | **never run** (cells exist in the driver, were not executed) |
| few-shot `9.2 / 13.1 / 15.9` vs `4.6 / 6.9 / 9.5` @ 1/5/10% | any CSV | **stdout-only, no file** |
| month-decoding `61.3%` vs `46.3%`, DOY MAE `30.4` vs `41.8` d | any CSV | **stdout-only, no file** |
| VICReg-off PASTIS collapse, `erank -> 2.4`, `std -> 0.04` | a `tjepa_noreg` PASTIS run | **never run** — logged `SKIPPED (budget)` in `run_s0/s1/s2.log` |
| test-fold column (`22.1`, `16.1`, ...) | a `--test` pass | **absent** |

## Why the seed CSVs are gone

`run_s1.log` and `run_s2.log` both die at **step 0** of the first cell:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.00 GiB.
GPU 1 has a total capacity of 47.40 GiB of which 1.83 GiB is free.
Process 767897 has 5.23 GiB memory in use. ... this process has 37.07 GiB in use.
```

Paths in the traceback are `/home/jagat/Atharv/multi_temporal_jepa` — a shared 48 GB card with four
other tenants. `run_s0.log` got further (through `mae`) and then took a `KeyboardInterrupt` inside
`linear_probe_segmentation`. **Seeds 1 and 2 produced no numbers at all.** There is nothing to
recover; P1 is a re-run, not a search.

## Consequence for the pre-registration

`evidence/PREREGISTRATION.md` rule **P1** requires `tjepa_h1` conv mIoU to exceed both `random` and
`raw_features`, by more than the across-seed sd at n=3. Right now **neither side of that inequality
exists**: no floors, no seed sd. P1 is not failed — it is unevaluated, which under the committed
reporting rule means the headline claim is not yet entitled to be stated in its strong form.

That is the whole reason P0 and P1 outrank everything else.

---

# Compute and data reality (checked 2026-08-25, before launching anything)

> **UPDATE, same day, after the user freed disk and the GPU.** Blockers 2 and 3 below are **cleared**:
> disk is now 72 GB free (was 25), and the GPU is idle with 7737 MiB free (the Kaggle job exited).
> Blocker 1 (PASTIS absent) stands. A **new** blocker appeared: **host RAM**, 3 GB available against a
> `num_workers=8` dataloader that wants ~5.4 GB. Measured throughput and the run schedule are in
> **`COMPUTE.md`** — read that, not the estimates below. The original readings are kept intact here
> because the seed-1/seed-2 OOM analysis still depends on them.

Nothing was launched. Three hard blockers, in the order they bite.

## 1. PASTIS is not on this machine

```
$ ls data_root/
CMAPSS  FINANCE          # PASTIS/ is gone
```

Every run in `protocol/` (P0, P1, P2, P3) reads PASTIS. **All four are blocked on this.**

`scripts/download_pastis.sh` pulls `PASTIS.zip` = **28.76 GB** (Zenodo 10.5281/zenodo.5012942,
md5 `cfc441bf18137ff0bbf4fad58828fb98`), then unzips to ~29 GB alongside it — **~58 GB peak**.

## 2. There is not enough disk for it

```
/dev/nvme1n1p2  468G  420G  25G  95% /      # single volume, no second mount
```

25 GB free against a 58 GB peak (29 GB if the zip is deleted the instant extraction succeeds, which
still exceeds free space). Freeing ~40 GB, or attaching external storage, is a **prerequisite to P0**.
Nothing inside this project helps: `.venv` 5.2 GB, `runs/` 1.5 GB, whole repo 6.7 GB.

## 3. The GPU is a laptop 4060, not the 48 GB server card

```
NVIDIA GeForce RTX 4060 Laptop    6424MiB / 8188MiB    98% util
  PID 299272  python3 -u kaggle/cuhkx_224_kaggle.py --stage train ...   6344MiB   (running 1h04m)
```

- **8 GB total, ~1.7 GB free right now.** An unrelated Kaggle job owns the card.
- The committed PASTIS runs used a shared **47.4 GB** card at `/home/jagat/Atharv/...` and still
  peaked at 6.65 GB for `tjepa_h1`.
- `configs/model/tjepa_8gb.yaml` exists for exactly this card — same quality (patch 8, embed 512,
  effective batch 192 via `grad_accum: 12`), per-step batch 16, ~6 GB peak. Its own comment says it
  **"needs the full card"**. So the Kaggle job must finish or be moved first.

### What this does to the priority order

| | needs pretraining? | server gpu-h | on a free 4060 |
|---|---|---|---|
| **P0** `raw_features` | **no** — `RawPatchEncoder`, patch-mean bands, probe only | ~0 | probe only, cheap |
| **P0** `random` | **no** — `build_model(cfg).target_encoder`, never updated, probe only | ~0 | probe only, cheap |
| **P1** `tjepa_h1` x5 seeds | yes | 2.12 each | the real cost, likely 3–5x slower |
| **P1** `spatial_jepa` x5 seeds | yes | 0.58 each | " |
| **P1** MAE / BYOL / SimCLR x5 | yes | 0.49 / 9.93 / 7.81 each | BYOL+SimCLR alone ≈ 88 server-gpu-h at n=5 |
| **P2** temporal-order pretext | yes | ~= spatial_jepa | + it does not exist yet, must be written |

**P0 is far cheaper than the "~1 day" estimate** — verified in `scripts/run_matrix.py:208-218`, both
floor cells skip training entirely and go straight to the frozen probe. Once PASTIS is on disk, P0 is
hours, not a day.

**P1 at 5 seeds across all seven cells is not affordable on this card.** The defensible reduction:
5 seeds on `tjepa_h1` + `spatial_jepa` + the two floors (the cells the pre-registered P1 rule and the
headline claim actually depend on), and report MAE / BYOL / SimCLR at n=1 with the `n` stated per row
in Table 1. That is a real limitation and belongs in the Limitations paragraph, not hidden.
