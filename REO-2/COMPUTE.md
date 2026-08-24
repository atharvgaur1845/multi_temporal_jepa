# Compute — measured on this machine, 2026-08-25

**Deadline: Sep 2, 2026 AoE — 8 days.**

Not estimated. Benchmarked by running the real `engine.train_jepa.train_one_epoch` against synthetic
tensors of the exact PASTIS batch shape `(16, 32, 10, 128, 128)`, so no dataset was required.

## Answer: yes, it runs, and it fits — with one unresolved blocker

| | temporal JEPA | spatial JEPA |
|---|---|---|
| s / micro-step (batch 16) | **2.642** | 0.152 |
| samples / s | 6.1 | 105.3 |
| **peak GPU allocated** | **5.85 GiB** | 1.31 GiB |
| peak GPU reserved | 6.69 GiB | 1.39 GiB |
| **100 epochs, N_train=1461** | **6.7 GPU-h** | **0.4 GPU-h** |

Card: RTX 4060 Laptop, 8188 MiB, **7737 MiB free**, no compute processes. Stack verified:
`torch 2.12.0+cu130`, CUDA 13.0, device detected.

- **It fits, with ~1 GiB of headroom.** `configs/model/tjepa_8gb.yaml` (batch 16, `grad_accum` 12,
  effective batch 192 — mathematically identical to the server's 32x6) peaks at 6.69 GiB reserved
  against 7.55 GiB free. The config's own comment ("needs the full card") is accurate: **nothing else
  may touch the GPU while this runs.**
- **This card is ~3.2x slower than the server** for temporal JEPA (6.7 h vs the 2.12 GPU-h recorded in
  `runs/matrix_results.csv`).
- **Spatial JEPA is 17x cheaper than temporal here**, versus 3.7x on the server — the laptop is
  disproportionately penalised by the 32-frame temporal batch, which is memory-bandwidth bound.

## The blocker is now host RAM, not the GPU

```
Mem:  total 15 GB   used 11 GB   available 3 GB   |   Swap: 2 of 3 GB already used
```

No single large consumer — Chrome, VSCode and Spotify at ~0.5 GB each. **The first benchmark attempt
was SIGKILLed by the host OOM killer**, not by CUDA.

`scripts/run_matrix.py:205` hardcodes `num_workers=8` (and `engine/train_jepa.py:141` likewise). Each
worker assembles a full batch — post-subsample `16 x 32 x 10 x 128 x 128 x 4 B = 335 MB`, and up to
`61` frames before subsampling — with `prefetch_factor=2`:

```
8 workers x 2 prefetch x 335 MB  =  ~5.4 GB    against 3 GB available
```

**This will OOM-kill the run**, most likely hours in, exactly as it killed the benchmark.

**Required change before P0:** drop `num_workers` to **2** (~1.3 GB) in `scripts/run_matrix.py:205`.
Close Chrome and Spotify while training. This is a one-line edit and it is not optional.

## Disk

```
/dev/nvme1n1p2  468G  373G  72G  16% avail
```

72 GB free against the ~58 GB PASTIS peak (28.76 GB zip + ~29 GB extracted). **Clears, with ~14 GB
margin** — thin. Delete `PASTIS.zip` the moment `unzip` exits 0.

## Budget to Sep 2

| item | GPU-h | note |
|---|---|---|
| PASTIS download + md5 + unzip | — | 1–3 h wall clock, not GPU |
| **P0** `random` + `raw_features` | ~0 pretrain | probe-only; verified at `run_matrix.py:208-218` |
| **P1** temporal JEPA, per seed | **6.7** | the dominant cost |
| **P1** spatial JEPA, per seed | 0.4 | |
| Fig 1 `tjepa_noreg` | ~1.3 | collapse happens early; **20 epochs is enough for the curve** |
| **P2** temporal-order pretext | ~6.7 | full-sequence, so temporal-priced; **plus ~0.5 day to write** |
| MAE / BYOL / SimCLR reruns | — | **do not.** Server cost was 0.49 / 9.93 / 7.81 h; BYOL alone would be ~30–100 h here. Keep them at n=1 and label the row. |

Probe time is *not* included in any of the above — `GpuHourMeter` stops before the probe
(`run_matrix.py`), so the committed `gpu_hours` column excludes it. Every cell, floors included, pays
it. Budget slack accordingly.

- **n=3:** `3 x 6.7 + 3 x 0.4 = 21.3 h` + noreg 1.3 + P2 6.7 = **~29 GPU-h**
- **n=5:** `5 x 6.7 + 5 x 0.4 = 35.5 h` + noreg 1.3 + P2 6.7 = **~44 GPU-h**

Over 8 days on a laptop that is also your daily driver, and which cannot be used for anything else
GPU-bound while a run is live, n=3 is comfortable and n=5 is tight.

## Scheduling rule: make early stopping safe

Run seeds **strictly sequentially**, one complete seed at a time (`--seed N --resume`), never in
parallel — parallel tenants on one card are precisely what killed seeds 1 and 2 on the server.

Order the queue so that **stopping at any point still leaves an honest table**:

1. P0 floors (probe-only, cheapest, and the one non-negotiable item)
2. `tjepa_h1` + `spatial_jepa` seed 0 — reproduces the existing result on this hardware
3. seed 1, then seed 2 — at this point the paper's *already written* "3 seeds, p=0.041" becomes true
4. `tjepa_noreg`, 20 epochs — unlocks Fig 1
5. P2 temporal-order — the sharpest reviewer defence
6. seeds 3 and 4 **only if the calendar allows**

Then report the `n` you actually reached, per row. Note that steps 1–3 make the submitted text
*honest*; steps 4–6 make it *stronger*. Do them in that order.
