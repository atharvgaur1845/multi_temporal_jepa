# P1 — recover or re-run the seed CSVs

## The seed CSVs cannot be recovered. This is a re-run.

`run_s1.log` and `run_s2.log` both die at **step 0 of the first cell**:

```
[run_matrix] device = cuda:1  seed = 1
step 0 loss 1.9508 ... effrank 16.3 varratio 0.062
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.00 GiB.
GPU 1 has a total capacity of 47.40 GiB of which 1.83 GiB is free.
```

Seeds 1 and 2 produced **no numbers whatsoever**. `run_s0.log` reached `mae` and then took a
`KeyboardInterrupt` inside `linear_probe_segmentation`. Paths are `/home/jagat/Atharv/...`, a shared
card with four other tenants.

So `22.3 ± 1.8` and `p = 0.041` (and `0.036 / 0.009 / 0.001`) have **no artifact behind them in this
checkout**. There is nothing to search for.

## Why n=3 is not enough anyway

At n=3 the Wilcoxon signed-rank test has a **minimum attainable two-sided p of 0.25** — it cannot
reach significance at any effect size. The reported `p = 0.041` is a paired *t*-test on three points,
which is one unlucky seed away from nothing. **Go to 5 seeds if compute allows.**

## Compute reality (see `../STATUS.md`)

Server gpu-h per cell, from the committed CSV: `tjepa_h1` 2.12, `spatial_jepa` 0.58, `mae` 0.49,
`byol` 9.93, `simclr` 7.81. The card here is an **8 GB laptop 4060**, currently at 6.4/8.2 GB under an
unrelated Kaggle job, and slower than the 47.4 GB card those hours were measured on.

**5 seeds x 7 cells is not affordable.** BYOL + SimCLR alone are ~88 server-gpu-h at n=5.

### The defensible reduction

Multi-seed the cells the pre-registered rule and the headline claim actually depend on:

| cell | seeds | why |
|---|---|---|
| `tjepa_h1` | 5 | the claim |
| `spatial_jepa` | 5 | the comparison that carries the paper |
| `random`, `raw_features` | 5 | P0's floors; the rule compares the margin against the seed sd |
| `mae`, `byol`, `simclr` | 1 | 15–16 mIoU behind; error bars will not change the ordering |

Then **state `n` per row in Table 1** and name the asymmetry in Limitations. A stated n=1 is honest;
an unstated n=1 dressed as `±` is the thing that gets caught.

## Commands

```bash
python scripts/migrate_matrix_csv.py runs/matrix_results.csv   # 8-col -> 10-col header, once

for s in 1 2 3 4; do            # seed 0 already exists for the trained cells
  python scripts/run_matrix.py \
      --config configs/model/tjepa_8gb.yaml \
      --data   configs/data/pastis.yaml \
      --only   tjepa_h1,spatial_jepa,random,raw_features \
      --seed   "$s" --resume
done
```

`--seed` tags outputs (`runs/matrix_results__s<N>.csv`, `runs/matrix/<cell>__s<N>.pt`) so runs cannot
collide, and the default no-`--seed` paths stay intact for the existing seed-0 pass.

Run them **sequentially**, not in parallel — parallel tenants on one card are precisely what killed
seeds 1 and 2. Use `--resume` so an OOM or a reboot continues instead of restarting.

## Aggregation

```bash
python scripts/aggregate.py    # means, sds, paired tests across the seed-tagged CSVs
```

Report the paired *t*-test **and** the sign of every per-seed difference. At n=5 also report Wilcoxon
(min two-sided p = 0.0625 — still weak, but at least attainable). If the temporal-vs-spatial gap does
not survive, that is the result, and the pre-registration commits us to saying so.

## Acceptance

- `runs/matrix_results__s{1,2,3,4}.csv` exist and are committed.
- `aggregate.py` reproduces every `±` and every p-value that appears in the paper.
- No number in the paper that cannot be regenerated from a committed CSV.
