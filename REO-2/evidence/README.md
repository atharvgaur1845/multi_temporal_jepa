# evidence/ — frozen provenance

Copies, not symlinks, taken 2026-08-25 from the parent repo so the submission's inputs cannot drift
underneath it. Nothing here is edited. See `../STATUS.md` for what these files do and do not support.

| file | copied from | what it is |
|---|---|---|
| `PREREGISTRATION.md` | `PREREGISTRATION.md` | decision rules fixed before the confirmatory runs. Rule **P1** is the one that governs this submission. |
| `pastis_matrix_results.csv` | `runs/matrix_results.csv` | the **only** committed PASTIS results. 5 cells, val fold, one seed, **no floors**, old 8-column header. |
| `configs/pastis.yaml` | `configs/data/pastis.yaml` | dataset config. `root: ./data_root/PASTIS` — that path does not exist on this machine. |
| `configs/tjepa.yaml` | `configs/model/tjepa.yaml` | the server config the committed numbers came from. |
| `configs/tjepa_server.yaml` | `configs/model/tjepa_server.yaml` | server variant. |
| `configs/tjepa_8gb.yaml` | `configs/model/tjepa_8gb.yaml` | **the one to use here** — same quality (patch 8, embed 512, effective batch 192 via `grad_accum: 12`), per-step batch 16, ~6 GB peak. |
| `logs/run_s0.log` | `run_s0.log` | seed-0 pass. Reached `mae`, then `KeyboardInterrupt` in the probe. |
| `logs/run_s1.log`, `logs/run_s2.log` | `run_s1.log`, `run_s2.log` | seeds 1 and 2. **Both OOM at step 0.** This is why P1 is a re-run, not a recovery. |
| `logs/run_main_matrix_cells.txt` | `grep '[run_matrix]' run.log` | per-cell result lines, including ablation cells that never reached the CSV. |
| `logs/run_main_effrank_steps.txt` | `grep 'effrank' run.log` | 2792 per-step traces — the VICReg-**on** half of Fig 1. |

Not copied: `*.pt` / `*.ckpt` (gitignored, and `runs/` totals 1.5 GB), and every artifact belonging to
finance, C-MAPSS, the alignment testbed or the structured-predictor work, all of which are out of
scope for a 4-page EO submission.

## Reading the effrank traces

```bash
python ../figures/fig1_effective_rank.py --logs logs/run_s0.log ../../run.log --list
```

VICReg-on `tjepa_h1` climbs **15.2 -> 454.0** over 2950 steps (of 512 dims), consistent with the
"~430/512" figure quoted in the parent README. There is no `tjepa_noreg` trace in any of these files —
it is logged `SKIPPED (budget)` in all three seed logs.
