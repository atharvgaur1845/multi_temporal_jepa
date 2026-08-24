# P0 — `random` + `raw_features` floors on PASTIS

**Priority: non-negotiable.** Everything else is optional relative to this.

## Why

`evidence/PREREGISTRATION.md` rule **P1**:

> **Entitled iff** `tjepa_h1` conv mIoU exceeds **both** `random` and `raw_features`, and the margin
> over the larger floor exceeds the across-seed standard deviation (n=3).
> - If `tjepa_h1` <= `random`: the win is attributable to the temporal-attention **architecture**, not
>   the objective. We will say so, and the headline claim is retracted to "the architecture helps."
> - If `tjepa_h1` <= `raw_features`: no representation learning is demonstrated on PASTIS at all.

Neither floor has ever been run (`runs/matrix_results.csv` has five rows, none of them a floor). The
central claim is currently **unassessable by our own protocol** — and the venue is run by the author
of the dataset. That combination is not submittable.

## Blockers (see `../STATUS.md`)

1. `data_root/PASTIS` does not exist — ~29 GB download.
2. 25 GB free disk against a ~58 GB extraction peak. **Free ~40 GB or attach external storage first.**
3. GPU busy: PID 299272 (Kaggle job) holds 6.3 of 8 GB.

Blocker 3 matters less here than anywhere else — see "cost" below.

## Cost — much lower than the "~1 day" estimate

Verified at `scripts/run_matrix.py:208-218`: **neither floor trains anything.**

- `raw_features` -> `models/raw_encoder.RawPatchEncoder`, patch-mean over raw bands, no parameters.
- `random` -> `build_model(cfg).target_encoder`, weights never updated (the untrained control that
  isolates the objective from the temporal-attention architecture).

Both go straight to `linear_probe_segmentation`. Cost is two frozen probes, not two pretrainings.

## Commands

```bash
# 0. prerequisite: PASTIS present and configs/data/pastis.yaml:root pointing at it
bash scripts/download_pastis.sh ./data_root       # ~29 GB, verifies size + md5

# 1. floors, same base config and same val fold as the existing five cells
python scripts/run_matrix.py \
    --config configs/model/tjepa_8gb.yaml \
    --data   configs/data/pastis.yaml \
    --only   random,raw_features \
    --resume \
    --probe-epochs 15
```

`--only` is the right switch: it runs these two cells against the **same base config** as the
committed baselines without re-running or contaminating them (see the flag's own docstring). `--resume`
appends rather than rewriting the header.

**Note the CSV header.** The committed `runs/matrix_results.csv` has the old 8-column header; the
driver now writes 10 (`seed`, `cv_fold` added). Commit `55175b6` added `scripts/migrate_matrix_csv.py`
for exactly this. Run it **before** appending, or the new rows will misalign under `csv.DictReader`:

```bash
python scripts/migrate_matrix_csv.py runs/matrix_results.csv   # writes a .bak, idempotent
```

## Acceptance

- Two new rows in `runs/matrix_results.csv`, `eval_split=val`, same probe budget (15 epochs), both
  heads (`linear` + `conv`).
- Copy the updated CSV into `../evidence/` and re-run the P1 rule.
- **Report the outcome whichever way it goes.** The pre-registration commits to this in advance. If
  `tjepa_h1` does not clear `random`, the paper's claim becomes "the temporal-attention architecture
  helps" and Table 1 says so.

---

## Addendum — the Fig 1 gap, folded in here because it is the same kind of hole

Fig 1 (effective rank, VICReg **on vs off**) is designated the money figure. The **off** curve does
not exist for PASTIS. `tjepa_noreg` is logged `SKIPPED (budget)` in `run_s0.log`, `run_s1.log` and
`run_s2.log`; there is no PASTIS noreg run in any committed log or CSV. The `erank -> 2.4` collapse
number in `README.md:1675` is not traceable to a file in this checkout.

`tjepa_noreg` **does** require pretraining (it is temporal JEPA with `var_coeff=cov_coeff=0`), so it
costs about the same as one `tjepa_h1` seed — but it collapses fast, and the figure only needs the
early steps where `effrank` crashes. A short run is enough for the curve.

```bash
python scripts/run_matrix.py --config configs/model/tjepa_8gb.yaml \
    --data configs/data/pastis.yaml --only tjepa_noreg --resume
```

If it is not run, **Fig 1 cannot be drawn honestly** and the slot goes to Fig 2 + Table 1.
`figures/fig1_effective_rank.py` refuses to plot the missing series rather than invent it.
