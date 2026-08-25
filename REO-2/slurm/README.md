# SLURM, from zero — for this project

You have never used SLURM. This file assumes that. Read it top to bottom once, then use the
scripts. **Deadline: Sep 2, 2026 AoE.**

Everything here supersedes `../COMPUTE.md`, which was written for your 8 GB laptop. That laptop is
now the fallback, not the plan.

---

## 0. The one idea

A cluster is many machines. You SSH into a **login node** — a small, shared, non-GPU machine. You do
**not** run training there. You write a script describing the resources you want, hand it to SLURM
with `sbatch`, and SLURM runs it later on a **compute node** that actually has the GPU.

> **Never run training on the login node.** It has no GPU, it is shared by everyone, and admins kill
> jobs found there. Login node = edit files, submit jobs, move data. That is all.

Three commands do 90% of the work:

| command | what it does |
|---|---|
| `sbatch script.sbatch` | submit a job, returns a job ID, exits immediately |
| `squeue -u $USER` | what are my jobs doing |
| `scancel <jobid>` | kill a job |

## 1. Pick a partition

A **partition** is a named queue pointing at a set of machines. Yours:

| partition | GPUs | per GPU | time limit | use it for |
|---|---|---|---|---|
| `gpu_a100_8` | 8x A100 SXM | **80 GB** | **5 days** | **primary.** 8 GPUs on one node + the longest limit |
| `gpu_h100_4` | 4x H100 x 2 nodes | 80 GB | 3 days | faster per GPU, 8 total |
| `gpu_h200_8` | 8x H200 NVL | ~140 GB | **1 day** | fastest, but the 1-day cap bites long jobs |
| `gpu_rtx_pro_6000_6_csis_hyd` | 6x RTX PRO 6000 | ~96 GB | 2 days | good fallback |
| `gpu_v100_2` | 2x V100 x 2 nodes | 32 GB | 3 days | older; fine for this model |
| `gpu_v100_1` | 1x V100 | 32 GB | **infinite** | the no-time-limit escape hatch |

**Use `gpu_a100_8`.** Our model peaks at 6.7 GiB on a laptop at batch 16 — an 80 GB A100 swallows it
whole, and 8 GPUs on one node means all five seeds run *at once*.

The `compute` / `big_compute*` partitions have **no GPUs** (`GRES` is `(null)`). Do not send training
there. They are useful for one thing: the PASTIS download, which is I/O not GPU.

## 2. What this changes about the plan

Measured on your laptop: temporal JEPA = **6.7 GPU-h** per seed. An A100 is roughly 10–15x that card
for this workload, so expect **~30 min/seed**, and an H100 ~15–20 min.

| | laptop | A100 (expected) |
|---|---|---|
| temporal JEPA, 1 seed | 6.7 h | ~0.5 h |
| P1 at 5 seeds (temporal + spatial) | ~36 h | **~3 GPU-h, under 1 h wall-clock across 8 GPUs** |
| MAE / BYOL / SimCLR at 5 seeds | unaffordable (~90 h) | affordable |

**So: do not reduce to n=3, and do not leave the baselines at n=1.** Both compromises in
`../protocol/P1_seeds.md` existed only because of the laptop. Run the full matrix at **5 seeds**.
That deletes an entire Limitations sentence.

## 3. Anatomy of a job script

```bash
#!/bin/bash
#SBATCH --job-name=p1-seeds        # shows up in squeue
#SBATCH --partition=gpu_a100_8     # which queue
#SBATCH --gres=gpu:1               # ONE gpu. this is the line that gets you a GPU.
#SBATCH --cpus-per-task=8          # CPU cores -> dataloader workers
#SBATCH --mem=64G                  # host RAM. the laptop's killer; free here.
#SBATCH --time=04:00:00            # HH:MM:SS. job is KILLED at this wall.
#SBATCH --output=logs/%x-%j.out    # %x=job-name %j=jobid
#SBATCH --error=logs/%x-%j.err
```

`#SBATCH` lines are comments to bash and directives to SLURM. They must be at the **top**, before any
real command. Everything after them is an ordinary bash script that runs on the compute node.

**`--time` is a hard kill, not a hint.** Ask for more than you need; unused time is refunded, an
overrun is a lost job. But a shorter request usually starts sooner.

## 4. Two things that will bite you

**Compute nodes usually have no internet.** So the 29 GB PASTIS download runs from the **login node**
(or a `compute` partition job), never from inside a GPU job. `01_stage_pastis.sh` handles this.

**Home directories usually have a quota**, often 20–50 GB. PASTIS is 29 GB and needs ~58 GB during
extraction. Find your scratch space before downloading:

```bash
quota -s 2>/dev/null; df -h $HOME; ls -ld /scratch/$USER /scratch/users/$USER /home/$USER/scratch 2>/dev/null
echo $SCRATCH
```

Put the dataset and all `runs/` output on **scratch**, not home. Scratch is often purged after N days
— fine for PASTIS, so **copy the result CSVs back to home** when a job finishes. The scripts do this.

## 5. Do this in order

> **Always `sbatch` from the repo root**, exactly as written below. The scripts resolve themselves
> via `$SLURM_SUBMIT_DIR`, and the `--output=REO-2/slurm/logs/...` paths are relative to it. Submitting
> from inside `REO-2/slurm/` will fail to find the repo.


```bash
# --- on the login node ---
ssh <you>@<cluster>
git clone <your repo>  &&  cd multi_temporal_jepa       # or rsync it up
bash REO-2/slurm/00_env_setup.sh                        # builds the venv, ~5 min
bash REO-2/slurm/02_pick_storage.sh                     # quota check -> where to put the data
bash REO-2/slurm/01_stage_pastis.sh $SCR                # 29 GB download, 1-3 h. tmux this.

# --- then submit, in this order ---
sbatch REO-2/slurm/10_fit_batch.sbatch                  # 10 min: find the batch size for an A100
#   read its log, set BATCH/ACCUM in the scripts below (must multiply to 192)
sbatch REO-2/slurm/20_p0_floors.sbatch                  # P0. non-negotiable. probe-only, fast.
sbatch REO-2/slurm/30_p1_seeds.sbatch                   # P1. array of 5 seeds, runs concurrently.
sbatch REO-2/slurm/40_noreg.sbatch                      # Fig 1's missing half.
sbatch REO-2/slurm/50_baselines.sbatch                  # MAE/BYOL/SimCLR at 5 seeds.
```

Use `tmux` (or `screen`) for anything long on the login node, so an SSH drop does not kill it:
`tmux new -s pastis` … detach with `Ctrl-b d` … return with `tmux attach -t pastis`.

## 6. Job arrays — how five seeds run at once

```bash
#SBATCH --array=0-4
```

submits **five** near-identical jobs. Each gets `$SLURM_ARRAY_TASK_ID` = 0,1,2,3,4, which we pass
straight through as `--seed`. SLURM schedules them on whatever GPUs free up. This is exactly the
multi-seed problem, and it is why P1 is now cheap.

Add `%2` (`--array=0-4%2`) to cap how many run at once — polite if the cluster is busy.

## 7. Watching a job

```bash
squeue -u $USER                      # queued/running. ST: PD=pending R=running CG=finishing
squeue -u $USER --start              # estimated start time for pending jobs
tail -f logs/p1-seeds-12345_0.out    # live output
scancel 12345                        # kill a job
scancel 12345_3                      # kill one array task
scancel -u $USER                     # kill everything of mine
sacct -j 12345 --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
seff 12345                           # efficiency report AFTER it finishes — did I over-ask?
```

`ST=PD` with `REASON=(Resources)` means "waiting for a free GPU" — normal. `(QOSMaxJobsPerUser)` or
`(AssocGrpGpuLimit)` means you hit a per-user cap; submit fewer at once.

**Read `seff` after your first job.** It tells you the memory and time you actually used, so the next
`--time` and `--mem` are informed rather than guessed.

## 8. Interactive debugging

When a job fails instantly and the log is unhelpful, get a shell on a GPU node and run it by hand:

```bash
srun --partition=gpu_a100_8 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
     --time=01:00:00 --pty bash
# now you are ON the compute node, with a GPU:
nvidia-smi
source .venv/bin/activate
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Do this **once**, early, before submitting the real jobs. Ten minutes here saves a day of failed
submissions.

## 9. Project-specific rules

- **Always `--device cuda`, never `cuda:1`.** `--gres=gpu:1` makes SLURM set `CUDA_VISIBLE_DEVICES`
  so your one allocated GPU is *always* device 0. Hardcoding an index targets someone else's GPU or
  a nonexistent one. `utils/device.resolve_device` already does the right thing with plain `cuda`.
- **Effective batch must stay 192.** `batch_size * grad_accum = 192` matches the committed baselines.
  Change the per-step batch freely for the bigger card; change `grad_accum` to compensate. Break this
  and your new numbers are not comparable to the old ones.
- **Migrate the CSV header once** before appending anything:
  `python scripts/migrate_matrix_csv.py runs/matrix_results.csv`
- **`num_workers`** is hardcoded to `8` at `scripts/run_matrix.py:205`. On the cluster that is fine
  and even low — set `--cpus-per-task=8` to match. (It was a problem only on the 15 GB laptop.)
- **Run one seed per job.** Never two training processes on one GPU. Sharing a card is what killed
  seeds 1 and 2 on the old server.
