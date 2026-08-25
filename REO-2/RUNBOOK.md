# Runbook — copy-paste, in order

`$CLUSTER` = your login host, `$USER` = your cluster username. Deadline **Sep 2, 2026 AoE**.

---

## 0 · Get the code onto the cluster (laptop)

The repo is 3 MB without `.venv/`, `runs/` and `data_root/` (all gitignored).

```bash
cd /home/atharv/Desktop/projects/multi_temporal_jepa
git add REO-2 && git commit -m "REO-2 workshop submission workspace" && git push
```

Then on the cluster:

```bash
ssh $USER@$CLUSTER
git clone https://github.com/atharvgaur1845/multi_temporal_jepa.git
cd multi_temporal_jepa
```

**Or skip git entirely** and push it straight over:

```bash
# from the laptop
rsync -avz --exclude .venv --exclude runs --exclude data_root --exclude .git \
      /home/atharv/Desktop/projects/multi_temporal_jepa/ \
      $USER@$CLUSTER:~/multi_temporal_jepa/
```

> That GitHub URL is your real name. It must not appear in the paper — `anonymous.4open.science`
> or no link at all.

## 1 · Environment (login node, ~5 min)

```bash
cd ~/multi_temporal_jepa
module avail python 2>&1 | head -20        # if the system python3 is < 3.10, load a newer one
bash REO-2/slurm/00_env_setup.sh
source .venv/bin/activate
pytest -q                                  # 102 passed, 3 skipped — offline, no GPU, no data
```

## 2 · Decide where the data lives (login node)

```bash
cd ~/multi_temporal_jepa
bash REO-2/slurm/02_pick_storage.sh
```

It prints your quota, every writable candidate with its free space **and its filesystem**, and tells
you which to use. Then export what it recommends:

```bash
export SCR=<what the script printed>
mkdir -p $SCR && df -h $SCR | tail -1
```

### Why not just put it in the repo directory?

You can — the scripts support both, it is one variable. The default leans to scratch for four
reasons, in order of how likely each is to bite:

1. **Home quotas.** Most HPC homes have a hard quota, often 20-50 GB. PASTIS peaks at **58 GB**
   (29 GB zip + 29 GB extracted, briefly both on disk). Hitting a quota mid-`unzip` leaves a
   half-extracted dataset and a job that fails hours later. `df` shows the *filesystem*, not your
   personal limit -- read the quota block, not just the free space.
2. **Home is usually the wrong filesystem.** Home is typically NFS, tuned for many small files.
   Scratch is typically Lustre/GPFS, tuned for throughput. Training reads ~20-40 MB per sample,
   randomly, ~1,400 samples/epoch, 100 epochs -- and `--array=0-4` means **five jobs doing it at
   once**. On NFS home that is slower for you and degrades home access for everyone else.
3. **Checkpoints outgrow the dataset.** Each encoder is 43.1M params ~ **172 MB**; ~40 of them
   across the seed sweep ~ 7 GB, on top of the dataset.
4. Home is backed up; a re-downloadable public dataset should not consume backup.

**Use the repo directory instead** if your quota allows ~75 GB. It is simpler -- `PASTIS_ROOT` then
resolves to `./data_root/PASTIS`, the config's own default, and `runs/` is a real directory rather
than a symlink:

```bash
export SCR=$HOME/multi_temporal_jepa
```

**The one real cost of scratch:** it is usually **purged** after 30-90 days without access, and is
not backed up. Whichever you pick, copy the result CSVs back to `$HOME` when jobs finish -- the
sbatch scripts do this, and step 12 pulls them to your laptop. Checkpoints can be regenerated;
the CSVs are the paper.

## 3 · Stage PASTIS (login node, 1–3 h — use tmux)

Compute nodes usually have no internet, so this cannot be an sbatch job.

```bash
tmux new -s pastis
cd ~/multi_temporal_jepa
bash REO-2/slurm/01_stage_pastis.sh $SCR
# detach: Ctrl-b then d      reattach: tmux attach -t pastis
```

When it finishes:

```bash
ls $SCR/data_root/PASTIS/metadata.geojson && du -sh $SCR/data_root/PASTIS
```

## 4 · Point the scripts at it

```bash
sed -i "s|^SCRATCH_ROOT=.*|SCRATCH_ROOT=\${SCRATCH_ROOT:-$SCR}|" REO-2/slurm/_common.sh
grep -n "SCRATCH_ROOT=\|^BATCH=\|^ACCUM=\|^PARTITION" REO-2/slurm/_common.sh
```

## 5 · Prove a GPU works before submitting anything (10 min)

```bash
srun --partition=gpu_a100_8 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
     --time=00:15:00 --pty bash

# now on the compute node:
nvidia-smi
cd ~/multi_temporal_jepa && source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
exit
```

If that prints an A100, everything below will work. If it doesn't, fix it here — not in a queue.

## 6 · Batch size for an 80 GB card

```bash
cd ~/multi_temporal_jepa                    # ALWAYS sbatch from the repo root
sbatch REO-2/slurm/10_fit_batch.sbatch
squeue -u $USER
cat REO-2/slurm/logs/fit-batch-*.out        # read the recommendation
```

Set what it says — **the two must multiply to 192**:

```bash
sed -i 's/^BATCH=.*/BATCH=${BATCH:-96}/;  s/^ACCUM=.*/ACCUM=${ACCUM:-2}/' REO-2/slurm/_common.sh
```

## 7 · P0 — the floors. Non-negotiable.

```bash
sbatch REO-2/slurm/20_p0_floors.sbatch
squeue -u $USER
tail -f REO-2/slurm/logs/p0-floors-*.out
```

When it lands, this is the pre-registered test — **report it whichever way it goes**:

```bash
column -s, -t < runs/matrix_results.csv
python scripts/audit_claims.py
```

`tjepa_h1` must beat **both** `random` and `raw_features`. If it doesn't, the headline claim
retracts to "the architecture helps" and the paper leads with that.

## 8 · P1 — five seeds, concurrently

```bash
sbatch REO-2/slurm/30_p1_seeds.sbatch       # --array=0-4 -> five jobs at once
squeue -u $USER                             # five rows: p1-seeds_0 .. _4
```

```bash
ls -la runs/matrix_results__s*.csv
python scripts/aggregate.py                 # means, sds, paired tests vs tjepa_h1
python scripts/aggregate.py --metric miou_linear    # and the strict linear head
```

## 9 · Figure 1's missing half, and the baselines

```bash
sbatch REO-2/slurm/40_noreg.sbatch          # VICReg-off collapse curve
sbatch REO-2/slurm/50_baselines.sbatch      # MAE/BYOL/SimCLR at n=5 (up to 1 day)
```

## 10 · Regenerate what was stdout-only

Few-shot and month-decoding write no files — tee them or they are lost again:

```bash
srun --partition=gpu_a100_8 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
  bash -c 'source .venv/bin/activate && \
    python scripts/evaluate.py --encoder-ckpt runs/matrix/tjepa_h1.pt --head conv --fewshot --knn \
      --config configs/model/tjepa.yaml --data $SCR/pastis_cluster.yaml 2>&1 | tee fs_temporal.log && \
    python scripts/evaluate.py --encoder-ckpt runs/matrix/spatial_jepa.pt --head conv --fewshot --knn \
      --config configs/model/tjepa.yaml --data $SCR/pastis_cluster.yaml 2>&1 | tee fs_spatial.log && \
    python scripts/mechanistic.py \
      --encoder-ckpt runs/matrix/tjepa_h1.pt runs/matrix/spatial_jepa.pt \
      --config configs/model/tjepa.yaml --data $SCR/pastis_cluster.yaml 2>&1 | tee mechanistic.log'
```

## 11 · Build the figures

```bash
python REO-2/figures/fig1_effective_rank.py \
    --logs REO-2/slurm/logs/noreg-fig1-*.out REO-2/slurm/logs/p1-seeds-*_0.out

python REO-2/figures/fig2_label_efficiency.py \
    --temporal fs_temporal.log --spatial fs_spatial.log \
    --full-temporal <conv mIoU from the CSV> --full-spatial <same>
```

Both refuse to draw if the data isn't there. That is deliberate — it is the guard against shipping
a number with nothing behind it.

## 12 · Bring the results home (from the laptop)

```bash
rsync -avz $USER@$CLUSTER:~/multi_temporal_jepa/runs/matrix_results*.csv \
           /home/atharv/Desktop/projects/multi_temporal_jepa/REO-2/evidence/
rsync -avz $USER@$CLUSTER:~/multi_temporal_jepa/REO-2/paper/figures/ \
           /home/atharv/Desktop/projects/multi_temporal_jepa/REO-2/paper/figures/
```

Then fill the paper from committed CSVs only:

```bash
grep -n 'NUM{' REO-2/paper/main.tex     # 19 placeholders; must reach zero
```

---

## Monitoring cheat-sheet

```bash
squeue -u $USER                  # PD = pending, R = running, CG = finishing
squeue -u $USER --start          # when will my pending job start
tail -f REO-2/slurm/logs/<name>-<jobid>.out
scancel <jobid>                  # kill one job
scancel <jobid>_3                # kill one array task
scancel -u $USER                 # kill everything
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
seff <jobid>                     # after it finishes: did I over-ask for time/memory?
```

## If a job fails

```bash
cat REO-2/slurm/logs/<name>-<jobid>.err
```

| symptom | cause | fix |
|---|---|---|
| `FATAL: PASTIS not found` | step 3 incomplete or `SCRATCH_ROOT` wrong | check `$SCR/data_root/PASTIS/metadata.geojson` |
| `FATAL: BATCH * ACCUM != 192` | step 6 half-done | the guard is intentional; fix `_common.sh` |
| `CUDA out of memory` | batch too big for the card you got | lower `BATCH`, raise `ACCUM`, keep the product at 192 |
| job vanishes at the wall-clock | `--time` too short | raise `#SBATCH --time`, resubmit; `--resume` skips finished cells |
| `PD ... (QOSMaxJobsPerUser)` | per-user job cap | `--array=0-4%2` to run two at a time |
| `command not found: python` | venv not active | `_common.sh` sources it; check `module load` in step 1 |
