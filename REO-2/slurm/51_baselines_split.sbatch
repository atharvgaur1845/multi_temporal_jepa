#!/bin/bash
#SBATCH --job-name=base3
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:30:00
#SBATCH --array=0-8
#SBATCH --requeue
#SBATCH --output=/scratch/pabitra/reo2/logs/%x-%A_%a.out
#SBATCH --error=/scratch/pabitra/reo2/logs/%x-%A_%a.err
#
# C2: MAE / BYOL / SimCLR at 3 seeds, at the SAME per-step batch (12x16) as the
# multi-seed cells. Removes both remaining caveats at once: n=1, and the batch
# mismatch that currently forces a "do not pool" note into Setup and Limitations.
#
# One array task per (cell, seed) rather than per seed, so the nine runs spread
# across the node's 8 GPUs instead of running three-at-a-time sequentially. BYOL
# is the long pole (~9.9 GPU-h on the old server card), so serialising it three
# times behind MAE and SimCLR would roughly triple the wall clock.
#
# TIME LIMIT IS SET BY A CLUSTER RESERVATION, NOT BY THE WORKLOAD. Reservation
# maint_sept1 takes every node at 2026-09-01T00:00 for 365 days, so nothing that
# would overlap midnight will be scheduled at all. 8h30 submitted ~14:00 fits.
# (A partition list was rejected by this scheduler, so a single partition it is.)
#
# --ckpt-every only checkpoints the JEPA branch of run_matrix.py; the
# MAE/BYOL/SimCLR trainers have no mid-training save, so a cell killed at the wall
# is lost. Per-seed CSVs are written independently, so partial completion is still
# usable: report whatever n actually landed.
CELLS=(mae byol simclr)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))

source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

echo "[c2] cell=$CELL seed=$SEED batch=$BATCH accum=$ACCUM"
mk_cfg configs/model/tjepa.yaml "$SCRATCH_ROOT/reo2/cfg_${CELL}_s${SEED}.yaml"

python scripts/run_matrix.py \
    --config "$SCRATCH_ROOT/reo2/cfg_${CELL}_s${SEED}.yaml" \
    --data   "$DATA_CFG" \
    --only   "$CELL" \
    --seed   "$SEED" \
    --device cuda \
    --knn \
    --probe-epochs 15 \
    --resume

echo "[c2] done cell=$CELL seed=$SEED"
