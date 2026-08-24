#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=REO-2/slurm/logs/%x-%A_%a.out
#SBATCH --error=REO-2/slurm/logs/%x-%A_%a.err
#
# MAE / BYOL / SimCLR at 5 seeds.
#
# On the laptop these were going to stay at n=1 (BYOL alone was ~30-100 h there),
# which would have forced an "unequal n" line into Limitations. On an A100 they are
# affordable, so run them properly and delete that limitation.
#
# BYOL and SimCLR are the expensive cells (9.93 and 7.81 GPU-h on the old server
# vs 2.12 for temporal), hence --time=1-00:00:00 (one day) rather than 8 h.
#
# Note the SimCLR caveat that stays in Limitations regardless: grad accumulation
# equalizes the OPTIMIZATION batch but not the NT-Xent negative count, so SimCLR
# is a same-budget baseline, not a tuned one. Say so; do not quietly drop it.
source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

SEED=$SLURM_ARRAY_TASK_ID
mk_cfg configs/model/tjepa.yaml "$SCRATCH_ROOT/tjepa_base_s${SEED}.yaml"

python scripts/run_matrix.py \
    --config "$SCRATCH_ROOT/tjepa_base_s${SEED}.yaml" \
    --data   "$DATA_CFG" \
    --only   mae,byol,simclr \
    --seed   "$SEED" \
    --device cuda \
    --knn \
    --probe-epochs 15 \
    --resume

echo "[baselines] seed $SEED done"
