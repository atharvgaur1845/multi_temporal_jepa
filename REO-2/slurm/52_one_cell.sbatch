#!/bin/bash
#SBATCH --job-name=c2cell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/pabitra/reo2/logs/%x-%A_%a.out
#SBATCH --error=/scratch/pabitra/reo2/logs/%x-%A_%a.err
#
# One cell, one seed per array task. CELL comes in via --export so the same file
# can be aimed at whichever partition currently has free GPUs: the A100 node is
# fully allocated with a queue of other users ahead of us, and reservation
# maint_sept1 takes every node at midnight for 365 days, so anything that does not
# start in the next few hours never runs.
SEED=$SLURM_ARRAY_TASK_ID
: "${CELL:?CELL must be passed via --export}"

source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

echo "[c2] cell=$CELL seed=$SEED batch=$BATCH accum=$ACCUM part=$SLURM_JOB_PARTITION"
mk_cfg configs/model/tjepa.yaml "$SCRATCH_ROOT/reo2/cfg_${CELL}_s${SEED}.yaml"

python scripts/run_matrix.py \
    --config "$SCRATCH_ROOT/reo2/cfg_${CELL}_s${SEED}.yaml" \
    --data   "$DATA_CFG" \
    --only   "$CELL" \
    --seed   "$SEED" \
    --device cuda --knn --probe-epochs 15 --resume
echo "[c2] done cell=$CELL seed=$SEED"
