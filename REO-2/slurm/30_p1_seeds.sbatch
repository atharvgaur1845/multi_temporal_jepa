#!/bin/bash
#SBATCH --job-name=p1-seeds
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=REO-2/slurm/logs/%x-%A_%a.out
#SBATCH --error=REO-2/slurm/logs/%x-%A_%a.err
#
# P1 -- the multi-seed re-run. THIS IS A JOB ARRAY: --array=0-4 submits five
# independent jobs, each with a different $SLURM_ARRAY_TASK_ID, which we pass
# straight through as --seed. On gpu_a100_8 all five land on the node's 8 GPUs
# and run CONCURRENTLY.
#
# Why re-run rather than recover: run_s1.log and run_s2.log both die at step 0
# with CUDA OOM on a shared card. Seeds 1 and 2 produced no numbers at all, so
# the committed "22.3 +- 1.8, p=0.041, 3 seeds" has no artifact behind it.
#
# n=5, not 3: Wilcoxon cannot reach significance at n=3 (minimum two-sided
# p = 0.25). At n=5 the minimum is 0.0625.
#
#   sbatch REO-2/slurm/30_p1_seeds.sbatch
#   squeue -u $USER          # five rows: p1-seeds_0 .. p1-seeds_4
source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

SEED=$SLURM_ARRAY_TASK_ID
echo "[p1] seed = $SEED"

mk_cfg configs/model/tjepa.yaml "$SCRATCH_ROOT/tjepa_s${SEED}.yaml"

# Floors are included per-seed on purpose: rule P1 compares the margin over the
# larger floor against the ACROSS-SEED sd, so the floors need a sd too.
python scripts/run_matrix.py \
    --config "$SCRATCH_ROOT/tjepa_s${SEED}.yaml" \
    --data   "$DATA_CFG" \
    --only   tjepa_h1,spatial_jepa,spatial_jepa_matched,random,raw_features \
    --seed   "$SEED" \
    --device cuda \
    --knn \
    --probe-epochs 15 \
    --resume

echo "[p1] seed $SEED done -> runs/matrix_results__s${SEED}.csv"
