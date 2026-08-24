#!/bin/bash
#SBATCH --job-name=fit-batch
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=REO-2/slurm/logs/%x-%j.out
#SBATCH --error=REO-2/slurm/logs/%x-%j.err
#
# RUN THIS FIRST. The laptop measurement (batch 16 -> 6.7 GiB) says nothing about
# what fits on an 80 GB A100. scripts/fit_batch.py sweeps batch sizes until OOM and
# recommends batch_size/grad_accum for a target effective batch of 192.
#
#   sbatch REO-2/slurm/10_fit_batch.sbatch
#   ... then put its recommendation into BATCH/ACCUM in _common.sh
#
# BATCH/ACCUM are unset here on purpose -- this job is what determines them.
BATCH=192 ACCUM=1 source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

python scripts/fit_batch.py \
    --config configs/model/tjepa.yaml \
    --device cuda \
    --eff-batch 192 \
    --margin-gb 8 \
    --batches 32,48,64,96,128,192

echo "[fit_batch] Set BATCH and ACCUM in REO-2/slurm/_common.sh from the line above."
echo "[fit_batch] They MUST multiply to 192. _common.sh refuses to run otherwise."
