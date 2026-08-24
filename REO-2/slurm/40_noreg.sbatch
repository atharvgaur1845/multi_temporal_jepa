#!/bin/bash
#SBATCH --job-name=noreg-fig1
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=REO-2/slurm/logs/%x-%j.out
#SBATCH --error=REO-2/slurm/logs/%x-%j.err
#
# Figure 1's missing half.
#
# Fig 1 (effective rank, VICReg on vs off) is the designated money figure and the
# hook of the whole paper. The VICReg-OFF curve DOES NOT EXIST: tjepa_noreg is
# logged "SKIPPED (budget)" in run_s0.log, run_s1.log and run_s2.log, and there is
# no PASTIS noreg run in any committed CSV. The "effective rank -> 2.4" number in
# the old README is not traceable to a file.
#
# Without this job, ../figures/fig1_effective_rank.py refuses to draw (by design)
# and Fig 1 has to be cut.
#
# Collapse happens EARLY, so the curve does not need 100 epochs -- but on an A100
# the full run is cheap, so we do it properly and get the endpoint too.
source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

mk_cfg configs/model/tjepa.yaml "$SCRATCH_ROOT/tjepa_noreg.yaml"

python scripts/run_matrix.py \
    --config "$SCRATCH_ROOT/tjepa_noreg.yaml" \
    --data   "$DATA_CFG" \
    --only   tjepa_noreg \
    --device cuda \
    --probe-epochs 15 \
    --resume

echo "[noreg] now build Fig 1 from the per-step traces in this job's .out file:"
echo "  python REO-2/figures/fig1_effective_rank.py \\"
echo "      --logs REO-2/slurm/logs/noreg-fig1-${SLURM_JOB_ID}.out REO-2/slurm/logs/p1-seeds-*_0.out"
