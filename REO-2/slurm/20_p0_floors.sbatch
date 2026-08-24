#!/bin/bash
#SBATCH --job-name=p0-floors
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=REO-2/slurm/logs/%x-%j.out
#SBATCH --error=REO-2/slurm/logs/%x-%j.err
#
# P0 -- THE NON-NEGOTIABLE RUN.
#
# Pre-registration rule P1 (../evidence/PREREGISTRATION.md) retracts the headline
# claim to "the architecture helps" unless tjepa_h1 clears BOTH of these floors.
# Neither has ever been run. Until they are, the central claim is unassessable by
# our own protocol -- at a venue run by the dataset's author.
#
#   random       : identical architecture, weights NEVER updated. Isolates the
#                  objective from the temporal-attention architecture itself.
#   raw_features : patch-mean raw bands, no encoder. The true no-representation floor.
#
# NEITHER TRAINS (verified, scripts/run_matrix.py:208-218) -- both go straight to
# the frozen probe. 2 h is generous.
source "${SLURM_SUBMIT_DIR:-$PWD}/REO-2/slurm/_common.sh"

# One-time: the committed CSV has the old 8-column header; the driver writes 10.
# Appending without this misaligns every new row under csv.DictReader.
if [[ -f runs/matrix_results.csv ]]; then
  python scripts/migrate_matrix_csv.py runs/matrix_results.csv
fi

mk_cfg configs/model/tjepa.yaml "$SCRATCH_ROOT/tjepa_cluster.yaml"

python scripts/run_matrix.py \
    --config "$SCRATCH_ROOT/tjepa_cluster.yaml" \
    --data   "$DATA_CFG" \
    --only   random,raw_features \
    --device cuda \
    --knn \
    --probe-epochs 15 \
    --resume

echo "[p0] floors written to runs/matrix_results.csv"
echo "[p0] REPORT THE OUTCOME EITHER WAY -- the pre-registration commits to this in advance."
cp runs/matrix_results.csv "$HOME/matrix_results_p0_$(date +%F).csv" 2>/dev/null || true
