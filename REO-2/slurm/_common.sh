# Sourced by every job script. EDIT THE TOP BLOCK ONCE, then never again.
# ---------------------------------------------------------------------
PARTITION_DEFAULT=gpu_a100_8

# Per-step batch and grad accumulation. THEY MUST MULTIPLY TO 192 --
# that is the effective batch every committed baseline used. Run
# 10_fit_batch.sbatch first and set these from its recommendation.
BATCH=${BATCH:-96}
ACCUM=${ACCUM:-2}

# Where the dataset and the run outputs live. Scratch, not home -- PASTIS is
# 29 GB and needs ~58 GB while extracting.
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
PASTIS_ROOT=${PASTIS_ROOT:-$SCRATCH_ROOT/data_root/PASTIS}

# Module names differ per cluster. `module avail cuda` on the login node to
# see yours; leave empty if the venv's bundled CUDA is enough (usually it is).
MODULES=${MODULES:-}
# ---------------------------------------------------------------------

set -euo pipefail

# Repo root = two levels up from this file, regardless of where you submitted from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "$MODULES" ]]; then module load $MODULES; fi
source .venv/bin/activate

if (( BATCH * ACCUM != 192 )); then
  echo "FATAL: BATCH($BATCH) * ACCUM($ACCUM) = $((BATCH*ACCUM)), not 192." >&2
  echo "       The effective batch must match the committed baselines or the new" >&2
  echo "       numbers are not comparable to the old ones. Fix _common.sh." >&2
  exit 1
fi

# `runs/` must land on scratch (checkpoints are large) but the code writes to a
# relative path, so symlink it once.
mkdir -p "$SCRATCH_ROOT/runs"
if [[ ! -e runs ]]; then ln -s "$SCRATCH_ROOT/runs" runs; fi

# Point the data config at wherever PASTIS actually is, without editing the
# committed config (keeps the repo clean and the job reproducible).
DATA_CFG="$SCRATCH_ROOT/pastis_cluster.yaml"
sed "s|^root:.*|root: $PASTIS_ROOT|" configs/data/pastis.yaml > "$DATA_CFG"

if [[ ! -f "$PASTIS_ROOT/metadata.geojson" ]]; then
  echo "FATAL: PASTIS not found at $PASTIS_ROOT" >&2
  echo "       Run  bash REO-2/slurm/01_stage_pastis.sh  on the LOGIN node first." >&2
  exit 1
fi

echo "=============================================================="
echo " job     : ${SLURM_JOB_NAME:-interactive}  id=${SLURM_JOB_ID:-none}"
echo " node    : $(hostname)"
echo " repo    : $REPO_ROOT"
echo " data    : $PASTIS_ROOT"
echo " batch   : $BATCH x accum $ACCUM = $((BATCH*ACCUM)) effective"
echo " gpu     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo " started : $(date -Is)"
echo "=============================================================="

# Build a config with the cluster batch size, again without touching the repo copy.
mk_cfg () {   # mk_cfg <src-yaml> <dst-yaml>
  sed -e "s/^\( *\)batch_size:.*/\1batch_size: $BATCH/" \
      -e "s/^\( *\)grad_accum:.*/\1grad_accum: $ACCUM/" "$1" > "$2"
}
