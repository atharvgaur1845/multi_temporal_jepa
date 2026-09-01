# Sourced by every job script. EDIT THE TOP BLOCK ONCE, then never again.
# ---------------------------------------------------------------------
PARTITION_DEFAULT=gpu_a100_8

# Per-step batch and grad accumulation. THEY MUST MULTIPLY TO 192 --
# that is the effective batch every committed baseline used. Run
# 10_fit_batch.sbatch first and set these from its recommendation.
# 12 x 16 = 192. This MUST match the multi-seed cells (tjepa_h1, spatial_jepa, floors),
# not merely the effective batch: SimCLR's negative count and the variance-covariance
# term are both per-micro-batch, so a different per-step batch is a different experiment.
BATCH=${BATCH:-12}
ACCUM=${ACCUM:-16}

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

# Activate whichever environment this machine has. On the cluster this is a CLONE of an
# existing conda env, living on scratch so it never touches the 40 GB HOME quota, and the
# user's own envs are left unmodified. sbatch runs a non-interactive shell, so the conda
# profile hook must be sourced first or `conda activate` silently does nothing.
CONDA_ENV="${CONDA_ENV:-$SCRATCH_ROOT/reo2/envs/tjepa}"
if [[ -x "$CONDA_ENV/bin/python" ]]; then
  # conda's activate.d hooks reference unset vars (e.g. MKL_INTERFACE_LAYER), which
  # `set -u` turns into a fatal error and kills the job before training starts.
  # Disable nounset across activation only, then restore it.
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  set -u
  echo "[env] conda: $CONDA_ENV  python=$(command -v python)"
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
  echo "[env] venv: $PWD/.venv"
else
  echo "FATAL: no environment found (looked for $CONDA_ENV and .venv)" >&2; exit 1
fi

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
