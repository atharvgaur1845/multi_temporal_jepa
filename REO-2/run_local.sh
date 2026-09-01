#!/bin/bash
# Laptop fallback: run the REO-2 queue on the RTX 4060, sequentially, resumable.
#
#   bash REO-2/run_local.sh              # run the whole queue
#   bash REO-2/run_local.sh p0           # just the floors
#   SEEDS="0 1 2" bash REO-2/run_local.sh
#
# Ctrl-C at any point is safe: --resume skips cells whose checkpoint already
# exists, so re-running continues instead of restarting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CFG=configs/model/tjepa_8gb.yaml     # batch 16 x accum 12 = 192 effective, ~6.7 GiB peak
DATA=configs/data/pastis.yaml        # root: ./data_root/PASTIS
SEEDS="${SEEDS:-0 1 2}"              # n=3. add "3 4" if the calendar allows.
WHAT="${1:-all}"

# 15 GB host RAM with ~3 GB free cannot feed 8 dataloader workers (8 x 2 x 335 MB
# ~ 5.4 GB). This is what OOM-killed the benchmark. 2 workers ~ 1.3 GB.
export TJEPA_WORKERS="${TJEPA_WORKERS:-2}"
# workers x prefetch batches live in host RAM at once; a PASTIS batch is ~42 MB/sample.
# 2 workers x 1 prefetch x 12 samples ~ 1.0 GB, against ~2.0 GB at the default prefetch 2.
export TJEPA_PREFETCH="${TJEPA_PREFETCH:-1}"
# Memory-map the .npy and select frames BEFORE the float32 cast: ~31 MB/sample instead of
# ~42 MB. Verified bit-identical to the default path (same indices, same RNG draw).
export TJEPA_MMAP="${TJEPA_MMAP:-1}"

# Seed-0 tjepa_h1 OOMed at step ~100 with 1.56 GiB "reserved but unallocated" --
# textbook allocator fragmentation, and the card only exposes 7.62 of 8.19 GiB
# (Xorg holds the rest). Two mitigations, both needed:
#   1. expandable_segments lets the allocator grow segments instead of stranding
#      them at fixed sizes. Recommended by the OOM message itself.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
#   2. smaller per-step batch, MORE accumulation. 12 x 16 = 192 -- the effective
#      batch is unchanged, so results stay comparable to every committed cell.
#      This is exactly what tjepa_8gb.yaml's own comment prescribes on OOM.
BATCH="${BATCH:-12}"; ACCUM="${ACCUM:-16}"
(( BATCH * ACCUM == 192 )) || { echo "[local] BATCH*ACCUM must be 192" >&2; exit 1; }

command -v nvidia-smi >/dev/null && {
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[local] GPU free: ${FREE} MiB (temporal JEPA needs ~6900)"
  if (( FREE < 6900 )); then
    echo "[local] NOT ENOUGH FREE VRAM. Something else is on the card:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
    echo "[local] Stop it first, then re-run. Aborting." >&2
    exit 1
  fi
}
[[ -f data_root/PASTIS/metadata.geojson ]] || {
  echo "[local] PASTIS not found at ./data_root/PASTIS" >&2; exit 1; }

# Write the batch override into a scratch config; never edit the committed one.
LCFG=".tjepa_local.yaml"
sed -e "s/^\( *\)batch_size:.*/\1batch_size: $BATCH/" \
    -e "s/^\( *\)grad_accum:.*/\1grad_accum: $ACCUM/" "$CFG" > "$LCFG"
CFG="$LCFG"
echo "[local] workers=$TJEPA_WORKERS  batch=$BATCH x accum=$ACCUM (=192)  seeds='$SEEDS'"
echo "[local] alloc: $PYTORCH_CUDA_ALLOC_CONF"
[[ -f runs/matrix_results.csv ]] && python scripts/migrate_matrix_csv.py runs/matrix_results.csv

# --ckpt-every 1 writes ONE rotating mid-training checkpoint per cell (atomic replace, so the
# previous is deleted, never accumulated) and --resume picks it up. A crash now costs at most
# one epoch instead of every epoch.
run () { echo; echo "=== $* ==="; date -Is; time python scripts/run_matrix.py \
           --config "$CFG" --data "$DATA" --device cuda --resume --probe-epochs 15 \
           --ckpt-every "${CKPT_EVERY:-1}" "$@"; }

if [[ "$WHAT" == "all" || "$WHAT" == "p0" ]]; then
  # P0 -- the non-negotiable floors. Neither trains; both go straight to the probe.
  run --only random,raw_features --knn
  echo "[local] P0 done. THE PRE-REGISTERED TEST:"
  column -s, -t < runs/matrix_results.csv
fi
[[ "$WHAT" == "p0" ]] && exit 0

if [[ "$WHAT" == "all" || "$WHAT" == "p1" ]]; then
  for s in $SEEDS; do
    # ~6.7 h for tjepa_h1 + ~0.4 h for spatial, per seed, measured on this card.
    run --only tjepa_h1,spatial_jepa,random,raw_features --seed "$s" --knn
  done
  python scripts/aggregate.py || true
fi
[[ "$WHAT" == "p1" ]] && exit 0

if [[ "$WHAT" == "all" || "$WHAT" == "noreg" ]]; then
  run --only tjepa_noreg          # Fig 1's missing VICReg-off curve
fi
echo; echo "[local] queue finished $(date -Is)"
