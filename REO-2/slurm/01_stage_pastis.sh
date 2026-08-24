#!/bin/bash
# RUN ON THE LOGIN NODE, inside tmux. Compute nodes usually have NO INTERNET,
# so the download cannot happen inside a GPU job.
#
#   tmux new -s pastis
#   bash REO-2/slurm/01_stage_pastis.sh /scratch/$USER
#   Ctrl-b d          (detach; reattach with: tmux attach -t pastis)
#
# 28.76 GB download + ~29 GB extracted = ~58 GB peak. Takes 1-3 h.
set -euo pipefail
DEST="${1:-/scratch/$USER}/data_root"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "[stage] destination: $DEST"
mkdir -p "$DEST"
echo "[stage] free space here:"; df -h "$DEST" | tail -1
echo "[stage] you need ~58 GB free. Ctrl-C now if that line says less."
sleep 5

# The repo's own downloader: resumable wget, size check, md5 check, unzip.
bash scripts/download_pastis.sh "$DEST"

# The zip is 29 GB of dead weight the moment extraction succeeds.
if [[ -f "$DEST/PASTIS.zip" && -f "$DEST/PASTIS/metadata.geojson" ]]; then
  echo "[stage] extraction verified; removing the 29 GB zip"
  rm -f "$DEST/PASTIS.zip"
fi

echo "[stage] done. PASTIS at $DEST/PASTIS"
echo "[stage] set SCRATCH_ROOT in REO-2/slurm/_common.sh to: $(dirname "$DEST")"
du -sh "$DEST/PASTIS"
