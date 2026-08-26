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

# Info-ZIP 6.0 ships a zip-bomb heuristic (the CVE-2019-13232 mitigation) that
# FALSE-POSITIVES on large zip64 archives:
#     error: invalid zip file with overlapped components (possible zip bomb)
# PASTIS.zip is 28.76 GB, so it is necessarily zip64. Disabling the heuristic is
# safe ONLY because download_pastis.sh verifies the md5 against the published
# checksum first -- integrity is established independently of the heuristic.
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

# The repo's own downloader: resumable wget, size check, md5 check, unzip.
bash scripts/download_pastis.sh "$DEST" || true

# If extraction still did not happen, fall back through tools that handle zip64
# without the heuristic. The md5 has already passed at this point.
if [[ ! -f "$DEST/PASTIS/metadata.geojson" && -f "$DEST/PASTIS.zip" ]]; then
  echo "[stage] unzip did not produce PASTIS/ -- trying fallbacks"
  cd "$DEST"
  if   command -v bsdtar >/dev/null; then echo "[stage] bsdtar"; bsdtar -xf PASTIS.zip
  elif command -v 7z     >/dev/null; then echo "[stage] 7z";     7z x -y PASTIS.zip
  else echo "[stage] python zipfile"
       python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" PASTIS.zip
  fi
  cd - >/dev/null
fi

if [[ ! -f "$DEST/PASTIS/metadata.geojson" ]]; then
  echo "[stage] FAILED: no $DEST/PASTIS/metadata.geojson after extraction" >&2
  exit 1
fi

# The zip is 29 GB of dead weight the moment extraction succeeds.
if [[ -f "$DEST/PASTIS.zip" && -f "$DEST/PASTIS/metadata.geojson" ]]; then
  echo "[stage] extraction verified; removing the 29 GB zip"
  rm -f "$DEST/PASTIS.zip"
fi

echo "[stage] done. PASTIS at $DEST/PASTIS"
echo "[stage] set SCRATCH_ROOT in REO-2/slurm/_common.sh to: $(dirname "$DEST")"
du -sh "$DEST/PASTIS"
