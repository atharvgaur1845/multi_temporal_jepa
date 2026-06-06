#!/usr/bin/env bash
# Download + extract PASTIS (~28.76 GB). Zenodo DOI 10.5281/zenodo.5012942
# PASTIS-R (adds Sentinel-1 SAR, ~54 GB): DOI 10.5281/zenodo.5735646
#
# Usage:  bash scripts/download_pastis.sh [DEST]
#   DEST defaults to ./data_root  (the archive extracts a top-level PASTIS/ folder)
# After extraction, point configs/data/pastis.yaml:root at "$DEST/PASTIS".
#
# Sentinel-2 band order (confirmed from VSainteuf/utae-paps PASTIS_Dataset):
#   [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]  -> 10 bands, this is the order in DATA_S2/*.npy.
set -euo pipefail

DEST="${1:-./data_root}"
URL="https://zenodo.org/records/5012942/files/PASTIS.zip?download=1"
ZIP="$DEST/PASTIS.zip"
EXPECTED_MD5="cfc441bf18137ff0bbf4fad58828fb98"
EXPECTED_BYTES=28760245504

mkdir -p "$DEST"

echo "[download_pastis] Downloading PASTIS.zip (~28.76 GB) into $DEST ..."
# -c resumes a partial download if the script is re-run.
wget -c -O "$ZIP" "$URL"

echo "[download_pastis] Verifying size ..."
ACTUAL_BYTES=$(stat -c%s "$ZIP")
if [[ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ]]; then
  echo "[download_pastis] SIZE MISMATCH: got $ACTUAL_BYTES, expected $EXPECTED_BYTES" >&2
  echo "[download_pastis] Re-run to resume the download." >&2
  exit 1
fi

echo "[download_pastis] Verifying md5 (reads ~29 GB, takes a while) ..."
ACTUAL_MD5=$(md5sum "$ZIP" | awk '{print $1}')
if [[ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]]; then
  echo "[download_pastis] MD5 MISMATCH: got $ACTUAL_MD5, expected $EXPECTED_MD5" >&2
  exit 1
fi
echo "[download_pastis] md5 OK."

echo "[download_pastis] Extracting ..."
unzip -q -o "$ZIP" -d "$DEST"

echo "[download_pastis] Done. PASTIS extracted under $DEST/PASTIS"
echo "[download_pastis] Next:  export PASTIS_ROOT=$DEST/PASTIS"
echo "[download_pastis] And point configs/data/pastis.yaml:root at it."
