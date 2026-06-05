#!/usr/bin/env bash
# Download + extract PASTIS (~29 GB). Zenodo DOI 10.5281/zenodo.5012942
# PASTIS-R (adds Sentinel-1 SAR, ~54 GB): DOI 10.5281/zenodo.5735646
#
# TODO (M0):
#   - fill in the exact Zenodo file URL(s) for the PASTIS archive,
#   - download (curl/wget), VERIFY the file size / md5,
#   - extract into $DEST, then point configs/data/pastis.yaml:root at it.
#   - after extraction, confirm the Sentinel-2 BAND ORDER from the official dataloader
#     and record it in pastis.yaml (the one fact the research could not fully verify).
set -euo pipefail

DEST="${1:-./data_root/PASTIS}"
mkdir -p "$DEST"

echo "TODO: implement PASTIS download into $DEST"
echo "Zenodo record: https://zenodo.org/records/5012942"
exit 1
