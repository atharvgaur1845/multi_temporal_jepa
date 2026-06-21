"""Fetch the NASA C-MAPSS turbofan degradation dataset -> data_root/CMAPSS/.

The industrial analogue of scripts/download_pastis.sh / download_finance.py. Three sources, tried in
this order:

  1. --zip /path/to/CMAPSSData.zip   : extract a locally-provided NASA zip (offline, authoritative).
  2. mirror download                 : pull the 12 plain-text files (train/test/RUL x FD001-FD004)
                                       from a public GitHub mirror with retry/backoff.
  3. (neither)                       : do nothing; data/cmapss_dataset.py then synthesizes a
                                       documented monotonic-degradation panel so the pipeline + tests
                                       run reproducibly offline.

Each FDxxx file is whitespace-delimited: columns = engine_id, cycle, op_setting_1..3, sensor_1..21.
RUL_FDxxx.txt = one integer per TEST engine (true RUL at its last available cycle).
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request
import zipfile

MIRROR = ("https://raw.githubusercontent.com/hankroark/"
          "Turbofan-Engine-Degradation/master/CMAPSSData")
FDS = ["FD001", "FD002", "FD003", "FD004"]
KINDS = ["train", "test", "RUL"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def _fetch(url, dest, retries=4, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            return len(data)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 3.0 * (2 ** attempt)
            print(f"  retry {attempt+1}/{retries} after {wait:.0f}s ({type(e).__name__})")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}")


def from_zip(zip_path, root):
    """Extract the C-MAPSS txt files out of a NASA zip (handles a nested folder layout)."""
    os.makedirs(root, exist_ok=True)
    want = {f"{k}_{fd}.txt" for fd in FDS for k in KINDS}
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            base = os.path.basename(member)
            if base in want:
                with z.open(member) as src, open(os.path.join(root, base), "wb") as dst:
                    dst.write(src.read())
                n += 1
    print(f"[download_cmapss] extracted {n} files from {zip_path} -> {root}")
    return n


def from_mirror(root):
    os.makedirs(root, exist_ok=True)
    n = 0
    for fd in FDS:
        for k in KINDS:
            name = f"{k}_{fd}.txt"
            dest = os.path.join(root, name)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                print(f"  {name}: cached"); n += 1; continue
            size = _fetch(f"{MIRROR}/{name}", dest)
            print(f"  {name}: {size} bytes"); n += 1
            time.sleep(0.5)
    print(f"[download_cmapss] mirror -> {n} files in {root}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CMAPSS_ROOT", "data_root/CMAPSS"))
    ap.add_argument("--zip", default=None, help="path to a locally-downloaded CMAPSSData.zip")
    args = ap.parse_args()
    if args.zip:
        from_zip(args.zip, args.root)
    else:
        print("[download_cmapss] no --zip given; pulling plain-text files from the mirror ...")
        from_mirror(args.root)
    # sanity print
    for fd in FDS:
        p = os.path.join(args.root, f"train_{fd}.txt")
        if os.path.isfile(p):
            with open(p) as f:
                rows = sum(1 for _ in f)
            print(f"  train_{fd}: {rows} rows")


if __name__ == "__main__":
    main()
