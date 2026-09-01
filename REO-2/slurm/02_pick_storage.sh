#!/bin/bash
# RUN ON THE LOGIN NODE, before staging PASTIS.
#   bash REO-2/slurm/02_pick_storage.sh
#
# Decides where the 29 GB dataset and the ~10 GB of checkpoints should live.
# Needs ~58 GB free at peak (29 GB zip + 29 GB extracted, briefly both on disk).
NEED_GB=58
MARGIN_GB=15   # checkpoints: ~172 MB per encoder, ~40 across the seed sweep

say () { printf '%s\n' "$*"; }
free_gb () { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
# Filesystem identity, so a "scratch" path that is really just a folder inside
# $HOME is not offered as an alternative to $HOME.
fsid () { df --output=source "$1" 2>/dev/null | tail -1 | tr -d ' '; }

say "=== quota ==="
# CRITICAL: df reports the FILESYSTEM, not your personal limit. A Lustre scratch
# can show 152 TB free and still stop you at 40 GB. Always read the quota, not df.
got_quota=0
if command -v lfs >/dev/null; then
  for m in "$HOME" /scratch "/scratch/$USER" ${SCRATCH:+"$SCRATCH"}; do
    [[ -d "$m" ]] || continue
    out=$(lfs quota -h -u "$USER" "$m" 2>/dev/null) && [[ -n "$out" ]] && {
      say "--- lfs quota on $m ---"; say "$out"; got_quota=1; }
  done
fi
command -v quota >/dev/null && { quota -s 2>/dev/null && got_quota=1; }
if [[ "$got_quota" -eq 0 ]]; then
  say "  !! COULD NOT READ ANY QUOTA."
  say "  !! Do NOT trust the free-space numbers below. Ask your admin for your"
  say "  !! scratch and home quotas before downloading 29 GB."
fi
say ""

HOME_FREE=$(free_gb "$HOME")
HOME_FS=$(fsid "$HOME")
say "=== candidate locations ==="
CANDIDATES=("$HOME" ${SCRATCH:+"$SCRATCH"} "/scratch/$USER" "/scratch/users/$USER" \
            "/lustre/$USER" "/work/$USER" "/data/$USER" "$HOME/scratch")
BEST=""; BEST_FREE=0
for d in "${CANDIDATES[@]}"; do
  [[ -z "$d" ]] && continue
  parent="$d"; [[ -d "$parent" ]] || parent="$(dirname "$d")"
  [[ -d "$parent" ]] || continue
  f=$(free_gb "$parent"); [[ -z "$f" ]] && continue
  w="no "; [[ -w "$parent" ]] && w="yes"
  fs=$(fsid "$parent"); same=""
  [[ "$fs" == "$HOME_FS" && "$d" != "$HOME" ]] && same="   <- same filesystem as \$HOME, not an alternative"
  printf '  %-28s free %5s GB   writable %s   [%s]%s\n' "$d" "$f" "$w" "$fs" "$same"
  if [[ "$w" == "yes" && "$f" -gt "$BEST_FREE" && "$d" != "$HOME" && "$fs" != "$HOME_FS" ]]; then
    BEST="$d"; BEST_FREE="$f"
  fi
done
say ""

say "=== recommendation ==="
if [[ -n "$HOME_FREE" && "$HOME_FREE" -ge $((NEED_GB + MARGIN_GB)) ]]; then
  say "  \$HOME has ${HOME_FREE} GB free — over the ${NEED_GB} GB peak plus ${MARGIN_GB} GB of checkpoints."
  say "  Simplest option: keep everything in the repo directory."
  say ""
  say "    export SCR=\$HOME/multi_temporal_jepa"
  say ""
  say "  CHECK THE QUOTA BLOCK ABOVE FIRST. 'df' shows the filesystem, not your"
  say "  personal limit — a quota well under ${NEED_GB} GB overrides this advice."
elif [[ -n "$BEST" ]]; then
  say "  \$HOME has ${HOME_FREE:-?} GB free — under the ${NEED_GB}+${MARGIN_GB} GB needed."
  say "  Use scratch:"
  say ""
  say "    export SCR=$BEST"
  say ""
  say "  Scratch is usually PURGED after 30-90 days and is not backed up."
  say "  Copy result CSVs back to \$HOME when jobs finish (the sbatch scripts do)."
else
  say "  Could not find a writable location with ${NEED_GB} GB free."
  say "  Ask your cluster admin / docs where project or scratch space lives."
  exit 1
fi
say ""
say "=== then wire it in ==="
say "  sed -i \"s|^SCRATCH_ROOT=.*|SCRATCH_ROOT=\\\${SCRATCH_ROOT:-\$SCR}|\" REO-2/slurm/_common.sh"
say "  grep -n 'SCRATCH_ROOT=' REO-2/slurm/_common.sh"
