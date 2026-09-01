#!/bin/bash
# Pull C2 results off the cluster and regenerate everything downstream.
# Run once the array finishes (or once it is clear no more will land before the
# maint_sept1 reservation takes the cluster at 2026-09-01T00:00).
#
#   bash REO-2/sync_from_cluster.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REMOTE=${REMOTE:-sharanga}
RDIR=${RDIR:-/scratch/pabitra/runs}

echo "[sync] pulling seed CSVs (results only; checkpoints stay on the cluster)"
rsync -avz --include='matrix_results*.csv' --exclude='*' "$REMOTE:$RDIR/" runs/

echo "[sync] guarding against the headerless-CSV failure that silently ate seed 1"
for f in runs/matrix_results__s*.csv; do
  [ -s "$f" ] || { echo "  !! $f is EMPTY"; continue; }
  head -1 "$f" | grep -q '^cell,' || echo "  !! $f HAS NO HEADER -- csv.DictReader will eat its first data row"
done

echo "[sync] regenerating consolidated CSVs"
./.venv/bin/python REO-2/make_results_csv.py

echo
echo "[sync] === DOES THE HEADLINE SURVIVE? ==="
./.venv/bin/python REO-2/check_headline.py

echo
echo "[sync] === NEW TABLE 1 BODY (paste into paper/main.tex) ==="
./.venv/bin/python REO-2/make_table1.py

echo
echo "[sync] regenerating Figure 1"
./.venv/bin/python REO-2/figures/fig1_floors.py

echo
echo "[sync] next: update paper/main.tex, then rebuild and re-strip metadata:"
echo "  cd REO-2/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
echo "  cp main.pdf main_prestrip.pdf.bak && python -c \"import pikepdf;p=pikepdf.open('main_prestrip.pdf.bak');[p.docinfo.__delitem__(k) for k in list(p.docinfo.keys())];p.save('main.pdf')\""
