"""Migrate runs/matrix_results.csv from the old 8-col schema to the current 10-col one.

run_matrix.py gained `seed` and `cv_fold` columns; --resume APPENDS without rewriting the header, so
a CSV started before that change ends up with an 8-col header and 10-col rows. csv.DictReader then
misaligns (eval_split reads as the seed, miou_conv reads the literal split name) -> aggregate.py
either crashes on float('test') or silently drops the row. This rewrites the header and backfills
seed=0 / cv_fold='' on the old rows. Writes a .bak first. Idempotent.

    python scripts/migrate_matrix_csv.py runs/matrix_results.csv
"""
import csv, shutil, sys

if len(sys.argv) != 2:
    raise SystemExit(__doc__.strip().splitlines()[-1].strip())
p = sys.argv[1]
NEW = ["cell","objective","seed","cv_fold","eval_split","miou_linear","miou_conv",
       "knn_acc","gpu_hours","peak_mem_gb"]
shutil.copy(p, p + ".bak")
rows = []
with open(p) as f:
    for r in csv.reader(f):
        if not r or r[0] == "cell":
            continue                              # drop any header (old or new)
        if len(r) == 8:                           # OLD schema -> insert seed=0, cv_fold=''
            r = r[:2] + ["0", ""] + r[2:]
        if len(r) != len(NEW):
            raise SystemExit(f"unexpected row width {len(r)}: {r}")
        rows.append(r)
with open(p, "w", newline="") as f:
    w = csv.writer(f); w.writerow(NEW); w.writerows(rows)
print(f"[migrate] {len(rows)} rows -> 10-col schema (backup: {p}.bak)")
