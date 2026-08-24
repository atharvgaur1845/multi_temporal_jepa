# Submission checklist

**Deadline: Sep 2, 2026 AoE.** Research track, 4 pages, double-blind, non-archival.
Compute budget and run order: `COMPUTE.md`. What the data can defend: `STATUS.md`.


## Today

- [ ] **Create the CMT3 account** — `cmt3.research.microsoft.com/REO2026`.
      Different portal from the main conference; this is not OpenReview. Do it before anything
      else, because account/registration friction is the failure that has no workaround at 23:59.
- [ ] Confirm the deadline, the timezone, and whether abstract registration precedes full submission.
- [x] ~~Pull `neurips_2026.sty`~~ — in `paper/`. `main.tex` compiles clean (3 pp, room for 4).

## Blockers before any run (see `COMPUTE.md`)

- [x] ~~Free disk~~ — now 72 GB free vs a ~58 GB peak. Thin: delete `PASTIS.zip` the instant `unzip` exits 0.
- [x] ~~Free the GPU~~ — idle, 7737 MiB free. Benchmarked: temporal JEPA fits at 6.69 GiB reserved.
- [ ] **Set `num_workers=2`** in `scripts/run_matrix.py:205` (and `engine/train_jepa.py:141`).
      At `8` the dataloader wants ~5.4 GB against **3 GB of available host RAM** and will be
      OOM-killed hours in. This already killed a benchmark run today. Not optional.
- [ ] Close Chrome / Spotify while training. Nothing else may touch the GPU.
- [ ] Re-download PASTIS (~29 GB, Zenodo 10.5281/zenodo.5012942, md5 `cfc441bf18137ff0bbf4fad58828fb98`).
- [ ] `python scripts/migrate_matrix_csv.py runs/matrix_results.csv` (8-col header -> 10-col) before
      appending any new rows.

## Runs, strict priority

- [ ] **P0** — `random` + `raw_features` floors. Non-negotiable. `protocol/P0_floors.md`
- [ ] **P1** — seed re-run, 5 seeds if compute allows. `protocol/P1_seeds.md`
- [ ] **P2** — temporal-order pretext baseline. `protocol/P2_temporal_ssl_baseline.md`
- [ ] **P3** — fine-tuning cell. Stretch. `protocol/P3_finetune.md`
- [ ] P0-addendum — `tjepa_noreg` on PASTIS, or Fig 1 cannot be drawn.

**If only two land, they are P0 and P1.**

## The paper currently states numbers that have not been run

`paper/main.tex` (the dropped template) hardcodes `22.3 ± 1.8`, `p=0.041`, `3 seeds`,
`15.8 ± 1.2`, `9.2/13.1/15.9` vs `4.6/6.9/9.5`, `61.3%` vs `46.3%`, and
`effective rank 2.4 of 512`. **None of these has an artifact in this checkout** (`STATUS.md`).
The abstract asserts all of them.

- [ ] Every one of those numbers either regenerated from a committed CSV, or removed.
- [ ] `grep -n "TBD" paper/main.tex` returns nothing.
- [ ] The `n` actually reached is what the paper says — not `3` because the template said `3`.

## Content

- [ ] 4 pages, references and appendix excluded. No margin or font games.
- [ ] Exactly 2 figures + 1 table. No architecture diagram. Month-decoding is inline prose.
- [ ] Every number traceable to a committed CSV.
- [ ] `grep -n PENDING paper/main.tex` returns nothing.
- [ ] Table 1 states `n` **per row**.
- [ ] Limitations paragraph names, in order: the floors, `n`, the val-vs-test protocol, the SimCLR
      negative-count caveat, and the missing temporal-SSL baseline if P2 did not land.
      Not sold as a contribution.
- [ ] Everything from the parent README that is out of scope is actually gone: finance, C-MAPSS,
      V1–V5, the self-audit table, the alignment testbed, Koopman/ODE/LKF, the graph negative,
      the monograph, the pre-registration.

## Citations

- [ ] Every arXiv ID **resolves** and the returned **title matches**. Resolving is not enough.
- [ ] `2601.14354` and `2603.20111` resolved or the claim deleted — see the `[DANGER]` block at the
      bottom of `paper/references.bib`. These are the two a reviewer is most likely to check.
- [ ] The PASTIS citation is exactly right — authors, venue, year, DOI. The workshop is run by its
      author.

## Double-blind and mechanics

- [ ] No names, no affiliations, no acknowledgements.
- [ ] No `github.com/atharvgaur1845` anywhere. Anonymous link or none.
- [x] PDF metadata blanked at the LaTeX level via `\hypersetup{...}` in `main.tex`;
      `pdfinfo main.pdf` verified — Title/Author/Creator/Producer all empty.
      **`exiftool` is not installed on this machine**, so that fallback is unavailable;
      re-run `pdfinfo` after the final build rather than trusting the earlier check.
- [ ] `\newpage` and `\input{checklist.tex}` deleted from the template.
- [ ] `\usepackage[dblblindworkshop]{neurips_2026}` and the `\workshoptitle` line present and exact.
- [ ] Source bundle (if required) scrubbed of identifying comments.
