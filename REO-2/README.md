# REO-2 — workshop submission workspace

**Venue:** 2nd Workshop on Advances in Representation Learning for Earth Observation (REO), NeurIPS 2026
**Portal:** CMT3 — `cmt3.research.microsoft.com/REO2026` (**not** OpenReview)
**Length:** 4 pages, references and appendix excluded
**Review:** double-blind

This directory is self-contained. Nothing outside it is edited; source artifacts are *copied* into
`evidence/` so the submission has a frozen provenance trail.

**Deadline: Sep 2, 2026 AoE.**

Read `STATUS.md` first (what the data can defend), then **`slurm/README.md`** (how to actually run
it — you have cluster access, so that is the plan; `COMPUTE.md` is the laptop fallback).

`paper/main.tex` is your dropped template with **every unrun number stripped out** and replaced by a
red `\NUM{...}` placeholder naming the run that produces it. Nothing in it is assumed from the
drafting values. Gate before upload: `grep -n 'NUM{' paper/main.tex` must return nothing.

---

## Scope: PASTIS only

The parent repo is a three-modality investigation plus a methodology paper. A 4-page EO workshop
paper is roughly 2,500 words. **~95% of `README.md` is cut.**

**Gone entirely:** finance (Phase 2), C-MAPSS (Phase 3), the V1–V5 validity criteria, the 7-claim
self-audit table, the alignment testbed and `alignment_index`, Koopman / Neural-ODE / LKF structured
predictors, the graph-JEPA negative, `report_full.md`, and the pre-registration as a contribution.

**The one exception:** a tight **Limitations** paragraph naming — the floors, `n`, the val-vs-test
protocol, the SimCLR negative-count caveat, and the missing temporal-SSL baseline. It is not sold as
a contribution. It is there so the paper reads as visibly rigorous at a venue run by the dataset's
own author. That is the correct dose of the honesty instinct at this length.

## Runs, in strict priority order

| | run | why it is where it is | est. |
|---|---|---|---|
| **P0** | `random` + `raw_features` floors on PASTIS | **Non-negotiable.** Pre-registration rule P1 retracts the headline claim to "the architecture helps" if `tjepa_h1` does not clear both. Submitting a paper whose central claim your own protocol marks unassessable, to this reviewer pool, is not an option. | ~1 day |
| **P1** | Re-run the seed CSVs | Seeds 1 and 2 died at step 0 (OOM, shared card). The `±1.8` and `p=0.041` are currently unsubstantiable. **Go to 5 seeds if compute allows** — Wilcoxon is dead at n=3 (min two-sided p = 0.25) and p=0.041 is one unlucky seed from nothing. | ~2 days |
| **P2** | A temporal-SSL baseline (frame-shuffle / temporal-order pretext, same encoder) | Without it, "temporal prediction ≻ spatial" is indistinguishable from "uses the time axis ≻ doesn't." Sharpest attack a reviewer has. | ~0.5 day |
| **P3** | Fine-tuning cell | Stretch. | — |

**If only two land, they are P0 and P1.**

Commands and acceptance criteria: `protocol/P0_floors.md` … `protocol/P3_finetune.md`.

## Figures — 2 maximum at this length

| slot | content | data status |
|---|---|---|
| **Fig 1** | Effective-rank curve, VICReg **on vs off**. The hook. Make it the money figure. | ⚠️ **on-curve exists** (`evidence/logs/`); **off-curve was never run on PASTIS** — see `protocol/P0_floors.md` §addendum |
| **Fig 2** | Label efficiency, temporal vs spatial at 1 / 5 / 10 / 100% | ⚠️ numbers are stdout-only, no CSV — regenerate |
| **Table 1** | Main comparison **with floors** | blocked on P0 + P1 |
| inline | Month-decoding (61.3 vs 46.3, chance 8.3) — **prose, not a figure** | ⚠️ stdout-only — regenerate |
| cut | Architecture diagram | — |

Builders: `figures/fig1_effective_rank.py`, `figures/fig2_label_efficiency.py`. Both refuse to plot
missing series rather than interpolate — no invented curves.

## Desk-reject mechanics

Full list with tick-boxes in `CHECKLIST.md`. The four that actually kill submissions:

1. **Make the CMT3 account today.** Different portal from the main conference.
2. **NeurIPS 2026 Overleaf template**, unmodified:
   ```latex
   \workshoptitle{2nd Workshop on Advances in Representation Learning for Earth Observation}
   \usepackage[dblblindworkshop]{neurips_2026}
   ```
   Delete `\newpage` and `\input{checklist.tex}` — the checklist is not required.
3. **Double-blind:** no names, no affiliations, no acknowledgements, no `github.com/atharvgaur1845`.
   Use `anonymous.4open.science` or omit the link. **Strip PDF metadata** — LaTeX leaks author names
   into PDF properties and that is a real desk-reject cause.
4. **Verify every arXiv ID resolves.** Under the NeurIPS LLM policy you own citation correctness, and
   a fabricated reference caught by *this* reviewer pool costs far more than the citation is worth.
   Two are already flagged as high-risk in `paper/references.bib` — read that file before compiling.

## Layout

```
REO-2/
├── README.md                this file — plan of record
├── STATUS.md                provenance audit: what the checkout can defend
├── CHECKLIST.md             desk-reject mechanics, tick-boxes
├── COMPUTE.md               measured laptop throughput — FALLBACK ONLY
├── slurm/                   SLURM guide + job scripts — THE PLAN
│   ├── README.md            SLURM from zero: partitions, sbatch, arrays, gotchas
│   ├── _common.sh           shared env; edit BATCH/ACCUM here once
│   ├── 00_env_setup.sh      login node: build the venv
│   ├── 01_stage_pastis.sh   login node: 29 GB download (compute nodes have no internet)
│   ├── 10_fit_batch.sbatch  find the A100 batch size before anything else
│   ├── 20_p0_floors.sbatch  P0 — non-negotiable
│   ├── 30_p1_seeds.sbatch   P1 — job array, 5 seeds concurrently
│   ├── 40_noreg.sbatch      Fig 1's missing VICReg-off curve
│   └── 50_baselines.sbatch  MAE/BYOL/SimCLR at 5 seeds
├── paper/
│   ├── main.tex             the dropped REO-2 template — canonical. Compiles.
│   ├── neurips_2026.sty     official style file, in place
│   ├── refs.bib             canonical bibliography, per-entry VERIFY notes
│   ├── main_ALT_skeleton.tex.bak   my earlier skeleton, superseded — kept, not used
│   ├── references_ALT.bib.bak      its bibliography, ditto
│   ├── TEMPLATE_NOTES.md    .sty provenance, metadata stripping, pre-upload greps
│   └── figures/             compiled PDFs land here
├── protocol/                P0–P3 run recipes: exact commands + acceptance criteria
├── evidence/                frozen copies: prereg, PASTIS CSV, configs, logs
└── figures/                 figure builders
```
