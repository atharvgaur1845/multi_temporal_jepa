# P3 — fine-tuning cell (stretch)

**Cut this before cutting anything else.** It lands only if P0, P1 and P2 are all done.

## Why it is wanted

Every PASTIS number in the project is a **frozen probe** (linear + conv head on a frozen encoder).
That is the right protocol for measuring representation quality, and it is what the paper claims. But
the practitioner question at an EO venue is "does pretraining on this help my downstream model" —
which is a fine-tuning question. Supervised U-TAE at 63.1 mIoU sits far above every frozen probe here;
without a fine-tuned cell the paper cannot say anything about that gap.

## Why it is last

- It is a **new evaluation path** — full backbone unfreezing, its own LR schedule, its own overfitting
  behavior on 3 training folds. Not a config flip.
- It introduces a fresh comparison-hygiene surface (V4): a fine-tuned temporal encoder vs a frozen
  baseline is not a comparison, so **every** cell in the table would need fine-tuning to match.
- At 4 pages there is no room to report both protocols properly.

## If it lands

- Fine-tune `tjepa_h1` **and** `spatial_jepa` and `random` — a fine-tuned learned encoder against a
  fine-tuned random init is the only version of this that means anything.
- Same fold, same schedule, same budget, seeds matched to P1.
- Report as a separate small table or two inline numbers. It does **not** get a figure.

## If it does not land

One clause in Limitations: "all results are frozen-probe; we do not evaluate fine-tuning."
That is complete and costs nothing. Do not gesture at unrun fine-tuning results.
