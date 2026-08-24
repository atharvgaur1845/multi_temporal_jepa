# P2 — a temporal-SSL baseline

**The single sharpest attack a reviewer has.** Estimated half a day of coding; it does not exist yet.

## The hole

Every baseline in the matrix is **spatial**: `spatial_jepa`, `mae`, `byol`, `simclr` all consume a
single frame (`use_temporal = False` in `scripts/run_matrix.py`). Confirmed: `objectives/baselines/`
contains only `mae.py`, `byol.py`, `simclr.py`.

So the comparison actually run is:

> temporal-JEPA (uses the time axis) vs four methods that **do not use the time axis at all**

which cannot separate

- **the claim:** future-latent *prediction* beats spatial pretext, from
- **the confound:** *any* use of the time axis beats not using it.

A reviewer at an EO workshop — where multi-temporal is the whole point — will ask this first.

## The fix

A temporal pretext task on the **same encoder** (`models/temporal_encoder.py`, same ViT backbone, same
`use_temporal=True` probe path), so the only thing that changes is the objective. Pick one:

- **Temporal-order verification** — sample a frame pair/triplet, binary-classify whether it is in
  chronological order. Cheap, standard, and the natural "uses time, does not predict latents" control.
- **Frame shuffle detection** — shuffle a fraction of the sequence, classify shuffled vs not.

Order verification is the better choice: it has a fixed 2-way head, no masking machinery, and a
trivially interpretable chance level (50%).

Watch the confound in the other direction: PASTIS uses **DOY positional encoding**
(`configs/data/pastis.yaml: date_encoding: doy`). If the encoder can read the timestamp off the
positional embedding, order verification is solvable without learning anything about the imagery —
the same circularity `scripts/mechanistic.py` already dodges by probing `encode_full` rather than the
temporal pathway. **Strip or randomize DOY inside the pretext head**, and say so in the paper.

## Sketch

1. `objectives/baselines/temporal_order.py` — a `TRAINERS`-compatible entry
   (`engine/train_baselines.TRAINERS`), signature `(loader, cfg, device) -> encoder`, matching the
   existing three.
2. Register `temporal_order` in `TRAINERS`.
3. Add a cell in `scripts/run_matrix.py: enumerate_cells()`:
   `cells.append(("temporal_order", {"objective": "temporal_order"}))`
4. Route it through the `use_temporal = True` probe branch, not the spatial `else` branch — otherwise
   it is probed differently from `tjepa_h1` and V4 (comparison hygiene) is violated, which is the exact
   split/protocol mismatch the project already caught once.
5. Match the compute budget to `spatial_jepa` and log gpu-h like every other cell.

```bash
python scripts/run_matrix.py --config configs/model/tjepa_8gb.yaml \
    --data configs/data/pastis.yaml --only temporal_order --resume
```

## Acceptance

- One new row in the matrix CSV, same val fold, same probe budget, `use_temporal=True`.
- Table 1 gains a row that is unambiguously "uses the time axis, does not predict future latents."
- **If `temporal_order` lands near `tjepa_h1`, that is the finding**, and the paper's claim narrows
  from "future-latent prediction wins" to "using the temporal axis wins." Write it that way.

If P2 does not land, it goes in Limitations by name — "we do not compare against a non-predictive
temporal pretext, so we cannot separate the objective from the use of the time axis." Naming it costs
one sentence and defuses the objection; omitting it invites the reviewer to make it for you.
