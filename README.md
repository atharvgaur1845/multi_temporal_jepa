# Temporal JEPA for Satellite Image Time Series

Self-supervised representation learning for satellite image time series (SITS) by **latent prediction**:
predict the *future* latent state of a location from its *past* observations — a causal, world-model-flavored
JEPA objective — and compare it against *spatial* JEPA (I-JEPA-style masked-region prediction) and
reconstruction/contrastive baselines (MAE, BYOL, SimCLR) on **PASTIS** (Sentinel-2 crop time series).

> **Research question.** Can a *temporal* JEPA objective learn more useful representations for remote
> sensing than *spatial* JEPA, under equal compute?

## ⚠️ This is a mentorship scaffold, not a finished repo

The core algorithms are intentionally **left unimplemented** (`raise NotImplementedError` + `TODO`).
You implement them; docstrings spell out the math so you build from understanding. The reference
implementation is shown only *after* you submit your attempt. See the design doc/plan for the contract.

## Milestones (implement in order)

| M | Goal | Hard gate |
|---|------|-----------|
| M0 | Data pipeline (PASTIS loader, variable-length collate, DOY, splits) | `tests/test_dataset.py` green |
| M1 | JEPA core as **Spatial JEPA** + **correctness harness** | overfit-8 → loss↓ AND collapse diagnostics healthy |
| M2 | **Temporal JEPA** (causal past→future, horizon Δ) | trains without collapse; causal-structure sanity |
| M3 | Eval harness (linear probe→mIoU, k-NN, few-shot, t-SNE/UMAP) | probe sanity (supervised≈U-TAE, random≈chance) |
| M4 | Baselines (MAE, BYOL, SimCLR) | each trains stably |
| M5 | Experiments + ablations (full matrix, GPU-hours logged) | reproducible from config+seed |
| M6 | Write-up (lit review, TDD, figures, repro checklist) | — |

## Quickstart

```bash
pip install -r requirements.txt
bash scripts/download_pastis.sh                 # ~29 GB
pytest tests/                                   # start by making these pass (TDD)
python scripts/overfit8_smoketest.py            # M1 gate
```

## Layout

See the directory tree in the design doc. Every module has a docstring stating its responsibility,
the math, and the `TODO`s you must fill in.
