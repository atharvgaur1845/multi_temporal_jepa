"""Few-shot evaluation: train the probe head on 1% / 5% / 10% of labels.

Self-supervised pretraining should shine in the low-label regime — this experiment is where
temporal vs spatial JEPA differences are often most visible. Use data/splits.fewshot_subset for
deterministic, stratified subsets.
"""
from __future__ import annotations


def fewshot_eval(encoder, full_train, val_loader, fractions=(0.01, 0.05, 0.10), seed=0):
    """For each fraction: build a stratified labeled subset, run the (frozen-encoder) probe,
    record metrics. Returns {fraction: metrics}. TODO: implement (reuse linear_probe) (M3)."""
    raise NotImplementedError("M3")
