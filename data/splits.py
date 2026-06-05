"""Official 5-fold CV splits + few-shot label samplers.

PASTIS ships an official 5-fold assignment — USE IT (don't invent random splits, or your
numbers won't be comparable to U-TAE and the literature).
"""
from __future__ import annotations


def fold_indices(root, folds):
    """Return the list of patch ids assigned to `folds` per the official PASTIS metadata.
    TODO: read the fold table and filter."""
    raise NotImplementedError("M0")


def fewshot_subset(dataset, fraction, seed):
    """Deterministically sample `fraction` of labeled samples for few-shot eval.

    Math/spec
        Stratify by crop class if possible so rare classes aren't dropped at 1%.
        Same seed -> same subset (reproducibility).

    Returns: a Subset or list of indices. TODO: implement stratified sampling.
    """
    raise NotImplementedError("M3")
