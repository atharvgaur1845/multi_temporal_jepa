"""Official 5-fold CV splits + few-shot label samplers.

PASTIS ships an official 5-fold assignment — USE IT (don't invent random splits, or your
numbers won't be comparable to U-TAE and the literature).
"""
from __future__ import annotations

import json
import os

import numpy as np


def cv_split(fold):
    """PASTIS 5-fold CV rotation. `fold` in 1..5 selects which fold is the held-out TEST set.

    Returns (train_folds, val_folds, test_folds): test = {fold}, val = the next fold (wraps),
    train = the remaining three. Averaging metrics over fold=1..5 gives the 5-fold CV result and
    the per-fold spread for error bars. (The single-split default in pastis.yaml — train[1,2,3]
    val[4] test[5] — is one such rotation.)
    """
    assert 1 <= fold <= 5, "fold must be in 1..5"
    allf = [1, 2, 3, 4, 5]
    test = [fold]
    val = [fold % 5 + 1]
    train = [f for f in allf if f not in test + val]
    return train, val, test


def fold_indices(root, folds):
    """Return the list of ID_PATCH assigned to `folds` per the official PASTIS metadata."""
    with open(os.path.join(str(root), "metadata.geojson"), "r") as f:
        meta = json.load(f)
    folds = set(int(x) for x in folds)
    ids = [int(feat["properties"]["ID_PATCH"]) for feat in meta["features"]
           if int(feat["properties"]["Fold"]) in folds]
    return sorted(ids)


def fewshot_subset(dataset, fraction, seed):
    """Deterministically sample `fraction` of labeled samples for few-shot eval.

    Stratify by the patch's DOMINANT crop class so rare classes aren't dropped at 1%: bucket
    each sample by its most-frequent non-background label, then take ceil(fraction * n) from
    each bucket (a fixed RNG keyed by `seed` makes the subset reproducible).

    Returns: a list of dataset indices.
    """
    rng = np.random.default_rng(seed)
    # bucket samples by dominant class
    buckets = {}
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if label is None:
            buckets.setdefault(-1, []).append(i)
            continue
        vals, counts = np.unique(label.numpy(), return_counts=True)
        # ignore background (0) when picking the dominant class, unless that's all there is
        order = np.argsort(-counts)
        dom = 0
        for j in order:
            if vals[j] != 0:
                dom = int(vals[j])
                break
        buckets.setdefault(dom, []).append(i)

    chosen = []
    for cls, idxs in buckets.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        k = max(1, int(np.ceil(fraction * len(idxs))))
        chosen.extend(idxs[:k].tolist())
    return sorted(chosen)
