"""Few-shot evaluation: train the probe head on 1% / 5% / 10% of labels.

Self-supervised pretraining should shine in the low-label regime — this experiment is where
temporal vs spatial JEPA differences are often most visible. Uses data/splits.fewshot_subset
for deterministic, stratified subsets.
"""
from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from data.pastis_dataset import collate_variable_length
from data.splits import fewshot_subset
from .linear_probe import linear_probe_segmentation


def fewshot_eval(encoder, full_train, val_loader, fractions=(0.01, 0.05, 0.10), seed=0,
                 num_classes=20, ignore_index=19, epochs=20, batch_size=16, use_temporal=True):
    """For each fraction: build a stratified labeled subset, run the (frozen-encoder) probe,
    record mIoU. Returns {fraction: {"miou": ...}}.
    """
    results = {}
    for frac in fractions:
        idx = fewshot_subset(full_train, frac, seed)
        subset = Subset(full_train, idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_variable_length)
        res = linear_probe_segmentation(encoder, loader, val_loader, num_classes=num_classes,
                                        ignore_index=ignore_index, epochs=epochs,
                                        use_temporal=use_temporal)
        results[frac] = {"miou": res["miou"], "n_samples": len(idx)}
    return results
