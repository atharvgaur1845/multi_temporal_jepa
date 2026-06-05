"""Band normalization, temporal sampling, and augmentations.

Normalization is per-band (each of the 10 Sentinel-2 bands has its own scale).
Augmentations are only needed for the BYOL/SimCLR baselines (two-view invariance);
JEPA itself does not require photometric augmentation.
"""
from __future__ import annotations


def compute_band_stats(dataset):
    """Compute per-band mean/std over the TRAIN folds only.

    Why train-only: using val/test statistics leaks information and inflates results.
    Returns: mean (C,), std (C,).

    TODO: stream over the dataset accumulating sum / sumsq per band (Welford or two-pass).
    """
    raise NotImplementedError("M0")


def normalize_bands(data, mean, std):
    """(T,C,H,W) -> normalized (T,C,H,W) using per-band mean/std (broadcast over T,H,W).
    TODO: implement; guard against std==0."""
    raise NotImplementedError("M0")


def temporal_subsample(data, dates, max_len):
    """Optionally subsample/truncate a long series to `max_len` acquisitions.

    Keep it deterministic for eval, optionally random for pretraining augmentation.
    Must keep `data` and `dates` aligned. TODO: implement.
    """
    raise NotImplementedError("M0")


def two_view_augment(data):
    """Produce two augmented views of a series for BYOL/SimCLR (M4).

    Typical ops for SITS: random temporal crop, random spatial crop/flip, mild
    band jitter. Keep label-irrelevant. TODO: implement (M4, not needed for JEPA).
    """
    raise NotImplementedError("M4")
