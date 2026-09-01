"""Band normalization, temporal sampling, and augmentations.

Normalization is per-band (each of the 10 Sentinel-2 bands has its own scale).
Augmentations are only needed for the BYOL/SimCLR baselines (two-view invariance);
JEPA itself does not require photometric augmentation.
"""
from __future__ import annotations

import torch


def compute_band_stats(dataset, max_samples=None):
    """Compute per-band mean/std over the TRAIN folds only.

    Why train-only: using val/test statistics leaks information and inflates results.

    Streaming two-pass-free estimator: accumulate sum and sum-of-squares per band over all
    pixels and timesteps, then mean = S1/N, std = sqrt(S2/N - mean^2). `max_samples` caps how
    many patches to read (the stats converge quickly; full pass is fine but slower).

    Returns: mean (C,), std (C,)  as float tensors.
    """
    n = 0
    s1 = None
    s2 = None
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in range(limit):
        data, _, _ = dataset[i]              # (T, C, H, W)
        x = data.permute(1, 0, 2, 3).reshape(data.shape[1], -1)  # (C, T*H*W)
        if s1 is None:
            s1 = x.sum(dim=1).double()
            s2 = (x.double() ** 2).sum(dim=1)
        else:
            s1 += x.sum(dim=1).double()
            s2 += (x.double() ** 2).sum(dim=1)
        n += x.shape[1]
    mean = (s1 / n)
    var = (s2 / n) - mean ** 2
    std = var.clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def normalize_bands(data, mean, std):
    """(T,C,H,W) -> normalized (T,C,H,W) using per-band mean/std (broadcast over T,H,W)."""
    mean = torch.as_tensor(mean, dtype=data.dtype, device=data.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=data.dtype, device=data.device).view(1, -1, 1, 1)
    return (data - mean) / std.clamp_min(1e-6)


def temporal_subsample_indices(T, max_len, train=False, generator=None):
    """The index selection used by `temporal_subsample`, factored out so callers that hold a
    memory-mapped array can pick frames BEFORE materializing them (see PASTIS.__getitem__).

    Consumes the RNG exactly as `temporal_subsample` does, so both paths agree given the same
    seed state. Returns a LongTensor of sorted indices, length min(T, max_len).
    """
    if T <= max_len:
        return torch.arange(T)
    if train:
        perm = torch.randperm(T, generator=generator)[:max_len]
        return torch.sort(perm).values
    return torch.linspace(0, T - 1, steps=max_len).round().long()


def temporal_subsample(data, dates, max_len, train=False, generator=None):
    """Optionally subsample/truncate a long series to `max_len` acquisitions.

    Deterministic for eval (evenly spaced indices), optionally random for pretraining
    augmentation (random sorted subset). Keeps `data` and `dates` aligned.

    Returns: (data', dates') with T' = min(T, max_len).
    """
    T = data.shape[0]
    if T <= max_len:
        return data, dates
    if train:
        perm = torch.randperm(T, generator=generator)[:max_len]
        idx = torch.sort(perm).values
    else:
        idx = torch.linspace(0, T - 1, steps=max_len).round().long()
    return data[idx], dates[idx]


def two_view_augment(data, generator=None):
    """Produce two augmented views of a series for BYOL/SimCLR (M4).

    Ops (label-irrelevant): random horizontal/vertical flip, random temporal subsample to ~75%,
    mild per-band multiplicative jitter. Returns (view1, view2), each (T', C, H, W).
    """
    def _one(x):
        T = x.shape[0]
        # random temporal crop to ~75%
        k = max(1, int(round(0.75 * T)))
        keep = torch.sort(torch.randperm(T, generator=generator)[:k]).values
        x = x[keep]
        # spatial flips
        if torch.rand(1, generator=generator).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        if torch.rand(1, generator=generator).item() < 0.5:
            x = torch.flip(x, dims=[-2])
        # band jitter
        jitter = 1.0 + 0.1 * (torch.rand(x.shape[1], generator=generator) - 0.5)
        x = x * jitter.view(1, -1, 1, 1)
        return x
    return _one(data), _one(data)
