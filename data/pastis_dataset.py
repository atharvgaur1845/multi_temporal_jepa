"""PASTIS dataset + variable-length collate.

Responsibility
--------------
Turn the on-disk PASTIS patches into batched tensors the models can consume, while
respecting two awkward facts about satellite image time series (SITS):

1. Sequences are VARIABLE LENGTH (38..61 acquisitions) and must be padded into a batch
   together with a boolean padding mask so temporal attention can ignore pad frames.
2. Acquisitions are IRREGULARLY SPACED in time. We therefore carry the day-of-year (DOY)
   of each acquisition so the model can build a *date-based* temporal positional encoding
   (see models/pos_embed.py). Integer frame index is NOT a substitute.

Tensor shapes (per sample)
    data   : (T, C, H, W)   float   C=10 bands, H=W=128
    dates  : (T,)           long    day-of-year in [1, 366]
    label  : (H, W)         long    per-pixel crop class in [0, 18]  (0 = background/void)

Batched (after collate, max length T_max)
    data    : (B, T_max, C, H, W)
    dates   : (B, T_max)
    pad_mask: (B, T_max)   bool   True where the frame is real, False where padded
    label   : (B, H, W)

On-disk PASTIS layout (Zenodo 5012942, mirrors VSainteuf/utae-paps)
    root/metadata.geojson                  # per-patch: ID_PATCH, Fold, dates-S2 (idx->YYYYMMDD)
    root/DATA_S2/S2_<ID_PATCH>.npy         # (T, 10, 128, 128) float
    root/ANNOTATIONS/TARGET_<ID_PATCH>.npy # (3, 128, 128) int; channel 0 = semantic class
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _yyyymmdd_to_doy(d: int) -> int:
    """Convert an integer date YYYYMMDD to its calendar day-of-year in [1, 366]."""
    d = int(d)
    year, month, day = d // 10000, (d // 100) % 100, d % 100
    return _dt.date(year, month, day).timetuple().tm_yday


def _parse_dates_field(val) -> list:
    """PASTIS stores dates-S2 as {idx_str: YYYYMMDD}. Across versions it may be a dict or a
    JSON string. Normalize to a list of YYYYMMDD ints in acquisition order."""
    if isinstance(val, str):
        val = json.loads(val)
    items = sorted(((int(k), int(v)) for k, v in val.items()), key=lambda kv: kv[0])
    return [v for _, v in items]


class PASTIS(Dataset):
    """PASTIS satellite image time series dataset.

    Parameters
    ----------
    root : str            path to the extracted PASTIS folder
    folds : list[int]     which of the 5 official folds to include
    norm_mean, norm_std : per-band normalization stats (compute from TRAIN folds only).
                          If None, raw reflectance is returned (used when computing the stats).
    return_label : bool   pretraining can skip labels; eval needs them
    sem_channel : int     which channel of TARGET_*.npy holds the semantic class (PASTIS = 0)

    DOY convention: calendar day-of-year in [1, 366] (NOT days-since-start), used consistently
    by models/pos_embed.doy_sincos_pos_embed.
    """

    def __init__(self, root, folds, norm_mean=None, norm_std=None, return_label=False,
                 sem_channel=0, max_seq_len=None, subsample_train=False):
        super().__init__()
        self.root = str(root)
        self.folds = list(folds)
        self.return_label = return_label
        self.sem_channel = sem_channel
        # Cap the number of acquisitions per series to bound memory (B*T frames go through the
        # spatial ViT). subsample_train=True => random subset (pretraining aug); else evenly
        # spaced (deterministic, for eval). None => keep all frames.
        self.max_seq_len = max_seq_len
        self.subsample_train = subsample_train
        self.norm_mean = None if norm_mean is None else torch.as_tensor(norm_mean).float()
        self.norm_std = None if norm_std is None else torch.as_tensor(norm_std).float()

        meta_path = os.path.join(self.root, "metadata.geojson")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"metadata.geojson not found under {self.root!r}. Run scripts/download_pastis.sh "
                "and point root at the extracted PASTIS folder."
            )
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # self.samples = list of (id_patch, [YYYYMMDD,...]) for patches in the requested folds.
        self.samples = []
        for feat in meta["features"]:
            props = feat["properties"]
            if int(props["Fold"]) not in self.folds:
                continue
            pid = int(props["ID_PATCH"])
            dates = _parse_dates_field(props["dates-S2"])
            self.samples.append((pid, dates))
        self.samples.sort(key=lambda s: s[0])
        if not self.samples:
            raise RuntimeError(f"No patches for folds {self.folds} under {self.root!r}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, raw_dates = self.samples[idx]
        path = os.path.join(self.root, "DATA_S2", f"S2_{pid}.npy")
        dates = torch.tensor([_yyyymmdd_to_doy(d) for d in raw_dates], dtype=torch.long)

        if os.environ.get("TJEPA_MMAP", "0") == "1":
            # Memory-lean path: memory-map the (T,10,128,128) int16 array, pick the frames we
            # will actually keep, and only THEN materialize float32. The default path below
            # loads all T frames and float32-casts all of them before discarding most, costing
            # ~42 MB per sample against ~31 MB here (T=43, max_seq_len=32).
            # Frame selection is identical -- same indices, same RNG draw, same order -- so the
            # returned tensor is bit-for-bit what the default path returns.
            s2 = np.load(path, mmap_mode="r")                       # no full read
            T = s2.shape[0]
            if self.max_seq_len is not None and T > self.max_seq_len:
                from .transforms import temporal_subsample_indices
                idx_t = temporal_subsample_indices(T, self.max_seq_len, train=self.subsample_train)
                s2 = np.ascontiguousarray(s2[idx_t.numpy()])
                dates = dates[idx_t]
            else:
                s2 = np.ascontiguousarray(s2)
            data = torch.from_numpy(s2.astype(np.float32))
        else:
            s2 = np.load(path)                                       # (T, 10, 128, 128)
            data = torch.from_numpy(s2.astype(np.float32))
            if self.max_seq_len is not None and data.shape[0] > self.max_seq_len:
                from .transforms import temporal_subsample
                data, dates = temporal_subsample(data, dates, self.max_seq_len,
                                                 train=self.subsample_train)

        if self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean.view(1, -1, 1, 1)
            std = self.norm_std.view(1, -1, 1, 1).clamp_min(1e-6)
            data = (data - mean) / std

        if not self.return_label:
            return data, dates, None

        tgt = np.load(os.path.join(self.root, "ANNOTATIONS", f"TARGET_{pid}.npy"))
        label = torch.from_numpy(tgt[self.sem_channel].astype(np.int64))  # (128, 128)
        return data, dates, label


def collate_variable_length(batch):
    """Pad a list of variable-length samples into a batch + build the padding mask.

    Math/spec
        T_max = max sequence length in the batch.
        Pad `data` and `dates` along the time axis up to T_max.
        pad_mask[b, t] = True iff frame t of sample b is a REAL acquisition (not padding).

    Returns
        dict(data, dates, pad_mask, label)   (label is None if samples carry no labels).

    Padding choices: data padded with 0.0, dates padded with DOY=0 (never a real DOY) so the
    temporal positional encoding + pad_mask jointly ignore those frames (Common Mistake #8).
    """
    datas, dates_list, labels = zip(*batch)
    B = len(datas)
    T_max = max(d.shape[0] for d in datas)
    C, H, W = datas[0].shape[1:]

    data = datas[0].new_zeros((B, T_max, C, H, W))
    dates = torch.zeros((B, T_max), dtype=torch.long)
    pad_mask = torch.zeros((B, T_max), dtype=torch.bool)

    for b, (x, d) in enumerate(zip(datas, dates_list)):
        t = x.shape[0]
        data[b, :t] = x
        dates[b, :t] = d
        pad_mask[b, :t] = True

    out = {"data": data, "dates": dates, "pad_mask": pad_mask}
    out["label"] = None if labels[0] is None else torch.stack(labels, dim=0)
    return out
