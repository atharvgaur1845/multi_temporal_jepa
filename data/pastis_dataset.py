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
    data   : (B, T_max, C, H, W)
    dates  : (B, T_max)
    pad_mask: (B, T_max)    bool    True where the frame is real, False where padded
    label  : (B, H, W)
"""
from __future__ import annotations

from torch.utils.data import Dataset


class PASTIS(Dataset):
    """PASTIS satellite image time series dataset.

    Parameters
    ----------
    root : str            path to the extracted PASTIS folder
    folds : list[int]     which of the 5 official folds to include
    norm_mean, norm_std : per-band normalization stats (compute from TRAIN folds only)
    return_label : bool   pretraining can skip labels; eval needs them

    TODO
    ----
    - Index the patches belonging to `folds` (read the official metadata, e.g. the
      geojson / fold table shipped with PASTIS).
    - In __getitem__: load the (T,C,H,W) array and the per-acquisition dates, apply
      band normalization, return (data, dates, label).
    - Decide your DOY convention and document it (calendar DOY vs days-since-start).
    """

    def __init__(self, root, folds, norm_mean=None, norm_std=None, return_label=False):
        super().__init__()
        # TODO: build self.samples (list of patch ids/paths) for the requested folds.
        raise NotImplementedError("M0: implement PASTIS indexing")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: load patch -> normalize -> (data[T,C,H,W], dates[T], label[H,W])
        raise NotImplementedError


def collate_variable_length(batch):
    """Pad a list of variable-length samples into a batch + build the padding mask.

    Math/spec
        T_max = max sequence length in the batch.
        Pad `data` and `dates` along the time axis up to T_max.
        pad_mask[b, t] = True iff frame t of sample b is a REAL acquisition (not padding).

    Returns
        dict(data, dates, pad_mask, label)

    Pitfalls
        - If you forget pad_mask, temporal attention will attend to zero-frames and
          corrupt the representation (Common Mistake #8).
        - Pad `dates` with a value the positional encoding will ignore (and that the
          pad_mask covers) — do NOT pad with a real DOY.

    TODO: implement padding + mask construction.
    """
    raise NotImplementedError("M0: implement variable-length collate")
