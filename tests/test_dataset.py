"""Data pipeline invariants. Skipped until PASTIS is downloaded (set PASTIS_ROOT)."""
import os

import pytest
import torch

from data.pastis_dataset import PASTIS, collate_variable_length

pytestmark = pytest.mark.skipif(
    not os.environ.get("PASTIS_ROOT"),
    reason="set PASTIS_ROOT to the extracted PASTIS folder to run data tests",
)


def test_sample_shapes():
    """A sample yields (T,C,H,W) data, (T,) dates, (H,W) label with C=10, H=W=128."""
    ds = PASTIS(os.environ["PASTIS_ROOT"], folds=[1], return_label=True)
    data, dates, label = ds[0]
    assert data.ndim == 4 and data.shape[1] == 10 and data.shape[2:] == (128, 128)
    assert dates.shape == (data.shape[0],)
    assert dates.dtype == torch.long
    assert torch.all((dates >= 1) & (dates <= 366))
    # NOTE: dates are CALENDAR day-of-year, and PASTIS spans Sep-2018..Nov-2019, so DOY WRAPS
    # at the year boundary (e.g. 350 -> 17). Acquisitions are chronological by index (which is
    # what the temporal split uses); DOY is a periodic positional signal, not a sort key.
    assert label.shape == (128, 128) and label.dtype == torch.long
    assert int(label.min()) >= 0 and int(label.max()) <= 19  # PASTIS: 0=bg, 1..18 crops, 19=void


def test_collate_padding_mask():
    """A batch of variable-length series pads to T_max with a correct boolean pad_mask."""
    ds = PASTIS(os.environ["PASTIS_ROOT"], folds=[1], return_label=True)
    batch = [ds[i] for i in range(4)]
    out = collate_variable_length(batch)
    B = 4
    T_max = max(s[0].shape[0] for s in batch)
    assert out["data"].shape == (B, T_max, 10, 128, 128)
    assert out["pad_mask"].shape == (B, T_max)
    for b, (x, d, _) in enumerate(batch):
        t = x.shape[0]
        assert out["pad_mask"][b, :t].all()
        assert not out["pad_mask"][b, t:].any()
        assert torch.count_nonzero(out["data"][b, t:]) == 0  # padded frames are zero
