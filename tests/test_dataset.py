"""Data pipeline invariants. Skipped until PASTIS is downloaded (set PASTIS_ROOT)."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PASTIS_ROOT"),
    reason="set PASTIS_ROOT to the extracted PASTIS folder to run data tests",
)


def test_sample_shapes():
    """A sample yields (T,C,H,W) data, (T,) dates, (H,W) label with C=10, H=W=128.

    TODO: instantiate PASTIS(os.environ['PASTIS_ROOT'], folds=[1], return_label=True),
    pull one sample, assert the shapes and dtypes above and that dates are sorted/in [1,366].
    """
    raise NotImplementedError("M0")


def test_collate_padding_mask():
    """A batch of variable-length series pads to T_max with a correct boolean pad_mask.

    TODO: build a few samples of different T, run collate_variable_length, assert
    data is (B,T_max,...), pad_mask True only on real frames, and pad frames are zero.
    """
    raise NotImplementedError("M0")
