"""Probe sanity (M3). Skipped until data + a built encoder exist.

These guard that the PROBE itself is meaningful before you trust any number from it:
    - random-init frozen encoder -> mIoU near chance (head can't be doing the work alone).
    - supervised-trained encoder -> mIoU approaches the U-TAE ballpark (~63 mIoU).
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PASTIS_ROOT"),
    reason="set PASTIS_ROOT and provide encoders to run probe sanity",
)


def test_random_encoder_near_chance():
    """TODO (M3): linear-probe a random-init frozen encoder; assert mIoU << supervised."""
    raise NotImplementedError("M3")
