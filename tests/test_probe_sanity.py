"""Probe sanity (M3). Skipped until data exists (set PASTIS_ROOT).

These guard that the PROBE itself is meaningful before you trust any number from it:
    - random-init frozen encoder -> mIoU well below the supervised ceiling (head alone can't
      do the work). We use a loose upper bound here; the strong claim (supervised ~ U-TAE 63)
      is checked separately once you have a supervised encoder.
"""
import os

import pytest
from torch.utils.data import DataLoader

from data.pastis_dataset import PASTIS, collate_variable_length
from eval.linear_probe import linear_probe_segmentation
from models.jepa import SITSEncoder

pytestmark = pytest.mark.skipif(
    not os.environ.get("PASTIS_ROOT"),
    reason="set PASTIS_ROOT and provide encoders to run probe sanity",
)


def test_random_encoder_near_chance():
    """A random-init frozen encoder probed with a linear head should score far below U-TAE (63)."""
    root = os.environ["PASTIS_ROOT"]
    tr = PASTIS(root, folds=[1], return_label=True)
    val = PASTIS(root, folds=[2], return_label=True)
    tl = DataLoader(tr, batch_size=4, shuffle=True, collate_fn=collate_variable_length)
    vl = DataLoader(val, batch_size=4, collate_fn=collate_variable_length)
    enc = SITSEncoder(embed_dim=128, depth=2, temporal_depth=2)
    res = linear_probe_segmentation(enc, tl, vl, num_classes=19, ignore_index=0, epochs=2)
    assert res["miou"] < 0.45, f"random encoder probe too high ({res['miou']:.3f}) — probe leaks?"
