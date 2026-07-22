"""Alignment testbed — the generator must decouple predictability from task-relevance.

These guard the EXPERIMENTAL DESIGN, not just the code: if the observation's predictability moved
with alpha, the sweep would confound the very thing it exists to separate, so that invariance is
asserted here as a first-class property.
"""
import numpy as np
import pytest

from data.synthetic_dynamics import generate_aligned
from eval.predictability import alignment_index, spectral_predictability, past_future_mi


def test_shapes_and_meta():
    g = generate_aligned(alpha=0.5, T=1200, W=16, obs_dim=6, seed=0)
    M = g["data"].shape[0]
    assert g["data"].shape == (M, 16, 6, 1)
    assert g["label"].shape == (M,)
    assert g["meta"]["alpha"] == 0.5
    assert np.isfinite(g["data"]).all() and np.isfinite(g["label"]).all()


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        generate_aligned(alpha=1.5, T=400)


def test_slow_block_is_predictable_and_fast_block_is_not():
    """The two latent blocks must actually sit at opposite ends of the predictability axis."""
    g = generate_aligned(alpha=1.0, T=4000, seed=0)
    assert spectral_predictability(g["z_slow"]) > 3 * spectral_predictability(g["z_fast"])
    assert past_future_mi(g["z_slow"]) > past_future_mi(g["z_fast"])


def test_input_predictability_is_INVARIANT_to_alpha():
    """THE design property: alpha moves the LABEL only, never the observation.

    x is rendered from both latent blocks regardless of alpha, so any downstream change across
    alpha isolates alignment with predictability held fixed.
    """
    oms, mis = [], []
    for a in (0.0, 0.5, 1.0):
        g = generate_aligned(alpha=a, T=4000, seed=0)
        oms.append(spectral_predictability(g["x_full"]))
        mis.append(past_future_mi(g["x_full"]))
    assert max(oms) - min(oms) < 1e-9, f"Omega moved with alpha: {oms}"
    assert max(mis) - min(mis) < 1e-9, f"past->future MI moved with alpha: {mis}"


def test_observation_identical_across_alpha_but_label_differs():
    g0 = generate_aligned(alpha=0.0, T=1200, seed=0)
    g1 = generate_aligned(alpha=1.0, T=1200, seed=0)
    np.testing.assert_allclose(g0["data"], g1["data"])                 # same inputs
    assert np.corrcoef(g0["label"], g1["label"])[0, 1] < 0.5           # different targets


def test_alignment_index_tracks_alpha():
    """The proposed label-aware index must rise with alpha while Omega stays flat."""
    idx = [alignment_index(g["data"], g["label"])
           for g in (generate_aligned(alpha=a, T=4000, seed=0) for a in (0.0, 0.5, 1.0))]
    assert 0.0 <= min(idx) and max(idx) <= 1.0
    assert idx[0] < idx[2], f"alignment_index did not increase with alpha: {idx}"


def test_alignment_index_rejects_degenerate_windows():
    with pytest.raises(ValueError):
        alignment_index(np.zeros((10, 1, 4)), np.zeros(10))             # W=1: no past to predict from
