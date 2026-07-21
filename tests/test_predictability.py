"""Predictability metrics + synthetic-dynamics generator (Part 7 backbone).

Guards that the predictability indices ORDER the regimes correctly (periodic ≫ white; AR(1) monotone
in φ) — the whole falsification design rests on these being trustworthy — and that the generator emits
the panel shapes the JEPA stack consumes. Pure numpy/torch, fully offline.
"""
import numpy as np

from data.synthetic_dynamics import generate, DynamicsWindows, collate
from eval.predictability import (spectral_predictability, ar_forecast_r2, past_future_mi,
                                 permutation_entropy_pred, predictability_report, intrinsic_dimension)


def _z(regime, phi=0.0, T=4000, seed=0):
    return generate(regime=regime, phi=phi, T=T, seed=seed)["z_full"]


def test_spectral_omega_orders_regimes():
    assert spectral_predictability(_z("periodic")) > spectral_predictability(_z("ar1", 0.9))
    assert spectral_predictability(_z("ar1", 0.9)) > spectral_predictability(_z("white"))
    assert spectral_predictability(_z("white")) < 0.2          # white ~ the low-predictability floor


def test_ar_r2_high_for_predictable_low_for_noise():
    assert ar_forecast_r2(_z("periodic")) > 0.9
    assert ar_forecast_r2(_z("ar1", 0.9)) > 0.5
    assert abs(ar_forecast_r2(_z("white"))) < 0.1              # white ~ unpredictable


def test_ar1_phi_sweep_monotone():
    omegas = [spectral_predictability(_z("ar1", p)) for p in (0.95, 0.7, 0.4, 0.1)]
    assert omegas[0] > omegas[1] > omegas[2] >= omegas[3] - 1e-3   # predictability decreases with φ


def test_past_future_mi_predictive_information():
    assert past_future_mi(_z("periodic")) > past_future_mi(_z("white"))
    assert past_future_mi(_z("white")) < 0.5                   # ~ no predictive information in noise


def test_permutation_entropy_pred_range_and_order():
    pp, pw = permutation_entropy_pred(_z("periodic")), permutation_entropy_pred(_z("white"))
    assert 0.0 <= pw <= pp <= 1.0


def test_lorenz_low_intrinsic_dimension():
    # the Lorenz attractor lives on a ~2-D manifold in 3-D -> participation ratio < 3
    assert intrinsic_dimension(_z("lorenz")) < 2.8


def test_generator_panel_shapes():
    g = generate(regime="ar1", phi=0.8, T=2000, obs_dim=8, W=32, seed=0)
    M = g["meta"]["n_windows"]
    assert g["data"].shape == (M, 32, 8, 1)
    assert g["latent"].shape[0] == M and g["latent"].shape[1] == g["meta"]["d_lat"]
    assert np.isfinite(g["data"]).all()


def test_generator_collate_matches_jepa_batch():
    g = generate(regime="ar1", phi=0.8, T=1500, W=24, seed=0)
    ds = DynamicsWindows(g["data"])
    b = collate([ds[i] for i in range(8)])
    assert b["data"].shape[0] == 8 and b["pad_mask"].all() and b["labels"] is None


def test_report_has_all_keys():
    r = predictability_report(_z("ar1", 0.7))
    for k in ("spectral_omega", "ar_r2", "past_future_mi", "intrinsic_dim", "largest_lyapunov"):
        assert k in r and np.isfinite(r[k])
