"""Structured latent-dynamics predictors + LKF (Part 6 #1/#3/#4).

Guards the drop-in predictors (shapes, grad routing, Koopman diagnostics, ODE identity-init) and the
Kalman filter (it must actually denoise). Pure synthetic; offline.
"""
import numpy as np
import pytest
import torch

from models.structured_predictors import KoopmanPredictor, NeuralODEPredictor
from models.finance_jepa import FinanceJEPA
from models.latent_filter import kalman_filter, lkf_report
from objectives.jepa_loss import jepa_latent_loss


def _batch(B=6, W=24, N=8, Fc=1):
    g = torch.Generator().manual_seed(0)
    return {"data": torch.randn(B, W, N, Fc, generator=g),
            "dates": torch.randint(1, 400, (B, W), generator=g),
            "pad_mask": torch.ones(B, W, dtype=torch.bool)}


def test_koopman_forward_and_diagnostics():
    p = KoopmanPredictor(enc_dim=32)
    ctx = torch.randn(4, 8, 32)
    out = p(ctx, None, horizon=1)
    assert out.shape == ctx.shape
    assert p(ctx, None, horizon=5).shape == ctx.shape       # K^5 still valid shape
    assert isinstance(p.spectral_radius(), float)
    assert p.operator().shape == (32, 32)                    # process operator for the LKF


def test_koopman_grad_flows():
    p = KoopmanPredictor(enc_dim=16)
    out = p(torch.randn(3, 5, 16, requires_grad=True), None, horizon=1)
    out.pow(2).mean().backward()
    assert p.K.grad is not None


def test_ode_identity_at_init():
    """Zero-init last layer -> the flow is ~identity at init (a safe, near-copy starting point)."""
    p = NeuralODEPredictor(enc_dim=16)
    z = torch.randn(4, 8, 16)
    assert torch.allclose(p(z, None, horizon=1), z, atol=1e-4)


@pytest.mark.parametrize("ptype", ["koopman", "ode"])
def test_finance_jepa_structured_grad_routing(ptype):
    m = FinanceJEPA(objective="temporal_jepa", num_assets=8, num_features=1, embed_dim=32, depth=2,
                    num_heads=4, temporal_depth=2, pred_dim=16, pred_depth=2, pred_heads=4,
                    min_context=4, horizon=1, predictor_type=ptype)
    pred, target, ctx = m(_batch())
    jepa_latent_loss(pred, target).backward()
    assert all(p.grad is None for p in m.target_encoder.parameters())        # EMA target frozen
    assert any(p.grad is not None for p in m.predictor.parameters())         # predictor trains


def test_structured_rejects_distributional():
    with pytest.raises(AssertionError):
        FinanceJEPA(embed_dim=32, pred_dim=16, predictor_type="koopman", distributional=True)


def test_kalman_filter_denoises():
    """On a persistent AR(1) trajectory with a good process model, filtering must beat the raw noisy
    measurement (RMSE lower)."""
    rng = np.random.default_rng(0)
    T, D = 300, 3
    phi = 0.95
    z = np.zeros((T, D)); z[0] = rng.standard_normal(D)
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(D)
    A = phi * np.eye(D)
    out = lkf_report(z, A, meas_noise=0.6, seed=1)
    assert out["rmse_filtered"] < out["rmse_measurement"]     # the filter denoises
    assert out["filter_gain_vs_measurement"] > 0


def test_kalman_filter_shapes():
    m = np.random.default_rng(0).standard_normal((50, 4))
    assert kalman_filter(m, 0.9 * np.eye(4)).shape == (50, 4)


# ---- Part 6 #7 hierarchical / multi-timescale ----
def test_hierarchical_shapes_and_grad():
    """horizons=[1,5,20] predicts K=3 future frames -> pred (B, K·N, D); single horizon unchanged."""
    single = FinanceJEPA(objective="temporal_jepa", num_assets=8, num_features=1, embed_dim=32,
                         depth=2, num_heads=4, temporal_depth=2, pred_dim=16, pred_depth=2,
                         pred_heads=4, min_context=6, horizon=1)
    p1, _, _ = single(_batch(W=40))
    assert p1.shape == (6, 8, 32)                                    # (B, N, D) — backward compatible
    hier = FinanceJEPA(objective="temporal_jepa", num_assets=8, num_features=1, embed_dim=32,
                       depth=2, num_heads=4, temporal_depth=2, pred_dim=16, pred_depth=2,
                       pred_heads=4, min_context=6, horizons=[1, 5, 20])
    p, t, _ = hier(_batch(W=40))
    assert p.shape == (6, 24, 32) == t.shape                        # (B, K·N, D), K=3
    jepa_latent_loss(p, t).backward()
    assert all(pp.grad is None for pp in hier.target_encoder.parameters())    # EMA target frozen
    assert any(pp.grad is not None for pp in hier.context_encoder.parameters())


def test_hierarchical_rejects_distributional():
    with pytest.raises(AssertionError):
        FinanceJEPA(embed_dim=32, pred_dim=16, horizons=[1, 5], distributional=True)
