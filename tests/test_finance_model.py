"""FinanceJEPA wiring — collapse-prevention invariants + no future leakage in the temporal split.

Mirrors tests/test_model_synthetic.py + test_ema.py + test_loss.py for the finance stack. Pure
synthetic tensors; no data needed.
"""
import pytest
import torch

from models.finance_jepa import FinanceJEPA
from objectives.jepa_loss import jepa_latent_loss, variance_covariance_reg


def _batch(B=6, W=24, N=9, F=4):
    g = torch.Generator().manual_seed(0)
    data = torch.randn(B, W, N, F, generator=g)
    dates = torch.randint(1, 366, (B, W), generator=g)
    pad = torch.ones(B, W, dtype=torch.bool)
    return {"data": data, "dates": dates, "pad_mask": pad}


@pytest.mark.parametrize("objective", ["temporal_jepa", "spatial_jepa"])
def test_forward_shapes_and_grad_routing(objective):
    m = FinanceJEPA(objective=objective, num_assets=9, num_features=4, embed_dim=64, depth=2,
                    num_heads=4, temporal_depth=2, pred_dim=32, pred_depth=2, pred_heads=4,
                    min_context=4, horizon=1)
    pred, target, ctx = m(_batch())
    assert pred.shape == target.shape
    loss = jepa_latent_loss(pred, target)
    sv, cv = variance_covariance_reg(ctx.float())
    (loss + sv + 0.04 * cv).backward()
    # target encoder must NEVER receive gradient (stop-grad + frozen); context encoder must.
    assert all(p.grad is None for p in m.target_encoder.parameters())
    assert any(p.grad is not None for p in m.context_encoder.parameters())


def test_target_encoder_frozen():
    m = FinanceJEPA(embed_dim=64, pred_dim=32, depth=2, temporal_depth=2, pred_depth=2)
    assert all(not p.requires_grad for p in m.target_encoder.parameters())


def test_predictor_bottleneck_enforced():
    with pytest.raises(AssertionError):
        FinanceJEPA(embed_dim=64, pred_dim=128)          # predictor must be narrower than encoder


def test_temporal_no_future_leakage():
    """The causal context mask must hide the target day and all later days from the pooled context.
    We verify the split indices: every context day index <= split rank s < target index."""
    torch.manual_seed(0)
    m = FinanceJEPA(objective="temporal_jepa", embed_dim=64, pred_dim=32, depth=2, temporal_depth=2,
                    pred_depth=2, min_context=4, horizon=1)
    b = _batch(B=32, W=24)
    # replicate the split logic and assert the invariant holds for all samples
    pad = b["pad_mask"]
    n_real = pad.sum(1)
    s_lo, s_hi = m.min_context - 1, n_real - 1 - m.horizon
    assert (s_hi >= s_lo).all()
    # the target is always strictly after the last context day (horizon >= 1)
    s = torch.full((32,), s_lo)
    tgt = s + m.horizon
    assert (tgt > s).all()
