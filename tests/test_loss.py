"""JEPA loss invariants — stop-gradient on the target is what prevents collapse/cheating."""
import torch

from objectives.jepa_loss import jepa_latent_loss


def test_zero_when_equal():
    x = torch.randn(4, 16, 32)
    # with target LayerNorm, equality may not give exactly 0; test the no-norm path == 0
    loss = jepa_latent_loss(x.clone(), x.clone(), norm_target=False)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_no_gradient_reaches_target():
    pred = torch.randn(4, 16, 32, requires_grad=True)
    target = torch.randn(4, 16, 32, requires_grad=True)
    loss = jepa_latent_loss(pred, target)
    loss.backward()
    assert pred.grad is not None, "predictor branch must receive gradient"
    assert target.grad is None, "target must be detached (stop-grad) — else the teacher collapses"
