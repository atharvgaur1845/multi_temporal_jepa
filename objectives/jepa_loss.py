"""JEPA latent loss — L2 in representation space (NOT pixel reconstruction).

This is the "JEPA thesis": predict in representation space. Two non-negotiable details:

1. STOP-GRADIENT on the target. The target comes from the EMA encoder and must be detached;
   if gradient flows into the target branch the system collapses (or cheats). The asymmetry
   (trainable context encoder + narrow predictor) vs (detached EMA target) is the anti-collapse
   mechanism.

2. LayerNorm the target before the loss. The target-encoder output is normalized so the
   regression isn't dominated by a few high-variance dimensions and the scale is stable.

    loss = mean_tokens || predictor_output - sg(LayerNorm(target_encoder_output)) ||^2
"""
from __future__ import annotations

import torch.nn.functional as F


def jepa_latent_loss(pred, target, norm_target=True, loss_type="l2"):
    """Mean error between predicted and target latents.

    Args
        pred   : (B, N_tgt, D)   predictor outputs at target locations (gradient flows here)
        target : (B, N_tgt, D)   target-encoder outputs (detached here -> stop-grad)
        norm_target : apply LayerNorm (over feature dim) to `target` before the loss
        loss_type   : 'l2' (I-JEPA) or 'l1' (V-JEPA ablation); reduction is identical so the
                      two variants are directly comparable.

    Returns: scalar loss.

    Stop-grad: `target` is detached unconditionally. Forgetting this is the #1 collapse bug;
    tests/test_loss.py asserts target.grad is None after backward.
    """
    target = target.detach()
    if norm_target:
        # LayerNorm over the feature dim only (per-token), no learnable affine.
        target = F.layer_norm(target, (target.shape[-1],))

    if loss_type == "l2":
        per_elem = (pred - target) ** 2
    elif loss_type == "l1":
        per_elem = (pred - target).abs()
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}")
    # mean over feature dim -> tokens -> batch
    return per_elem.mean()
