"""JEPA latent loss — L2 in representation space (NOT pixel reconstruction).

This is the "JEPA thesis": predict in representation space. Two non-negotiable details:

1. STOP-GRADIENT on the target. The target comes from the EMA encoder and must be detached;
   if gradient flows into the target branch the system collapses (or cheats). The asymmetry
   (trainable context encoder + narrow predictor) vs (detached EMA target) is the anti-collapse
   mechanism.

2. LayerNorm the target before the loss. The target-encoder output is normalized so the
   regression isn't dominated by a few high-variance dimensions and the scale is stable.

    loss = mean_tokens || predictor_output - sg(LayerNorm(target_encoder_output)) ||^2
           (averaged over target tokens, then over the M target blocks)
"""
from __future__ import annotations


def jepa_latent_loss(pred, target, norm_target=True):
    """Mean squared error between predicted and target latents.

    Args
        pred   : (B, N_tgt, D)   predictor outputs at target locations (gradient flows here)
        target : (B, N_tgt, D)   target-encoder outputs (MUST be detached / stop-grad)
        norm_target : apply LayerNorm to `target` before the loss

    Returns: scalar loss.

    TODO
        - apply LayerNorm over the feature dim of `target` if norm_target.
        - detach the target (assert/ensure no grad). Forgetting this is the #1 collapse bug.
        - reduce: mean over feature dim -> mean over tokens -> mean over batch/blocks.
    Ablation hook: an L1 variant (à la V-JEPA) — keep the reduction identical so it's comparable.
    """
    raise NotImplementedError("M1")
