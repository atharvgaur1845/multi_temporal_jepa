"""The predictor — a NARROW transformer. This bottleneck is half of the anti-collapse trick.

Why narrow (width ~384, regardless of encoder dim): the asymmetry between a wide context
encoder and a narrow predictor, combined with the EMA target + stop-gradient, is what makes
the trivial constant solution NOT a stable fixed point (same mechanism family as BYOL/SimSiam).
If you widen the predictor to match the encoder you weaken this and risk collapse.

Flow
    context tokens (B, N_ctx, D_enc)
        -> project to predictor width D_pred (=384)
        -> concat with MASK TOKENS, one per target location:
               mask_token (one shared learnable vector, broadcast)
             + positional embedding of the target location (spatial pos AND/OR target DOY)
        -> transformer blocks (depth ~6)
        -> project back to D_enc and read out the predictions at the target slots
    returns predicted target latents (B, N_tgt, D_enc)
"""
from __future__ import annotations

import torch.nn as nn


class Predictor(nn.Module):
    """Narrow transformer predictor with a shared learnable mask token.

    Parameters
        enc_dim   : encoder embedding dim (input/output width at the boundary)
        pred_dim  : predictor internal width (KEEP ~384 — the bottleneck)
        depth, num_heads

    TODO
    ----
    - input projection enc_dim -> pred_dim and output projection pred_dim -> enc_dim.
    - one learnable mask token (shape (1,1,pred_dim)); broadcast to N_tgt slots.
    - add target positional info to each mask token:
        * Spatial JEPA: 2D spatial pos of the target block locations.
        * Temporal JEPA: the target frame's DOY embedding (predicting a FUTURE frame).
    - run blocks over [context_tokens ++ mask_tokens]; return only the mask-token outputs.

    Pitfall: if mask tokens carry NO positional info, every target prediction is identical
    and the task degenerates.
    """

    def __init__(self, enc_dim=256, pred_dim=384, depth=6, num_heads=12):
        super().__init__()
        raise NotImplementedError("M1")

    def forward(self, context_tokens, target_pos, n_targets):
        raise NotImplementedError("M1")
