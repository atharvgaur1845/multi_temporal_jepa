"""JEPA assembly: context encoder + EMA target encoder (frozen) + predictor.

This is the heart of the method. The asymmetry between the two encoders — one trained by
gradient descent, one an EMA copy that receives NO gradient (stop-grad) — is what prevents
representation collapse. Getting the gradient flow and the EMA wiring exactly right is the
single most important correctness property; tests/test_ema.py and tests/test_loss.py guard it.

Both Spatial JEPA and Temporal JEPA share this class; they differ only in WHICH view goes to
the target encoder and what positional info the predictor's mask tokens carry:

    Spatial JEPA : target = masked target BLOCKS of the same frame; mask tokens carry spatial pos.
    Temporal JEPA: target = a FUTURE frame's latent; mask tokens carry the future frame's DOY.
"""
from __future__ import annotations

import torch.nn as nn


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture.

    Components
        context_encoder : trainable (ViT [+ temporal encoder for the temporal variant])
        target_encoder  : EMA copy of context_encoder, requires_grad=False, eval-ish
        predictor       : narrow transformer

    Construction TODO
        - build context_encoder; deep-copy it into target_encoder.
        - set every target_encoder parameter requires_grad=False (NEVER optimized).
        - keep an external EMA updater (engine/ema.py) — do the EMA step in the train loop,
          not inside forward.

    forward(batch, mask_spec) TODO
        1. context branch: encode the VISIBLE/context view with context_encoder.
        2. predictor: predict target-location latents from context + positional mask tokens.
        3. target branch: under torch.no_grad(), encode the TARGET view with target_encoder;
           that output (after LayerNorm + detach in the loss) is the regression target.
        return (pred, target) for objectives/jepa_loss.py.

    Collapse-prevention invariants (assert these in tests):
        - target_encoder params have requires_grad == False
        - no gradient reaches target_encoder (the target is detached)
        - predictor width < encoder width (asymmetry preserved)
    """

    def __init__(self, context_encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        # TODO: self.context_encoder = ...; self.target_encoder = deepcopy(...) frozen;
        #       self.predictor = predictor
        raise NotImplementedError("M1")

    def forward(self, batch, mask_spec):
        raise NotImplementedError("M1")
