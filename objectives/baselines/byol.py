"""BYOL baseline — online + EMA-target with a predictor head, no negatives.

Closest cousin to JEPA among the baselines: it also uses an EMA target and stop-gradient, but
operates on two AUGMENTED views of the same input (invariance) rather than predicting a
masked/future region. Good for isolating "what does the JEPA prediction task add over plain
EMA self-distillation?".

Spec: two views -> online encoder+projector+predictor on view1; target encoder+projector
(EMA, stop-grad) on view2; loss = 2 - 2*cos_sim(predict(view1), sg(target(view2))), symmetrized.

TODO (M4): projector/predictor MLP heads + symmetric cosine loss + EMA (reuse engine/ema.py).
"""
from __future__ import annotations


def byol_loss(online_pred, target_proj):
    """Negative cosine similarity (symmetrized by the caller). target_proj is detached.
    TODO: implement (M4)."""
    raise NotImplementedError("M4")
