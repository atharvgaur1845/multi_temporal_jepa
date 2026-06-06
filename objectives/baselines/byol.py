"""BYOL baseline — online + EMA-target with a predictor head, no negatives.

Closest cousin to JEPA among the baselines: it also uses an EMA target and stop-gradient, but
operates on two AUGMENTED views of the same input (invariance) rather than predicting a
masked/future region. Good for isolating "what does the JEPA prediction task add over plain
EMA self-distillation?".

Spec: two views -> online encoder+projector+predictor on view1; target encoder+projector
(EMA, stop-grad) on view2; loss = 2 - 2*cos_sim(predict(view1), sg(target(view2))), symmetrized.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def mlp_head(in_dim, hidden_dim, out_dim):
    """Standard BYOL projector/predictor MLP: Linear -> BN -> ReLU -> Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def byol_loss(online_pred, target_proj):
    """Negative cosine similarity (the caller symmetrizes over the two view orderings).

    online_pred : (B, D) predictor output on view1 (gradient flows)
    target_proj : (B, D) EMA target projection on view2 (caller must pass a detached tensor)
    Returns the mean of 2 - 2*cos_sim, i.e. in [0, 4].
    """
    p = F.normalize(online_pred, dim=-1)
    z = F.normalize(target_proj.detach(), dim=-1)
    return (2 - 2 * (p * z).sum(dim=-1)).mean()
