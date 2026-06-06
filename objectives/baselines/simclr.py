"""SimCLR baseline — contrastive (NT-Xent) over two augmented views.

The "contrastive objective" comparison point. Needs negatives (the rest of the batch), so
batch size matters. Contrast with JEPA/BYOL which use no negatives.

Spec: two views -> encoder + projector -> L2-normalized embeddings z; NT-Xent:
    for each anchor, positive = its other view, negatives = all other 2(B-1) embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def projector(in_dim, hidden_dim, out_dim):
    """SimCLR projection head: Linear -> ReLU -> Linear."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, out_dim))


def nt_xent_loss(z, temperature=0.5):
    """NT-Xent over a (2B, D) batch where rows [0:B] are view1 and [B:2B] are view2, aligned by
    sample (anchor i's positive is i+B and vice-versa).

    z is L2-normalized inside. Returns the mean cross-entropy to the positive.
    """
    z = F.normalize(z, dim=-1)
    n = z.shape[0]
    B = n // 2
    sim = (z @ z.t()) / temperature                    # (2B, 2B)
    # mask self-similarity
    sim.fill_diagonal_(float("-inf"))
    # positive index for row i: (i + B) % 2B
    pos = (torch.arange(n, device=z.device) + B) % n
    return F.cross_entropy(sim, pos)
