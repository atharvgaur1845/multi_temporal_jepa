"""SimCLR baseline — contrastive (NT-Xent) over two augmented views.

The "contrastive objective" comparison point. Needs negatives (the rest of the batch), so
batch size matters. Contrast with JEPA/BYOL which use no negatives.

Spec: two views -> encoder + projector -> L2-normalized embeddings z; NT-Xent:
    for each anchor, positive = its other view, negatives = all other 2(B-1) embeddings;
    loss = -log( exp(sim(z_i,z_j)/τ) / sum_{k≠i} exp(sim(z_i,z_k)/τ) ), averaged.

TODO (M4): projector head + NT-Xent with temperature τ.
"""
from __future__ import annotations


def nt_xent_loss(z, temperature=0.5):
    """NT-Xent over a (2B, D) batch of L2-normalized embeddings (views interleaved/stacked).
    TODO: implement the similarity matrix, mask the self-term, cross-entropy to positives (M4)."""
    raise NotImplementedError("M4")
