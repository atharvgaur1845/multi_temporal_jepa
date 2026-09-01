"""Temporal-order verification: a NON-PREDICTIVE temporal pretext task.

This is the control the paper needs. Temporal JEPA beats spatial JEPA, but that comparison
alone cannot separate two explanations:

    (a) predicting the FUTURE LATENT is what helps, or
    (b) merely USING THE TIME AXIS at all is what helps.

Order verification uses the time axis and predicts nothing. It trains the same encoder,
through the same temporal pathway, for the same budget. If it lands near temporal JEPA,
explanation (b) is live and the paper's claim must narrow. If it lands near spatial JEPA,
(a) survives.

THE DOY LEAKAGE TRAP. `SITSEncoder.encode_temporal(data, dates, pad_mask)` adds an explicit
day-of-year encoding. If we permuted the frames AND their dates together, the task would be
solvable by reading the positional encoding alone, with no reference to the imagery, and the
encoder would learn nothing about phenology. So we permute the IMAGE FRAMES ONLY and leave
`dates` in their original chronological order. The model must then decide whether the image
content is consistent with the claimed date order, which cannot be answered from the encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def order_head(embed_dim):
    """Binary head over the pooled spatiotemporal embedding: ordered vs permuted."""
    return nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(inplace=True),
                         nn.Linear(embed_dim // 2, 2))


def make_order_batch(batch, generator=None):
    """Return (data, label). Half the batch keeps chronological frame order, half is permuted.

    `dates` and `pad_mask` are returned UNCHANGED by the caller: only the image frames move.
    A permutation is redrawn if it happens to be the identity, so positives and negatives are
    never the same input with different labels.
    """
    data, pad_mask = batch["data"], batch["pad_mask"]
    B, T = data.shape[:2]
    out = data.clone()
    label = torch.zeros(B, dtype=torch.long, device=data.device)
    for b in range(B):
        if torch.rand(1, generator=generator).item() < 0.5:
            continue                                   # keep in order, label 0
        n = int(pad_mask[b].sum().item())               # permute only REAL frames
        if n < 3:
            continue                                   # too short to be a meaningful negative
        for _ in range(8):
            perm = torch.randperm(n, generator=generator)
            if not torch.equal(perm, torch.arange(n)):
                break
        else:
            continue                                   # never got a non-identity permutation
        out[b, :n] = data[b, perm.to(data.device)]
        label[b] = 1
    return out, label


def pool_spatiotemporal(tokens, pad_mask):
    """(B, F, N, D) + (B, F) -> (B, D). Mean over patches, padding-masked mean over frames."""
    m = pad_mask.float()[:, :, None]                    # (B, F, 1)
    per_frame = tokens.mean(dim=2)                      # (B, F, D)
    return (per_frame * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
