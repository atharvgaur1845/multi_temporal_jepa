"""Collapse diagnostics — your early-warning system. Log these EVERY N steps.

A falling JEPA loss is NOT evidence of success: a collapsed model (constant embeddings) also
has ~0 loss. You must watch representation health directly and PAIR loss with a probe metric.
This module is what lets you catch silent collapse on slow satellite data within minutes
instead of days (the reason we built the M1 correctness harness).

All functions take a 2D tensor of embeddings (N, D). Callers should flatten token/batch dims
(e.g. (B, N_tgt, D) -> (B*N_tgt, D)) before passing in.
"""
from __future__ import annotations

import torch


def _flatten(embeddings):
    z = embeddings.detach()
    if z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
    return z.float()


def per_dim_std(embeddings):
    """Mean over dims of the per-dimension std across the batch.

    s_d = std_N(z[:, d]); return mean_d(s_d).
    Healthy: bounded away from 0. Collapse: -> 0 (rule of thumb: < ~0.1 is suspicious).
    """
    z = _flatten(embeddings)
    return z.std(dim=0, unbiased=False).mean().item()


def effective_rank(embeddings):
    """Effective rank of the embedding covariance (spectral entropy).

    cov = centered z^T z / N; eigenvalues λ_i >= 0; p_i = λ_i / Σλ;
    effective_rank = exp(-Σ p_i log p_i).
    Healthy: high (many active dims). Dimensional collapse: -> 1.
    """
    z = _flatten(embeddings)
    z = z - z.mean(dim=0, keepdim=True)
    n = z.shape[0]
    cov = (z.t() @ z) / max(1, n)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eig.sum()
    if total <= 0:
        return 1.0
    p = eig / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return torch.exp(entropy).item()


def variance_ratio(pred, target):
    """Ratio of predictor-output total variance to target total variance.

    If the predictor learned a constant, this -> 0. Healthy: O(1).
    """
    p = _flatten(pred)
    t = _flatten(target)
    vp = p.var(dim=0, unbiased=False).sum()
    vt = t.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    return (vp / vt).item()


def offdiag_covariance(embeddings):
    """Mean squared off-diagonal of the correlation matrix (VICReg-style).

    Rising off-diagonal => informational/dimensional collapse even if per-dim std looks fine.
    """
    z = _flatten(embeddings)
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = z / std
    n = z.shape[0]
    corr = (z.t() @ z) / max(1, n)
    d = corr.shape[0]
    off = corr - torch.diag(torch.diag(corr))
    return (off.pow(2).sum() / (d * (d - 1) + 1e-12)).item()


def collapse_metrics(embeddings, pred=None, target=None):
    """Bundle the diagnostics into a flat dict for logging."""
    out = {
        "per_dim_std": per_dim_std(embeddings),
        "effective_rank": effective_rank(embeddings),
        "offdiag_cov": offdiag_covariance(embeddings),
    }
    if pred is not None and target is not None:
        out["variance_ratio"] = variance_ratio(pred, target)
    return out
