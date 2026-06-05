"""Collapse diagnostics — your early-warning system. Log these EVERY N steps.

A falling JEPA loss is NOT evidence of success: a collapsed model (constant embeddings) also
has ~0 loss. You must watch representation health directly and PAIR loss with a probe metric.
This module is what lets you catch silent collapse on slow satellite data within minutes
instead of days (the reason we built the M1 correctness harness).
"""
from __future__ import annotations


def per_dim_std(embeddings):
    """Mean over dims of the per-dimension std across the batch.

    Math: for embeddings z (N, D): s_d = std_N(z[:, d]); return mean_d(s_d).
    Healthy: bounded away from 0. Collapse: -> 0 (rule of thumb: < ~0.1 is suspicious).
    TODO: implement.
    """
    raise NotImplementedError("M1")


def effective_rank(embeddings):
    """Effective rank of the embedding covariance (a.k.a. spectral entropy).

    Math: cov = centered z^T z / N; singular values σ_i; p_i = σ_i / Σσ;
          effective_rank = exp(-Σ p_i log p_i).
    Healthy: high (many active dimensions). Dimensional collapse: drops toward 1.
    TODO: implement.
    """
    raise NotImplementedError("M1")


def variance_ratio(pred, target):
    """Ratio of predictor-output variance to target variance.

    If the predictor learned a constant, this -> 0. Healthy: O(1).
    TODO: implement (use total variance across dims).
    """
    raise NotImplementedError("M1")


def offdiag_covariance(embeddings):
    """Mean squared off-diagonal of the (normalized) covariance (VICReg-style).

    Rising off-diagonal => informational/dimensional collapse even if per-dim std looks fine.
    TODO: implement (optional but informative).
    """
    raise NotImplementedError("M1")


def collapse_metrics(embeddings, pred=None, target=None):
    """Bundle the above into a dict for logging. TODO: assemble + return."""
    raise NotImplementedError("M1")
