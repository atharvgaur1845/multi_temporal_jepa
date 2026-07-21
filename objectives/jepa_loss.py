"""JEPA latent loss — L2 in representation space (NOT pixel reconstruction).

This is the "JEPA thesis": predict in representation space. Two non-negotiable details:

1. STOP-GRADIENT on the target. The target comes from the EMA encoder and must be detached;
   if gradient flows into the target branch the system collapses (or cheats). The asymmetry
   (trainable context encoder + narrow predictor) vs (detached EMA target) is the anti-collapse
   mechanism.

2. LayerNorm the target before the loss. The target-encoder output is normalized so the
   regression isn't dominated by a few high-variance dimensions and the scale is stable.

    loss = mean_tokens || predictor_output - sg(LayerNorm(target_encoder_output)) ||^2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_latent_loss(pred, target, norm_target=True, loss_type="l2", sample_weight=None):
    """Mean error between predicted and target latents.

    Args
        pred   : (B, N_tgt, D)   predictor outputs at target locations (gradient flows here)
        target : (B, N_tgt, D)   target-encoder outputs (detached here -> stop-grad)
        norm_target : apply LayerNorm (over feature dim) to `target` before the loss
        loss_type   : 'l2' (I-JEPA) or 'l1' (V-JEPA ablation); reduction is identical so the
                      two variants are directly comparable.
        sample_weight : optional (B,) per-sample weights (Part 6 #2, predictability-conditioned
                      curriculum). If given, the loss is a weighted mean over samples — windows whose
                      dynamics are more predictable get more of the encoder's capacity. Normalized to
                      mean 1 so the overall loss scale (and thus the LR interaction) is preserved.

    Returns: scalar loss. Stop-grad: `target` is detached unconditionally (the #1 collapse bug).
    """
    target = target.detach()
    if norm_target:
        # LayerNorm over the feature dim only (per-token), no learnable affine.
        target = F.layer_norm(target, (target.shape[-1],))

    if loss_type == "l2":
        per_elem = (pred - target) ** 2
    elif loss_type == "l1":
        per_elem = (pred - target).abs()
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}")
    if sample_weight is None:
        return per_elem.mean()                                        # mean over feat -> tokens -> batch
    per_sample = per_elem.mean(dim=tuple(range(1, per_elem.dim())))   # (B,) mean over all but batch
    w = sample_weight / sample_weight.mean().clamp_min(1e-6)          # normalize to mean 1
    return (per_sample * w).mean()


def jepa_beta_nll_loss(mu, logvar, target, beta=0.5, norm_target=True, eps=1e-6):
    """Heteroscedastic beta-NLL latent loss (Phase 4 — the distributional JEPA objective).

    The predictor emits a Gaussian N(mu, sigma^2=exp(logvar)) over the future latent instead of a
    point. Motivation (report_finance.md §12): when the future is unpredictable (markets), a POINT L2
    target is noise and the objective erases usable structure. With a variance head the model can
    output large sigma^2 where the future is unpredictable, which DOWN-WEIGHTS the un-learnable mean
    gradient (sigma^2 divides the squared error), so capacity flows to what IS predictable — the
    variance itself (volatility clusters). sigma^2 thus becomes a learned volatility signal.

    Standard Gaussian NLL per element (target detached + LayerNorm'd, as in the point loss):
        NLL_i = 0.5 * [ (y_i - mu_i)^2 / sigma_i^2 + log sigma_i^2 ]

    beta-NLL (Seitzer et al., ICLR 2022, "On the Pitfalls of Heteroscedastic Uncertainty Estimation")
    fixes the known gradient pathology (sigma^2 couples into the mean gradient, hurting the mean fit)
    by weighting each element's loss by a STOP-GRADIENT variance term:
        L = mean_i [ sg(sigma_i^2)^beta * NLL_i ]
    so d L / d mu_i  ∝  (mu_i - y_i) / sigma_i^(2 - 2*beta).  beta=0 -> pure NLL; beta=1 -> the mean
    gradient ignores the variance (robust mean, variance still learned via the log term); beta=0.5
    (default, recommended) balances the two. Reference impl: martius-lab/beta-nll.

    Args
        mu, logvar : (..., D)  predictor mean and log-variance (gradient flows through both)
        target     : (..., D)  target-encoder output (detached here -> stop-grad)
        beta       : beta-NLL exponent in [0, 1]
        norm_target: LayerNorm the target over the feature dim (matches jepa_latent_loss)
    Returns: scalar loss.
    """
    target = target.detach()
    if norm_target:
        target = F.layer_norm(target, (target.shape[-1],))
    var = logvar.exp().clamp_min(eps)
    nll = 0.5 * ((target - mu) ** 2 / var + logvar)             # (..., D)
    if beta > 0:
        nll = nll * var.detach() ** beta                        # stop-grad variance weighting
    return nll.mean()


def variance_covariance_reg(z, gamma=1.0, eps=1e-4):
    """VICReg-style anti-collapse regularizer on a trainable embedding `z` (..., D).

    Why this is needed for *temporal* SITS JEPA: consecutive acquisitions of the same field are
    nearly identical, so "predict the future latent" is trivially solvable by collapsing the
    encoder to a constant (loss -> 0, std -> 0). EMA + predictor + stop-grad alone don't prevent
    this when the two views are so correlated. These two terms push back directly:

      std_loss = mean_d relu(gamma - std_d)   # keep each dim's batch-std >= gamma (anti-collapse)
      cov_loss = sum_{i != j} cov_ij^2 / D     # decorrelate dims (anti dimensional-collapse)

    Returns (std_loss, cov_loss). Apply to the trainable context embedding (and optionally the
    predictor output); the EMA target is detached and not regularized.
    """
    z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.t() @ z) / max(1, n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / d
    return std_loss, cov_loss
