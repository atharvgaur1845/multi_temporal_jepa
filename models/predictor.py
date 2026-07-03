"""The predictor — a NARROW transformer. This bottleneck is half of the anti-collapse trick.

Why narrow (width ~384, regardless of encoder dim): the asymmetry between a wide context
encoder and a narrow predictor, combined with the EMA target + stop-gradient, is what makes
the trivial constant solution NOT a stable fixed point (same mechanism family as BYOL/SimSiam).

Flow
    context tokens (B, N_ctx, D_enc)
        -> project to predictor width D_pred (=384)
        -> concat with MASK TOKENS, one per target location:
               mask_token (one shared learnable vector, broadcast)
             + projected positional embedding of the target location (spatial pos AND/OR DOY)
        -> transformer blocks (depth ~6)
        -> project back to D_enc and read out predictions at the target slots
    returns predicted target latents (B, N_tgt, D_enc)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vit import Block


class Predictor(nn.Module):
    """Narrow transformer predictor with a shared learnable mask token.

    Parameters
        enc_dim   : encoder embedding dim (input/output width at the boundary)
        pred_dim  : predictor internal width (KEEP ~384 — the bottleneck)
        depth, num_heads
    """

    def __init__(self, enc_dim=256, pred_dim=384, depth=6, num_heads=12,
                 predict_variance=False, logvar_range=(-6.0, 4.0)):
        super().__init__()
        self.enc_dim = enc_dim
        self.pred_dim = pred_dim
        self.in_proj = nn.Linear(enc_dim, pred_dim)
        self.pos_proj = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([Block(pred_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, enc_dim)
        # Distributional (heteroscedastic) head — Phase 4. When on, a SECOND head predicts the
        # per-dim log-variance of the future latent, so the predictor emits a Gaussian N(mu, sigma^2)
        # instead of a point. Default OFF -> byte-for-byte the original point predictor (the satellite
        # and finance/industrial results stay reproducible). logvar is clamped to a sane range so the
        # variance can neither explode (mean ignored) nor vanish (division blow-up) — see beta-NLL loss.
        self.predict_variance = predict_variance
        self.logvar_min, self.logvar_max = logvar_range
        if predict_variance:
            self.out_proj_var = nn.Linear(pred_dim, enc_dim)
            # init variance head near log sigma^2 = 0 (sigma ~ 1), matching the LayerNorm'd target scale
            nn.init.zeros_(self.out_proj_var.weight)
            nn.init.zeros_(self.out_proj_var.bias)

    def forward(self, context_tokens, target_pos, n_targets=None):
        """
        context_tokens : (B, N_ctx, enc_dim)   encoded visible context
        target_pos     : (B, N_tgt, enc_dim)   positional embedding of each target slot
                         (spatial pos for Spatial JEPA; spatial + future DOY for Temporal JEPA)
        Returns        : (B, N_tgt, enc_dim) predicted target latents (point mode), OR
                         (mu, logvar) each (B, N_tgt, enc_dim) when predict_variance=True.

        The mask token carries no positional info by itself — every target slot is the SAME
        learnable vector PLUS its positional embedding, so predictions differ per location.
        """
        B, N_tgt, _ = target_pos.shape
        ctx = self.in_proj(context_tokens)                       # (B, N_ctx, pred_dim)
        masks = self.mask_token.expand(B, N_tgt, -1) + self.pos_proj(target_pos)
        x = torch.cat([ctx, masks], dim=1)                       # (B, N_ctx+N_tgt, pred_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = x[:, -N_tgt:]                                     # read out the mask-token slots
        mu = self.out_proj(pred)                                # (B, N_tgt, enc_dim)
        if not self.predict_variance:
            return mu
        logvar = self.out_proj_var(pred).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar
