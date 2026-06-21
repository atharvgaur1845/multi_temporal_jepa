"""FinanceJEPA — JEPA assembly over a financial panel (temporal or spatial).

A faithful parallel of models/jepa.JEPA, kept SEPARATE from it on purpose (CLAUDE.md: keep the old
implementation untouched, minimal changes) so the validated satellite results stay reproducible.
Everything modality-agnostic is REUSED, not reimplemented: the narrow Predictor, the JEPA latent
loss + VICReg regularizer, the EMA machinery, the collapse diagnostics, and the transformer stacks.

The two objectives differ exactly as in the satellite version:
    Temporal JEPA: target = a FUTURE day's latent; predictor mask tokens carry asset-pos + future DOY.
    Spatial  JEPA: target = a masked subset of ONE day's assets; mask tokens carry asset-pos.

Anti-collapse note carries over and is, if anything, sharper for markets: consecutive trading days
are highly correlated (tomorrow ≈ today), so "predict the next day's latent" is trivially solvable
by emitting a constant. The VICReg variance/covariance term on the trainable context embedding
(objectives/jepa_loss.variance_covariance_reg, on by default) prevents that collapse.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from masking.asset_mask import sample_asset_mask
from .finance_encoder import PanelEncoder
from .pos_embed import doy_sincos_pos_embed
from .predictor import Predictor


class FinanceJEPA(nn.Module):
    """JEPA over a financial panel. Collapse-prevention invariants match models/jepa.JEPA:
    target encoder is frozen (requires_grad False, never optimized, detached in the loss) and the
    predictor width is strictly narrower than the encoder width.
    """

    def __init__(self, objective="temporal_jepa", num_assets=9, num_features=4,
                 embed_dim=128, depth=4, num_heads=4, temporal_depth=4,
                 pred_dim=64, pred_depth=4, pred_heads=4,
                 horizon=1, min_context=8, n_targets=None, grad_checkpoint=False,
                 temporal_period=366):
        super().__init__()
        assert objective in ("spatial_jepa", "temporal_jepa")
        assert pred_dim < embed_dim, (
            f"predictor width ({pred_dim}) must be NARROWER than encoder width ({embed_dim}) "
            "— the asymmetry bottleneck is half the anti-collapse mechanism.")
        self.objective = objective
        self.horizon = horizon
        self.min_context = min_context
        self.num_assets = num_assets
        self.temporal_period = temporal_period
        self.n_targets = n_targets if n_targets is not None else max(1, num_assets // 2)

        self.context_encoder = PanelEncoder(num_assets, num_features, embed_dim, depth, num_heads,
                                            temporal_depth, grad_checkpoint=grad_checkpoint,
                                            temporal_period=temporal_period)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(embed_dim, pred_dim, pred_depth, pred_heads)
        self.embed_dim = embed_dim

    # ---------------------------------------------------------------- forward
    def forward(self, batch):
        """Returns (pred, target, context_repr). context_repr is the TRAINABLE embedding to watch
        for collapse (the EMA target lags and can hide an in-progress collapse)."""
        if self.objective == "spatial_jepa":
            return self._forward_spatial(batch)
        return self._forward_temporal(batch)

    def _pick_frame(self, pad_mask):
        min_real = int(pad_mask.sum(dim=1).min().item())
        return int(torch.randint(0, max(1, min_real), (1,)).item())

    def _forward_spatial(self, batch):
        data, pad_mask = batch["data"], batch["pad_mask"]
        f = self._pick_frame(pad_mask)
        frame = data[:, f]                                            # (B, N, F)
        ctx_idx, tgt_idx = sample_asset_mask(self.num_assets, self.n_targets)
        ctx_idx, tgt_idx = ctx_idx.to(frame.device), tgt_idx.to(frame.device)

        z_ctx = self.context_encoder.encode_subset(frame, ctx_idx)   # (B, Nc, D)
        spat = self.context_encoder.spatial_pos                      # (N, D)
        target_pos = spat[tgt_idx].unsqueeze(0).expand(frame.shape[0], -1, -1)  # (B, Nt, D)
        pred = self.predictor(z_ctx, target_pos)                     # (B, Nt, D)
        with torch.no_grad():
            full = self.target_encoder.encode_full(frame)            # (B, N, D)
            z_tgt = full[:, tgt_idx]                                  # (B, Nt, D)
        return pred, z_tgt, z_ctx

    def _forward_temporal(self, batch):
        data, dates, pad_mask = batch["data"], batch["dates"], batch["pad_mask"]
        B, T = pad_mask.shape
        device = data.device
        n_real = pad_mask.sum(dim=1)                                 # (B,)

        # PER-SAMPLE causal split: context = days [0..s], target = day s+horizon. Windows are fixed
        # length and front-packed (all real), so position == chronological trading-day rank.
        s_lo = self.min_context - 1
        s_hi = n_real - 1 - self.horizon                             # (B,) inclusive
        if (s_hi < s_lo).any():
            raise ValueError(f"window too short: min n_real={int(n_real.min())}, "
                             f"min_context={self.min_context}, horizon={self.horizon}")
        r = torch.rand(B, device=device)
        s = s_lo + (r * (s_hi - s_lo + 1).float()).floor().long()
        s = torch.minimum(s, s_hi)
        tgt_idx = s + self.horizon                                   # (B,) future day

        # context-only mask: hide the target day and everything after it from temporal attention
        # (no future leakage) and from the masked-mean pool.
        ctx_mask = (torch.arange(T, device=device)[None, :] <= s[:, None]) & pad_mask  # (B,T)
        ctx_tok = self.context_encoder.encode_temporal(data, dates, ctx_mask)  # (B,T,N,D)
        m = ctx_mask.float()[:, :, None, None]
        ctx_repr = (ctx_tok * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)       # (B,N,D) over PAST

        bidx = torch.arange(B, device=device)
        tgt_frame = data[bidx, tgt_idx]                              # (B, N, F)
        tgt_date = dates[bidx, tgt_idx]                              # (B,)
        spat = self.context_encoder.spatial_pos.unsqueeze(0)        # (1, N, D)
        doy = doy_sincos_pos_embed(tgt_date.unsqueeze(1), self.embed_dim,
                                   period=self.temporal_period).squeeze(1)  # (B, D)
        target_pos = spat + doy.unsqueeze(1)                         # (B, N, D)
        pred = self.predictor(ctx_repr, target_pos)                 # (B, N, D)
        with torch.no_grad():
            z_tgt = self.target_encoder.encode_full(tgt_frame)       # (B, N, D)
        return pred, z_tgt, ctx_repr


def build_finance_model(cfg, meta):
    """Construct a FinanceJEPA from an fjepa.yaml-style config + dataset meta (assets/features).

    Enforces the predictor bottleneck (clamp to half the encoder width if mis-set), mirroring
    models.jepa.build_model so the embed-dim ablation grid stays valid without per-cell tuning.
    """
    enc = cfg["encoder"]
    pred = cfg["predictor"]
    temp = cfg.get("temporal", {})
    edim = enc["embed_dim"]
    pdim = pred["embed_dim"]
    pheads = pred["num_heads"]
    if pdim >= edim:
        pdim = edim // 2
        pheads = max(1, pdim // 32)
        print(f"[build_finance_model] predictor width clamped to {pdim} (heads {pheads}).")
    if pdim % pheads != 0:
        pheads = max(1, pdim // 16)
    return FinanceJEPA(
        objective=cfg.get("objective", "temporal_jepa"),
        num_assets=meta["num_assets"], num_features=meta["num_features"],
        embed_dim=edim, depth=enc["depth"], num_heads=enc["num_heads"],
        temporal_depth=enc.get("temporal_depth", 4),
        pred_dim=pdim, pred_depth=pred["depth"], pred_heads=pheads,
        horizon=temp.get("horizon", 1), min_context=temp.get("min_context", 8),
        grad_checkpoint=enc.get("grad_checkpoint", False),
        temporal_period=temp.get("period", 366),
    )
