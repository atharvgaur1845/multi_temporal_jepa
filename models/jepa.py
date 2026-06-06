"""JEPA assembly: context encoder + EMA target encoder (frozen) + predictor.

This is the heart of the method. The asymmetry between the two encoders — one trained by
gradient descent, one an EMA copy that receives NO gradient (stop-grad) — is what prevents
representation collapse. Getting the gradient flow and the EMA wiring exactly right is the
single most important correctness property; tests/test_ema.py and tests/test_loss.py guard it.

Both Spatial JEPA and Temporal JEPA share this class; they differ only in WHICH view goes to
the target encoder and what positional info the predictor's mask tokens carry:

    Spatial JEPA : target = masked target BLOCKS of one frame; mask tokens carry spatial pos.
    Temporal JEPA: target = a FUTURE frame's latent; mask tokens carry spatial pos + future DOY.

Batch-shared masking note: real frames are front-packed by collate_variable_length, so a
batch-shared causal split rank `s` selects valid real frames for every sample (the target frame
R[s+horizon] exists because s+horizon <= min_real-1). This keeps the temporal path vectorized.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from masking.multiblock import sample_multiblock_mask
from masking.temporal_mask import split_past_future  # noqa: F401  (kept for reference/tests)
from .patch_embed import PatchEmbed
from .pos_embed import build_2d_sincos_pos_embed, doy_sincos_pos_embed
from .predictor import Predictor
from .temporal_encoder import TemporalEncoder
from .vit import ViTEncoder


class SITSEncoder(nn.Module):
    """Shared encoder: per-frame patch embed + spatial ViT (+ temporal transformer).

    Used as BOTH the trainable context encoder and (deep-copied, frozen) the EMA target encoder.
    """

    def __init__(self, img_size=128, patch_size=16, in_chans=10, embed_dim=256,
                 depth=6, num_heads=8, temporal_depth=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_hw = self.patch_embed.grid_hw
        self.num_patches = self.patch_embed.num_patches
        pos = build_2d_sincos_pos_embed(self.grid_hw, embed_dim)  # (N, D)
        self.register_buffer("spatial_pos", pos, persistent=False)
        self.spatial_vit = ViTEncoder(embed_dim, depth, num_heads, mlp_ratio)
        self.temporal_enc = TemporalEncoder(embed_dim, temporal_depth, num_heads, mlp_ratio)
        self.embed_dim = embed_dim

    # ---- spatial-only paths (single frame) ----
    def encode_full(self, frame):
        """frame (B, C, H, W) -> full-frame spatial tokens (B, N, D)."""
        tok = self.patch_embed(frame)               # (B, N, D)
        return self.spatial_vit(tok, pos_embed=self.spatial_pos)

    def encode_subset(self, frame, idx):
        """Encode only the tokens in `idx` (I-JEPA context encoder). Returns (B, |idx|, D)."""
        tok = self.patch_embed(frame) + self.spatial_pos.unsqueeze(0)  # (B, N, D) pos added pre-select
        tok = tok[:, idx]
        return self.spatial_vit(tok, pos_embed=None)

    # ---- spatiotemporal path (sequence) ----
    def encode_temporal(self, data, dates, pad_mask):
        """data (B, F, C, H, W) -> time-aware tokens (B, F, N, D)."""
        B, F, C, H, W = data.shape
        flat = data.reshape(B * F, C, H, W)
        tok = self.patch_embed(flat)                                  # (B*F, N, D)
        tok = self.spatial_vit(tok, pos_embed=self.spatial_pos)       # (B*F, N, D)
        tok = tok.reshape(B, F, self.num_patches, self.embed_dim)
        return self.temporal_enc(tok, dates, pad_mask)                # (B, F, N, D)


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture (spatial or temporal).

    Collapse-prevention invariants (asserted in tests):
        - target_encoder params have requires_grad == False (NEVER optimized)
        - no gradient reaches target_encoder (target is detached in the loss)
        - predictor width (pred_dim) < encoder width (embed_dim)
    """

    def __init__(self, objective="temporal_jepa", img_size=128, patch_size=16, in_chans=10,
                 embed_dim=256, depth=6, num_heads=8, temporal_depth=4,
                 pred_dim=384, pred_depth=6, pred_heads=12,
                 horizon=1, min_context=4, n_targets=4):
        super().__init__()
        assert objective in ("spatial_jepa", "temporal_jepa")
        self.objective = objective
        self.horizon = horizon
        self.min_context = min_context
        self.n_targets = n_targets

        assert pred_dim < embed_dim, (
            f"predictor width ({pred_dim}) must be NARROWER than encoder width ({embed_dim}) "
            "— the asymmetry bottleneck is half the anti-collapse mechanism."
        )
        self.context_encoder = SITSEncoder(img_size, patch_size, in_chans, embed_dim,
                                           depth, num_heads, temporal_depth)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(embed_dim, pred_dim, pred_depth, pred_heads)
        self.grid_hw = self.context_encoder.grid_hw
        self.embed_dim = embed_dim

    # ---------------------------------------------------------------- forward
    def forward(self, batch, mask_spec=None):
        """Returns (pred, target, context_repr).

        context_repr is the TRAINABLE context-encoder output used for the prediction. It (not
        the EMA target) is the right thing to watch for collapse: the target is EMA-smoothed and
        lags, so it can look healthy while the student is already collapsing.
        """
        if self.objective == "spatial_jepa":
            return self._forward_spatial(batch, mask_spec)
        return self._forward_temporal(batch)

    def _pick_frame(self, pad_mask):
        """Pick one real frame index shared across the batch (real frames are front-packed)."""
        min_real = int(pad_mask.sum(dim=1).min().item())
        return int(torch.randint(0, max(1, min_real), (1,)).item())

    def _forward_spatial(self, batch, mask_spec):
        data, pad_mask = batch["data"], batch["pad_mask"]
        f = self._pick_frame(pad_mask)
        frame = data[:, f]                                   # (B, C, H, W)

        if mask_spec is None:
            ctx_idx, target_blocks = sample_multiblock_mask(self.grid_hw, self.n_targets)
        else:
            ctx_idx, target_blocks = mask_spec
        tgt_idx = torch.cat(target_blocks).to(frame.device)
        ctx_idx = ctx_idx.to(frame.device)

        z_ctx = self.context_encoder.encode_subset(frame, ctx_idx)        # (B, Nc, D)
        spat = self.context_encoder.spatial_pos                           # (N, D)
        target_pos = spat[tgt_idx].unsqueeze(0).expand(frame.shape[0], -1, -1)  # (B, Nt, D)
        pred = self.predictor(z_ctx, target_pos)                          # (B, Nt, D)

        with torch.no_grad():
            full = self.target_encoder.encode_full(frame)                 # (B, N, D)
            z_tgt = full[:, tgt_idx]                                      # (B, Nt, D)
        return pred, z_tgt, z_ctx

    def _forward_temporal(self, batch):
        data, dates, pad_mask = batch["data"], batch["dates"], batch["pad_mask"]
        B, T = pad_mask.shape
        device = data.device
        n_real = pad_mask.sum(dim=1)                                      # (B,)

        # PER-SAMPLE causal split (richer than a batch-shared rank). Real frames are
        # front-packed by the collate, so position == chronological rank.
        s_lo = self.min_context - 1
        s_hi = n_real - 1 - self.horizon                                  # (B,) inclusive
        if (s_hi < s_lo).any():
            raise ValueError(f"a sample is too short: min n_real in batch={int(n_real.min())}, "
                             f"min_context={self.min_context}, horizon={self.horizon}")
        r = torch.rand(B, device=device)
        s = (s_lo + (r * (s_hi - s_lo + 1).float()).floor().long())
        s = torch.minimum(s, s_hi)                                        # safety clamp (B,)
        tgt_idx = s + self.horizon                                        # (B,) real future frame

        # context-only mask: hide the target + everything after it from temporal attention
        # (no future leakage) AND from the masked-mean pool below.
        ctx_mask = (torch.arange(T, device=device)[None, :] <= s[:, None]) & pad_mask  # (B,T)
        ctx_tok = self.context_encoder.encode_temporal(data, dates, ctx_mask)  # (B,T,N,D)
        m = ctx_mask.float()[:, :, None, None]
        ctx_repr = (ctx_tok * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B,N,D) mean over PAST

        bidx = torch.arange(B, device=device)
        tgt_frame = data[bidx, tgt_idx]                                  # (B, C, H, W)
        tgt_date = dates[bidx, tgt_idx]                                  # (B,)
        spat = self.context_encoder.spatial_pos.unsqueeze(0)             # (1, N, D)
        doy = doy_sincos_pos_embed(tgt_date.unsqueeze(1), self.embed_dim).squeeze(1)  # (B, D)
        target_pos = spat + doy.unsqueeze(1)                             # (B, N, D)
        pred = self.predictor(ctx_repr, target_pos)                     # (B, N, D)

        with torch.no_grad():
            z_tgt = self.target_encoder.encode_full(tgt_frame)           # (B, N, D)
        return pred, z_tgt, ctx_repr


def build_model(cfg):
    """Construct a JEPA from a tjepa.yaml-style config dict.

    Enforces the predictor bottleneck: if the configured predictor width is >= the encoder
    width (which would break the asymmetry that prevents collapse), it is clamped to half the
    encoder width and the head count is adjusted to ~32-dim heads. This keeps the invariant
    valid across the embed-dim ablation grid {128,256,512,768} without manual per-cell tuning.
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
        print(f"[build_model] predictor width clamped to {pdim} (heads {pheads}) "
              f"to stay narrower than encoder width {edim}.")
    if pdim % pheads != 0:
        pheads = max(1, pdim // 32)
    return JEPA(
        objective=cfg.get("objective", "temporal_jepa"),
        patch_size=enc["patch_size"], embed_dim=edim, depth=enc["depth"],
        num_heads=enc["num_heads"], temporal_depth=enc.get("temporal_depth", 4),
        pred_dim=pdim, pred_depth=pred["depth"], pred_heads=pheads,
        horizon=temp.get("horizon", 1), min_context=temp.get("min_context", 4),
    )
