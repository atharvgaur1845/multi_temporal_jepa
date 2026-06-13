"""Training drivers for the MAE / BYOL / SimCLR baselines (M4).

All three train the SAME shared `SITSEncoder` spatial backbone (patch_embed + spatial_vit) so
the downstream linear probe reads them through one uniform pathway
(`extract_dense_features(..., use_temporal=False)`) — a fair, equal-compute comparison against
the JEPA variants. None of them trains the temporal encoder (that is the JEPA contribution).

Each `train_*` returns the trained backbone, ready for eval/linear_probe.
"""
from __future__ import annotations

import torch

from engine.ema import ema_update, momentum_schedule
from models.jepa import SITSEncoder
from objectives.baselines.byol import byol_loss, mlp_head
from objectives.baselines.mae import MAEModel
from objectives.baselines.simclr import nt_xent_loss, projector


def _new_backbone(cfg, device):
    enc = cfg["encoder"]
    return SITSEncoder(patch_size=enc["patch_size"], embed_dim=enc["embed_dim"],
                       depth=enc["depth"], num_heads=enc["num_heads"],
                       temporal_depth=enc.get("temporal_depth", 4),
                       grad_checkpoint=enc.get("grad_checkpoint", False)).to(device)


def _two_views(batch):
    """Two augmented views of the batch (flips + band jitter); dates/pad_mask shared."""
    def aug(data):
        if torch.rand(1).item() < 0.5:
            data = torch.flip(data, dims=[-1])
        if torch.rand(1).item() < 0.5:
            data = torch.flip(data, dims=[-2])
        jitter = 1.0 + 0.1 * (torch.rand(data.shape[2], device=data.device) - 0.5)
        return data * jitter.view(1, 1, -1, 1, 1)
    v1 = dict(batch); v1["data"] = aug(batch["data"])
    v2 = dict(batch); v2["data"] = aug(batch["data"])
    return v1, v2


def _pool_global(backbone, batch, frame_chunk=64):
    """Global embedding (B, D): spatial-encode each real frame, mean over tokens, masked-mean
    over time. Spatial-only (matches the probe's use_temporal=False pathway).

    Frames are encoded in CHUNKS of `frame_chunk` so peak memory is bounded by one chunk, not by
    B*T (which is huge for BYOL: two backbones x two views). With gradient checkpointing on, the
    autograd graph keeps only the small per-frame embeddings, so this caps memory without changing
    the effective batch or the result.
    """
    data, pad_mask = batch["data"], batch["pad_mask"]
    B, T = data.shape[:2]
    flat = data.reshape(B * T, *data.shape[2:])
    chunks = [backbone.encode_full(flat[i:i + frame_chunk]).mean(dim=1)
              for i in range(0, flat.shape[0], frame_chunk)]
    tok = torch.cat(chunks, dim=0).reshape(B, T, backbone.embed_dim)                 # (B,T,D)
    m = pad_mask.float()[:, :, None]
    return (tok * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)                        # (B, D)


def _opt(params, cfg):
    return torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                             weight_decay=cfg["optim"].get("weight_decay_start", 0.04))


def train_byol(loader, cfg, device, logger=print):
    """BYOL: online backbone+projector+predictor vs EMA target backbone+projector (symmetric)."""
    D = cfg["encoder"]["embed_dim"]
    online = _new_backbone(cfg, device)
    target = _new_backbone(cfg, device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    proj_o = mlp_head(D, 4 * 256, 256).to(device)
    pred_o = mlp_head(256, 4 * 256, 256).to(device)
    proj_t = mlp_head(D, 4 * 256, 256).to(device)
    proj_t.load_state_dict(proj_o.state_dict())
    for p in proj_t.parameters():
        p.requires_grad_(False)

    opt = _opt(list(online.parameters()) + list(proj_o.parameters()) + list(pred_o.parameters()), cfg)
    epochs = cfg["optim"]["epochs"]
    ga = max(1, cfg["optim"].get("grad_accum", 1))            # match JEPA's effective batch
    opt_steps_total = epochs * (len(loader) // ga)
    opt.zero_grad(set_to_none=True)
    micro = ostep = 0
    for ep in range(epochs):
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            v1, v2 = _two_views(batch)
            p1 = pred_o(proj_o(_pool_global(online, v1)))
            p2 = pred_o(proj_o(_pool_global(online, v2)))
            with torch.no_grad():
                t1 = proj_t(_pool_global(target, v1))
                t2 = proj_t(_pool_global(target, v2))
            loss = byol_loss(p1, t2) + byol_loss(p2, t1)
            (loss / ga).backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
                # EMA after the optimizer step, momentum scheduled over optimizer steps
                m = momentum_schedule(ostep, opt_steps_total,
                                      cfg["ema"]["base_momentum"], cfg["ema"]["final_momentum"])
                ema_update(online, target, m); ema_update(proj_o, proj_t, m)
                if ostep % cfg["log"].get("diagnostics_every", 50) == 0:
                    logger(f"[byol] opt-step {ostep} loss {loss.item():.4f}")
                ostep += 1
        logger(f"[byol] epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    return online


def train_simclr(loader, cfg, device, logger=print):
    """SimCLR: backbone + projector, NT-Xent over two views.

    NOTE: grad-accum below matches JEPA's *optimization* batch (192), but NT-Xent negatives are
    still per-micro-batch (2*batch_size), since accumulation can't pool negatives across steps —
    a true large-negative SimCLR needs a memory bank / all-gather (out of scope). So SimCLR's
    contrastive batch stays at the loader batch size; we note this in the report.
    """
    D = cfg["encoder"]["embed_dim"]
    backbone = _new_backbone(cfg, device)
    proj = projector(D, 4 * 256, 256).to(device)
    temp = cfg.get("simclr", {}).get("temperature", 0.5)
    opt = _opt(list(backbone.parameters()) + list(proj.parameters()), cfg)
    epochs = cfg["optim"]["epochs"]
    ga = max(1, cfg["optim"].get("grad_accum", 1))
    opt.zero_grad(set_to_none=True)
    micro = ostep = 0
    for ep in range(epochs):
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            v1, v2 = _two_views(batch)
            z = torch.cat([proj(_pool_global(backbone, v1)), proj(_pool_global(backbone, v2))], dim=0)
            loss = nt_xent_loss(z, temperature=temp)
            (loss / ga).backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
                if ostep % cfg["log"].get("diagnostics_every", 50) == 0:
                    logger(f"[simclr] opt-step {ostep} loss {loss.item():.4f}")
                ostep += 1
        logger(f"[simclr] epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    return backbone


def train_mae(loader, cfg, device, logger=print):
    """MAE: per-frame masked pixel reconstruction over the shared backbone."""
    backbone = _new_backbone(cfg, device)
    model = MAEModel(backbone, patch_size=cfg["encoder"]["patch_size"],
                     mask_ratio=cfg.get("mae", {}).get("mask_ratio", 0.75)).to(device)
    opt = _opt(model.parameters(), cfg)
    epochs = cfg["optim"]["epochs"]
    ga = max(1, cfg["optim"].get("grad_accum", 1))            # match JEPA's effective batch
    opt.zero_grad(set_to_none=True)
    micro = ostep = 0
    for ep in range(epochs):
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            data, pad_mask = batch["data"], batch["pad_mask"]
            f = int(torch.randint(0, int(pad_mask.sum(1).min().item()), (1,)).item())
            loss = model(data[:, f])
            (loss / ga).backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
                if ostep % cfg["log"].get("diagnostics_every", 50) == 0:
                    logger(f"[mae] opt-step {ostep} loss {loss.item():.4f}")
                ostep += 1
        logger(f"[mae] epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    return backbone


TRAINERS = {"mae": train_mae, "byol": train_byol, "simclr": train_simclr}
