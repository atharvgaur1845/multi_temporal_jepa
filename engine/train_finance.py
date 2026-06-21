"""Training drivers for the finance pipeline: Temporal/Spatial FinanceJEPA + MAE/BYOL/SimCLR.

All five objectives train the SAME PanelEncoder backbone (cross-asset ViT + temporal transformer)
so the downstream probes read every method through one uniform pathway — the equal-everything-but-
the-objective protocol from report.md §2, transferred to markets. The JEPA loop reuses the satellite
schedulers, EMA, latent loss, VICReg regularizer and collapse diagnostics unchanged; only the data
modality and the (label-free) augmentation differ.

Each trainer returns (encoder, use_temporal) ready for eval/finance_tasks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.diagnostics import collapse_metrics
from engine.ema import ema_update, momentum_schedule
from engine.train_jepa import _lr_at, _wd_at
from models.finance_encoder import PanelEncoder
from models.finance_jepa import build_finance_model
from objectives.baselines.byol import byol_loss, mlp_head
from objectives.baselines.simclr import nt_xent_loss, projector
from objectives.jepa_loss import jepa_latent_loss, variance_covariance_reg


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _new_backbone(cfg, meta, device):
    enc = cfg["encoder"]
    # Baselines (esp. BYOL: 2 backbones x 2 views; SimCLR: 2 views) encode every frame of the window
    # for the global pool, so their activation memory is several x the JEPA path. Force gradient
    # checkpointing on the baseline backbone to bound it (numerically identical, ~25% slower) — this
    # lets BYOL/SimCLR fit the same batch as the JEPA cells on an 8 GB card.
    return PanelEncoder(num_assets=meta["num_assets"], num_features=meta["num_features"],
                        embed_dim=enc["embed_dim"], depth=enc["depth"], num_heads=enc["num_heads"],
                        temporal_depth=enc.get("temporal_depth", 4),
                        grad_checkpoint=True,
                        temporal_period=cfg.get("temporal", {}).get("period", 366)).to(device)


def _jitter(data, sigma):
    """Label-free pretraining augmentation: additive Gaussian noise on the standardized features
    (analogue of the satellite flip aug; price-series have no spatial symmetry to flip)."""
    if sigma <= 0:
        return data
    return data + sigma * torch.randn_like(data)


def _pool_global(backbone, batch):
    """Global window embedding (B, D): spatial-encode each day, mean over assets, masked-mean over
    days. Spatial-only (matches the probe's use_temporal=False pathway for baselines)."""
    data, pad_mask = batch["data"], batch["pad_mask"]
    B, W, N, Fc = data.shape
    flat = data.reshape(B * W, N, Fc)
    tok = backbone.encode_full(flat).mean(dim=1).reshape(B, W, backbone.embed_dim)   # (B,W,D)
    m = pad_mask.float()[:, :, None]
    return (tok * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)                         # (B, D)


def _two_views(batch, sigma=0.1):
    """Two augmented views (feature jitter + per-feature scale jitter); dates/pad_mask shared."""
    def aug(data):
        scale = 1.0 + 0.1 * (torch.rand(data.shape[-1], device=data.device) - 0.5)
        return data * scale + sigma * torch.randn_like(data)
    v1 = dict(batch); v1["data"] = aug(batch["data"])
    v2 = dict(batch); v2["data"] = aug(batch["data"])
    return v1, v2


def _opt(params, cfg):
    return torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                             weight_decay=cfg["optim"].get("weight_decay_start", 0.04))


# ------------------------------------------------------------------ JEPA (temporal / spatial)
def train_finance_jepa(loader, cfg, meta, device, logger=print):
    """Pretrain Temporal or Spatial FinanceJEPA. Returns (target_encoder, use_temporal=True)."""
    model = build_finance_model(cfg, meta).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"].get("weight_decay_start", 0.04))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["optim"].get("amp", True))
    loss_cfg, ema_cfg, log_cfg, optim_cfg = cfg["loss"], cfg["ema"], cfg["log"], cfg["optim"]
    epochs = optim_cfg["epochs"]
    ga = max(1, optim_cfg.get("grad_accum", 1))
    total_steps = epochs * len(loader)
    warmup_steps = optim_cfg.get("warmup_epochs", 0) * len(loader)
    base_lr, min_lr = optim_cfg["lr"], optim_cfg.get("min_lr", 0.0)
    wd0, wd1 = optim_cfg.get("weight_decay_start", 0.04), optim_cfg.get("weight_decay_end", 0.04)
    sigma = optim_cfg.get("jitter", 0.05)
    var_c, cov_c = loss_cfg.get("var_coeff", 1.0), loss_cfg.get("cov_coeff", 0.04)

    step = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        model.train()
        for it, batch in enumerate(loader):
            batch = _to_device(batch, device)
            if optim_cfg.get("augment", True):
                batch = dict(batch); batch["data"] = _jitter(batch["data"], sigma)
            lr = _lr_at(step, total_steps, warmup_steps, base_lr, min_lr)
            wd = _wd_at(step, total_steps, wd0, wd1)
            for g in opt.param_groups:
                g["lr"], g["weight_decay"] = lr, wd
            with torch.autocast(device_type=device.type, enabled=optim_cfg.get("amp", True)):
                pred, target, ctx = model(batch)
                loss = jepa_latent_loss(pred, target,
                                        norm_target=loss_cfg.get("target_layernorm", True),
                                        loss_type=loss_cfg.get("type", "l2"))
                if var_c or cov_c:
                    std_l, cov_l = variance_covariance_reg(ctx.float())
                    loss = loss + var_c * std_l + cov_c * cov_l
            scaler.scale(loss / ga).backward()
            if (it + 1) % ga == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                m = momentum_schedule(step, total_steps, ema_cfg["base_momentum"],
                                      ema_cfg["final_momentum"])
                ema_update(model.context_encoder, model.target_encoder, m)
            if step % log_cfg.get("diagnostics_every", 50) == 0:
                diag = collapse_metrics(ctx, pred=pred, target=target)
                logger(f"[{cfg['objective']}] step {step} loss {loss.item():.4f} lr {lr:.2e} "
                       f"std {diag['per_dim_std']:.3f} effrank {diag['effective_rank']:.1f} "
                       f"varratio {diag.get('variance_ratio', float('nan')):.3f}")
            step += 1
    return model.target_encoder, True


# ------------------------------------------------------------------ MAE
def train_mae_fin(loader, cfg, meta, device, logger=print):
    """Masked cross-section autoencoder: hide a fraction of a day's assets, reconstruct their raw
    features from the (full-frame, masked-input) spatial encoding. Trains the spatial backbone."""
    backbone = _new_backbone(cfg, meta, device)
    decoder = nn.Linear(cfg["encoder"]["embed_dim"], meta["num_features"]).to(device)
    mask_ratio = cfg.get("mae", {}).get("mask_ratio", 0.5)
    N = meta["num_assets"]
    n_mask = max(1, int(round(mask_ratio * N)))
    opt = _opt(list(backbone.parameters()) + list(decoder.parameters()), cfg)
    epochs, ga = cfg["optim"]["epochs"], max(1, cfg["optim"].get("grad_accum", 1))
    micro = ostep = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            data, pad_mask = batch["data"], batch["pad_mask"]
            f = int(torch.randint(0, int(pad_mask.sum(1).min().item()), (1,)).item())
            frame = data[:, f]                                       # (B, N, F)
            idx = torch.randperm(N, device=device)[:n_mask]
            masked = frame.clone(); masked[:, idx] = 0.0             # hide masked assets' features
            tok = backbone.encode_full(masked)                      # (B, N, D)
            recon = decoder(tok)                                    # (B, N, F)
            loss = F.mse_loss(recon[:, idx], frame[:, idx])
            (loss / ga).backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
                if ostep % cfg["log"].get("diagnostics_every", 50) == 0:
                    logger(f"[mae] opt-step {ostep} loss {loss.item():.4f}")
                ostep += 1
        logger(f"[mae] epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    return backbone, False


# ------------------------------------------------------------------ BYOL
def train_byol_fin(loader, cfg, meta, device, logger=print):
    D = cfg["encoder"]["embed_dim"]
    online = _new_backbone(cfg, meta, device)
    target = _new_backbone(cfg, meta, device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    proj_o, pred_o = mlp_head(D, 4 * 256, 256).to(device), mlp_head(256, 4 * 256, 256).to(device)
    proj_t = mlp_head(D, 4 * 256, 256).to(device)
    proj_t.load_state_dict(proj_o.state_dict())
    for p in proj_t.parameters():
        p.requires_grad_(False)
    opt = _opt(list(online.parameters()) + list(proj_o.parameters()) + list(pred_o.parameters()), cfg)
    epochs, ga = cfg["optim"]["epochs"], max(1, cfg["optim"].get("grad_accum", 1))
    opt_steps_total = epochs * (len(loader) // ga + 1)
    micro = ostep = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            v1, v2 = _two_views(batch)
            p1, p2 = pred_o(proj_o(_pool_global(online, v1))), pred_o(proj_o(_pool_global(online, v2)))
            with torch.no_grad():
                t1, t2 = proj_t(_pool_global(target, v1)), proj_t(_pool_global(target, v2))
            loss = byol_loss(p1, t2) + byol_loss(p2, t1)
            (loss / ga).backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
                m = momentum_schedule(ostep, opt_steps_total, cfg["ema"]["base_momentum"],
                                      cfg["ema"]["final_momentum"])
                ema_update(online, target, m); ema_update(proj_o, proj_t, m)
                ostep += 1
        logger(f"[byol] epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    return online, False


# ------------------------------------------------------------------ SimCLR
def train_simclr_fin(loader, cfg, meta, device, logger=print):
    D = cfg["encoder"]["embed_dim"]
    backbone = _new_backbone(cfg, meta, device)
    proj = projector(D, 4 * 256, 256).to(device)
    temp = cfg.get("simclr", {}).get("temperature", 0.5)
    opt = _opt(list(backbone.parameters()) + list(proj.parameters()), cfg)
    epochs, ga = cfg["optim"]["epochs"], max(1, cfg["optim"].get("grad_accum", 1))
    micro = ostep = 0
    opt.zero_grad(set_to_none=True)
    for ep in range(epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            v1, v2 = _two_views(batch)
            z = torch.cat([proj(_pool_global(backbone, v1)), proj(_pool_global(backbone, v2))], dim=0)
            loss = nt_xent_loss(z, temperature=temp)
            (loss / ga).backward()
            micro += 1
            if micro % ga == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
                ostep += 1
        logger(f"[simclr] epoch {ep + 1}/{epochs} loss {loss.item():.4f}")
    return backbone, False


FIN_TRAINERS = {"mae": train_mae_fin, "byol": train_byol_fin, "simclr": train_simclr_fin}
