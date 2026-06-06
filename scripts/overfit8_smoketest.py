"""M1 HARD GATE: overfit 8 samples and prove the model is learning, not collapsing.

Because we skipped the CIFAR warm-up, this is your fast feedback loop. It must pass before you
train on full PASTIS. Two conditions, BOTH required:

    (1) LEARNING:    loss drops substantially on 8 fixed samples within a few hundred steps.
    (2) NO COLLAPSE: per-dim std stays bounded away from 0, effective rank stays high.

A model can satisfy (1) by collapsing (predicting a constant that the EMA target also drifts
toward) — that is why (2) is non-negotiable.

By default this runs on a SYNTHETIC PASTIS-shaped batch so it is runnable without the 29 GB
download (correctness of the wiring, not of the data). Pass --pastis to use 8 real samples.

Usage:
    python scripts/overfit8_smoketest.py                 # synthetic, temporal_jepa
    python scripts/overfit8_smoketest.py --objective spatial_jepa
    python scripts/overfit8_smoketest.py --pastis        # needs PASTIS_ROOT
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.diagnostics import collapse_metrics  # noqa: E402
from engine.ema import ema_update, momentum_schedule  # noqa: E402
from models.jepa import JEPA  # noqa: E402
from objectives.jepa_loss import jepa_latent_loss  # noqa: E402


def make_synthetic_batch(B=8, T=12, C=10, H=128, W=128, seed=0):
    """A PASTIS-shaped batch: spatially-structured frames so the task is learnable, irregular
    DOY dates, all-real pad mask. Real frames front-packed (as the real collate guarantees)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, 1, C, H // 8, W // 8, generator=g)
    base = base.repeat(1, T, 1, 8, 8)  # upsample so neighboring pixels correlate
    drift = 0.1 * torch.randn(B, T, C, H, W, generator=g)
    data = base + drift
    dates = torch.stack([torch.sort(torch.randint(1, 366, (T,), generator=g)).values
                         for _ in range(B)])
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    return {"data": data, "dates": dates, "pad_mask": pad_mask, "label": None}


def load_pastis_batch(B=8):
    from torch.utils.data import DataLoader
    from data.pastis_dataset import PASTIS, collate_variable_length
    root = os.environ["PASTIS_ROOT"]
    ds = PASTIS(root, folds=[1], return_label=False)
    loader = DataLoader(ds, batch_size=B, shuffle=True, collate_fn=collate_variable_length)
    return next(iter(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="temporal_jepa",
                    choices=["temporal_jepa", "spatial_jepa"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pastis", action="store_true")
    ap.add_argument("--std-floor", type=float, default=0.05)
    ap.add_argument("--rank-floor", type=float, default=2.0)
    ap.add_argument("--device", default=None, help="e.g. cuda:1 (default: auto)")
    args = ap.parse_args()

    from utils.device import resolve_device
    device = resolve_device(args.device)
    batch = load_pastis_batch() if args.pastis else make_synthetic_batch()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # small model so it overfits 8 samples quickly
    model = JEPA(objective=args.objective, embed_dim=128, depth=2, num_heads=4,
                 temporal_depth=2, pred_dim=96, pred_depth=2, pred_heads=4,
                 horizon=1, min_context=4).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    first_loss = None
    last = {}
    for step in range(args.steps):
        model.train()
        pred, target, ctx = model(batch)
        loss = jepa_latent_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        m = momentum_schedule(step, args.steps)
        ema_update(model.context_encoder, model.target_encoder, m)
        if first_loss is None:
            first_loss = loss.item()
        if step % 50 == 0 or step == args.steps - 1:
            # collapse measured on the trainable context embedding (see diagnostics rationale)
            diag = collapse_metrics(ctx, pred=pred, target=target)
            last = diag
            print(f"step {step:4d}  loss {loss.item():.4f}  std {diag['per_dim_std']:.3f}  "
                  f"effrank {diag['effective_rank']:.2f}  varratio {diag['variance_ratio']:.3f}")

    learned = loss.item() < 0.5 * first_loss
    healthy = last["per_dim_std"] > args.std_floor and last["effective_rank"] > args.rank_floor
    print(f"\nLEARNING (loss {first_loss:.3f} -> {loss.item():.3f}): {learned}")
    print(f"NO-COLLAPSE (std>{args.std_floor}, effrank>{args.rank_floor}): {healthy}")
    if learned and healthy:
        print("M1 GATE: PASS")
        sys.exit(0)
    print("M1 GATE: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
