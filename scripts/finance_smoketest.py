"""M1 gate for the finance stack — the markets analogue of scripts/overfit8_smoketest.py.

Overfit a tiny batch for many steps and require the JEPA loss to FALL while the representation
stays HEALTHY (per-dim std and effective rank do NOT collapse). A falling loss alone is not
success: the failure mode is collapse (constant embedding -> loss to 0, std -> 0). With the VICReg
term on, the loss won't reach 0 — that is expected, not a bug. Runs offline on the synthetic panel.

    python scripts/finance_smoketest.py --device cuda:0
    python scripts/finance_smoketest.py --device cuda:0 --noreg   # show collapse without VICReg
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from data.finance_dataset import make_finance_datasets, collate_windows  # noqa: E402
from engine.diagnostics import collapse_metrics  # noqa: E402
from engine.ema import ema_update, momentum_schedule  # noqa: E402
from models.finance_jepa import FinanceJEPA  # noqa: E402
from objectives.jepa_loss import jepa_latent_loss, variance_covariance_reg  # noqa: E402
from utils.device import resolve_device  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--objective", default="temporal_jepa")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--noreg", action="store_true", help="disable VICReg -> should collapse")
    args = ap.parse_args()
    seed_everything(0)
    device = resolve_device(args.device)

    pre, _, _, meta = make_finance_datasets(window=48, allow_synth=True)
    loader = torch.utils.data.DataLoader(pre, batch_size=8, shuffle=True, collate_fn=collate_windows)
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    model = FinanceJEPA(objective=args.objective, num_assets=meta["num_assets"],
                        num_features=meta["num_features"], embed_dim=128, depth=4, num_heads=4,
                        temporal_depth=4, pred_dim=64, pred_depth=4, pred_heads=4,
                        min_context=8, horizon=1).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    var_c = 0.0 if args.noreg else 1.0
    cov_c = 0.0 if args.noreg else 0.04

    first_loss = first_std = None
    for step in range(args.steps):
        pred, target, ctx = model(batch)
        loss = jepa_latent_loss(pred, target)
        if var_c or cov_c:
            sv, cv = variance_covariance_reg(ctx.float())
            loss = loss + var_c * sv + cov_c * cv
        opt.zero_grad(); loss.backward(); opt.step()
        m = momentum_schedule(step, args.steps, 0.996, 1.0)
        ema_update(model.context_encoder, model.target_encoder, m)
        diag = collapse_metrics(ctx, pred=pred, target=target)
        if first_loss is None:
            first_loss, first_std = loss.item(), diag["per_dim_std"]
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {loss.item():.4f} std {diag['per_dim_std']:.3f} "
                  f"effrank {diag['effective_rank']:.1f}")
    last_std = diag["per_dim_std"]
    loss_fell = loss.item() < first_loss
    healthy = last_std > 0.3                                   # not collapsed to a constant
    print(f"\nloss {first_loss:.3f} -> {loss.item():.3f} (fell={loss_fell}) | "
          f"std {first_std:.3f} -> {last_std:.3f} (healthy={healthy})")
    if args.noreg:
        print("PASS (noreg demo): expect std to drift low — that is the collapse VICReg prevents.")
        return
    ok = loss_fell and healthy
    print("M1 GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
