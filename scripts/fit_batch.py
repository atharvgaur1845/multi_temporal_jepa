"""Find the largest per-step batch that fits the GPU for a given config, and print the
batch_size / grad_accum to set (keeping a target effective batch).

Memory scales steeply with batch here because the temporal encoder folds the N spatial tokens
into the batch (B*N sequences), so don't guess — measure. This sweeps batch sizes, stops at the
first OOM, and recommends the largest that fits under your free memory minus a safety margin.

Usage:
    python scripts/fit_batch.py --config configs/model/tjepa_server.yaml --device cuda:1
    python scripts/fit_batch.py --device cuda:1 --eff-batch 192 --margin-gb 5
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa import build_model  # noqa: E402
from objectives.jepa_loss import jepa_latent_loss, variance_covariance_reg  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.device import resolve_device  # noqa: E402


def _measure(cfg, B, T, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    m = build_model(cfg).to(device)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    g = torch.Generator().manual_seed(0)
    b = {"data": torch.randn(B, T, 10, 128, 128, generator=g).to(device),
         "dates": torch.stack([torch.sort(torch.randint(1, 366, (T,), generator=g)).values
                               for _ in range(B)]).to(device),
         "pad_mask": torch.ones(B, T, dtype=torch.bool, device=device), "label": None}
    for _ in range(3):
        with torch.autocast("cuda"):
            pr, tg, cx = m(b)
            loss = jepa_latent_loss(pr, tg)
            sl, cl = variance_covariance_reg(cx.float())
            loss = loss + sl + 0.04 * cl
        opt.zero_grad(); loss.backward(); opt.step()
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    del m, opt, b
    gc.collect(); torch.cuda.empty_cache()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/tjepa_server.yaml")
    ap.add_argument("--data", default="configs/data/pastis.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--eff-batch", type=int, default=192, help="effective batch to keep (sets grad_accum)")
    ap.add_argument("--margin-gb", type=float, default=5.0, help="headroom to leave free (probe + others)")
    ap.add_argument("--batches", default="16,32,48,64,80,96,128",
                    help="comma list of per-step batches to try")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    T = load_yaml(args.data).get("max_seq_len", 32)
    device = resolve_device(args.device or cfg.get("device"))
    free, total = (x / 1e9 for x in torch.cuda.mem_get_info(device))
    budget = free - args.margin_gb
    print(f"[fit_batch] {device}: {free:.1f} GB free / {total:.1f} GB total; "
          f"target ≤ {budget:.1f} GB (margin {args.margin_gb}); T={T}")

    best = None
    for B in [int(x) for x in args.batches.split(",")]:
        try:
            peak = _measure(cfg, B, T, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch {B:>4} -> OOM"); torch.cuda.empty_cache(); break
            raise
        fits = peak <= budget
        print(f"  batch {B:>4} -> {peak:5.1f} GB {'OK' if fits else '> budget'}")
        if fits:
            best = B
        else:
            break

    if best is None:
        print("[fit_batch] even the smallest batch exceeds budget — free more memory or lower the model.")
        return
    # grad_accum to hit the effective batch (>=1)
    ga = max(1, round(args.eff_batch / best))
    print(f"\n[fit_batch] RECOMMEND: batch_size: {best}   grad_accum: {ga}   "
          f"(effective {best*ga}; target {args.eff_batch})")
    print(f"[fit_batch] edit {args.config} optim.batch_size/grad_accum accordingly.")


if __name__ == "__main__":
    main()
