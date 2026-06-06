"""Drive the full experiment matrix (M5) and log cost honestly.

Matrix (full, as specified):
    objectives : {temporal_jepa, spatial_jepa, mae, byol, simclr}
    horizon    : Δ ∈ {1, 2, 4, 8}                 (temporal_jepa only)
    ablations  : predictor depth {1,2,4,6}, embed dim {128,256,512,768}

Each cell = (pretrain -> freeze -> {linear_probe mIoU, k-NN, few-shot}) with a fixed seed and a
GpuHourMeter. THIS IS GPU-WEEKS on one card — order cells to get signal early.

    IMPORTANT: if you cap/skip cells for budget, LOG exactly which cells were skipped. A matrix
    that silently drops cells looks complete but isn't (plan §5 M5).

This driver builds per-cell config dicts, runs pretrain + eval, and appends a row to a CSV.
Use --dry-run to print the cell list (and which are skipped by --max-cells) without training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_yaml  # noqa: E402


def enumerate_cells():
    """Yield (name, overrides) cells for the full matrix. Overrides patch the base tjepa config."""
    cells = []
    # horizon study (temporal jepa)
    for h in (1, 2, 4, 8):
        cells.append((f"tjepa_h{h}", {"objective": "temporal_jepa", "temporal": {"horizon": h}}))
    # baselines
    for obj in ("spatial_jepa", "mae", "byol", "simclr"):
        cells.append((obj, {"objective": obj}))
    # ablations: predictor depth + embed dim (temporal jepa, horizon 1)
    for d in (1, 2, 4, 6):
        cells.append((f"tjepa_preddepth{d}",
                      {"objective": "temporal_jepa", "predictor": {"depth": d}}))
    for dim in (128, 256, 512, 768):
        cells.append((f"tjepa_dim{dim}",
                      {"objective": "temporal_jepa", "encoder": {"embed_dim": dim}}))
    return cells


def _deep_update(base, overrides):
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/tjepa.yaml")
    ap.add_argument("--data", default="configs/data/pastis.yaml")
    ap.add_argument("--out", default="runs/matrix_results.csv")
    ap.add_argument("--max-cells", type=int, default=None, help="cap cells (logs the rest as SKIPPED)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_yaml(args.config)
    cells = enumerate_cells()
    run = cells if args.max_cells is None else cells[: args.max_cells]
    skipped = [] if args.max_cells is None else cells[args.max_cells:]

    print(f"[run_matrix] {len(cells)} cells total; running {len(run)}; skipping {len(skipped)}")
    for name, _ in skipped:
        print(f"[run_matrix] SKIPPED (budget): {name}")

    if args.dry_run:
        for name, ov in run:
            print(f"[run_matrix] would run: {name}  overrides={ov}")
        return

    # heavy imports only when actually training
    import torch
    from torch.utils.data import DataLoader
    from data.pastis_dataset import PASTIS, collate_variable_length
    from data.transforms import compute_band_stats
    from engine.train_jepa import train_one_epoch
    from eval.linear_probe import linear_probe_segmentation
    from models.jepa import build_model
    from utils.gpu_hours import GpuHourMeter
    from utils.seed import seed_everything

    data_cfg = load_yaml(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False)
    mean, std = compute_band_stats(train, max_samples=200)
    train.norm_mean, train.norm_std = mean, std
    val = PASTIS(data_cfg["root"], folds=data_cfg["val_folds"], return_label=True,
                 norm_mean=mean, norm_std=std)
    probe_tr = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=True,
                      norm_mean=mean, norm_std=std)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "objective", "miou", "gpu_hours", "peak_mem_gb"])
        for name, ov in run:
            cfg = _deep_update(base, ov)
            seed_everything(cfg["log"].get("seed", 0))
            loader = DataLoader(train, batch_size=cfg["optim"]["batch_size"], shuffle=True,
                                num_workers=8, collate_fn=collate_variable_length, drop_last=True)
            # NOTE: JEPA-family cells only here; MAE/BYOL/SimCLR need their own train fns (M4).
            if cfg["objective"] not in ("temporal_jepa", "spatial_jepa"):
                print(f"[run_matrix] {name}: baseline training not wired in this driver; SKIPPED")
                continue
            model = build_model(cfg).to(device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg["optim"]["lr"])
            scaler = torch.cuda.amp.GradScaler(enabled=cfg["optim"].get("amp", True))
            total = cfg["optim"]["epochs"] * len(loader)
            meter = GpuHourMeter(); meter.start()
            step = 0
            for _ in range(cfg["optim"]["epochs"]):
                step = train_one_epoch(model, loader, opt, scaler, cfg, step, total, device)
            stats = meter.stop()

            from torch.utils.data import DataLoader as DL
            vl = DL(val, batch_size=8, collate_fn=collate_variable_length)
            tl = DL(probe_tr, batch_size=8, shuffle=True, collate_fn=collate_variable_length)
            res = linear_probe_segmentation(model.target_encoder, tl, vl,
                                            num_classes=data_cfg["num_classes"],
                                            ignore_index=data_cfg["ignore_index"], epochs=10)
            writer.writerow([name, cfg["objective"], res["miou"],
                             stats["gpu_hours"], stats["peak_mem_gb"]])
            f.flush()
            print(f"[run_matrix] {name}: mIoU={res['miou']:.3f} gpu_h={stats['gpu_hours']:.2f}")


if __name__ == "__main__":
    main()
