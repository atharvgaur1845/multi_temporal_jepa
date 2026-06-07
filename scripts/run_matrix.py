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
    """Yield (name, overrides) cells, FRONT-LOADED by importance so --max-cells gets the headline
    result first. Order: (1) main equal-compute comparison [temporal h1 vs spatial vs baselines],
    (2) horizon study, (3) ablations. Overrides patch the base tjepa config."""
    cells = []
    # 1) MAIN COMPARISON (the contribution) — temporal@h1 vs spatial JEPA vs MAE/BYOL/SimCLR.
    cells.append(("tjepa_h1", {"objective": "temporal_jepa", "temporal": {"horizon": 1}}))
    for obj in ("spatial_jepa", "mae", "byol", "simclr"):
        cells.append((obj, {"objective": obj}))
    # 2) horizon study (h1 already covered above)
    for h in (2, 4, 8):
        cells.append((f"tjepa_h{h}", {"objective": "temporal_jepa", "temporal": {"horizon": h}}))
    # 3) ablations: predictor depth + embed dim (temporal jepa, horizon 1)
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
    ap.add_argument("--device", default=None, help="override config device, e.g. cuda:1")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-epochs", type=int, default=15)
    ap.add_argument("--knn", action="store_true", help="also record parcel k-NN per cell")
    ap.add_argument("--test", action="store_true", help="probe on test_folds (default: val_folds)")
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
    from engine.train_baselines import TRAINERS
    from engine.train_jepa import train_one_epoch
    from eval.linear_probe import linear_probe_segmentation
    from models.jepa import build_model
    from utils.device import resolve_device
    from utils.gpu_hours import GpuHourMeter
    from utils.seed import seed_everything

    data_cfg = load_yaml(args.data)
    device = resolve_device(args.device or base.get("device"))
    print(f"[run_matrix] device = {device}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    msl = data_cfg.get("max_seq_len")
    eval_folds = data_cfg["test_folds"] if args.test else data_cfg["val_folds"]
    eval_split = "test" if args.test else "val"
    print(f"[run_matrix] probing on {eval_split} folds {eval_folds}")
    train = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False,
                   max_seq_len=msl, subsample_train=True)
    mean, std = compute_band_stats(train, max_samples=200)
    train.norm_mean, train.norm_std = mean, std
    eval_ds = PASTIS(data_cfg["root"], folds=eval_folds, return_label=True,
                     norm_mean=mean, norm_std=std, max_seq_len=msl)
    probe_tr = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=True,
                      norm_mean=mean, norm_std=std, max_seq_len=msl)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "objective", "eval_split", "miou_linear", "miou_conv",
                         "knn_acc", "gpu_hours", "peak_mem_gb"])
        from torch.utils.data import DataLoader as DL
        for name, ov in run:
            cfg = _deep_update(base, ov)
            obj = cfg["objective"]
            # per-cell memory safety: heavy cells (large embed_dim, esp. at patch_size 8) get a
            # smaller per-step batch with proportionally more grad_accum (effective batch held).
            if cfg["encoder"]["embed_dim"] >= 768:
                cfg["optim"]["batch_size"] = max(8, cfg["optim"]["batch_size"] // 2)
                cfg["optim"]["grad_accum"] = cfg["optim"]["grad_accum"] * 2
                print(f"[run_matrix] {name}: embed>=768 -> batch {cfg['optim']['batch_size']}, "
                      f"accum {cfg['optim']['grad_accum']} (memory safety)")
            seed_everything(cfg["log"].get("seed", 0))
            loader = DataLoader(train, batch_size=cfg["optim"]["batch_size"], shuffle=True,
                                num_workers=8, collate_fn=collate_variable_length, drop_last=True)
            meter = GpuHourMeter(device); meter.start()

            if obj in ("temporal_jepa", "spatial_jepa"):
                model = build_model(cfg).to(device)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                        lr=cfg["optim"]["lr"])
                scaler = torch.amp.GradScaler(device.type, enabled=cfg["optim"].get("amp", True))
                total = cfg["optim"]["epochs"] * len(loader)
                step = 0
                for _ in range(cfg["optim"]["epochs"]):
                    step = train_one_epoch(model, loader, opt, scaler, cfg, step, total, device)
                encoder, use_temporal = model.target_encoder, True
            else:  # mae / byol / simclr — spatial-only backbone, probed without temporal encoder
                encoder = TRAINERS[obj](loader, cfg, device)
                use_temporal = False
            stats = meter.stop()

            el = DL(eval_ds, batch_size=8, collate_fn=collate_variable_length)
            tl = DL(probe_tr, batch_size=8, shuffle=True, collate_fn=collate_variable_length)
            # both heads, matched across all methods -> linear (strict) + conv (headline) columns
            mious = {}
            for h in ("linear", "conv"):
                res = linear_probe_segmentation(encoder, tl, el, num_classes=data_cfg["num_classes"],
                                                ignore_index=data_cfg["ignore_index"],
                                                epochs=args.probe_epochs, device=device,
                                                use_temporal=use_temporal, head=h)
                mious[h] = res["miou"]

            knn_acc = ""
            if args.knn:
                from eval.knn import knn_accuracy, parcel_embeddings
                Xtr, ytr = parcel_embeddings(encoder, tl, device=device)
                Xev, yev = parcel_embeddings(encoder, el, device=device)
                knn_acc = round(knn_accuracy(Xtr, ytr, Xev, yev, k=20), 4)

            writer.writerow([name, obj, eval_split, round(mious["linear"], 4),
                             round(mious["conv"], 4), knn_acc,
                             round(stats["gpu_hours"], 3), round(stats["peak_mem_gb"], 2)])
            f.flush()
            print(f"[run_matrix] {name}: linear={mious['linear']*100:.2f} conv={mious['conv']*100:.2f} "
                  f"knn={knn_acc} gpu_h={stats['gpu_hours']:.2f}")


if __name__ == "__main__":
    main()
