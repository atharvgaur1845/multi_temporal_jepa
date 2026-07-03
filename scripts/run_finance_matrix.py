"""Drive the finance experiment matrix: pretrain every objective on the SAME PanelEncoder, freeze,
and evaluate the five downstream tasks — the markets analogue of scripts/run_matrix.py.

Each cell = (pretrain -> freeze -> evaluate_all) with a fixed seed and a GpuHourMeter; one CSV row
per cell with every downstream metric. Front-loaded so --max-cells gets the headline first:

    1) main comparison : temporal JEPA vs spatial JEPA vs MAE/BYOL/SimCLR (+ random-init control)
    2) horizon study   : Δ ∈ {5, 20} trading days
    3) ablation        : VICReg off (collapse check)

    python scripts/run_finance_matrix.py --config configs/model/fjepa.yaml \
        --data configs/data/finance.yaml --device cuda:0 --max-cells 6 --resume
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_yaml  # noqa: E402


def enumerate_cells():
    cells = []
    # 1) MAIN COMPARISON (the contribution)
    cells.append(("tjepa_h1", {"objective": "temporal_jepa", "temporal": {"horizon": 1}}))
    # Phase 4: distributional (heteroscedastic beta-NLL) temporal JEPA — the finance-rescue candidate.
    cells.append(("tjepa_dist", {"objective": "temporal_jepa", "temporal": {"horizon": 1},
                                 "predictor": {"distributional": True},
                                 "loss": {"type": "beta_nll", "beta": 0.5}}))
    for obj in ("spatial_jepa", "mae", "byol", "simclr"):
        cells.append((obj, {"objective": obj}))
    cells.append(("random", {"objective": "temporal_jepa", "_random_init": True}))  # control
    # 2) horizon study
    for h in (5, 20):
        cells.append((f"tjepa_h{h}", {"objective": "temporal_jepa", "temporal": {"horizon": h}}))
    # 3) VICReg ablation (expected to collapse -> downstream tasks degrade)
    cells.append(("tjepa_noreg",
                  {"objective": "temporal_jepa", "loss": {"var_coeff": 0.0, "cov_coeff": 0.0}}))
    return cells


def _deep_update(base, overrides):
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


METRICS = ["regime_acc", "regime_f1", "vol_r2", "vol_ic", "anom_auroc", "anom_ap",
           "clust_ari", "clust_nmi", "clust_silhouette", "fcast_dir_acc", "fcast_ret_ic", "emb_std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/fjepa.yaml")
    ap.add_argument("--data", default="configs/data/finance.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default="runs/finance")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (quick runs)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    base = load_yaml(args.config)
    data_cfg = load_yaml(args.data)
    seed = args.seed if args.seed is not None else int(base["log"].get("seed", 0))
    if args.epochs is not None:
        base["optim"]["epochs"] = args.epochs
    base.setdefault("log", {})["seed"] = seed
    tag = "" if args.seed is None else f"__s{seed}"
    out_path = args.out or f"runs/finance_results{tag}.csv"

    cells = enumerate_cells()
    run = cells if args.max_cells is None else cells[: args.max_cells]
    skipped = [] if args.max_cells is None else cells[args.max_cells:]
    print(f"[run_finance] {len(cells)} cells; running {len(run)}; skipping {len(skipped)}")
    for name, _ in skipped:
        print(f"[run_finance] SKIPPED (budget): {name}")
    if args.dry_run:
        for name, ov in run:
            print(f"[run_finance] would run: {name}  overrides={ov}")
        return

    import torch
    from torch.utils.data import DataLoader
    from data.finance_dataset import make_finance_datasets, collate_windows
    from engine.train_finance import FIN_TRAINERS, train_finance_jepa
    from eval.finance_tasks import evaluate_all
    from models.finance_jepa import build_finance_model
    from utils.device import resolve_device
    from utils.gpu_hours import GpuHourMeter
    from utils.seed import seed_everything

    device = resolve_device(args.device or base.get("device"))
    pre_ds, ptr_ds, pte_ds, meta = make_finance_datasets(
        root=data_cfg["root"], window=data_cfg["window"],
        train_stride=data_cfg.get("train_stride", 1), eval_stride=data_cfg.get("eval_stride", 5),
        train_end=data_cfg.get("train_end", 20171231), vol_horizon=data_cfg.get("vol_horizon", 20),
        anom_horizon=data_cfg.get("anom_horizon", 5), allow_synth=data_cfg.get("allow_synth", True),
        seed=seed)
    print(f"[run_finance] device={device} seed={seed} source={meta['source']} "
          f"assets={meta['num_assets']} feats={meta['num_features']} "
          f"train_windows={meta['n_train_windows']} test_windows={meta['n_test_windows']}")
    ptr_loader = DataLoader(ptr_ds, batch_size=128, collate_fn=collate_windows)
    pte_loader = DataLoader(pte_ds, batch_size=128, collate_fn=collate_windows)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    resume_existing = args.resume and os.path.exists(out_path)
    with open(out_path, "a" if resume_existing else "w", newline="") as f:
        writer = csv.writer(f)
        if not resume_existing:
            writer.writerow(["cell", "objective", "seed", *METRICS, "gpu_hours"])
        for name, ov in run:
            ckpt = os.path.join(args.ckpt_dir, f"{name}{tag}.pt")
            if args.resume and os.path.exists(ckpt):
                print(f"[run_finance] {name}: already complete -> skip")
                continue
            cfg = _deep_update(base, ov)
            random_init = cfg.pop("_random_init", False)
            obj = cfg["objective"]
            seed_everything(seed)
            loader = DataLoader(pre_ds, batch_size=cfg["optim"]["batch_size"], shuffle=True,
                                collate_fn=collate_windows, drop_last=True)
            meter = GpuHourMeter(device); meter.start()

            if random_init:                                   # untrained control
                model = build_finance_model(cfg, meta).to(device)
                encoder, use_temporal = model.target_encoder, True
                del model
            elif obj in ("temporal_jepa", "spatial_jepa"):
                encoder, use_temporal = train_finance_jepa(loader, cfg, meta, device)
            else:
                encoder, use_temporal = FIN_TRAINERS[obj](loader, cfg, meta, device)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stats = meter.stop()

            res = evaluate_all(encoder, use_temporal, ptr_loader, pte_loader, meta, device, seed)
            writer.writerow([name, obj, seed, *[round(res.get(k, float("nan")), 4) for k in METRICS],
                             round(stats["gpu_hours"], 4)])
            f.flush()
            torch.save({"encoder": encoder.state_dict(), "cfg": cfg, "objective": obj,
                        "use_temporal": use_temporal, "meta": meta, "seed": seed}, ckpt)
            print(f"[run_finance] {name}: regime_acc={res['regime_acc']:.3f} "
                  f"vol_r2={res['vol_r2']:.3f} anom_auroc={res['anom_auroc']:.3f} "
                  f"clust_nmi={res['clust_nmi']:.3f} fcast_ic={res['fcast_ret_ic']:.3f} "
                  f"gpu_h={stats['gpu_hours']:.3f}")
            del encoder, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[run_finance] wrote {out_path}")


if __name__ == "__main__":
    main()
