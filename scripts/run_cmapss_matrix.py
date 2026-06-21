"""Drive the C-MAPSS experiment matrix across FD subsets: pretrain every objective on the SAME
PanelEncoder, freeze, and evaluate the five degradation tasks — the industrial analogue of
scripts/run_finance_matrix.py.

Cells (front-loaded): temporal JEPA (Δ=1) vs spatial JEPA vs MAE/BYOL/SimCLR, plus the two
finance-lesson FLOORS — `random` (untrained encoder) and `raw_features` (probes on pooled raw
sensors, no encoder) — then a horizon sweep (Δ=5,20) and the VICReg-off ablation. One CSV row per
(fd, cell) with every downstream metric.

    python scripts/run_cmapss_matrix.py --config configs/model/cjepa.yaml \
        --data configs/data/cmapss.yaml --device cuda:0 --fds FD001 --resume
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
    cells = [
        ("tjepa_h1", {"objective": "temporal_jepa", "temporal": {"horizon": 1}}),
        ("spatial_jepa", {"objective": "spatial_jepa"}),
        ("mae", {"objective": "mae"}),
        ("byol", {"objective": "byol"}),
        ("simclr", {"objective": "simclr"}),
        ("random", {"objective": "temporal_jepa", "_random_init": True}),
        ("raw_features", {"_raw_features": True}),
        ("tjepa_h5", {"objective": "temporal_jepa", "temporal": {"horizon": 5}}),
        ("tjepa_h20", {"objective": "temporal_jepa", "temporal": {"horizon": 20}}),
        ("tjepa_noreg", {"objective": "temporal_jepa", "loss": {"var_coeff": 0.0, "cov_coeff": 0.0}}),
    ]
    return cells


def _deep_update(base, overrides):
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


METRICS = ["rul_r2", "rul_rmse", "rul_ic", "rul_std_rmse", "rul_phm08", "health_acc", "health_f1",
           "anom_auroc", "anom_ap", "clust_nmi", "clust_ari", "clust_silhouette",
           "retr_health_prec", "retr_rul_ic", "emb_std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/cjepa.yaml")
    ap.add_argument("--data", default="configs/data/cmapss.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default="runs/cmapss")
    ap.add_argument("--fds", nargs="*", default=None, help="subset list, e.g. FD001 FD004 (default: config)")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
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
    out_path = args.out or f"runs/cmapss_results{tag}.csv"
    fds = args.fds or data_cfg.get("fds", ["FD001"])

    cells = enumerate_cells()
    run = cells if args.max_cells is None else cells[: args.max_cells]
    skipped = [] if args.max_cells is None else cells[args.max_cells:]
    print(f"[run_cmapss] fds={fds} {len(cells)} cells; running {len(run)}; skipping {len(skipped)}")
    for name, _ in skipped:
        print(f"[run_cmapss] SKIPPED (budget): {name}")
    if args.dry_run:
        for fd in fds:
            for name, ov in run:
                print(f"[run_cmapss] would run: {fd}/{name}  overrides={ov}")
        return

    import torch
    from torch.utils.data import DataLoader
    from data.cmapss_dataset import make_cmapss_datasets, collate_cmapss_windows
    from engine.train_finance import FIN_TRAINERS, train_finance_jepa
    from eval.cmapss_tasks import evaluate_all_cmapss, evaluate_raw_features
    from models.finance_jepa import build_finance_model
    from utils.device import resolve_device
    from utils.gpu_hours import GpuHourMeter
    from utils.seed import seed_everything

    device = resolve_device(args.device or base.get("device"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    resume_existing = args.resume and os.path.exists(out_path)
    f = open(out_path, "a" if resume_existing else "w", newline="")
    writer = csv.writer(f)
    if not resume_existing:
        writer.writerow(["fd", "cell", "objective", "seed", *METRICS, "gpu_hours"])

    for fd in fds:
        pre_ds, ptr_ds, pte_ds, std_ds, meta = make_cmapss_datasets(
            root=data_cfg["root"], fd=fd, window=data_cfg["window"],
            train_stride=data_cfg.get("train_stride", 1), eval_stride=data_cfg.get("eval_stride", 3),
            rul_cap=data_cfg.get("rul_cap", 125), anom_rul=data_cfg.get("anom_rul", 20),
            health_thr=tuple(data_cfg.get("health_thr", (100, 50, 20))),
            allow_synth=data_cfg.get("allow_synth", True), seed=seed)
        print(f"[run_cmapss] {fd}: source={meta['source']} sensors={meta['num_assets']} "
              f"feats={meta['num_features']} train_eng={meta['n_train_engines']} "
              f"test_eng={meta['n_test_engines']} pretrain_win={meta['n_pretrain_windows']} "
              f"std_protocol_eng={len(std_ds)} (engines >= window kept)")
        ptr_loader = DataLoader(ptr_ds, batch_size=256, collate_fn=collate_cmapss_windows)
        pte_loader = DataLoader(pte_ds, batch_size=256, collate_fn=collate_cmapss_windows)
        std_loader = DataLoader(std_ds, batch_size=256, collate_fn=collate_cmapss_windows)

        for name, ov in run:
            ckpt = os.path.join(args.ckpt_dir, f"{fd}_{name}{tag}.pt")
            if args.resume and os.path.exists(ckpt):
                print(f"[run_cmapss] {fd}/{name}: already complete -> skip")
                continue
            cfg = _deep_update(base, ov)
            random_init = cfg.pop("_random_init", False)
            raw_features = cfg.pop("_raw_features", False)
            seed_everything(seed)
            meter = GpuHourMeter(device); meter.start()

            if raw_features:
                res = evaluate_raw_features(ptr_loader, pte_loader, std_loader, meta, seed)
                stats = meter.stop()
                writer.writerow([fd, name, "raw", seed,
                                 *[round(res.get(k, float("nan")), 4) for k in METRICS],
                                 round(stats["gpu_hours"], 4)]); f.flush()
                open(ckpt, "w").close()       # marker so --resume skips this encoder-free floor cell
                print(f"[run_cmapss] {fd}/{name}: rul_r2={res['rul_r2']:.3f} "
                      f"health_acc={res['health_acc']:.3f} (floor)")
                continue

            obj = cfg["objective"]
            loader = DataLoader(pre_ds, batch_size=cfg["optim"]["batch_size"], shuffle=True,
                                collate_fn=collate_cmapss_windows, drop_last=True,
                                num_workers=4, persistent_workers=True)
            if random_init:
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

            res = evaluate_all_cmapss(encoder, use_temporal, ptr_loader, pte_loader, std_loader,
                                      meta, device, seed)
            writer.writerow([fd, name, obj, seed,
                             *[round(res.get(k, float("nan")), 4) for k in METRICS],
                             round(stats["gpu_hours"], 4)]); f.flush()
            torch.save({"encoder": encoder.state_dict(), "cfg": cfg, "objective": obj, "fd": fd,
                        "use_temporal": use_temporal, "meta": meta, "seed": seed}, ckpt)
            print(f"[run_cmapss] {fd}/{name}: rul_r2={res['rul_r2']:.3f} rmse={res['rul_rmse']:.2f} "
                  f"phm08={res.get('rul_phm08', float('nan')):.0f} health_acc={res['health_acc']:.3f} "
                  f"anom={res['anom_auroc']:.3f} clust_nmi={res['clust_nmi']:.3f} "
                  f"retr={res['retr_health_prec']:.3f} gpu_h={stats['gpu_hours']:.3f}")
            del encoder, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    f.close()
    print(f"[run_cmapss] wrote {out_path}")


if __name__ == "__main__":
    main()
