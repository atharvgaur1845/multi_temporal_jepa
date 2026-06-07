"""Evaluate a pretrained JEPA checkpoint: freeze the encoder and report downstream metrics.

This is the step that produces your headline result. It loads runs/.../last.ckpt, freezes the
(EMA target) encoder, and runs:
    - linear probe -> dense per-pixel mIoU   (compare vs supervised U-TAE = 63.1)
    - k-NN on parcel-mean features (optional, fast sanity)

Usage:
    python scripts/evaluate.py --ckpt runs/tjepa/last.ckpt
    python scripts/evaluate.py --ckpt runs/tjepa/last.ckpt --device cuda:1 --probe-epochs 30 --knn

Notes
    - use_temporal is inferred from the objective (JEPA -> temporal pathway; MAE/BYOL/SimCLR ->
      spatial-only). Band-normalization stats are recomputed from the train folds (cheap, and the
      training run didn't persist them in the checkpoint).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pastis_dataset import PASTIS, collate_variable_length  # noqa: E402
from data.transforms import compute_band_stats  # noqa: E402
from engine.diagnostics import collapse_metrics  # noqa: E402
from eval.linear_probe import extract_dense_features, linear_probe_segmentation  # noqa: E402
from models.jepa import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.device import resolve_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/tjepa.yaml")
    ap.add_argument("--data", default="configs/data/pastis.yaml")
    ap.add_argument("--ckpt", default="runs/tjepa/last.ckpt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-batch", type=int, default=8)
    ap.add_argument("--knn", action="store_true", help="also run parcel k-NN")
    ap.add_argument("--test", action="store_true", help="evaluate on test_folds instead of val_folds")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)
    device = resolve_device(args.device or cfg.get("device"))
    obj = cfg.get("objective", "temporal_jepa")
    use_temporal = obj in ("temporal_jepa", "spatial_jepa")
    msl = data_cfg.get("max_seq_len")
    print(f"[evaluate] ckpt={args.ckpt} objective={obj} use_temporal={use_temporal} device={device}")

    # model + weights
    model = build_model(cfg).to(device)
    step = load_checkpoint(args.ckpt, model, map_location=device)
    print(f"[evaluate] loaded checkpoint at step {step}")
    encoder = model.target_encoder  # the EMA representation

    # band stats from train folds (recomputed; cheap)
    train_unlab = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False,
                         max_seq_len=msl)
    mean, std = compute_band_stats(train_unlab, max_samples=200)

    eval_folds = data_cfg["test_folds"] if args.test else data_cfg["val_folds"]
    probe_tr = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=True,
                      norm_mean=mean, norm_std=std, max_seq_len=msl)
    eval_ds = PASTIS(data_cfg["root"], folds=eval_folds, return_label=True,
                     norm_mean=mean, norm_std=std, max_seq_len=msl)
    tl = DataLoader(probe_tr, batch_size=args.probe_batch, shuffle=True,
                    collate_fn=collate_variable_length, num_workers=4)
    el = DataLoader(eval_ds, batch_size=args.probe_batch, collate_fn=collate_variable_length,
                    num_workers=4)

    # collapse sanity on a val batch (did pretraining stay healthy?)
    with torch.no_grad():
        b = next(iter(el))
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        feat = extract_dense_features(encoder, b, use_temporal=use_temporal)  # (B,D,H,W)
        z = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        diag = collapse_metrics(z)
    print(f"[evaluate] feature health: per_dim_std={diag['per_dim_std']:.3f} "
          f"effective_rank={diag['effective_rank']:.1f} offdiag_cov={diag['offdiag_cov']:.4f}")

    # linear probe -> mIoU
    res = linear_probe_segmentation(encoder, tl, el, num_classes=data_cfg["num_classes"],
                                    ignore_index=data_cfg["ignore_index"], epochs=args.probe_epochs,
                                    device=device, use_temporal=use_temporal)
    split = "test" if args.test else "val"
    print(f"\n[evaluate] === RESULT ({split}) ===")
    print(f"[evaluate] linear-probe mIoU = {res['miou']*100:.2f}   (U-TAE supervised ref = 63.1)")
    print(f"[evaluate] per-class IoU = {[round(float(x), 3) for x in res['per_class_iou']]}")

    if args.knn:
        from eval.knn import knn_accuracy, parcel_embeddings
        Xtr, ytr = parcel_embeddings(encoder, tl, device=device)
        Xva, yva = parcel_embeddings(encoder, el, device=device)
        acc = knn_accuracy(Xtr, ytr, Xva, yva, k=20)
        print(f"[evaluate] parcel k-NN acc = {acc*100:.2f}")


if __name__ == "__main__":
    main()
