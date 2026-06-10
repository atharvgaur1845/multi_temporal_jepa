"""Qualitative + quantitative feature-space figure for a frozen encoder.

Extracts parcel-mean embeddings, projects to 2D (t-SNE or UMAP) colored by dominant crop class,
and reports cluster purity + silhouette. This is the qualitative figure for the write-up: a good
representation shows class-coherent clusters.

Usage:
    python scripts/feature_figure.py --encoder-ckpt runs/matrix/tjepa_h1.pt \
        --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml \
        --device cuda:0 --method tsne --out runs/figures/tjepa_h1_tsne.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pastis_dataset import PASTIS, collate_variable_length  # noqa: E402
from data.transforms import compute_band_stats  # noqa: E402
from eval.knn import parcel_embeddings  # noqa: E402
from eval.feature_analysis import cluster_purity, project_2d, silhouette  # noqa: E402
from models.jepa import SITSEncoder, build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.device import resolve_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model/tjepa_8gb.yaml")
    ap.add_argument("--data", default="configs/data/pastis.yaml")
    ap.add_argument("--encoder-ckpt", default=None, help="per-cell encoder from run_matrix")
    ap.add_argument("--ckpt", default=None, help="full JEPA checkpoint from train_jepa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    ap.add_argument("--test", action="store_true", help="use test folds (default: val)")
    ap.add_argument("--out", default="runs/figures/features.png")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data)
    device = resolve_device(args.device or cfg.get("device"))
    msl = data_cfg.get("max_seq_len")

    if args.encoder_ckpt:
        blob = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        ec = blob["cfg"]["encoder"]
        encoder = SITSEncoder(patch_size=ec["patch_size"], embed_dim=ec["embed_dim"],
                              depth=ec["depth"], num_heads=ec["num_heads"],
                              temporal_depth=ec.get("temporal_depth", 4)).to(device)
        encoder.load_state_dict(blob["encoder"])
    else:
        model = build_model(cfg).to(device)
        load_checkpoint(args.ckpt, model, map_location=device)
        encoder = model.target_encoder

    folds = data_cfg["test_folds"] if args.test else data_cfg["val_folds"]
    train_unlab = PASTIS(data_cfg["root"], folds=data_cfg["train_folds"], return_label=False,
                         max_seq_len=msl)
    mean, std = compute_band_stats(train_unlab, max_samples=200)
    ds = PASTIS(data_cfg["root"], folds=folds, return_label=True,
                norm_mean=mean, norm_std=std, max_seq_len=msl)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_variable_length, num_workers=4)

    X, y = parcel_embeddings(encoder, loader, device=device)
    X, y = X.numpy(), y.numpy()
    print(f"[feature_figure] {len(y)} parcels, {len(np.unique(y))} classes")
    print(f"[feature_figure] cluster_purity = {cluster_purity(X, y):.3f}  "
          f"silhouette = {silhouette(X, y):.3f}")

    emb = project_2d(X, method=args.method)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab20", s=8, alpha=0.7)
    plt.colorbar(sc, label="crop class"); plt.title(f"{args.method.upper()} of parcel embeddings")
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[feature_figure] saved {args.out}")


if __name__ == "__main__":
    main()
