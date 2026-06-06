"""k-NN evaluation on parcel-level features (training-free probe).

Reduce each labeled patch to a single embedding (masked-mean of encoder features over its
non-background pixels), then classify val patches by k nearest neighbors in train-embedding
space. Cheap, hyperparameter-light signal of feature quality; complements the linear probe.
"""
from __future__ import annotations

import torch

from .linear_probe import extract_dense_features


@torch.no_grad()
def parcel_embeddings(encoder, loader, device=None):
    """Frozen encoder -> per-patch mean feature + patch dominant (non-bg) label.

    Returns (X, y): X (M, D) float, y (M,) long.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    feats, labels = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        f = extract_dense_features(encoder, batch)          # (B, D, H, W)
        lab = batch["label"]                                 # (B, H, W)
        B, D, H, W = f.shape
        for b in range(B):
            fg = lab[b] != 0
            if fg.sum() == 0:
                continue
            emb = f[b].permute(1, 2, 0)[fg].mean(dim=0)      # (D,)
            vals, counts = torch.unique(lab[b][fg], return_counts=True)
            dom = vals[counts.argmax()]
            feats.append(emb.cpu()); labels.append(dom.cpu())
    return torch.stack(feats), torch.stack(labels)


def knn_accuracy(X_train, y_train, X_val, y_val, k=20):
    """Cosine k-NN classification accuracy (majority vote over k neighbors)."""
    Xtr = torch.nn.functional.normalize(X_train, dim=-1)
    Xva = torch.nn.functional.normalize(X_val, dim=-1)
    sim = Xva @ Xtr.t()                                      # (Nval, Ntr)
    knn = sim.topk(min(k, Xtr.shape[0]), dim=1).indices      # (Nval, k)
    neigh = y_train[knn]                                      # (Nval, k)
    preds = torch.mode(neigh, dim=1).values
    return (preds == y_val).float().mean().item()
