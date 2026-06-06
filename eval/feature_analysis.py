"""Qualitative + quantitative feature-space analysis.

t-SNE / UMAP projections colored by crop class (qualitative), plus cluster-purity and silhouette
(quantitative) to back up the pictures with numbers. Run on parcel embeddings (eval/knn.py).
"""
from __future__ import annotations

import numpy as np


def project_2d(X, method="tsne", seed=0):
    """Project embeddings (N, D) to 2D for visualization. Returns (N, 2) ndarray."""
    X = np.asarray(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", random_state=seed).fit_transform(X)
    elif method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError(f"unknown method {method!r}")


def cluster_purity(X, y, n_clusters=None, seed=0):
    """KMeans cluster, then purity = (1/N) * sum_clusters max_class |cluster ∩ class|."""
    from sklearn.cluster import KMeans
    X = np.asarray(X); y = np.asarray(y)
    n_clusters = n_clusters or len(np.unique(y))
    assign = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
    total = 0
    for c in np.unique(assign):
        members = y[assign == c]
        if len(members):
            total += np.bincount(members).max()
    return total / len(y)


def silhouette(X, y):
    """Silhouette score of embeddings w.r.t. crop labels (inter-class separation)."""
    from sklearn.metrics import silhouette_score
    return silhouette_score(np.asarray(X), np.asarray(y))
