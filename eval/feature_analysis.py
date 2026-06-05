"""Qualitative + quantitative feature-space analysis.

t-SNE / UMAP projections colored by crop class (qualitative), plus cluster-purity and silhouette
(quantitative) to back up the pictures with numbers. Run on parcel embeddings (eval/knn.py).
"""
from __future__ import annotations


def project_2d(X, method="tsne"):
    """Project embeddings to 2D for visualization. TODO: wrap sklearn TSNE / umap-learn (M3)."""
    raise NotImplementedError("M3")


def cluster_purity(X, y, n_clusters=None):
    """KMeans cluster, then purity = mean over clusters of the majority-class fraction.
    TODO: implement (M3)."""
    raise NotImplementedError("M3")


def silhouette(X, y):
    """Silhouette score of embeddings w.r.t. crop labels (inter-class separation).
    TODO: wrap sklearn.metrics.silhouette_score (M3)."""
    raise NotImplementedError("M3")
