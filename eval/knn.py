"""k-NN evaluation on parcel-level features (training-free probe).

Reduce each labeled parcel/patch to a single embedding (e.g. masked-mean of encoder features
over the parcel's pixels), then classify val parcels by k nearest neighbors in train-embedding
space. Cheap, hyperparameter-light signal of feature quality; complements the linear probe.
"""
from __future__ import annotations


def parcel_embeddings(encoder, loader):
    """Frozen encoder -> per-parcel mean feature + parcel label.
    Returns (X, y). TODO: pool features within each parcel mask (M3)."""
    raise NotImplementedError("M3")


def knn_accuracy(X_train, y_train, X_val, y_val, k=20):
    """Cosine/L2 k-NN classification accuracy. TODO: implement or wrap sklearn (M3)."""
    raise NotImplementedError("M3")
