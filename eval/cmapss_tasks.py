"""Downstream evaluation of a FROZEN C-MAPSS encoder — five representation-quality tasks.

Same philosophy as the satellite/finance probes: freeze the encoder, reduce each window to one
mean-pooled embedding, fit light probes on TRAIN-engine windows, score on TEST-engine windows.
The finance lesson is baked in — the bar is not just the SSL baselines but the **raw-feature floor**
and a **random-init** encoder (both are run as cells in scripts/run_cmapss_matrix.py).

    1. RUL regression       — ridge -> R^2, RMSE, rank-IC (windowed) + the field-standard last-cycle
                              protocol (RMSE + PHM08 asymmetric score) vs RUL_FDxxx.txt
    2. Health classification— logistic -> accuracy, macro-F1   (4-stage degradation)
    3. Anomaly detection    — kNN-distance to train -> AUROC, AP   (near-failure)   [reused]
    4. Clustering           — KMeans vs health stage -> NMI, ARI, silhouette        [reused]
    5. NN retrieval         — does trajectory preserve similarity? kNN in train-embedding space ->
                              health precision@k + neighbour-RUL rank-IC
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler

# reuse the finance implementations unchanged (modality-agnostic)
from eval.finance_tasks import anomaly_detection, clustering


def _anomaly_vs_healthy(Xtr, health_tr, Xte, anom_te, k=20):
    """Novelty detection with a HEALTHY reference. Unlike finance (crashes are rare in train), every
    C-MAPSS train engine runs to failure, so near-failure states are NOT out-of-distribution for an
    all-train reference (that inverts the AUROC). The principled setup models 'healthy' (health
    stage 0) and flags deviation: near-failure test windows should be far from the healthy manifold."""
    ref = Xtr[health_tr == 0]
    if len(ref) <= k:
        ref = Xtr                                       # fallback if too few healthy windows
    return anomaly_detection(ref, Xte, anom_te, k=k)


@torch.no_grad()
def extract_window_embeddings(encoder, loader, device=None, use_temporal=True):
    """Frozen encoder -> one (D,) embedding per window (mean-pool over sensors AND cycles) + labels.
    use_temporal=True for JEPA encoders (encode_temporal); False for MAE/BYOL/SimCLR (encode_full)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    keys = ("rul", "rul_true", "health", "anomaly")
    feats, labs = [], {k: [] for k in keys}
    for batch in loader:
        data, dates, pad = batch["data"].to(device), batch["dates"].to(device), batch["pad_mask"].to(device)
        if use_temporal:
            tok = encoder.encode_temporal(data, dates, pad)                  # (B,W,N,D)
        else:
            B, W, N, Fc = data.shape
            tok = encoder.encode_full(data.reshape(B * W, N, Fc)).reshape(B, W, N, encoder.embed_dim)
        m = pad.float()[:, :, None, None]
        pooled = (tok * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * tok.shape[2]).clamp_min(1.0)
        feats.append(pooled.cpu().float())
        for k in keys:
            labs[k].append(batch["labels"][k])
    X = torch.cat(feats, 0).numpy()
    labels = {k: torch.cat(labs[k], 0).numpy() for k in keys}
    return X, labels


def phm08_score(pred, true):
    """NASA PHM08 asymmetric scoring: late predictions (pred>true) penalised more than early ones.
    Lower is better. d = pred - true; s = Σ exp(-d/13)-1 (d<0) or exp(d/10)-1 (d>=0)."""
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def rul_regression(Xtr, rul_tr, Xte, rul_te, Xstd=None, rul_true_std=None):
    """Ridge probe on (capped) RUL. Windowed: R^2 + rank-IC on the capped test RUL. Standard
    protocol (if Xstd/rul_true_std given): predict at each test engine's last cycle, compare to the
    TRUE uncapped RUL -> RMSE + PHM08 score (the field benchmark)."""
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=10.0).fit(sc.transform(Xtr), rul_tr)
    pred = reg.predict(sc.transform(Xte))
    ic = spearmanr(pred, rul_te).correlation
    out = {"rul_r2": float(r2_score(rul_te, pred)),
           "rul_rmse": float(np.sqrt(np.mean((pred - rul_te) ** 2))),
           "rul_ic": float(ic if ic == ic else 0.0)}
    if Xstd is not None and rul_true_std is not None and len(Xstd):
        pstd = np.clip(reg.predict(sc.transform(Xstd)), 0, None)         # predictions stay >= 0
        out["rul_std_rmse"] = float(np.sqrt(np.mean((pstd - rul_true_std) ** 2)))
        out["rul_phm08"] = phm08_score(pstd, rul_true_std)
    return out


def health_stage_classification(Xtr, ytr, Xte, yte, seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return {"health_acc": float(accuracy_score(yte, pred)),
            "health_f1": float(f1_score(yte, pred, average="macro"))}


def retrieval(Xtr, health_tr, rul_tr, Xte, health_te, rul_te, k=10):
    """Cosine kNN retrieval in TRAIN-embedding space. Reports health precision@k (fraction of the k
    neighbours sharing the query's health stage) and the rank-IC between each query's true RUL and
    its neighbours' mean RUL — i.e. does a similar embedding imply a similar degradation state."""
    Xtr_t = torch.nn.functional.normalize(torch.from_numpy(Xtr).float(), dim=-1)
    Xte_t = torch.nn.functional.normalize(torch.from_numpy(Xte).float(), dim=-1)
    sim = Xte_t @ Xtr_t.t()                                              # (Nte, Ntr)
    knn = sim.topk(min(k, Xtr_t.shape[0]), dim=1).indices.numpy()       # (Nte, k)
    neigh_health = health_tr[knn]                                        # (Nte, k)
    prec = float(np.mean(neigh_health == health_te[:, None]))
    neigh_rul = rul_tr[knn].mean(axis=1)
    ic = spearmanr(neigh_rul, rul_te).correlation
    return {"retr_health_prec": prec, "retr_rul_ic": float(ic if ic == ic else 0.0)}


def evaluate_all_cmapss(encoder, use_temporal, ptr_loader, pte_loader, std_loader, meta,
                        device=None, seed=0):
    """Run all five tasks. Returns a flat metric dict (+ embedding std diagnostic)."""
    Xtr, Ltr = extract_window_embeddings(encoder, ptr_loader, device, use_temporal)
    Xte, Lte = extract_window_embeddings(encoder, pte_loader, device, use_temporal)
    Xstd, Lstd = extract_window_embeddings(encoder, std_loader, device, use_temporal)
    res = {}
    res.update(rul_regression(Xtr, Ltr["rul"], Xte, Lte["rul"], Xstd, Lstd["rul_true"]))
    res.update(health_stage_classification(Xtr, Ltr["health"], Xte, Lte["health"], seed))
    res.update(_anomaly_vs_healthy(Xtr, Ltr["health"], Xte, Lte["anomaly"]))
    res.update(clustering(Xte, Lte["health"], meta["num_health"], seed))
    res.update(retrieval(Xtr, Ltr["health"], Ltr["rul"], Xte, Lte["health"], Lte["rul"]))
    res["emb_std"] = float(Xte.std(0).mean())
    res["n_train"], res["n_test"], res["n_std"] = len(Xtr), len(Xte), len(Xstd)
    return res


# raw-feature floor: probe the mean-pooled INPUT features directly (no encoder) — the finance lesson.
def raw_feature_embeddings(loader):
    keys = ("rul", "rul_true", "health", "anomaly")
    feats, labs = [], {k: [] for k in keys}
    for batch in loader:
        data, pad = batch["data"], batch["pad_mask"]                     # (B,W,N,F),(B,W)
        m = pad.float()[:, :, None, None]
        pooled = (data * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * data.shape[2]).clamp_min(1.0)
        feats.append(pooled.reshape(data.shape[0], -1).float())          # (B, N*F)
        for k in keys:
            labs[k].append(batch["labels"][k])
    X = torch.cat(feats, 0).numpy()
    return X, {k: torch.cat(labs[k], 0).numpy() for k in keys}


def evaluate_raw_features(ptr_loader, pte_loader, std_loader, meta, seed=0):
    Xtr, Ltr = raw_feature_embeddings(ptr_loader)
    Xte, Lte = raw_feature_embeddings(pte_loader)
    Xstd, Lstd = raw_feature_embeddings(std_loader)
    res = {}
    res.update(rul_regression(Xtr, Ltr["rul"], Xte, Lte["rul"], Xstd, Lstd["rul_true"]))
    res.update(health_stage_classification(Xtr, Ltr["health"], Xte, Lte["health"], seed))
    res.update(_anomaly_vs_healthy(Xtr, Ltr["health"], Xte, Lte["anomaly"]))
    res.update(clustering(Xte, Lte["health"], meta["num_health"], seed))
    res.update(retrieval(Xtr, Ltr["health"], Ltr["rul"], Xte, Lte["health"], Lte["rul"]))
    res["emb_std"] = float(Xte.std(0).mean())
    res["n_train"], res["n_test"], res["n_std"] = len(Xtr), len(Xte), len(Xstd)
    return res


METRIC_KEYS = ["rul_r2", "rul_rmse", "rul_ic", "rul_std_rmse", "rul_phm08", "health_acc",
               "health_f1", "anom_auroc", "anom_ap", "clust_nmi", "clust_ari", "clust_silhouette",
               "retr_health_prec", "retr_rul_ic"]
