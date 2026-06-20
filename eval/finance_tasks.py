"""Downstream evaluation of a FROZEN finance encoder — five tasks, one global window embedding.

Same philosophy as the satellite probes (eval/linear_probe.py, eval/knn.py): freeze the pretrained
encoder, reduce each window to a single embedding, and read it five independent ways so the verdict
can't hinge on one probe. Probe heads are fit on the TRAIN period and scored on the held-out TEST
period (temporal split) — no look-ahead.

    1. regime classification  — logistic probe -> accuracy + macro-F1   (4-way market regime)
    2. volatility prediction  — ridge probe -> R^2 + Spearman IC         (forward realized vol)
    3. anomaly detection      — kNN-distance score -> AUROC + AP         (forward crash/stress)
    4. clustering             — KMeans vs regime -> ARI + NMI + silhouette (training-free)
    5. downstream forecasting — logistic (dir) + ridge (ret) -> dir-acc + IC (next-day index)

A random-init encoder is the control: probes on it should be near chance, so a method "winning"
means the *pretext task* put usable structure in the representation (not the probe leaking).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             normalized_mutual_info_score, r2_score, roc_auc_score,
                             silhouette_score)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def extract_window_embeddings(encoder, loader, device=None, use_temporal=True):
    """Frozen encoder -> one (D,) embedding per window (mean-pool over assets AND days) + labels.

    use_temporal=True : spatiotemporal path (encode_temporal) for the JEPA encoders.
    use_temporal=False: spatial-only path (encode_full per day) for MAE/BYOL/SimCLR backbones,
                        which never trained the temporal transformer (fair eval).
    Returns (X (M,D) float32, labels dict of (M,) arrays).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    keys = ("regime", "anomaly", "fwd_dir", "fwd_vol", "fwd_ret")
    feats, labs = [], {k: [] for k in keys}
    for batch in loader:
        data = batch["data"].to(device)
        dates = batch["dates"].to(device)
        pad = batch["pad_mask"].to(device)
        if use_temporal:
            tok = encoder.encode_temporal(data, dates, pad)                  # (B,W,N,D)
        else:
            B, W, N, Fc = data.shape
            tok = encoder.encode_full(data.reshape(B * W, N, Fc)).reshape(B, W, N, encoder.embed_dim)
        m = pad.float()[:, :, None, None]
        pooled = (tok * m).sum(dim=(1, 2)) / (m.sum(dim=(1, 2)) * tok.shape[2]).clamp_min(1.0)  # (B,D)
        feats.append(pooled.cpu().float())
        for k in keys:
            labs[k].append(batch["labels"][k])
    X = torch.cat(feats, 0).numpy()
    labels = {k: torch.cat(labs[k], 0).numpy() for k in keys}
    return X, labels


# --------------------------------------------------------------------------- individual tasks
def regime_classification(Xtr, ytr, Xte, yte, seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return {"regime_acc": float(accuracy_score(yte, pred)),
            "regime_f1": float(f1_score(yte, pred, average="macro"))}


def volatility_prediction(Xtr, vtr, Xte, vte):
    # predict log realized vol (vol is ~log-normal); report R^2 in log space + rank IC
    ltr, lte = np.log(np.clip(vtr, 1e-6, None)), np.log(np.clip(vte, 1e-6, None))
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=10.0).fit(sc.transform(Xtr), ltr)
    pred = reg.predict(sc.transform(Xte))
    ic = spearmanr(pred, lte).correlation
    return {"vol_r2": float(r2_score(lte, pred)), "vol_ic": float(ic if ic == ic else 0.0)}


def anomaly_detection(Xtr, Xte, yte_anom, k=20):
    """Unsupervised anomaly score = mean distance to the k nearest TRAIN embeddings (train is
    mostly 'normal'). Higher distance => more anomalous. AUROC + average precision vs the label."""
    sc = StandardScaler().fit(Xtr)
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr))).fit(sc.transform(Xtr))
    dist, _ = nn.kneighbors(sc.transform(Xte))
    score = dist.mean(axis=1)
    if len(np.unique(yte_anom)) < 2:                        # degenerate (no anomalies in test)
        return {"anom_auroc": float("nan"), "anom_ap": float("nan")}
    return {"anom_auroc": float(roc_auc_score(yte_anom, score)),
            "anom_ap": float(average_precision_score(yte_anom, score))}


def clustering(Xte, yte_regime, n_clusters=4, seed=0):
    sc = StandardScaler().fit(Xte)
    Z = sc.transform(Xte)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Z)
    sil = silhouette_score(Z, km.labels_) if len(np.unique(km.labels_)) > 1 else float("nan")
    return {"clust_ari": float(adjusted_rand_score(yte_regime, km.labels_)),
            "clust_nmi": float(normalized_mutual_info_score(yte_regime, km.labels_)),
            "clust_silhouette": float(sil)}


def forecasting(Xtr, dtr, Xte, dte, rtr, rte, seed=0):
    """Next-day index: direction (logistic -> accuracy) + return (ridge -> rank IC)."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    if len(np.unique(dtr)) > 1:
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Ztr, dtr)
        out["fcast_dir_acc"] = float(accuracy_score(dte, clf.predict(Zte)))
    else:
        out["fcast_dir_acc"] = float("nan")
    reg = Ridge(alpha=10.0).fit(Ztr, rtr)
    ic = spearmanr(reg.predict(Zte), rte).correlation
    out["fcast_ret_ic"] = float(ic if ic == ic else 0.0)
    return out


def evaluate_all(encoder, use_temporal, probe_tr_loader, probe_te_loader, meta, device=None, seed=0):
    """Run all five downstream tasks. Returns a flat dict of metrics (+ embedding diagnostics)."""
    Xtr, Ltr = extract_window_embeddings(encoder, probe_tr_loader, device, use_temporal)
    Xte, Lte = extract_window_embeddings(encoder, probe_te_loader, device, use_temporal)
    res = {}
    res.update(regime_classification(Xtr, Ltr["regime"], Xte, Lte["regime"], seed))
    res.update(volatility_prediction(Xtr, Ltr["fwd_vol"], Xte, Lte["fwd_vol"]))
    res.update(anomaly_detection(Xtr, Xte, Lte["anomaly"]))
    res.update(clustering(Xte, Lte["regime"], meta["num_regimes"], seed))
    res.update(forecasting(Xtr, Ltr["fwd_dir"], Xte, Lte["fwd_dir"],
                           Ltr["fwd_ret"], Lte["fwd_ret"], seed))
    # representation health (mirrors the satellite collapse diagnostics)
    std = Xte.std(0).mean()
    res["emb_std"] = float(std)
    res["n_train"], res["n_test"] = len(Xtr), len(Xte)
    return res


METRIC_KEYS = ["regime_acc", "regime_f1", "vol_r2", "vol_ic", "anom_auroc", "anom_ap",
               "clust_ari", "clust_nmi", "clust_silhouette", "fcast_dir_acc", "fcast_ret_ic"]
