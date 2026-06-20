"""Downstream eval probes return sane values, and a SIGNAL embedding beats a NOISE embedding.

Uses synthetic embeddings with a planted regime structure so the probes have something to find;
guards against a probe that silently returns garbage or crashes on edge cases (rare anomaly class).
"""
import numpy as np

from eval.finance_tasks import (anomaly_detection, clustering, forecasting,
                                regime_classification, volatility_prediction)


def _planted(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)                                   # 4 regimes
    centers = rng.standard_normal((4, d)) * 3.0
    X = centers[y] + rng.standard_normal((n, d))               # separable-ish clusters
    return X, y


def test_regime_signal_beats_noise():
    X, y = _planted()
    Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]
    sig = regime_classification(Xtr, ytr, Xte, yte)["regime_acc"]
    noise = regime_classification(np.random.default_rng(1).standard_normal(Xtr.shape), ytr,
                                  np.random.default_rng(2).standard_normal(Xte.shape), yte)["regime_acc"]
    assert sig > noise
    assert sig > 0.5                                            # well above 4-class chance (0.25)


def test_volatility_regression_runs():
    X, y = _planted()
    v = np.exp(0.1 * (X[:, 0]) - 2.5)                          # vol correlated with feature 0
    out = volatility_prediction(X[:300], v[:300], X[300:], v[300:])
    assert "vol_r2" in out and "vol_ic" in out


def test_anomaly_auroc_in_range_and_handles_degenerate():
    X, _ = _planted()
    lab = (np.arange(len(X)) % 33 == 0).astype(int)           # ~3% anomalies
    out = anomaly_detection(X[:300], X[300:], lab[300:])
    assert 0.0 <= out["anom_auroc"] <= 1.0
    # degenerate: no anomalies in test -> NaN, not a crash
    deg = anomaly_detection(X[:300], X[300:], np.zeros(len(X) - 300, dtype=int))
    assert np.isnan(deg["anom_auroc"])


def test_clustering_recovers_planted_structure():
    X, y = _planted()
    out = clustering(X, y, n_clusters=4)
    assert out["clust_nmi"] > 0.3                              # should recover most of the structure


def test_forecasting_runs():
    X, y = _planted()
    d = (X[:, 1] > 0).astype(int)
    r = X[:, 1] * 0.01
    out = forecasting(X[:300], d[:300], X[300:], d[300:], r[:300], r[300:])
    assert "fcast_dir_acc" in out and "fcast_ret_ic" in out
