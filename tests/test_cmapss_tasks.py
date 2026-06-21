"""C-MAPSS downstream probes return sane values; PHM08 score has the right asymmetry; signal>noise."""
import numpy as np

from eval.cmapss_tasks import (phm08_score, rul_regression, health_stage_classification,
                               retrieval, _anomaly_vs_healthy)


def test_phm08_zero_and_asymmetry():
    assert abs(phm08_score([50.0], [50.0])) < 1e-9            # perfect -> 0
    late = phm08_score([60.0], [50.0])                        # d=+10 (predicted too healthy)
    early = phm08_score([40.0], [50.0])                       # d=-10 (predicted too worn)
    assert late > early > 0                                   # late predictions penalised more


def _planted(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    rul = rng.uniform(0, 125, n)
    centers = rng.standard_normal((1, d))
    X = centers + rng.standard_normal((n, d)) + (rul[:, None] / 125.0) * 4.0  # RUL drives a direction
    return X, rul


def test_rul_regression_signal_beats_noise():
    X, rul = _planted()
    Xtr, rtr, Xte, rte = X[:300], rul[:300], X[300:], rul[300:]
    sig = rul_regression(Xtr, rtr, Xte, rte)["rul_r2"]
    noise = rul_regression(np.random.default_rng(9).standard_normal(Xtr.shape), rtr,
                           np.random.default_rng(8).standard_normal(Xte.shape), rte)["rul_r2"]
    assert sig > noise and sig > 0.3


def test_rul_standard_protocol_metrics():
    X, rul = _planted()
    out = rul_regression(X[:300], rul[:300], X[300:], rul[300:],
                         Xstd=X[300:320], rul_true_std=rul[300:320])
    assert "rul_std_rmse" in out and "rul_phm08" in out and out["rul_std_rmse"] >= 0


def test_health_classification_runs():
    X, rul = _planted()
    health = np.digitize(rul, [20, 50, 100])                  # 4 stages
    out = health_stage_classification(X[:300], health[:300], X[300:], health[300:])
    assert 0.0 <= out["health_acc"] <= 1.0 and "health_f1" in out


def test_anomaly_vs_healthy_reference():
    """Near-failure windows must be FAR from the HEALTHY manifold (AUROC>0.5). Guards the C-MAPSS
    fix: fitting kNN on ALL train (which contains failures) would invert the AUROC."""
    rng = np.random.default_rng(0)
    d = 16
    healthy = rng.standard_normal((300, d))                   # health stage 0 cluster
    failing = rng.standard_normal((60, d)) + 6.0              # near-failure, displaced cluster
    Xtr = np.vstack([healthy, failing])
    health_tr = np.array([0] * 300 + [3] * 60)
    Xte = np.vstack([rng.standard_normal((50, d)), rng.standard_normal((10, d)) + 6.0])
    anom_te = np.array([0] * 50 + [1] * 10)
    out = _anomaly_vs_healthy(Xtr, health_tr, Xte, anom_te, k=10)
    assert out["anom_auroc"] > 0.7                            # healthy-reference -> correct direction


def test_retrieval_preserves_similarity():
    X, rul = _planted()
    health = np.digitize(rul, [20, 50, 100])
    out = retrieval(X[:300], health[:300], rul[:300], X[300:], health[300:], rul[300:], k=10)
    assert 0.0 <= out["retr_health_prec"] <= 1.0
    assert out["retr_rul_ic"] > 0.0                           # similar embeddings -> similar RUL
