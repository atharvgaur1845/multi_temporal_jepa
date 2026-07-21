"""Measurable predictability indices for a (latent) time series — the quantitative backbone of the
predictability hypothesis (report_full.md §18; Part 7 of the research agenda).

The project's central claim is that a causal future-latent-prediction (JEPA) objective helps *to the
degree the latent trajectory is predictable*. To make that falsifiable we need to *measure*
predictability, not just assert it. This module implements the standard indices:

  spectral_predictability(Ω)  — 1 - normalized spectral entropy; concentrated spectrum (periodic) -> 1,
                                flat spectrum (white noise) -> 0. (the "spectral predictability score";
                                operational low-predictability threshold Ω<0.2, arXiv:2507.13556)
  permutation_entropy_pred    — 1 - normalized Bandt-Pompe ordinal entropy; ordered -> 1, random -> 0.
  ar_forecast_r2              — out-of-sample R² of a linear AR(p) one-step predictor (learnable
                                structure a *linear* model can exploit).
  autocorr_time               — integrated / 1-e autocorrelation time (persistence).
  largest_lyapunov            — Rosenstein estimate of the largest Lyapunov exponent (>0 => chaotic:
                                short-term predictable, long-term not).
  past_future_mi              — Gaussian estimate of I(past block; future block) = excess entropy =
                                the *predictive information* (Bialek-Nemenman-Tishby 2001); the
                                information-theoretic definition of learnable structure.
  intrinsic_dimension         — participation ratio of the covariance spectrum (effective # of active
                                latent directions).

All operate on x of shape (T,) or (T, D) (per-dim then averaged). `predictability_report(x)` returns
the full dict. These are deliberately dependency-light (numpy only) so they run anywhere.
"""
from __future__ import annotations

import numpy as np


def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def spectral_predictability(x):
    """Ω = 1 - H_spectral / H_max, averaged over channels. Range [0,1], higher = more predictable."""
    X = _as2d(x)
    out = []
    for d in range(X.shape[1]):
        s = X[:, d] - X[:, d].mean()
        power = np.abs(np.fft.rfft(s)) ** 2
        power = power[1:]                                   # drop the DC component
        tot = power.sum()
        if tot <= 0:
            out.append(0.0); continue
        p = power / tot
        H = -(p * np.log(p + 1e-12)).sum()
        out.append(1.0 - H / np.log(len(p)))
    return float(np.mean(out))


def permutation_entropy_pred(x, m=4, tau=1):
    """1 - normalized permutation entropy (Bandt & Pompe). Higher = more predictable."""
    from math import factorial
    X = _as2d(x)
    out = []
    for d in range(X.shape[1]):
        s = X[:, d]
        n = len(s) - (m - 1) * tau
        if n <= 1:
            out.append(0.0); continue
        # ordinal pattern of each length-m window -> a permutation index
        patterns = {}
        for i in range(n):
            w = s[i:i + m * tau:tau]
            key = tuple(np.argsort(w, kind="stable"))
            patterns[key] = patterns.get(key, 0) + 1
        counts = np.array(list(patterns.values()), dtype=np.float64)
        p = counts / counts.sum()
        H = -(p * np.log(p + 1e-12)).sum()
        out.append(1.0 - H / np.log(factorial(m)))
    return float(np.mean(out))


def ar_forecast_r2(x, p=5, train_frac=0.7):
    """Out-of-sample R² of a linear AR(p) one-step-ahead predictor (per channel, averaged).
    Measures the structure a *linear* forecaster can exploit; ~1 periodic/AR, ~0 white noise."""
    X = _as2d(x)
    out = []
    for d in range(X.shape[1]):
        s = X[:, d]
        T = len(s)
        if T < 4 * p:
            out.append(0.0); continue
        A = np.stack([s[i:T - p + i] for i in range(p)], axis=1)     # (T-p, p) lag matrix
        y = s[p:]
        k = int(len(y) * train_frac)
        Atr, ytr, Ate, yte = A[:k], y[:k], A[k:], y[k:]
        Atr1 = np.concatenate([Atr, np.ones((len(Atr), 1))], 1)
        w, *_ = np.linalg.lstsq(Atr1, ytr, rcond=None)
        pred = np.concatenate([Ate, np.ones((len(Ate), 1))], 1) @ w
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
        out.append(max(-1.0, 1.0 - ss_res / ss_tot))
    return float(np.mean(out))


def autocorr_time(x, max_lag=None):
    """1/e autocorrelation time (lag where |acf| first drops below 1/e), averaged over channels."""
    X = _as2d(x)
    T = X.shape[0]
    max_lag = max_lag or min(200, T // 2)
    out = []
    for d in range(X.shape[1]):
        s = X[:, d] - X[:, d].mean()
        var = (s * s).mean()
        if var <= 0:
            out.append(0.0); continue
        acf = np.array([(s[:T - k] * s[k:]).mean() / var for k in range(1, max_lag)])
        below = np.where(np.abs(acf) < np.exp(-1))[0]
        out.append(float(below[0] + 1) if len(below) else float(max_lag))
    return float(np.mean(out))


def largest_lyapunov(x, m=3, tau=1, max_t=20, theiler=5):
    """Rosenstein's largest-Lyapunov-exponent estimate on the (per-channel-averaged) series. >0 =>
    chaotic. Approximate — meant to *rank* regimes (Lorenz >0, periodic/AR ~0, noise ill-defined)."""
    s = _as2d(x).mean(1)
    N = len(s) - (m - 1) * tau
    if N < 3 * max_t:
        return 0.0
    emb = np.stack([s[i:i + N] for i in range(0, m * tau, tau)], axis=1)   # (N, m)
    div = np.zeros(max_t)
    cnt = np.zeros(max_t)
    for i in range(N):
        d2 = ((emb - emb[i]) ** 2).sum(1)
        d2[max(0, i - theiler):i + theiler + 1] = np.inf                  # Theiler window
        j = int(np.argmin(d2))
        for t in range(max_t):
            if i + t < N and j + t < N:
                dist = np.linalg.norm(emb[i + t] - emb[j + t])
                if dist > 0:
                    div[t] += np.log(dist); cnt[t] += 1
    valid = cnt > 0
    if valid.sum() < 3:
        return 0.0
    curve = div[valid] / cnt[valid]
    ts = np.arange(max_t)[valid]
    slope = np.polyfit(ts, curve, 1)[0]                                   # divergence rate
    return float(slope)


def past_future_mi(x, k=5):
    """Gaussian estimate of predictive information I(past_k ; future_k) = excess entropy (nats).
    Bialek-Nemenman-Tishby 2001: the mutual information between the past and future of a series is
    exactly its learnable structure. For a jointly-Gaussian model:
        I = 0.5 * log( det(Σ_past) det(Σ_fut) / det(Σ_joint) ).
    """
    X = _as2d(x)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    T, D = X.shape
    # build past/future block vectors of length k over the flattened channels
    rows = []
    for i in range(k, T - k):
        past = X[i - k:i].reshape(-1)
        fut = X[i:i + k].reshape(-1)
        rows.append(np.concatenate([past, fut]))
    if len(rows) < 2 * k * D + 2:
        return 0.0
    M = np.stack(rows, 0)
    n = k * D
    C = np.cov(M, rowvar=False) + 1e-4 * np.eye(M.shape[1])
    Cp, Cf, = C[:n, :n], C[n:, n:]
    sign_j, ld_j = np.linalg.slogdet(C)
    sign_p, ld_p = np.linalg.slogdet(Cp)
    sign_f, ld_f = np.linalg.slogdet(Cf)
    mi = 0.5 * (ld_p + ld_f - ld_j)
    return float(max(0.0, mi))


def intrinsic_dimension(x):
    """Participation ratio of the covariance spectrum: (Σλ)² / Σλ² — the effective # of active dims."""
    X = _as2d(x)
    if X.shape[1] < 2:
        return 1.0
    lam = np.linalg.eigvalsh(np.cov(X, rowvar=False))
    lam = np.clip(lam, 0, None)
    return float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12))


def predictability_report(x):
    """All indices for a trajectory x (T,) or (T,D). Returns a flat dict."""
    return {
        "spectral_omega": spectral_predictability(x),
        "perm_entropy_pred": permutation_entropy_pred(x),
        "ar_r2": ar_forecast_r2(x),
        "autocorr_time": autocorr_time(x),
        "largest_lyapunov": largest_lyapunov(x),
        "past_future_mi": past_future_mi(x),
        "intrinsic_dim": intrinsic_dimension(x),
    }


PRED_KEYS = ["spectral_omega", "perm_entropy_pred", "ar_r2", "autocorr_time",
             "largest_lyapunov", "past_future_mi", "intrinsic_dim"]
