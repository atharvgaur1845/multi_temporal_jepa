"""LKF-JEPA — a latent Kalman filter over the encoder's measurements (Part 6 #1).

The JEPA pieces already form a state-space model: the ENCODER produces noisy *measurements* of the
latent state, and the (Koopman) PREDICTOR is a linear *process model* z_{t+1}=A z_t. A Kalman filter
then fuses the two — correcting an open-loop rollout online with each new measurement — which should
(a) denoise the latent trajectory and (b) help MORE when the process model is good (i.e. when the
dynamics are predictable), linking the filter's value to predictability.

We implement a standard (diagonal-init) linear Kalman filter in the encoder's latent space with
measurement matrix H=I (measurements are latents). Reports open-loop vs filtered RMSE against a clean
reference — the quantitative LKF result.

    predict:  x⁻ = A x ;                 P⁻ = A P Aᵀ + Q
    update:   K  = P⁻ (P⁻ + R)⁻¹ ;        x = x⁻ + K (m − x⁻) ;   P = (I−K) P⁻
"""
from __future__ import annotations

import numpy as np


def kalman_filter(measurements, A, q=1e-2, r=1.0, p0=1.0):
    """Filter a (T, D) measurement sequence under process model A (D,D). q,r = process/measurement
    noise scales (isotropic). Returns the filtered state estimate (T, D)."""
    T, D = measurements.shape
    Q, R = q * np.eye(D), r * np.eye(D)
    x = measurements[0].astype(np.float64).copy()
    P = p0 * np.eye(D)
    I = np.eye(D)
    out = [x.copy()]
    for t in range(1, T):
        x = A @ x                                        # predict state
        P = A @ P @ A.T + Q                              # predict covariance
        S = P + R                                        # innovation covariance (H=I)
        K = P @ np.linalg.solve(S, I)                    # Kalman gain = P S⁻¹
        x = x + K @ (measurements[t] - x)                # correct with the measurement
        P = (I - K) @ P
        out.append(x.copy())
    return np.stack(out)


def open_loop_rollout(x0, A, T):
    """Process-model-only rollout (no measurement corrections): x_t = A^t x0. Returns (T, D)."""
    D = len(x0)
    out = [x0.astype(np.float64).copy()]
    x = out[0].copy()
    for _ in range(1, T):
        x = A @ x
        out.append(x.copy())
    return np.stack(out)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def lkf_report(clean, A, meas_noise=0.5, r=None, seed=0):
    """Given a clean latent trajectory `clean` (T,D) and a process operator A, add measurement noise,
    then compare RMSE (vs clean) of: the raw noisy measurements, the open-loop process rollout, and
    the Kalman-filtered estimate. The process-noise variance Q is estimated DATA-DRIVEN from the
    operator's one-step residuals (a good A -> small Q -> the filter trusts the model and denoises;
    a bad A -> large Q -> the filter falls back to the measurements). `r` defaults to the (known)
    measurement-noise variance. A correctly-tuned KF is optimal, so filtered ≤ measurement whenever A
    explains variance; the gain scales with the process-model accuracy (≈ one-step predictability)."""
    rng = np.random.default_rng(seed)
    clean = clean.astype(np.float64)
    scale = clean.std() + 1e-8
    noisy = clean + meas_noise * scale * rng.standard_normal(clean.shape)
    r = (meas_noise * scale) ** 2 if r is None else r
    q = float(((clean[1:] - clean[:-1] @ A.T) ** 2).mean())    # data-driven process noise
    D = clean.shape[1]
    filt = kalman_filter(noisy, A, q=q, r=r)
    # STATIC baseline: the same filter with NO dynamics (A=0) = optimal shrinkage toward the prior
    # mean. Subtracting it isolates what the *process model* (dynamics) adds beyond static denoising.
    q_static = float((clean ** 2).mean())                      # prior variance (A=0 residual = signal)
    filt_static = kalman_filter(noisy, np.zeros((D, D)), q=q_static, r=r)
    openl = open_loop_rollout(noisy[0], A, len(clean))
    return {
        "rmse_measurement": _rmse(noisy, clean),
        "rmse_open_loop": _rmse(openl, clean),
        "rmse_static": _rmse(filt_static, clean),
        "rmse_filtered": _rmse(filt, clean),
        "filter_gain_vs_measurement": _rmse(noisy, clean) - _rmse(filt, clean),
        "dynamics_gain": _rmse(filt_static, clean) - _rmse(filt, clean),   # value of the DYNAMICS
    }
