"""Synthetic latent-dynamics generator for the predictability sweep (Part 7 falsification design).

We generate a low-dimensional LATENT trajectory z_t whose predictability we can *dial*, render it to
a higher-dim OBSERVATION x_t = z_t A + noise at a fixed SNR, cut it into windows, and define the
downstream task as **recovering the clean latent z at the window's last step** from a frozen encoder.

Regimes span the predictability axis (report_full.md §18):
    periodic  — sums of sinusoids                     -> fully predictable (Ω≈1, MI high, LLE≈0)
    ar1(φ)    — z_{t+1}=φ z_t + √(1-φ²) ε              -> CONTINUOUS knob: φ→1 predictable, φ→0 white
    lorenz    — the Lorenz attractor (deterministic)  -> chaotic (LLE>0): short-term predictable only
    white     — i.i.d. noise                          -> unpredictable (Ω≈0, MI≈0)

Why "recover z_t": on a predictable trajectory, past frames inform the present latent, so a temporal
model (JEPA) can *denoise* it far better than a per-frame/raw readout — and that advantage should GROW
with predictability and VANISH on white noise. The advantage-vs-predictability curve is the
falsifiable test: flat ⇒ hypothesis falsified.

Panel layout matches the JEPA stack: data (num_windows, W, N=obs_dim, F=1); the encoder treats the
obs_dim coordinates as the cross-section (tokens) and W as time.
"""
from __future__ import annotations

import numpy as np
import torch

REGIMES = ("periodic", "ar1", "lorenz", "white")


def _latent(regime, T, d_lat, phi, rng, burn=500):
    """Return a (T, d_lat) latent trajectory, ~unit variance per dim."""
    if regime == "white":
        z = rng.standard_normal((T, d_lat))
    elif regime == "ar1":
        z = np.zeros((T + burn, d_lat))
        z[0] = rng.standard_normal(d_lat)
        s = np.sqrt(max(1e-6, 1.0 - phi ** 2))
        for t in range(1, T + burn):
            z[t] = phi * z[t - 1] + s * rng.standard_normal(d_lat)       # stationary, unit var
        z = z[burn:]
    elif regime == "periodic":
        t = np.arange(T)
        z = np.zeros((T, d_lat))
        for d in range(d_lat):
            for _ in range(3):                                           # a few incommensurate tones
                f = rng.uniform(0.01, 0.12); ph = rng.uniform(0, 2 * np.pi)
                z[:, d] += np.sin(2 * np.pi * f * t + ph)
        z = (z - z.mean(0)) / (z.std(0) + 1e-8)
    elif regime == "lorenz":
        d_lat = 3                                                        # Lorenz is 3-D
        sig, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        steps = (T + burn) * 3                                          # subsample x3 for a fuller attractor
        s = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
        traj = np.zeros((steps, 3))
        for i in range(steps):
            x, y, zc = s
            s = s + dt * np.array([sig * (y - x), x * (rho - zc) - y, x * y - beta * zc])
            traj[i] = s
        z = traj[burn * 3::3][:T]
        z = (z - z.mean(0)) / (z.std(0) + 1e-8)
    else:
        raise ValueError(regime)
    return z.astype(np.float64)


def _render(z, obs_dim, snr, rng, nonlinear=True, gain=2.0):
    """x_t = φ(z_t A) + noise at target SNR (signal-power / noise-power). Returns (T, obs_dim).

    The NONLINEAR observation map φ=tanh(gain··) is deliberate: it handicaps a raw *linear* probe
    (it cannot invert the nonlinearity), so a learned encoder has something to do — otherwise, on a
    linear-Gaussian world, ridge on the raw input is near-optimal and no representation can beat it
    (exactly the 'features already sufficient' artifact we hit on finance). With φ nonlinear, the
    JEPA-vs-MAE gap then isolates the value of *temporal integration* (which should scale with
    predictability), controlling for the nonlinear-inversion value shared by both learned encoders."""
    d_lat = z.shape[1]
    A = rng.standard_normal((d_lat, obs_dim)) / np.sqrt(d_lat)
    signal = np.tanh(gain * (z @ A)) if nonlinear else z @ A
    sig_pow = signal.var()
    noise = rng.standard_normal(signal.shape) * np.sqrt(sig_pow / max(1e-6, snr))
    return (signal + noise).astype(np.float32)


def generate(regime="ar1", phi=0.9, T=6000, d_lat=3, obs_dim=8, W=32, snr=4.0, stride=2, seed=0):
    """Return dict(data (M,W,N,1), latent (M,d_lat), z_full (T,d_lat), x_full (T,obs_dim), meta).
    `data` is the panel of windows; `latent` is the clean z at each window's last step (probe target).
    `z_full` is used to MEASURE predictability (ground-truth latent dynamics)."""
    rng = np.random.default_rng(seed)
    z = _latent(regime, T, d_lat, phi, rng)
    x = _render(z, obs_dim, snr, rng)
    ends = list(range(W - 1, T, stride))
    data = np.stack([x[e - W + 1:e + 1] for e in ends], 0)[:, :, :, None]     # (M,W,N,1)
    latent = np.stack([z[e] for e in ends], 0)                                # (M,d_lat)
    meta = {"regime": regime, "phi": phi, "num_assets": obs_dim, "num_features": 1, "window": W,
            "d_lat": z.shape[1], "obs_dim": obs_dim, "snr": snr, "n_windows": len(ends)}
    return {"data": data.astype(np.float32), "latent": latent.astype(np.float32),
            "z_full": z, "x_full": x, "meta": meta}


class DynamicsWindows(torch.utils.data.Dataset):
    """Windows for the JEPA stack: __getitem__ -> (x (W,N,F), dates (W,), None). Dates = time index
    (monotonic), used with a large temporal_period so the sinusoidal encoding doesn't wrap."""

    def __init__(self, data):
        self.data = data                                                     # (M,W,N,F)
        self.W = data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = torch.from_numpy(self.data[i]).float()
        dates = torch.arange(1, self.W + 1, dtype=torch.long)
        return x, dates, None


def collate(batch):
    xs, dates, _ = zip(*batch)
    data = torch.stack(xs, 0)
    dts = torch.stack(dates, 0)
    return {"data": data, "dates": dts, "pad_mask": torch.ones(dts.shape, dtype=torch.bool),
            "labels": None}
