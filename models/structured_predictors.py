"""Structured latent-dynamics predictors (Part 6 #3 Koopman, #4 Neural-ODE).

The default JEPA predictor is a free-form transformer: given the context latent it predicts the future
latent with no dynamical structure. These two drop-in replacements impose a *dynamics prior* — the
future latent is the context latent EVOLVED FORWARD by an explicit dynamical system — which (a) is the
right inductive bias when the latent trajectory is smooth/low-dimensional, and (b) exposes analytic
diagnostics the transformer cannot: the Koopman operator's spectral radius (a drift/stability index),
and a continuous-time flow that handles irregular sampling by construction.

Both keep the predictor signature `forward(context, target_pos, horizon)` so they are drop-in for
`models.predictor.Predictor` inside FinanceJEPA (temporal objective only — they model *time* evolution,
so `target_pos` is unused). Point prediction only (no variance head).

    KoopmanPredictor  — learn a linear operator K in a (learned) observable space:  ẑ_{t+Δ} = dec(K^Δ enc(z_t))
                        ρ(K) = spectral radius; |λ|≈1 => slow/predictable, |λ|>1 => expansive/chaotic.
    NeuralODEPredictor— learn a latent vector field f; integrate  dz/dt = f(z)  from 0..Δ by RK4.
                        Δ is the *real* time gap, so irregular revisit times are handled natively.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KoopmanPredictor(nn.Module):
    """ẑ_{t+Δ} = dec( K^Δ · enc(z_t) ) + b.  enc/dec are learned observables (Koopman lift); K is a
    single linear operator whose Δ-step power advances the latent. Global-linear (locally-linear when
    enc/dec are nonlinear — here linear lifts, so a clean linear Koopman)."""

    def __init__(self, enc_dim=128, koop_dim=None, **_):
        super().__init__()
        koop_dim = koop_dim or enc_dim
        self.enc_dim, self.koop_dim = enc_dim, koop_dim
        self.lift = nn.Linear(enc_dim, koop_dim, bias=False) if koop_dim != enc_dim else nn.Identity()
        self.unlift = nn.Linear(koop_dim, enc_dim, bias=False) if koop_dim != enc_dim else nn.Identity()
        self.K = nn.Parameter(torch.eye(koop_dim) + 0.01 * torch.randn(koop_dim, koop_dim))
        self.b = nn.Parameter(torch.zeros(enc_dim))

    def forward(self, context_tokens, target_pos=None, n_targets=None, horizon=1):
        g = self.lift(context_tokens)                                  # (B, N, koop_dim)
        Kp = torch.linalg.matrix_power(self.K, max(1, int(horizon)))   # K^Δ
        g = g @ Kp.t()                                                 # advance Δ steps
        return self.unlift(g) + self.b                                 # (B, N, enc_dim)

    @torch.no_grad()
    def spectral_radius(self):
        return float(torch.linalg.eigvals(self.K.detach().float()).abs().max().item())

    @torch.no_grad()
    def operator(self):
        """Return the effective enc_dim×enc_dim one-step operator A s.t. ẑ_{t+1} ≈ A z_t (for the LKF
        process model). For identity lifts A = K; otherwise A = unlift∘K∘lift (linear)."""
        if isinstance(self.lift, nn.Identity):
            return self.K.detach().float()
        W_l = self.lift.weight.detach().float()          # (koop, enc)
        W_u = self.unlift.weight.detach().float()        # (enc, koop)
        return W_u @ self.K.detach().float() @ W_l       # (enc, enc)


def _rk4_step(f, z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class NeuralODEPredictor(nn.Module):
    """ẑ_{t+Δ} = z_t + ∫_0^Δ f(z) dt, integrated by fixed-step RK4 (no torchdiffeq dependency). The
    vector field f is a small MLP. `substeps` RK4 steps per unit of time -> continuous-time flow that
    is principled for irregular sampling (Δ is the real elapsed time)."""

    def __init__(self, enc_dim=128, hidden=None, substeps=4, max_steps=128, **_):
        super().__init__()
        hidden = hidden or 2 * enc_dim
        self.f = nn.Sequential(nn.Linear(enc_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, enc_dim))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)   # start at identity flow
        self.substeps, self.max_steps = substeps, max_steps

    def forward(self, context_tokens, target_pos=None, n_targets=None, horizon=1):
        z = context_tokens
        total = self.f  # alias
        n_steps = min(self.max_steps, self.substeps * max(1, int(horizon)))
        dt = float(max(1, int(horizon))) / n_steps
        for _ in range(n_steps):
            z = _rk4_step(total, z, dt)
        return z


def build_structured_predictor(kind, enc_dim, **kw):
    if kind == "koopman":
        return KoopmanPredictor(enc_dim, koop_dim=kw.get("koop_dim"))
    if kind == "ode":
        return NeuralODEPredictor(enc_dim, hidden=kw.get("hidden"), substeps=kw.get("substeps", 4))
    raise ValueError(f"unknown structured predictor {kind!r}")
