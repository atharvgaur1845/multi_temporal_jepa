# Causal Future-Latent Prediction as a Self-Supervised Objective for Temporally Evolving Systems

### A Three-Domain Study of Temporal JEPA on Satellite, Financial, and Industrial Time Series

**Technical Monograph — Multi-Temporal JEPA Project**
*Frozen-encoder representation-learning comparison across PASTIS (Sentinel-2), S&P-500 sector panels, and NASA C-MAPSS turbofan degradation.*

---

> **Reader's note.** This document is a self-contained research monograph. It assumes fluency with
> transformers, self-supervised learning, optimization, probability and linear algebra, and derives
> the non-obvious pieces (the JEPA objective, VICReg, EMA dynamics, effective rank, the PHM08 score)
> from first principles. The three per-domain engineering reports — [report.md](report.md) (satellite),
> [report_finance.md](report_finance.md) (finance), [report_cmapss.md](report_cmapss.md) (industrial) —
> are the primary sources for the numbers reproduced here; every figure in this monograph is traceable
> to `runs/*_results.csv`. Nothing is fabricated; where a number is single-seed it is marked.

---

## Notation

| Symbol | Meaning |
|---|---|
| $x$ | an input observation (a frame, a window, a token) |
| $X \in \mathbb{R}^{T\times \cdots}$ | a time series of $T$ frames |
| $N$ | number of cross-sectional tokens per frame (pixels-patches / assets / sensors) |
| $F$ | per-token input feature dimension |
| $D$ | encoder embedding width (`embed_dim`) |
| $D_p$ | predictor width (`pred_dim`), with the invariant $D_p < D$ |
| $T,\,W$ | sequence / window length (frames, trading days, cycles) |
| $\Delta$ | prediction horizon (how far ahead the future target lies) |
| $f_\theta$ | the trainable **context** encoder (online network) |
| $f_\xi$ | the **target** encoder (EMA copy of $f_\theta$, stop-grad) |
| $g_\phi$ | the predictor (narrow transformer) |
| $z = f(x)$ | a latent representation |
| $\hat z$ | a predicted latent |
| $\tau \in [0,1)$ | EMA momentum |
| $\eta$, $\lambda$ | learning rate; loss coefficients |
| $\mathrm{sg}[\cdot]$ | stop-gradient operator |
| $\mathrm{LN}$ | LayerNorm |
| $\mathbb{E},\,\mathbb{H},\,\mathbb{I}$ | expectation, entropy, mutual information |
| $\mathrm{erank}$ | effective rank of a covariance |

Domain dictionary (the single most important table in the project):

| abstract object | PASTIS satellite | S&P finance | C-MAPSS industrial |
|---|---|---|---|
| frame (one time step) | a Sentinel-2 acquisition ($10\times128\times128$) | one trading day | one operating cycle |
| cross-sectional token | a pixel patch ($P\times P$) | one sector ETF | one sensor |
| $N$ (tokens/frame) | 256 ($P{=}8$) | 9 | 14–17 |
| temporal position | day-of-year (periodic) | day-of-year | operating cycle (monotonic) |
| "predict the future" | next acquisition's latent | tomorrow's market latent | future cycle's sensor latent |
| downstream signal | crop type / phenology | regime / volatility | remaining useful life |

---

# 1. Executive Summary

**Problem.** Most self-supervised learning (SSL) for high-dimensional data learns by *spatial*
pretext tasks — reconstruct masked pixels (MAE), or enforce invariance between augmented views
(SimCLR, BYOL). For data that is natively a *time series of the same entities* (a field re-imaged
across a season; a market re-priced each day; an engine re-sensed each cycle) those tasks discard the
axis that carries the signal: **time**. We ask whether a *causal future-latent-prediction* objective
— predict the embedding of a future frame from the embeddings of past frames, in representation space,
à la JEPA — is a better pretext task for such systems.

**Hypothesis.** Future-from-past prediction forces the encoder to model the system's *latent
dynamics* (phenology, regime evolution, degradation), which is exactly what temporally-grounded
downstream tasks need. We predict this helps **iff** the latent trajectory is *predictable* — smooth
and persistent — and fails when the future is effectively a random walk.

**Method.** A single factorized space-then-time encoder (a per-frame cross-sectional ViT followed by
a temporal transformer), a narrow predictor, an EMA target encoder, an L2 latent loss, and a VICReg
variance–covariance regularizer to prevent collapse. The *same* architecture is instantiated on three
domains by swapping only the frame tokenizer and the temporal positional encoding. Every comparison
is a *frozen-encoder* probe: pretrain, freeze, fit a light linear/kNN probe, measure.

**Findings (three independent domains).**

1. **Satellite (PASTIS, predictable/seasonal):** Temporal JEPA **wins** decisively — conv mIoU
   $22.3\pm1.8$ vs Spatial JEPA $16.2\pm0.4$ (+6.0, paired $t$-test $p=0.041$, 3 seeds), and beats
   MAE/BYOL/SimCLR by **+15–16 mIoU** ($p<0.01$). A mechanistic probe shows the temporal objective
   makes the *spatial* features season-aware (month-decoding accuracy 61.3% vs 46.3%, chance 8.3%).
2. **Finance (S&P-500 sectors, non-stationary):** Temporal JEPA **loses** — it beats only Spatial
   JEPA (7/10 metrics) but falls below MAE/BYOL and, critically, **below a raw-feature linear probe
   and below its own random initialization** (regime accuracy 0.61 trained vs 0.80 untrained). An
   honest *negative / inverted-transfer* result; longer horizons make it monotonically worse.
3. **Industrial (NASA C-MAPSS, monotonic degradation):** Temporal JEPA **wins** — it is the best SSL
   objective across all four FD subsets (beats Spatial/MAE/BYOL/SimCLR on 43–51 of 52 metric-subsets)
   and **clears the raw-feature floor** (RUL $R^2$ 0.63–0.81 vs raw 0.18–0.34) — the bar finance
   failed. The honest nuance: an *untrained* network is competitive on the easiest single-condition
   subsets; learning's advantage grows with task difficulty (12/13 wins over random on FD002 vs 8/13
   on FD001).

**Scientific conclusion.** The three points are not a contradiction; they are a *curve*. Causal
temporal-prediction SSL helps **to the extent the modality has a predictable latent trajectory**.
PASTIS (periodic phenology) and C-MAPSS (monotone wear) sit on the predictable end and the objective
wins; finance (near-efficient, non-stationary) sits on the unpredictable end and it not only fails to
help but actively *erases* usable structure. The robust sub-finding is that **`temporal > spatial`
replicates on all three domains** — predicting forward in time is a better pretext than masking within
a frame whenever there is *any* temporal signal at all.

**Contributions.** (i) A modality-agnostic temporal-JEPA implementation reused verbatim across three
domains, isolating the objective as the only scientific variable; (ii) the first (to our knowledge)
controlled three-domain *predictability-spectrum* study of causal latent prediction with **raw-feature
and random-init floors** as the bar; (iii) an honest negative result on finance and its mechanistic
explanation; (iv) a falsifiable principle relating temporal persistence to SSL benefit (§18).

---

# 2. Background

## 2.1 Self-Supervised and Representation Learning

Supervised learning estimates $p(y\mid x)$ from labeled pairs; its appetite for labels is the
bottleneck in domains where labels are scarce and expensive (dense crop maps, regime annotations,
run-to-failure logs). **Representation learning** instead seeks a map $f:\mathcal X\to\mathbb R^D$
such that simple (e.g. linear) functions of $f(x)$ solve many downstream tasks. **Self-supervised
learning** trains $f$ with a *pretext* task whose targets are derived from $x$ itself, so no human
labels are needed. The empirical promise, repeatedly borne out since 2018, is that a good pretext
task yields features competitive with supervised pretraining and far better label efficiency.

The central design question of SSL is: *what pretext task forces the network to learn the structure
that downstream tasks need, without letting it cheat?* Four broad answers exist, and this project is
a controlled comparison of all four plus a fifth (ours):

- **Masked reconstruction** (denoising autoencoders → BERT → MAE): hide part of $x$, reconstruct it
  in input space.
- **Contrastive** (CPC, SimCLR): pull together representations of two views of the same $x$, push
  apart different $x$.
- **Self-distillation / non-contrastive** (BYOL, SimSiam, DINO): predict one view's representation
  from another's using an asymmetric online/target pair, with no negatives.
- **Joint-embedding predictive (JEPA)** (I-JEPA, V-JEPA): predict the *representation* of a masked
  region from the representation of a visible region — like masked reconstruction but in *latent*
  space, like distillation but *predicting a hidden part* rather than enforcing invariance.

Our objective is **causal temporal JEPA**: the "hidden part" is the *future*, and the split is by
time, not by spatial masking.

## 2.2 The Information-Bottleneck view

Why should *any* of these produce useful features? The Information Bottleneck (Tishby et al., 1999;
Tishby & Zaslavsky, 2015) frames a good representation $Z=f(X)$ as one that maximizes information
about a relevant variable $Y$ while minimizing information about nuisances:

$$
\min_{f}\; \mathbb I(Z;X) - \beta\, \mathbb I(Z;Y).
$$

SSL replaces the unavailable $Y$ with a self-derived target. In masked reconstruction $Y$ is the
masked pixels; in contrastive learning $Y$ is "which instance"; in JEPA $Y$ is the latent of the
masked/future region. The key insight that motivates this project: **the choice of $Y$ determines
which information is preserved.** If $Y$ is "the next frame's latent," then $Z$ must retain whatever
predicts temporal evolution — and discard whatever is unpredictable. On a predictable system this
filters *toward* the dynamics-relevant signal; on an unpredictable one the predictive target is noise
and the bottleneck filters *away* useful static structure (this is precisely the finance failure mode,
§12, §18).

## 2.3 Contrastive learning and the role of negatives

Contrastive SSL maximizes a lower bound on mutual information between views (InfoNCE; Oord et al.,
2018). For a batch of $2B$ views with positives $(i, i^+)$,

$$
\mathcal L_{\text{InfoNCE}} = -\sum_i \log \frac{\exp(\mathrm{sim}(z_i,z_{i^+})/\kappa)}{\sum_{j\neq i}\exp(\mathrm{sim}(z_i,z_j)/\kappa)},
$$

with $\mathrm{sim}$ cosine similarity and $\kappa$ a temperature. The denominator's *negatives* are
what prevent collapse: without them every $z$ could equal a constant and the numerator would be
maximal. SimCLR's well-known dependence on large batches is exactly the need for many negatives. A
recurring theme of this project's baselines: on small panels (9 sectors, 17 sensors) the contrastive
batch is small and SimCLR is the weakest method — consistent with theory.

## 2.4 Masked reconstruction and predictive coding

MAE (He et al., 2021) masks ~75% of patches and reconstructs *pixels* with an asymmetric
encoder–decoder; the encoder sees only visible patches. It is a high-capacity, low-prior objective:
reconstructing pixels forces the encoder to model fine appearance, which can be a *distraction* for
semantic tasks (a known MAE weakness — linear-probe accuracy lags its fine-tuning accuracy). **Predictive
coding** (Rao & Ballard, 1999; CPC, Oord et al., 2018) instead predicts *future* latents from past
context with an autoregressive model and an InfoNCE loss — the conceptual ancestor of temporal JEPA,
differing in that CPC is contrastive (needs negatives) whereas JEPA is regression-in-latent-space with
an EMA target and an explicit anti-collapse regularizer.

## 2.5 JEPA and World Models

The **Joint-Embedding Predictive Architecture** (LeCun, 2022; I-JEPA, Assran et al., 2023) predicts,
in representation space, the embedding of a masked target block from the embedding of a context block:

$$
\hat z_{\text{tgt}} = g_\phi\big(f_\theta(x_{\text{ctx}}),\, \text{pos}_{\text{tgt}}\big),\qquad
z_{\text{tgt}} = \mathrm{sg}\big[f_\xi(x_{\text{tgt}})\big],
$$

and minimizes $\|\hat z_{\text{tgt}} - z_{\text{tgt}}\|^2$. By predicting *abstract* representations
rather than pixels, JEPA can ignore unpredictable high-frequency detail (the thing MAE wastes capacity
on) and focus on semantically predictable structure. **V-JEPA** (Bardes et al., 2024) extends this to
video with spatiotemporal masking — but crucially **non-causally and bidirectionally** (masked tubes
can be anywhere in the clip). The distinction that defines *our* contribution: we make the split
**causal** (context = strictly past, target = strictly future) so the objective is a *forward
dynamics model*, aligning JEPA with the **World Models** program (Ha & Schmidhuber, 2018; Hafner et
al., 2019), which posits that learning to predict the future latent state of an environment yields
representations supporting planning and control. A temporally-evolving panel *is* a (passive)
dynamical system; "predict the next latent" is the simplest world-model objective.

## 2.6 Predictive State Representations & temporal representation learning

Predictive State Representations (Littman et al., 2001) formalize a system's state as a vector of
predictions about future observations — a state is *defined by* what it implies about the future. This
is the theoretical charter for temporal JEPA: the encoder is pushed toward a sufficient statistic for
forward prediction. Modern temporal-SSL methods (TS2Vec, TNC, TS-TCC) typically use contrastive
objectives over temporal crops; temporal JEPA differs by being *predictive and generative-in-latent*
rather than contrastive, which (§16) removes the negative-sampling dependence.

## 2.7 The three application modalities (why this triad)

- **Remote-sensing SITS (PASTIS).** Earth observation is natively a time series of the *same* place;
  crops are separated by *phenological stage* (growth, senescence, harvest), which tracks time. A
  spatial-only objective treats each acquisition independently and throws this away. Strong, smooth,
  *seasonal* temporal structure → the predictable end of the spectrum.
- **Financial panels (S&P sectors).** A market is a cross-section of assets re-priced daily. Returns
  are close to a martingale (efficient-market hypothesis); the cross-section co-moves through regimes,
  but the *next-day* latent is nearly unpredictable and the data are non-stationary (the 1999–2017
  distribution differs from 2018–2026). The unpredictable end.
- **Industrial PHM (C-MAPSS).** A turbofan degrades *monotonically* from healthy to failure; sensor
  trajectories drift smoothly. The most predictable end — the confirmation case.

These three were chosen precisely to *span the predictability axis*, turning a single result into a
falsifiable spectrum hypothesis (§18).

---

# 3. Mathematical Foundations

This section derives every mechanism used in the system, from probability primitives to the loss.

## 3.1 Probability, expectation, conditional probability

For a representation to be a *sufficient statistic* for forecasting, we need the language of
conditional distributions. Given jointly distributed $(X,Y)$, the conditional density is
$p(y\mid x)=p(x,y)/p(x)$, and the expectation of $h(Y)$ given $X{=}x$ is
$\mathbb E[h(Y)\mid x]=\int h(y)\,p(y\mid x)\,dy$. The **conditional expectation** $\mathbb E[Y\mid X]$
is the (a.s. unique) function of $X$ minimizing mean-squared error $\mathbb E\|Y-m(X)\|^2$ over all
measurable $m$; this is the formal object an L2 latent predictor approximates (it learns
$\hat z\approx\mathbb E[z_{\text{future}}\mid z_{\text{past}}]$). The **tower property**
$\mathbb E[\mathbb E[Y\mid X]]=\mathbb E[Y]$ and the **law of total variance**

$$
\mathrm{Var}(Y)=\mathbb E[\mathrm{Var}(Y\mid X)] + \mathrm{Var}(\mathbb E[Y\mid X])
$$

are the key tools: the *irreducible* term $\mathbb E[\mathrm{Var}(Y\mid X)]$ is the noise floor of any
predictor. On finance this term dominates (future ≈ unpredictable), so the best predictor is near the
unconditional mean and the objective's gradient signal is mostly noise — a fact we will quantify with
$R^2$ (§12).

## 3.2 Mutual information and why prediction shapes representations

The mutual information between representation $Z=f(X_{\text{past}})$ and future $X_{\text{fut}}$,

$$
\mathbb I(Z; X_{\text{fut}}) = \mathbb H(X_{\text{fut}}) - \mathbb H(X_{\text{fut}}\mid Z),
$$

is maximized when $Z$ retains everything about the past that predicts the future. Minimizing the L2
latent-prediction loss is (under Gaussian-residual assumptions) maximizing a lower bound on
$\mathbb I(Z; z_{\text{fut}})$: writing the optimal predictor's residual as Gaussian with covariance
$\Sigma$, the per-sample loss equals $\tfrac12(z_{\text{fut}}-\hat z)^\top\Sigma^{-1}(z_{\text{fut}}-\hat z)$
up to constants, and $-\mathbb E[\log p]$ is cross-entropy whose minimization tightens an
InfoNCE-style bound on $\mathbb I$. The qualitative consequence is the load-bearing one: **the pretext
target selects which information survives.** On a system whose future is a deterministic-plus-smooth
function of the past, $\mathbb I(Z;X_{\text{fut}})$ is large and aligns with the semantic signal; on a
martingale it is near zero and the objective provides no useful pressure.

## 3.3 Attention, from first principles

A transformer layer maps a set of $n$ tokens $H\in\mathbb R^{n\times d}$ to a new set, mixing
information by content-based routing. Self-attention computes queries, keys, values by linear maps
$Q=HW_Q,\;K=HW_K,\;V=HW_V$ ($W_\bullet\in\mathbb R^{d\times d}$) and forms

$$
\mathrm{Attn}(H)=\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_h}}\Big)V .
$$

**Derivation of the $1/\sqrt{d_h}$ scale.** If entries of $q,k\in\mathbb R^{d_h}$ are independent with
mean 0 and variance 1, then $q^\top k=\sum_{i=1}^{d_h} q_i k_i$ has mean 0 and variance $d_h$. Feeding
logits of magnitude $O(\sqrt{d_h})$ into softmax saturates it (gradients vanish), so we divide by
$\sqrt{d_h}$ to keep logit variance $O(1)$ regardless of head dimension — the standard scaled
dot-product argument (Vaswani et al., 2017). **Multi-head** attention runs $H_{\text{heads}}$ such maps
in parallel on $d_h=d/H_{\text{heads}}$-dimensional projections and concatenates, letting different
heads route on different subspaces (e.g. one head over assets, one over cycles). **Permutation
equivariance:** attention is equivariant to token permutation, which is why explicit positional
encodings are required — and why, for tokens that *do* have identity but no order (assets, sensors), a
*learned* per-token embedding is the right inductive bias (§7.3) rather than a sinusoid over an
arbitrary index.

**Key-padding mask.** For variable-length series we add $-\infty$ to logits of padded keys before the
softmax, so $\mathrm{softmax}$ assigns them zero weight: $\mathrm{logit}_{ij}\!\leftarrow\!-\infty$ if
key $j$ is padding. This is how the temporal transformer ignores pad frames (PASTIS) and how the
causal context-only mask blocks future leakage (all domains, §8.3).

**Cross-attention** (used implicitly by the predictor) lets queries from one set (mask tokens at the
target positions) attend to keys/values from another (the encoded context), i.e. $Q$ from mask tokens,
$K,V$ from context.

**Complexity.** Self-attention over $n$ tokens is $O(n^2 d)$ time and $O(n^2 + nd)$ memory. For a
$T\times N$ spatiotemporal grid, full 3-D attention is $O((TN)^2 d)$ — for PASTIS $T\!\le\!61$,
$N\!=\!256$ this is $\sim$15k tokens, $\sim$$2.3\times10^8$ pairwise terms per layer: infeasible on one
GPU. **Factorization** (§7) reduces this to $O(N^2 d)$ (spatial) $+\,O(T^2 d)$ (temporal) per token,
the single most important architectural decision for tractability.

## 3.4 LayerNorm and residual learning

**LayerNorm** normalizes each token across its feature dimension:

$$
\mathrm{LN}(h)=\gamma\odot\frac{h-\mu_h}{\sqrt{\sigma_h^2+\epsilon}}+\beta,\quad
\mu_h=\tfrac1d\!\sum_i h_i,\;\sigma_h^2=\tfrac1d\!\sum_i (h_i-\mu_h)^2 .
$$

It stabilizes the scale of activations independent of batch (unlike BatchNorm), which matters for the
small, variable batches here. In the JEPA loss the target is LayerNorm'd *without* affine
($\gamma{=}1,\beta{=}0$) so the regression is not dominated by a few high-variance latent dimensions
and the target scale is stationary as the EMA encoder drifts. **Pre-norm residual blocks**
$h\leftarrow h+\mathrm{Attn}(\mathrm{LN}(h))$, $h\leftarrow h+\mathrm{MLP}(\mathrm{LN}(h))$ give a
clean identity path: the Jacobian of a block is $I + \partial(\cdot)$, so gradients flow even through
deep stacks (He et al., 2016; Xiong et al., 2020) — the reason 6–12 layer stacks train without warmup
pathologies.

## 3.5 EMA target and why it prevents collapse

The target encoder is an exponential moving average of the online encoder:

$$
\xi \leftarrow \tau\,\xi + (1-\tau)\,\theta,
$$

updated **after** each optimizer step, with $\tau$ ramped on a schedule $\tau:\;0.996\to1.0$ (cosine
or linear over training). Two facts make this an anti-collapse mechanism. First, $f_\xi$ receives **no
gradient** (`requires_grad=False` + the target is detached in the loss), so the trivial solution
"both encoders output a constant" is not directly optimized. Second, $\xi$ *lags* $\theta$: the target
is a slowly-moving teacher, so "match the target" is a moving goalpost — the system cannot instantly
satisfy it by collapsing, which (combined with the predictor asymmetry, §3.7) makes the constant
solution a non-stationary, unstable fixed point (the BYOL/SimSiam analysis; Grill et al., 2020; Chen &
He, 2021; Tian et al., 2021). The momentum schedule starts loose (0.996, fast-moving teacher early
when the student is random) and tightens to 1.0 (frozen teacher late, for a stable target).

## 3.6 Optimization: AdamW, cosine schedule, decoupled weight decay

**AdamW** (Loshchilov & Hutter, 2019) maintains first/second moment estimates

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
$$

bias-corrects $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$, and updates

$$
\theta_t=\theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda_{\text{wd}}\,\theta_{t-1}\Big).
$$

The **decoupling** of weight decay from the adaptive denominator is the "W": in vanilla Adam, $L_2$
regularization gets divided by $\sqrt{\hat v_t}$ and is therefore applied unevenly across parameters;
AdamW applies $\lambda_{\text{wd}}\theta$ directly, recovering true weight decay. We **cosine-ramp**
$\lambda_{\text{wd}}$ from 0.04 → 0.40 over training (I-JEPA's recipe), increasing regularization as
features sharpen. The **learning rate** uses linear warmup then cosine decay,

$$
\eta_t=\begin{cases}\eta_{\max}\,t/t_{\text{warm}} & t<t_{\text{warm}}\\[4pt]
\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos\pi\tfrac{t-t_{\text{warm}}}{t_{\text{tot}}-t_{\text{warm}}}\big)&t\ge t_{\text{warm}},\end{cases}
$$

warmup avoiding the large early Adam steps that destabilize attention, cosine decay annealing into a
flat minimum.

## 3.7 The predictor bottleneck (asymmetry)

The predictor width $D_p$ is strictly **less** than the encoder width $D$ (default $384<512$ on
satellite; $64<128$ on finance/industrial). The asymmetry — wide encoder, narrow predictor — is the
second half of the anti-collapse mechanism. Intuition: a collapsed encoder ($f\equiv c$) makes the
prediction task trivial, but the *gradient* that the predictor sends back to the encoder, filtered
through a narrow bottleneck, cannot reinforce the collapse direction as effectively as a full-rank
predictor could; the SimSiam/BYOL line of work shows the predictor + stop-grad together approximate an
expectation-maximization that avoids the collapsed fixed point (Tian et al., 2021). We *assert*
$D_p<D$ in code (`build_model` clamps a mis-set config), having been bitten once by a config with
$D_p>D$.

## 3.8 VICReg: variance and covariance regularization

Even with EMA + predictor + stop-grad, *temporally adjacent frames are nearly identical* (tomorrow ≈
today; cycle $t{+}1\approx t$; consecutive acquisitions of a field), so "predict the future latent" is
trivially solvable by emitting a constant. We add a **VICReg** term (Bardes et al., 2022) on the
trainable context embedding $z\in\mathbb R^{B\times D}$ (flattened over tokens):

$$
\mathcal L_{\text{var}}=\frac1D\sum_{j=1}^{D}\max\!\big(0,\;\gamma-\sqrt{\mathrm{Var}(z_{:,j})+\epsilon}\big),\qquad
\mathcal L_{\text{cov}}=\frac1D\sum_{i\neq j}\big[\mathrm{Cov}(z)\big]_{ij}^2 .
$$

**Variance term derivation & intent.** $\mathcal L_{\text{var}}$ is a hinge that activates when a
dimension's batch standard deviation falls below $\gamma{=}1$; its gradient pushes that dimension's
spread back up, directly forbidding the constant solution ($\sigma{\to}0$). **Covariance term.**
$\mathrm{Cov}(z)=\tfrac1{B-1}(z-\bar z)^\top(z-\bar z)$; penalizing squared off-diagonals decorrelates
features, forbidding *dimensional* collapse (all dimensions encoding the same thing). We use
$\lambda_{\text{var}}{=}1.0$, $\lambda_{\text{cov}}{=}0.04$ — setting both to 0 recovers pure I-JEPA
and, on real data, reproduces collapse (we verified: effective rank $\sim$110 → $\sim$2.3 on finance,
and std → 0.04 on satellite). The full loss is

$$
\boxed{\;\mathcal L = \underbrace{\big\|\,g_\phi(z_{\text{ctx}})-\mathrm{sg}[\mathrm{LN}(f_\xi(x_{\text{tgt}}))]\,\big\|^2}_{\text{latent prediction}}
\;+\;\lambda_{\text{var}}\mathcal L_{\text{var}}(z_{\text{ctx}})\;+\;\lambda_{\text{cov}}\mathcal L_{\text{cov}}(z_{\text{ctx}})\;}
$$

## 3.9 Collapse diagnostics: effective rank and intrinsic dimensionality

A falling loss is *not* success — collapse also drives the loss down. We monitor the **effective
rank** of the embedding covariance. Let $\sigma_1\ge\cdots\ge\sigma_D\ge0$ be the singular values of
the centered embedding matrix and $p_i=\sigma_i/\sum_j\sigma_j$. The effective rank (Roy & Vetterli,
2007) is the exponential of the spectral entropy:

$$
\mathrm{erank}(z)=\exp\!\Big(-\sum_{i=1}^{D} p_i\log p_i\Big)\in[1,D].
$$

$\mathrm{erank}=1$ iff all variance is in one direction (rank-1 collapse); $\mathrm{erank}=D$ iff the
spectrum is flat (isotropic). We log per-dimension std, effective rank, and the
predictor/target variance ratio every $N$ steps on the *trainable* branch (the EMA target lags and can
mask an in-progress collapse). Healthy training shows erank *climbing* (e.g. C-MAPSS $2\to118/128$),
which is the signature of an enriching representation.

## 3.10 Temporal positional encodings

Tokens carry identity but attention is permutation-equivariant, so position must be injected.

**Sinusoidal (Vaswani).** For position $p$ and dimension $2i/2i{+}1$,
$\mathrm{PE}(p,2i)=\sin(p/10000^{2i/d})$, $\mathrm{PE}(p,2i{+}1)=\cos(\cdot)$. The geometric
frequencies give the network a basis to represent relative offsets via angle-addition identities.

**Day-of-year (DOY) — periodic.** PASTIS/finance acquisitions are irregularly spaced and span a year
boundary, so we encode the *calendar* DOY $d\in[1,366]$ with phase $\theta=2\pi d/366$, periodic over
a year:

$$
\mathrm{PE}_{\text{DOY}}(d)=\big[\sin(\theta f_1),\cos(\theta f_1),\dots\big],\quad \theta=2\pi d/366 .
$$

This makes "$\Delta$ steps ahead" a *physical* notion of elapsed time and lets the model exploit
seasonality.

**Cycle index — monotonic.** C-MAPSS operating cycles are monotone (1,2,…,543) and *not* periodic; a
period-366 phase would *wrap* (cycle 367 ≡ cycle 1), destroying ordering. The single code change for
the industrial domain is a configurable `temporal_period`; we set it to 1024 > the longest engine so
phases stay monotone and distinct. This one parameter is the entire modality-specific adaptation of
the encoder.

## 3.11 Masking and linear probing

**Masking** defines the pretext split. Spatial JEPA uses I-JEPA multi-block masking (sample target
blocks first, then a context block with overlap removed → disjoint sets, no trivial copy); on small
cross-sections (assets/sensors) this becomes a random disjoint partition of the $N$ tokens. Temporal
JEPA's "mask" is a *causal* split: a per-sample rank $s$ with context = frames $\le s$, target = frame
$s{+}\Delta$, enforced by a context-only attention mask. **Linear probing** is the evaluation contract:
freeze $f$, fit only a linear (or kNN) head on top, measure. A linear probe tests whether the needed
information is *linearly accessible* in $z$ — a strict, low-capacity readout that cannot manufacture
structure the encoder did not provide, which is what makes the random-init and raw-feature controls
meaningful (§10).

---

# 4. Literature Review

For each method: objective, architecture, loss, strengths, weaknesses, and its relation to this work.

**I-JEPA (Assran et al., 2023).** *Objective:* predict latent of masked target blocks from a visible
context block. *Arch:* ViT context/target encoders (target = EMA) + narrow predictor. *Loss:* L2 in
latent space. *Strengths:* avoids pixel-level detail; strong linear-probe features; no hand-crafted
augmentations. *Weaknesses:* purely spatial — discards time. *Relation:* our **direct ancestor**; we
revert item-1 (spatial → causal-temporal split) and add VICReg + DOY encoding.

**V-JEPA (Bardes et al., 2024).** *Objective:* latent prediction of masked spatiotemporal tubes in
video. *Arch:* video ViT + EMA target + predictor. *Loss:* L1/L2 latent. *Strengths:* learns motion
features without pixels. *Weaknesses:* **bidirectional / non-causal** masking — it is a *denoiser in
spacetime*, not a forward dynamics model; built for dense RGB video, not irregular multivariate
panels. *Relation:* we are the *causal* specialization (context strictly precedes target), which is
the world-model framing V-JEPA does not take.

**MAE (He et al., 2021).** *Objective:* reconstruct masked pixels. *Arch:* asymmetric ViT
encoder–decoder; encoder sees ~25% of patches. *Loss:* pixel MSE on masked patches. *Strengths:*
simple, scalable, great fine-tuning. *Weaknesses:* spends capacity on appearance; linear-probe lags;
no temporal modeling in the vanilla form. *Relation:* a **baseline** in all three domains (we
reconstruct masked sensors/assets/patches). It *wins on finance* — telling, because on a
non-stationary modality a generic reconstruction prior is more robust than a predictive one (§12,§14).

**BYOL (Grill et al., 2020).** *Objective:* online net predicts target net's projection of another
augmented view; no negatives. *Arch:* online (encoder+proj+pred) vs EMA target (encoder+proj). *Loss:*
$2-2\cos$. *Strengths:* no negatives, strong features. *Weaknesses:* relies on augmentations encoding
the right invariances; can collapse without care. *Relation:* a **baseline** and JEPA's closest cousin
(EMA + predictor + stop-grad) — isolating "what does *predicting a hidden region* add over *enforcing
view-invariance*?" On finance, BYOL is among the strongest; on satellite/industrial it loses to
temporal JEPA — invariance discards the temporal change that those tasks need.

**SimCLR (Chen et al., 2020).** *Objective:* contrastive NT-Xent over two views. *Arch:* encoder +
MLP projector. *Loss:* InfoNCE. *Strengths:* principled MI bound. *Weaknesses:* needs many negatives →
large batches; our small panels starve it. *Relation:* the contrastive **baseline**; consistently
weakest on the small-cross-section domains, as theory predicts.

**CPC (Oord et al., 2018).** *Objective:* predict future latents via InfoNCE with an autoregressive
context. *Relation:* the **conceptual ancestor** of temporal JEPA; we replace contrastive
future-prediction with regression-in-latent + EMA + VICReg, removing negative sampling.

**TS2Vec (Yue et al., 2022) / TS-TCC / TNC.** Contrastive temporal SSL with hierarchical/temporal
augmentations for generic time series. *Relation:* same problem family (industrial/financial TS); a
natural *additional* temporal baseline (future work, §19). Our objective is predictive, not
contrastive.

**SatMAE (Cong et al., 2022) / SSL4EO (Wang et al., 2023).** MAE-style and contrastive SSL for
satellite imagery, often with temporal/multispectral encodings. *Relation:* domain-specific
state-of-practice for PASTIS-like data; our temporal-JEPA is a *different objective* on the same
modality, and the mechanistic result (§15) explains *why* a temporal objective helps remote sensing.

**Industrial SSL (PHM).** RUL is usually tackled *supervised* (LSTM/CNN/transformer regressors,
end-to-end). Frozen-SSL comparisons are rarer; our C-MAPSS study is, to our knowledge, the first
controlled temporal-JEPA-vs-MAE/BYOL/SimCLR frozen-probe comparison with raw/random floors.

**Financial SSL.** Predominantly contrastive or autoencoding on returns; the efficient-market prior
makes representation gains hard to demonstrate. Our negative result (no SSL beats raw features
out-of-time) is consistent with that difficulty and quantifies it with controls.

---

# 5. Research Motivation

The reasoning chain that produces Temporal JEPA:

1. **The data is a time series of the same entities.** A PASTIS patch is one place re-imaged 38–61
   times; a market is one set of sectors re-priced daily; an engine is one unit re-sensed each cycle.
   The *identity* axis (which pixel/asset/sensor) and the *time* axis (which acquisition/day/cycle)
   are both present and meaningful.
2. **Spatial-only objectives discard the time axis.** MAE/SimCLR/BYOL/I-JEPA, applied per frame, treat
   acquisitions as i.i.d. images. For crop type — defined by *how a parcel changes through the season*
   — this throws away the discriminative signal.
3. **The downstream signal is temporal.** Crop phenology, market regime, engine health are all
   *trajectory* properties. A representation that encodes *where on the trajectory* a sample sits is
   what these tasks need.
4. **Forcing forward prediction encodes the trajectory.** To predict the future latent, the encoder
   must model the dynamics — phenological progression, regime persistence, degradation rate. This is
   the World-Models / PSR argument made concrete.
5. **But only if the future is predictable.** If the next latent is (nearly) a deterministic-plus-smooth
   function of the past, the objective has signal; if it is a martingale, the objective's target is
   noise and (Information-Bottleneck, §2.2/§3.2) the representation is filtered *away* from useful
   static structure. This is the crux that makes the hypothesis *falsifiable* and motivates spanning
   the predictability axis with three domains.

Hence: a **causal temporal JEPA**, tested on a predictable, an unpredictable, and a very-predictable
modality, with floors that bound how much *any* representation could win.

---

# 6. Research Hypotheses

We state hypotheses formally; expected outcomes are committed *before* the experiments.

- **H1 (objective superiority).** For temporally evolving panels, causal future-latent prediction
  yields frozen representations with higher downstream quality than spatial masking, reconstruction,
  contrastive, and self-distillation objectives, under matched architecture and epochs.
  *Expected:* true on predictable domains; **falsifiable** on unpredictable ones.
- **H2 (temporal ≻ spatial).** Causal temporal JEPA ≻ Spatial (I-JEPA-style) JEPA on the same encoder.
  *Expected:* true wherever any temporal signal exists; the most robust prediction.
- **H3 (persistence dependence).** The benefit of H1 is monotone in the *temporal persistence /
  predictability* of the latent trajectory. *Expected:* PASTIS (high) > C-MAPSS (very high but easy) ;
  finance (low) → benefit vanishes or inverts.
- **H4 (non-stationarity failure).** Under strong non-stationarity + near-martingale dynamics, temporal
  JEPA underperforms reconstruction/contrastive baselines and can fall **below a random-init encoder**
  (the objective is *actively harmful*). *Expected:* the finance outcome.
- **H5 (horizon).** On predictable domains, downstream quality is approximately horizon-insensitive
  (predicting $\Delta$ ahead is learnable for a range of $\Delta$); on unpredictable domains it
  degrades monotonically with $\Delta$. *Expected:* satellite flat, finance monotone-worse, C-MAPSS
  flat.
- **H6 (anti-collapse necessity).** VICReg is necessary whenever consecutive frames are highly
  correlated; its necessity scales with how much the predictive signal *fails* to constrain the
  representation. *Expected:* essential on satellite/finance; less critical (but still helpful) on
  C-MAPSS where one dominant signal self-stabilizes.

All six are evaluated in §11–§16; the scorecard is in §14.

---

# 7. Architecture

The architecture is a single factorized space–time encoder reused across domains; only the
tokenizer and temporal-position module change. We explain each block, why it exists, and why
alternatives were rejected.

```
                  ┌──────────────────────── CONTEXT path (trainable f_θ) ─────────────────────────┐
 X[B,T,N,F] ─────►│  FrameTokenizer  →  + token-pos  →  Spatial(cross-section) ViT  (per frame)   │
 (past frames)    │        │                                          │                            │
                  │        └──reshape (B,T,N,D)──►  + temporal-pos  →  Temporal Transformer (over T)│
                  │                                                    │  → masked-mean over past   │
                  └────────────────────────────────────────────────── z_ctx [B,N,D] ──────────────┘
                                                          │
              target_pos = token_pos + temporal_pos(future)│
                                                          ▼
                  ┌──── PREDICTOR g_φ (NARROW, D_p<D) ────┐
                  │ proj z_ctx→D_p ; append mask tokens   │ → ẑ_future [B,N,D]
                  │ (+ target_pos) ; transformer ; read   │
                  └───────────────────────────────────────┘
 future frame ──► TARGET encoder f_ξ (EMA, stop-grad) → LN → z_future [B,N,D]
                                                          │
   Loss = ‖ ẑ_future − sg(z_future) ‖²  + λv·Var(z_ctx) + λc·Cov(z_ctx)  ◄┘
```

## 7.1 Frame tokenizer (the only input-specific module)

A frame is mapped to $N$ tokens of width $D$.

- **PASTIS:** a $\mathrm{Conv2d}(C{=}10\to D,\ \text{kernel}{=}\text{stride}{=}P)$ patchifies a
  $10\times128\times128$ acquisition into $N=(128/P)^2$ tokens ($P{=}8\Rightarrow N{=}256$). Conv is
  the natural choice: it is a learnable linear projection of non-overlapping patches in one op, and the
  10-band input requires `in_channels=10` (not 3).
- **Finance / C-MAPSS:** a $\mathrm{Linear}(F\to D)$ applied per token (asset-day or sensor-cycle).
  There is no 2-D grid, so Conv is inapplicable; a shared linear projection is the analogue. $F$ is
  small (4 for finance, 3 for C-MAPSS), so each token is a learned $D$-vector modulated by a few causal
  features.

*Rejected alternative:* one token per frame ($N{=}1$, the whole cross-section concatenated). This
destroys the cross-sectional structure, makes the spatial ViT vacuous, and leaves Spatial JEPA with
nothing to mask — defeating the H2 comparison.

## 7.2 Spatial (cross-sectional) ViT

A pre-norm transformer stack over the $N$ tokens of a single frame (depth 6 satellite / 4
finance-industrial). It mixes information *within* a time step: across pixels (texture/context),
across assets (cross-sectional co-movement), across sensors (sensor correlations). *Why:* intra-frame
context is genuinely informative (a parcel's neighborhood; the market's breadth; correlated sensor
banks). *Rejected:* per-token MLP (no token mixing → cannot model cross-sectional structure).

## 7.3 Token positional encoding

- **PASTIS:** fixed 2-D sin/cos over the $(H',W')$ patch grid (row and column each get a 1-D table,
  concatenated) — patches have a true spatial order.
- **Finance / C-MAPSS:** a **learned** per-token embedding $\in\mathbb R^{N\times D}$. Assets and
  sensors have *identity but no metric order*, so a sinusoid over an arbitrary index would impose a
  false geometry; a learned embedding lets the model discover relationships (e.g. cyclical vs defensive
  sectors; correlated sensor groups). This is a deliberate inductive-bias choice, not an oversight.

## 7.4 Temporal transformer

After spatial encoding, tokens are reshaped to $(B,T,N,D)$ and a transformer attends **over time**,
*per token position* (we fold $N$ into the batch so each spatial location attends across its own
history). It adds the temporal positional encoding (DOY or cycle) along $T$ and honors the
per-frame key-padding mask. *Why factorize?* §3.3: full 3-D attention is $O((TN)^2)$ and infeasible;
factorized space-then-time is $O(N^2)+O(T^2)$ and tractable, at the cost of not modeling
space–time interactions in a single layer (recovered across stacked layers). *This temporal
transformer is the module the baselines never train* — MAE/BYOL/SimCLR are spatial-only — which is the
crux of the fair-evaluation pathway (§10).

## 7.5 Predictor (narrow transformer)

Given context tokens $z_{\text{ctx}}\in\mathbb R^{B\times N\times D}$ and target positions, the
predictor (i) projects $z_{\text{ctx}}$ to $D_p$, (ii) appends one shared learnable **mask token** per
target slot, each summed with that slot's positional embedding (spatial pos for Spatial JEPA; spatial
+ future-temporal pos for Temporal JEPA — the query must say *where and when* it predicts), (iii) runs
a $D_p$-width transformer (depth 6/4), (iv) reads out the mask-token slots and projects back to $D$.
*Why narrow:* §3.7, the asymmetry bottleneck is half the anti-collapse mechanism. *Why a transformer
(not an MLP):* targets at different positions must be predicted *jointly and conditioned on context*;
attention provides exactly this conditioning.

## 7.6 EMA target encoder

A deep copy of the context encoder with `requires_grad=False`, updated by EMA (§3.5). It encodes the
*target* (future frame / masked blocks) to produce the regression target, which is LayerNorm'd and
detached. *Why a separate EMA encoder (not the online encoder):* using the online encoder for the
target invites collapse and a degenerate "predict yourself" shortcut; the lagging teacher breaks it.

## 7.7 Pooling / representation extraction (evaluation)

Downstream, each sample is reduced to one vector. JEPA encoders use the **temporal pathway**
(`encode_temporal`) then mean-pool over (time × tokens); spatial-only baselines use the **per-frame
pathway** (`encode_full`) then masked-mean over time — each method read through the representation it
actually learned. For dense PASTIS segmentation the token grid is instead bilinearly upsampled to
pixel resolution and a $1\times1$-conv (or light 2-layer conv) probe predicts per-pixel logits. *Why
mean-pool:* a parameter-free, low-bias summary that does not give any method extra capacity; the
probe, not the pooling, is where signal is read.

---

# 8. Complete Mathematical Formulation

## 8.1 Forward pass

Let $X\in\mathbb R^{B\times T\times N\times F}$ (with $F$ folded into the tokenizer for PASTIS). The
context encoder computes, for the causal context mask $M\in\{0,1\}^{B\times T}$ (1 = past/visible):

$$
\begin{aligned}
H^{(0)}_{b,t} &= \mathrm{Tokenize}(X_{b,t}) + \mathrm{pos}_{\text{tok}} \in\mathbb R^{N\times D},\\
H^{(\ell)}_{b,t} &= \mathrm{Block}^{(\ell)}_{\text{sp}}(H^{(\ell-1)}_{b,t}),\quad \ell=1..L_{\text{sp}},\\
\tilde H_{b,:,n} &= \mathrm{TemporalStack}\big(H^{(L_{\text{sp}})}_{b,:,n} + \mathrm{pos}_{\text{time}}(d_{b,:}),\ \text{kpm}=M_b\big),\\
z^{\text{ctx}}_{b,n} &= \frac{\sum_t M_{b,t}\,\tilde H_{b,t,n}}{\sum_t M_{b,t}} \quad\text{(masked-mean over past)} .
\end{aligned}
$$

The predictor then forms, with future target position(s):

$$
\hat z_{b,n} = \mathrm{OutProj}\Big(\mathrm{PredStack}\big[\underbrace{\mathrm{InProj}(z^{\text{ctx}}_b)}_{\text{context}},\ \underbrace{\text{masktok}+\mathrm{PosProj}(\text{pos}_{\text{tgt},b})}_{\text{queries}}\big]\Big)_{\text{tgt slots}} .
$$

The target encoder (no grad) encodes the future frame $x^{\text{tgt}}_b=X_{b,\,s_b+\Delta}$:

$$
z^{\text{tgt}}_{b,n} = \mathrm{LN}\big(f_\xi(x^{\text{tgt}}_b)\big)_{n},\qquad \text{(then stop-grad)} .
$$

## 8.2 Attention (one block, explicit)

For tokens $H\in\mathbb R^{n\times d}$, heads $h=1..H_{\text{heads}}$, $d_h=d/H_{\text{heads}}$:

$$
Q^{(h)}=H W_Q^{(h)},\;K^{(h)}=HW_K^{(h)},\;V^{(h)}=HW_V^{(h)},\quad
A^{(h)}=\mathrm{softmax}\!\Big(\tfrac{Q^{(h)}K^{(h)\top}}{\sqrt{d_h}}+ \text{mask}\Big),
$$
$$
\mathrm{MHA}(H)=\big[A^{(1)}V^{(1)}\,\|\cdots\|\,A^{(H_{\text{heads}})}V^{(H_{\text{heads}})}\big]W_O,
$$
$$
H'=H+\mathrm{MHA}(\mathrm{LN}(H)),\qquad H''=H'+\mathrm{MLP}(\mathrm{LN}(H')),\;\;\mathrm{MLP}(u)=W_2\,\mathrm{GELU}(W_1 u).
$$

The additive $\text{mask}$ is $-\infty$ on padded/future keys (key-padding & causal masks).

## 8.3 Causal split (no future leakage) — the property that matters most

Per sample $b$, with $n_b=\sum_t M^{\text{real}}_{b,t}$ real frames, draw a split rank

$$
s_b \sim \mathrm{Unif}\{\,c_{\min}-1,\ \dots,\ n_b-1-\Delta\,\},\qquad \text{target index } = s_b+\Delta,
$$

and set the context mask $M_{b,t}=\mathbb 1[t\le s_b]\cdot M^{\text{real}}_{b,t}$. Because $\Delta\ge1$,
*every context time index $<$ target index*: no future information can reach the context (enforced both
in the attention mask and the pool). This is the single most important correctness invariant
(unit-tested: `test_temporal_mask`, `test_finance_model::test_temporal_no_future_leakage`). The split
is **per-sample** for temporal diversity.

## 8.4 Losses

$$
\mathcal L_{\text{pred}}=\frac1{BND}\sum_{b,n}\big\|\hat z_{b,n}-\mathrm{sg}[z^{\text{tgt}}_{b,n}]\big\|_2^2,
\qquad \text{(or }\ell_1\text{ for the V-JEPA ablation).}
$$

With $z\equiv z^{\text{ctx}}$ reshaped to $(BN)\times D$, centered $\bar z=z-\mathbb E[z]$,
$C=\tfrac1{BN-1}\bar z^\top\bar z$:

$$
\mathcal L_{\text{var}}=\tfrac1D\!\sum_j \max(0,1-\sqrt{C_{jj}+\epsilon}),\qquad
\mathcal L_{\text{cov}}=\tfrac1D\!\sum_{i\neq j} C_{ij}^2,
$$
$$
\mathcal L=\mathcal L_{\text{pred}}+\lambda_{\text{var}}\mathcal L_{\text{var}}+\lambda_{\text{cov}}\mathcal L_{\text{cov}}.
$$

## 8.5 EMA + optimization recap

$$
\theta\leftarrow\theta-\eta_t\,\mathrm{AdamW}(\nabla_\theta\mathcal L);\qquad
\xi\leftarrow\tau_t\xi+(1-\tau_t)\theta\ \text{(after the step)};\qquad
\tau_t:0.996\to1.0,\ \eta_t:\text{warmup→cosine},\ \lambda_{\text{wd}}:0.04\to0.40 .
$$

## 8.6 Complexity analysis

Let $L_{\text{sp}},L_{\text{te}},L_p$ be spatial/temporal/predictor depths.

- **Time per sample:** spatial $O(L_{\text{sp}}\,T\,N^2 D)$ + temporal $O(L_{\text{te}}\,N\,T^2 D)$ +
  predictor $O(L_p (N+N_{\text{tgt}})^2 D_p)$. Factorization replaces the infeasible
  $O((TN)^2D)$ with the sum of two quadratics.
- **Memory:** activations $O(L\,T\,N\,D)$ dominated by the spatial stack over all $T{\cdot}N$ frame-tokens;
  bounded by gradient checkpointing ($O(\sqrt L)$ activation memory) and frame-chunked pooling for the
  baselines. PASTIS at $P{=}8$ ($N{=}256$, $T{\le}32$) peaks $\sim$6 GB at batch 16; finance/industrial
  panels ($N\!\le\!17$) are tiny.
- **Parameters:** finance/industrial model $\approx$1.8 M trainable; satellite $\approx$ tens of M at
  $D{=}512$.

---

# 9. Implementation

```
configs/      yaml configs (data + model). model/{tjepa*,fjepa,cjepa}.yaml; data/{pastis,finance,cmapss}.yaml
data/         pastis_dataset.py | finance_dataset.py | cmapss_dataset.py + variable-length / window collates
masking/      multiblock.py (spatial I-JEPA) | temporal_mask.py (causal split) | asset_mask.py (sensor/asset masking)
models/       patch_embed, pos_embed, vit, temporal_encoder, predictor, jepa (satellite) ;
              finance_encoder (PanelEncoder), finance_jepa (FinanceJEPA, build_finance_model)  [reused for C-MAPSS]
objectives/   jepa_loss.py (latent loss + variance_covariance_reg) ; baselines/{mae,byol,simclr}.py
engine/       train_jepa.py | train_baselines.py | train_finance.py (JEPA + MAE/BYOL/SimCLR for panels) ; ema.py ; diagnostics.py
eval/         linear_probe.py, knn.py, fewshot.py (satellite) ; finance_tasks.py ; cmapss_tasks.py
scripts/      download_{pastis,finance,cmapss}.py ; run_{matrix,finance_matrix,cmapss_matrix}.py ; aggregate*.py ;
              {overfit8,finance,cmapss}_smoketest.py ; mechanistic.py ; feature_figure.py
utils/        seed, config, checkpoint (RNG state), gpu_hours (device-aware), device (single GPU knob)
tests/        masking/leakage/ema/loss/diagnostics + per-domain dataset/model/task tests
```

**Reuse is the engineering thesis.** The satellite JEPA (`models/jepa.py`) is left *untouched* so its
results stay reproducible; the finance stack (`PanelEncoder`, `FinanceJEPA`) is the generic
"panel of $N$ entities × $F$ features over $T$ steps" version, and **C-MAPSS reuses it verbatim** — the
only change is threading a `temporal_period` argument (default 366, behaviour-preserving; verified by
re-running the full prior 34-test suite) so monotonic cycle indices don't wrap.

**Training infra.** Mixed precision (`torch.autocast` + `GradScaler`); gradient accumulation to hit a
fixed effective batch (192 satellite, 128 finance, 256 industrial); cosine LR + cosine weight-decay;
EMA after the optimizer step; collapse diagnostics every $N$ steps **paired with the loss**.
**Checkpointing** saves model+optimizer+scaler+RNG state for exact resume. **GPU-hour metering** is
device-aware (a real bug fixed earlier: querying GPU0 while training on GPU2 reported 0 memory).
**Reproducibility:** `seed_everything` seeds Python/NumPy/torch/CUDA; probes are seeded so reported
numbers are reproducible to $\pm0.1$.

**Correctness harness.** (i) The **M1 gate** smoke tests overfit 8 samples / a tiny batch and require
loss ↓ *while* std/effective-rank stay healthy — catching collapse in minutes. (ii) Unit tests assert
the load-bearing invariants: masking disjointness, *no future leakage*, EMA frozen/ramping, loss
stop-gradient, diagnostics distinguishing collapse, full-forward gradient routing
(target encoder gets **no** grad, context encoder does). 47 pass / 3 skip across the project.

---

# 10. Experimental Design

**The contract.** Fix *everything* except the pretext objective. Every method trains the *same*
encoder backbone on the *same* train data for the *same* epochs; only the objective and its minimal
head differ. After pretraining the encoder is **frozen** and read by light probes. This isolates the
single scientific variable: *what does the pretext task buy?*

**Why frozen, not fine-tuned.** Fine-tuning conflates representation quality with the encoder's
plasticity and the head's capacity; freezing measures the representation *as learned*. Absolute numbers
are therefore below end-to-end supervised ceilings (e.g. U-TAE 63.1 mIoU on PASTIS; supervised RUL
RMSE ~12–16 on C-MAPSS) — the *ordering across objectives* is the result.

**Baselines and why each exists.**
- *Spatial JEPA* — the **direct** comparator: same JEPA machinery, spatial masking instead of temporal
  split. Isolates the value of the *temporal* objective (H2).
- *MAE / BYOL / SimCLR* — the three other SSL paradigms (reconstruction / self-distillation /
  contrastive), trained on the *same* backbone (H1, H3).
- *random-init* — the **floor that bounds the architecture**: the same network *untrained*. If a
  trained method does not beat it, the *learning* added nothing. (The control that exposed the finance
  failure.)
- *raw_features* — the **floor that bounds the data**: the probes on the mean-pooled raw input, *no
  encoder*. If no SSL beats it, pretraining buys nothing over the engineered features. (The bar finance
  failed and C-MAPSS cleared.)

**Leakage prevention.** *Satellite:* official 5-fold split (train {1,2,3}, val {4}, test {5});
probes fit on train folds, reported on val/test. *Finance:* strict **out-of-time** split (train ≤
2017, test ≥ 2018) with a **purge gap** of `window+max_horizon` days so no train window's forward
label reaches into test (unit-tested). *Industrial:* C-MAPSS ships **disjoint train/test engines**
(test truncated) — no contamination by construction. Windows never cross an engine/series boundary.

**Evaluation philosophy.** Read each representation *three or more independent ways* (dense mIoU + kNN
+ few-shot on satellite; five tasks each on finance/industrial) so no conclusion hinges on one probe's
quirks, and always against both floors. Monitor collapse rather than assume it.

---

# 11. Satellite Experiments (PASTIS — the predictable, *winning* case)

## 11.1 Dataset & statistics
**PASTIS** (Garnot & Landrieu, ICCV 2021; Zenodo 5012942): 2,433 Sentinel-2 patches of
$128\times128$ px, **10 spectral bands**, **38–61 irregularly-spaced acquisitions** per patch
(Sep-2018 → Nov-2019), with dense semantic labels: **0 = background, 1–18 = crop types, 19 = void**
(20 values; void ignored). Inputs $X\in\mathbb R^{T\times10\times128\times128}$ + acquisition
day-of-year $d\in[1,366]$. Because acquisitions span a year boundary, **DOY wraps** (350 → 17);
chronological order is the acquisition index, DOY is used only as a periodic time encoding. Official
5-fold CV: train {1,2,3}, val {4}, test {5}.

## 11.2 Architecture & training
$P{=}8$ patches ($N{=}256$ tokens/frame), $D{=}512$, spatial depth 6, temporal depth 4, 8 heads;
predictor $D_p{=}384$, depth 6, 12 heads; horizon $\Delta{=}1$, $c_{\min}{=}4$; VICReg
$\lambda_v{=}1.0,\lambda_c{=}0.04$; EMA $0.996\to1.0$; AdamW lr $10^{-3}$, 15-epoch warmup, wd
$0.04\to0.40$, **100 epochs**, effective batch 192 (batch 16 × grad-accum 12), gradient checkpointing
(fits 8 GB). Baselines train the same backbone, equalized to effective batch 192.

## 11.3 Evaluation
Frozen encoder; three independent probes: **dense conv-mIoU** (1×1 conv [linear] + a light 2-layer
conv head on per-pixel features), **parcel kNN** (mean-pool encoder features per field → 20-NN), and
**few-shot** (probe on 1/5/10 % labels). 3 seeds with a paired $t$-test vs temporal.

## 11.4 Results

**Main comparison — conv mIoU, val fold, mean ± std (3 seeds), paired $t$-test vs Temporal:**

| Method | conv mIoU | Δ vs temporal | $t$-test $p$ |
|---|---|---|---|
| **Temporal JEPA (Δ=1)** | **22.3 ± 1.8** | — | — |
| Spatial JEPA | 16.2 ± 0.4 | +6.0 | **0.041** |
| Spatial JEPA (compute-matched, 3.5× epochs) | 15.8 ± 1.2 | +6.5 | **0.036** |
| SimCLR | 7.3 ± 0.8 | +15.0 | **0.009** |
| BYOL | 7.1 ± 0.9 | +15.2 | **0.001** |
| MAE | 6.5 ± 1.1 | +15.8 | **0.009** |
| *Supervised U-TAE (end-to-end ceiling, not a frozen peer)* | *63.1* | — | — |

**Test fold (1 seed):** temporal **22.1**, spatial 16.1, compute-matched spatial 17.1, SimCLR 7.1,
BYOL 4.9, MAE 3.6. **Few-shot (test, 1/5/10 % labels):** temporal **9.2 / 13.1 / 15.9** vs spatial
4.6 / 6.9 / 9.5 — temporal wins at *every* fraction and the gap *widens* as labels shrink (+37 % at
full → **+100 % at 1 %**). **Parcel kNN (val):** temporal **65.5**, BYOL 62.7, spatial 58.7, SimCLR
54.6, MAE 54.4.

**Horizon study (3 seeds, conv mIoU):** Δ=1/2/4/8 = **22.3 / 20.8 / 21.8 / 22.6** — flat within noise,
none differs from Δ=1 ($p>0.1$), every horizon beats spatial (16.2). (H5: horizon-insensitive ✓.)

**VICReg ablation:** with $\lambda_v{=}\lambda_c{=}0$ (pure I-JEPA) the model **collapses on real
PASTIS** — loss → 0, per-dim std → 0.04, effective rank → 2.4 — because consecutive acquisitions of a
field are near-identical. VICReg-on holds std ~1.0. (H6 ✓: essential here.)

## 11.5 Interpretation & scientific conclusions
- **H1, H2, H3 all supported and significant.** Temporal beats Spatial by **+6.0 mIoU** ($p{=}0.041$)
  and every reconstruction/contrastive baseline by **+15–16 mIoU** ($p<0.01$), across 3 seeds, three
  independent probes, and it generalizes val→test. The compute-matched control (spatial trained 3.5×
  longer → no gain) shows the win is the **objective, not compute**.
- **Data-efficiency is the SSL story:** the temporal advantage grows as labels shrink (few-shot).
- **Why it wins (mechanism, §15):** the future-prediction objective makes the encoder
  *phenology/season-aware*, which is exactly the crop-discriminative signal.

---

# 12. Financial Experiments (S&P-500 — the unpredictable, *failing* case)

## 12.1 Why finance, and the pre-committed expectation
Finance was chosen as the **stress test**: a near-efficient, non-stationary market where the next-day
latent is close to a martingale. Pre-committed prediction (H3/H4): the temporal advantage should
shrink toward zero or invert. We did *not* tune to rescue it; the random/raw controls bound the
ceiling.

## 12.2 Data
Real Yahoo download: the **9 original Select-Sector SPDR ETFs** (XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY)
as the cross-section ($N{=}9$), + ^GSPC/^VIX for labels only; **6,908 trading days, 1998-12-31 →
2026-06-18** (dot-com, 2008 GFC, 2011, 2015–16, 2018-Q4, COVID 2020 [index −12.8 % single day, VIX
82.7], 2022 bear, 2023–26 bull). Annualized index vol 19.3 %; mean pairwise sector return correlation
0.60 (strong market factor + idiosyncrasy). Per-asset features ($F{=}4$, causal): log-return,
|log-return|, Δlog-volume, vol-standardized return. Window $W{=}64$ days. **Out-of-time split** (train
≤ 2017 ≈ 4,698 windows; test ≥ 2018 ≈ 2,044) with a purge gap.

## 12.3 Five tasks & protocol
Frozen encoder → mean-pool window embedding → probes fit on train-period, scored on test-period:
**regime classification** (4-way, logistic → acc/F1), **volatility prediction** (forward 20-day
realized vol, ridge → R²/IC), **anomaly detection** (forward crash, kNN-distance → AUROC/AP),
**clustering** (KMeans vs regime → NMI/ARI), **forecasting** (next-day direction/return → dir-acc/IC).
Config: $D{=}128$, 4+4 depth, predictor 64, 50 epochs, batch 128, VICReg on.

## 12.4 Results (out-of-time TEST 2018–2026, seed 0)

Higher is better; best **trained-SSL** per row in **bold**; floors in _italics_.

| task / metric | **Temporal JEPA** | Spatial JEPA | MAE | BYOL | SimCLR | _random_ | _raw_ |
|---|---|---|---|---|---|---|---|
| Regime accuracy | 0.609 | 0.758 | **0.797** | 0.787 | 0.790 | _0.802_ | _0.804_ |
| Regime macro-F1 | 0.528 | 0.564 | **0.710** | 0.703 | 0.689 | _0.715_ | _0.747_ |
| Volatility R² | −0.228 | −0.435 | 0.157 | **0.181** | 0.099 | _0.169_ | _0.112_ |
| Volatility rank-IC | 0.253 | 0.309 | **0.450** | 0.442 | 0.415 | _0.428_ | _0.439_ |
| Anomaly AUROC | 0.745 | 0.553 | **0.837** | 0.738 | 0.521 | _0.726_ | _0.837_ |
| Anomaly avg-prec | 0.189 | 0.047 | 0.171 | **0.269** | 0.033 | _0.161_ | _0.273_ |
| Clustering NMI | 0.157 | 0.132 | 0.333 | **0.367** | 0.252 | _0.141_ | _0.329_ |
| Clustering ARI | 0.130 | 0.085 | 0.379 | **0.417** | 0.230 | _0.092_ | _0.334_ |
| Forecast dir-acc | 0.523 | 0.479 | 0.499 | 0.511 | **0.523** | _0.496_ | _0.533_ |
| Forecast ret-IC | 0.085 | 0.045 | 0.076 | **0.094** | 0.051 | _0.116_ | _0.080_ |

**Horizon (Temporal JEPA):** regime acc **0.609 (Δ1) → 0.516 (Δ5) → 0.494 (Δ20)**; forecast ret-IC
**0.085 → 0.009 → −0.086** — *monotonically worse* (H5 ✓, the unpredictable signature). **VICReg-off:**
effective rank collapses ~110 → ~2.3 (H6 ✓: necessary on markets too).

## 12.5 Interpretation — observed failure, negative/inverted transfer
1. **Temporal JEPA loses to every reconstruction/contrastive baseline** (beats only Spatial JEPA,
   7/10 — so `temporal > spatial` *still replicates*). MAE and BYOL are the strongest trained encoders.
2. **No SSL beats the raw-feature floor.** A linear probe on the mean-pooled inputs matches/beats every
   pretrained encoder (regime 0.80, anomaly-AUROC 0.84). On this out-of-time benchmark, SSL buys
   essentially nothing over the engineered features.
3. **Temporal JEPA is uniquely *harmful*:** same architecture, the *untrained* encoder scores regime
   **0.80**, the temporal-JEPA-trained one **0.61**. Optimizing "predict tomorrow's latent" on
   non-stationary data **actively erases** the cross-sectional/volatility structure the tasks need —
   and longer horizons erase more.
4. **Why MAE/BYOL win here.** Reconstruction (MAE) and view-invariance (BYOL) learn *generic,
   distribution-robust* features that transfer across the 2018 regime shift; a *predictive* objective
   over-specializes to the (unpredictable, shifting) 1999–2017 dynamics. The efficient-market signature
   is visible in the forecasting row: direction ≈ 50 % for *all* methods (a leakage sanity check).

This is an **honest negative result**, not a tuning failure: the random and raw-feature controls bound
how much *any* method could have won (very little), and Temporal JEPA falling *below* both, under the
identical probe, is a clean within-architecture verdict.

---

# 13. Industrial Experiments (NASA C-MAPSS — the very-predictable, *winning* case)

## 13.1 Dataset
NASA C-MAPSS turbofan run-to-failure simulation; four subsets crossing operating conditions × fault
modes: **FD001** (1 cond, 1 fault, 100 train / 100 test engines), **FD002** (6 cond, 1 fault, 260/259),
**FD003** (1 cond, 2 faults, 100/100), **FD004** (6 cond, 2 faults, 249/248). Each engine: 21 sensors
+ 3 operating settings per cycle; engines run **128–543 cycles**. Train engines run to failure; test
engines are truncated with a separate `RUL_FDxxx.txt` giving true RUL at the last cycle.

## 13.2 Mapping & adaptation
**21 sensors = cross-section (tokens)**, **operating cycle = frame**, **window $W{=}40$ cycles**.
Reuses the finance `PanelEncoder`/`FinanceJEPA` *verbatim*; the only change is `temporal_period=1024`
(monotone cycles, §3.10). Condition-normalization: for the 6-condition FD002/FD004 we KMeans the 3
settings into 6 regimes and z-score each sensor within its regime (train-only); single-condition
subsets use a global z-score. Constant sensors (≈0 variance on train) dropped (15 kept FD001). Per-
sensor features ($F{=}3$, causal): normalized value, 1-step Δ, 5-cycle rolling mean. Labels: RUL
piecewise-linear capped at 125; 4-stage health from RUL thresholds (100/50/20); anomaly = RUL ≤ 20.
Disjoint train/test engines → no leakage.

## 13.3 Five tasks
RUL regression (ridge → R²/RMSE/rank-IC + the **standard last-cycle RMSE & PHM08 score**), health
classification (logistic → acc/F1), anomaly detection (kNN-distance to **healthy** train windows →
AUROC/AP), clustering (KMeans vs health → NMI/ARI), nearest-neighbor retrieval (cosine kNN → health
p@k, neighbor-RUL IC). **PHM08 score** (lower better): $\sum \exp(-d/13)-1$ if $d<0$ else
$\exp(d/10)-1$, $d=\text{pred}-\text{true}$ (penalizes *late* RUL more). Config $D{=}128$, 4+4 depth,
predictor 64, 20 epochs, batch 256.

## 13.4 Results — headline (Temporal JEPA Δ=1 win counts, of 13 metrics × 4 subsets = 52)

| Temporal JEPA beats… | FD001 | FD002 | FD003 | FD004 | **total** |
|---|---|---|---|---|---|
| SimCLR | 13 | 13 | 13 | 12 | **51/52** |
| MAE | 12 | 12 | 11 | 11 | **46/52** |
| raw features _(floor)_ | 11 | 13 | 10 | 11 | **45/52** |
| Spatial JEPA | 11 | 11 | 10 | 11 | **43/52** |
| BYOL | 12 | 11 | 10 | 10 | **43/52** |
| random-init _(floor)_ | 8 | 12 | 9 | 11 | **40/52** |

**RUL regression (the canonical task), per subset:**

| RUL metric | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **Temporal JEPA R²** | **0.677** | **0.634** | **0.806** | **0.667** |
| best SSL baseline R² | 0.577 (mae) | 0.551 (mae) | 0.662 (spatial) | 0.566 (mae) |
| _random R²_ | _0.651_ | _0.600_ | _0.714_ | _0.596_ |
| _raw R²_ | _0.344_ | _0.291_ | _0.344_ | _0.183_ |
| **Temporal JEPA last-cycle RMSE** ↓ | 16.4 | 26.2 | 14.8 | 27.0 |
| **Temporal JEPA PHM08** ↓ | 471 | 6 465 | 425 | 5 128 |

**Representation-quality tasks (Temporal / best-baseline / _random_ / _raw_):**

| metric ↑ | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Health acc | **0.74**/0.74/_0.78_/_0.66_ | **0.73**/0.69/_0.71_/_0.61_ | **0.86**/0.81/_0.83_/_0.73_ | **0.81**/0.77/_0.79_/_0.69_ |
| Anomaly AUROC | **0.99**/0.98/_0.98_/_0.93_ | **0.98**/0.97/_0.96_/_0.89_ | 0.98/**0.99**/_0.99_/_0.92_ | **0.97**/0.97/_0.95_/_0.81_ |
| Retrieval p@k | **0.66**/0.65/_0.63_/_0.58_ | **0.67**/0.61/_0.61_/_0.52_ | **0.77**/0.74/_0.72_/_0.70_ | **0.78**/0.72/_0.70_/_0.65_ |

*(Full $7\times13$ per-subset tables are in [report_cmapss.md](report_cmapss.md) §4.2.)*

**Random-control difficulty scaling** — Temporal JEPA beats random on {8, 12, 9, 11}/13 for
{FD001, FD002, FD003, FD004}: the margin over an *untrained* net is small on the easy single-condition
subsets and decisive on the hard multi-condition ones.

**Ablations (FD001):** horizon Δ=1/5/20 → RUL R² **0.677 / 0.671 / 0.655** (nearly flat — H5 ✓);
**VICReg-off** → RUL R² 0.703 (slightly *up*) but emb-std 0.38 → 0.26 and retrieval 0.66 → 0.64
(starting to collapse) — anti-collapse is **less critical** here because one dominant degradation
direction self-stabilizes (H6 nuance).

## 13.5 Per-subset discussion
- **FD001 (easiest).** Temporal JEPA best SSL (R² 0.68, PHM08 471) but *random* is within noise (R²
  0.65; random wins PHM08/health/NMI) — the signal is so strong a random temporal-attention projection
  suffices.
- **FD002 / FD004 (6 conditions).** Hardest; condition-normalization essential. Temporal JEPA pulls
  clearly ahead of *all* baselines *and* both floors (12/13 and 11/13 over random; 13/13 and 11/13 over
  raw) — learning matters when the signal is messy.
- **FD003 (2 faults, 1 condition).** Highest absolute RUL R² (0.81); temporal best on RUL/health/
  retrieval; on anomaly the spatial/random methods tie (0.99) — anomaly is near-saturated here.

## 13.6 Interpretation
H1 supported (best SSL across all subsets), and — unlike finance — **the raw-feature floor is cleared**
(45/52; RUL R² roughly double raw). `temporal > spatial` replicates a third time (43/52). The honest
limit: on the easy subsets the *learning* adds little over the architecture; its value scales with
difficulty.

---

# 14. Cross-Domain Analysis

The three domains arranged on the **predictability axis**:

| | PASTIS (satellite) | S&P (finance) | C-MAPSS (industrial) |
|---|---|---|---|
| temporal structure | periodic / seasonal | near-martingale, non-stationary | monotone degradation |
| next-latent predictability | high | ~zero | very high |
| stationarity (train→test) | high | **low** (regime shift) | high (disjoint engines, same physics) |
| **Temporal JEPA vs baselines** | **wins +15–16 mIoU** | **loses** (below MAE/BYOL) | **wins** (best SSL) |
| **vs raw-feature floor** | wins (huge) | **loses** | **wins** (R² ~2×) |
| **vs random-init** | wins | **loses** (0.61<0.80) | wins (margin ↑ with difficulty) |
| **temporal ≻ spatial** | ✓ (+6.0) | ✓ (7/10) | ✓ (43/52) |
| horizon behavior | flat | monotonically worse | flat |
| VICReg necessity | essential | essential | helpful, not essential |

**The unified explanation.** A single latent variable organizes all of it: *the predictability of the
system's latent trajectory*. Where the future is a smooth function of the past (PASTIS phenology,
C-MAPSS wear), "predict the next latent" injects exactly the trajectory information downstream tasks
need, and the objective wins — including over the raw and random floors. Where the future is a random
walk *and* the distribution shifts (finance), the predictive target is noise; the Information-
Bottleneck (§3.2) then filters the representation *away* from the useful static structure, so the
objective not only fails to help but underperforms a random projection. The two flat-vs-monotone
horizon curves are the cleanest evidence: predictability, not the objective's mechanics, sets the
outcome. The one invariant across all three — `temporal ≻ spatial` — says that *whenever there is any
temporal signal, predicting forward in time beats masking within a frame.*

**Why MAE/BYOL invert with Temporal JEPA between satellite and finance.** Reconstruction and
invariance are *generic* priors (model appearance / collapse nuisances); they are distribution-robust
but throw away temporal change. Temporal prediction is a *specific* prior (model the dynamics); it is
powerful when the dynamics are real and learnable, and a liability when they are noise. The crossover
is exactly the predictability axis.

---

# 15. Mechanistic Analysis

**Why temporal wins on PASTIS — H-mech-2, confirmed.** We probe the *frozen spatial* features
(`encode_full`, **not** the temporal pathway, to avoid the DOY-encoding circularity) to decode
acquisition time from a *single* frame (val, seed 0, `scripts/mechanistic.py`): 12-way month
classification (chance 8.3 %) + circular DOY regression.

| Encoder | month-acc | DOY circular MAE |
|---|---|---|
| **Temporal JEPA (Δ=1)** | **61.3 %** | **30.4 days** |
| Spatial JEPA | 46.3 % | 41.8 days |

Temporal-JEPA's *spatial* features decode the acquisition month **+15 points** better and the
day-of-year ~11 days more accurately. Since crops are separated by *phenological stage* (which tracks
time), this is direct evidence the future-prediction objective made the *spatial* representation
season-aware — the mechanism behind the segmentation win.

**Latent geometry & collapse.** Effective rank (§3.9) is the running collapse diagnostic. Healthy
runs show erank *climbing* (C-MAPSS $2.4\to118/128$; PASTIS train $\sim$430/512); VICReg-off runs show
it *crashing* (PASTIS $\to2.4$; finance $\to2.3$). The variance/covariance terms are visibly doing the
work the EMA+predictor alone cannot when consecutive frames are near-identical.

**Temporal smoothness & retrieval.** On C-MAPSS, nearest-neighbor retrieval in temporal-JEPA embedding
space returns windows of *similar RUL* (neighbor-RUL rank-IC 0.59–0.63, health p@k 0.66–0.78, best of
all methods) — i.e. the embedding trajectory preserves degradation similarity, the geometric signature
of a learned latent trajectory. On finance, retrieval/clustering are weak for *every* method —
consistent with there being no smooth latent trajectory to preserve.

**Recommended visualizations (future runs, `scripts/feature_figure.py`).** (i) t-SNE/UMAP of parcel
embeddings colored by crop — temporal should show tighter clusters; (ii) PCA of C-MAPSS window
embeddings colored by RUL — temporal should show a smooth 1-D-ish manifold (the degradation arc);
(iii) attention maps of the temporal transformer — heads attending to phenologically-active windows;
(iv) embedding-trajectory plots per engine over cycles — a smooth curve for temporal, a blob for raw.

---

# 16. Ablation Studies

Run and reported here (✓), or available in code and pending compute (∘):

- **Prediction horizon $\Delta$** (✓ all 3 domains). PASTIS flat (22.3/20.8/21.8/22.6); finance
  monotone-worse (regime 0.61/0.52/0.49); C-MAPSS flat (R² 0.677/0.671/0.655). *The key
  diagnostic of predictability.*
- **VICReg coefficients** (✓). $\lambda_v{=}\lambda_c{=}0$ → collapse on PASTIS/finance (erank → ~2);
  on C-MAPSS no catastrophic collapse (one dominant signal). Satellite grid `var0.5/var2.0` coded (∘).
- **Predictor width $D_p$** (∘ satellite grid `pred128/pred256`; the **invariant** $D_p<D$ is asserted,
  not ablated away — a config with $D_p>D$ once caused a real bug). The bottleneck is structural.
- **Encoder width $D$ / depth** (∘ satellite `dim{128,256,512,768}`, `preddepth{1,2,4,6}`).
- **Patch size $P$** (✓ qualitatively): $P{=}16$ ($N{=}64$) gave kNN 68.5 but coarse mIoU 14.7 (a
  resolution artifact, not a learning failure) → switched default to $P{=}8$.
- **Window length $W$** (config): 64 (finance), 40 (C-MAPSS, fits $\Delta{\le}20$).
- **Mask ratio** (MAE baseline): 0.75 (satellite), 0.5 (panels).
- **Pooling** (✓ by construction): masked-mean over time × tokens; dense upsample for PASTIS.
- **EMA momentum / LR / weight-decay schedules** follow I-JEPA; warmup is necessary (Adam early-step
  instability, §3.6).
- **Loss type** $\ell_2$ (I-JEPA) vs $\ell_1$ (V-JEPA) — coded, $\ell_2$ default.

The ablations that *answer a scientific question* are horizon (predictability) and VICReg (collapse);
the capacity ablations are engineering knobs and are scoped to compute.

---

# 17. Failure Analysis

**Where the method fails.** Finance — comprehensively (§12): below MAE/BYOL, below raw features, below
its own random init; horizon makes it worse. **What assumption breaks:** the implicit assumption that
$z_{\text{fut}}$ is a smooth, *stationary* function of $z_{\text{past}}$. Markets violate both: the
next-day latent is ~unpredictable (the conditional-variance floor of §3.1 dominates), and the
train→test distribution shifts (1999–2017 ≠ 2018–2026), so the learned predictor is both
low-signal and stale. The Information-Bottleneck consequence (§3.2): the representation is squeezed
toward the (noisy) predictable component and away from the static cross-sectional/vol structure the
tasks actually read — hence *worse than random projection*.

**The softer "failure" on C-MAPSS:** on the *easiest* subsets the learning adds little over a random
network. **Assumption:** that the task is hard enough that representation learning matters. When a
single dominant signal (monotone wear) survives any projection, the architecture + a linear probe is
most of the story; SSL's value is real only as difficulty rises.

**Not hidden — designed for.** Both failures are caught *because* of the random-init and raw-feature
floors; a study reporting only SSL-vs-SSL would have missed them.

**Phase 4 — the distributional rescue was tried and REJECTED (a controlled negative).** The most
obvious algorithmic fix for the finance failure is to predict a *distribution* over the future latent
instead of a point, so the model can output high variance where the future is unpredictable
(heteroscedastic β-NLL; Seitzer et al., ICLR 2022) and let the predicted variance become a volatility
signal (returns are unpredictable, but volatility clusters). We implemented this as an additive,
flag-gated `tjepa_dist` (predictor μ,σ² heads; β=0.5; VICReg retained; 52 tests still pass). The
*mechanism works* — during training the pooled predicted σ tracks realized volatility at rank-IC
≈ 0.75–0.82. But the **rescue fails on every downstream criterion**: at matched 50-epoch training the
distributional representation scores finance regime **0.53** / vol-R² **−0.23** (below point-JEPA's
0.61 and far below the ~0.80 raw/random floors); exposing the predicted variance as a probe feature
adds only a small anomaly bump and clears no floor; and on the *predictable* C-MAPSS FD001 it is a
mild net negative (RUL-R² 0.658 vs point 0.677). So predicting a distribution does not manufacture
predictable structure that isn't there — the finance failure **survives the obvious fix**, confirming
it is fundamental (non-stationarity + near-martingale returns), not a point-target artifact. *Novelty
note:* probabilistic JEPA is itself 2026 prior art (VJEPA, arXiv 2601.14354; Var-JEPA, arXiv
2603.20111, incl. tabular Var-T-JEPA), so this is a controlled mechanistic test, not a new method.
Details: [report_finance.md](report_finance.md) §8.

**Phase 5 — is it the shift or unpredictability? UNPREDICTABILITY (an evaluation-protocol fix also
fails).** One alternative remained: maybe SSL is fine and only the 1999–2017→2018–2026 *distribution
shift* hurts. We tested it by reusing the encoders with an *in-period* probe (fit+test both inside
2018–2026) and, definitively, by *re-pretraining the encoder on recent data (≤2019) and evaluating
fully in-period on 2020–2026 — no shift anywhere*. **With the shift entirely removed, raw features
still win** (regime 0.831) over every SSL method (MAE 0.685, temporal JEPA 0.460, still worst). So the
failure is **not** non-stationarity of the split — it is intrinsic task-hardness/unpredictability: the
regime/vol signal already lives in the engineered features, so a learned representation adds nothing
and temporal prediction subtracts. The finance negative is thus robust to *both* an algorithmic fix
(Phase 4) and a protocol fix (Phase 5) — the strongest form of the predictability-spectrum result
([report_finance.md](report_finance.md) §9).

**Remaining mitigations / future work:** (i) condition the predictor on a regime variable, or use
rolling train→test refits with shorter gaps (attack the *non-stationarity* directly, which the
distributional objective did not); (ii) predict genuinely-predictable *scalar* targets (realized vol,
via a supervised auxiliary) rather than the full future latent; (iii) the §18 stationarity go/no-go as
a gate before applying temporal JEPA at all.

---

# 18. Theoretical Discussion — a falsifiable principle

**Implicit assumptions of causal temporal JEPA.** (A1) *Predictability:* there exists a low-dimensional
latent $s_t$ with $s_{t+\Delta}\approx \Phi(s_t)$ for a smooth $\Phi$ (the latent trajectory is a
function of its past). (A2) *Stationarity:* $\Phi$ and the observation map are stable across the
train→deploy shift. (A3) *Relevance:* the downstream target is a function of position-on-trajectory
$s_t$ (phenological stage, health, regime).

**Claim (predictability ⇒ benefit).** Under A1–A3, minimizing $\mathbb E\|z_{t+\Delta}-g(z_{\le t})\|^2$
drives $z$ toward a sufficient statistic of $s_t$ for forecasting, which by A3 is sufficient for the
downstream task; the gain over a generic prior (MAE/BYOL) is monotone in the predictable fraction of
$\mathrm{Var}(s_{t+\Delta})$, i.e. in $1-\dfrac{\mathbb E[\mathrm{Var}(s_{t+\Delta}\mid s_t)]}{\mathrm{Var}(s_{t+\Delta})}$
(the trajectory's $R^2$ from §3.1). When that fraction → 1 (C-MAPSS, PASTIS) the objective is maximally
informative; when → 0 (efficient market) the objective's target is noise and the Information-Bottleneck
filters the representation *away* from the static signal — predicting **negative** transfer, observed.

**Which dynamical systems satisfy A1–A3.** Dissipative / degrading systems (monotone attractors:
engines, batteries, materials fatigue), seasonally-driven systems (phenology, climate, demand), and
inertial physical systems (weather over short horizons, robotics) satisfy them. (Near-)martingales,
chaotic systems past their Lyapunov horizon, and regime-switching processes with shifting parameters
violate them.

**When temporal prediction should beat reconstruction.** Reconstruction (MAE) optimizes a *generic*
sufficient statistic for the *current* observation (appearance); temporal prediction optimizes a
sufficient statistic for the *future* (dynamics). The latter dominates exactly when (a) the downstream
task is a trajectory property (A3) and (b) the trajectory is predictable (A1) and stable (A2). Our
three domains instantiate (predictable+relevant → win), (relevant but unpredictable/unstable → lose),
(predictable+relevant, very easy → win but small marginal value over architecture).

**A practical, falsifiable go/no-go.** Before applying temporal JEPA to a new domain, estimate the
**latent-trajectory $R^2$** — fit a cheap forward model (or even raw-feature ridge) for
$x_{t+\Delta}$ from $x_{\le t}$ and measure out-of-time $R^2$, and measure train→test distribution
shift (e.g. an MMD or a domain-classifier AUC). The prediction: temporal JEPA beats generic SSL **iff
that $R^2$ is materially positive *and* the shift is small.** This is directly testable on any new
modality and is the project's main transferable claim.

---

# 19. Future Work

- **General Temporal JEPA toolkit:** one config-driven panel-JEPA usable on any $(T,N,F)$ series; the
  satellite/finance/industrial stacks already share 90 % of the code.
- **World models / control:** apply the causal objective to RL observation streams; the PSR/World-Model
  link (§2.5/§2.6) suggests the learned latent supports planning.
- **Weather & climate:** the prototypical predictable-but-high-dimensional system; A1–A3 hold over
  short horizons.
- **Robotics & continuous-time:** Latent ODEs / continuous-time transformers for irregular sampling;
  C-MAPSS already exercises irregular-but-monotone time.
- **Multi-modal JEPA:** predict one modality's future latent from another's (e.g. sensor → image).
- **Distributional / diffusion predictors:** for stochastic systems (finance), predict a *distribution*
  over the future latent (vol is predictable even when returns are not) instead of a point — directly
  targets the §12 failure.
- **Stronger temporal baselines:** TS2Vec-style temporal contrastive and a temporal-order pretext, to
  test "temporal *prediction* ≻ other temporal SSL," not just ≻ spatial.
- **Rigor:** multi-seed error bars on finance/industrial (the satellite 3-seed protocol); 5-fold CV;
  500-constituent finance panel; end-to-end fine-tuning numbers alongside the frozen probes.

---

# 20. Conclusion

**Scientific contributions.** (1) A controlled, three-domain test of *causal future-latent prediction*
as an SSL objective, isolating the objective as the only variable. (2) Evidence that the objective's
value is governed by a single latent factor — **predictability of the latent trajectory** — with a
clean win (PASTIS), a clean loss (finance), and a clean win (C-MAPSS) arranged along that axis. (3) The
robust invariant `temporal ≻ spatial` on all three domains. (4) A falsifiable go/no-go criterion
(§18) relating trajectory-$R^2$ + distribution-shift to expected benefit.

**Engineering contributions.** A modality-agnostic factorized space–time JEPA with EMA target, narrow
predictor, and VICReg anti-collapse, reused verbatim across three modalities (the only per-domain
changes: the frame tokenizer and a one-line temporal-period knob); a frozen-probe evaluation harness
with **random-init and raw-feature floors**; per-domain downloaders (incl. a cookie+crumb Yahoo
fetcher and a C-MAPSS mirror/zip loader) with synthetic offline fallbacks; 47 passing tests and M1
collapse gates.

**Negative findings, stated plainly.** On out-of-time finance, *no* SSL method beats raw features, and
temporal JEPA falls below its own random initialization — the objective is actively harmful on a
non-stationary near-martingale. On easy C-MAPSS subsets, an untrained network is competitive — the
learning's marginal value is small when one signal dominates. Both were *caught by the controls*, which
is the methodological lesson: **always include the random and raw-feature floors.**

**General lessons.** (i) The pretext target selects which information survives; choose it to match the
downstream signal *and* the data's predictability. (ii) `temporal ≻ spatial` whenever time carries
signal. (iii) Report the floors, not just the SSL leaderboard. (iv) Horizon-sensitivity is a free
diagnostic of whether a domain is in-scope.

**Open problems.** A quantitative law linking trajectory-$R^2$ to the mIoU/RUL gap; distributional
predictors for stochastic systems; whether fine-tuning closes the finance gap; and scaling the win to
500-name panels and to weather/robotics.

---

# Figures (described for inclusion)

1. **Architecture diagram** — the factorized context/predictor/EMA-target pipeline of §7 (the ASCII
   schematic there, rendered).
2. **Training pipeline** — data → tokenizer → spatial ViT → temporal transformer → masked-mean →
   predictor; EMA arrow from online to target; loss node with the three terms.
3. **JEPA / EMA flow** — gradient paths (solid through online+predictor; **none** into target),
   stop-grad and LayerNorm on the target branch, EMA update after the optimizer step.
4. **Causal split illustration** — a length-$T$ strip with context $\le s$ shaded, target at $s{+}\Delta$,
   the context-only attention mask as a lower-triangular block.
5. **Predictability-spectrum schematic** — the three domains on a horizontal "latent-trajectory $R^2$"
   axis with win/loss markers (the core thesis figure).
6. **Per-domain bar charts** — PASTIS conv-mIoU; finance regime/vol/anomaly; C-MAPSS RUL-R²/PHM08, each
   with the random and raw floors drawn as horizontal lines.
7. **Horizon plots** — three small multiples (flat / monotone-down / flat) overlaying the metric vs Δ.
8. **Effective-rank curves** — erank vs step for VICReg-on (rising) vs off (collapsing), all domains.
9. **Mechanistic bars** — PASTIS month-decoding accuracy (temporal 61.3 vs spatial 46.3, chance 8.3).
10. **Embedding visualizations** — t-SNE (PASTIS crops), PCA-vs-RUL arc (C-MAPSS), blob (finance).
11. **Temporal-persistence illustration** — sensor/return/NDVI trajectories with their autocorrelation,
    annotated with each domain's next-step predictability.
12. **Cross-domain scorecard** — the §14 table as a heatmap (green wins / red losses).

---

# Appendices

## A. Repository walkthrough
See §9. Entry points: `scripts/run_matrix.py` (satellite), `scripts/run_finance_matrix.py`,
`scripts/run_cmapss_matrix.py` — each: pretrain → freeze → probe per cell → CSV (+ saved encoder),
resumable. `scripts/aggregate*.py` produce the comparison tables and per-task verdicts. `report.md`,
`report_finance.md`, `report_cmapss.md` are the per-domain sources of record.

## B. Configuration files (key knobs)
- `configs/model/tjepa_8gb.yaml` — satellite: $P8$, $D512$, predictor 384, 100 epochs, eff-batch 192,
  grad-checkpoint, VICReg 1.0/0.04.
- `configs/model/fjepa.yaml` — finance: $D128$, 4+4 depth, predictor 64, 50 epochs, batch 128, jitter 0.05.
- `configs/model/cjepa.yaml` — industrial: $D128$, 4+4 depth, predictor 64, **temporal.period 1024**,
  20 epochs, batch 256.
- `configs/data/{pastis,finance,cmapss}.yaml` — roots, windows, splits, label thresholds, synth toggle.

## C. Hyperparameters (consolidated)

| | PASTIS | finance | C-MAPSS |
|---|---|---|---|
| $N$ tokens / frame | 256 | 9 | 14–17 |
| $F$ input feats | 10 bands (conv) | 4 | 3 |
| $D$ / $D_p$ | 512 / 384 | 128 / 64 | 128 / 64 |
| spatial / temporal depth | 6 / 4 | 4 / 4 | 4 / 4 |
| heads | 8 | 4 | 4 |
| window $T/W$ | ≤32 | 64 | 40 |
| horizon $\Delta$ | 1 (sweep 1–8) | 1 (sweep 1/5/20) | 1 (sweep 1/5/20) |
| temporal period | 366 (DOY) | 366 (DOY) | **1024 (cycle)** |
| epochs / eff-batch | 100 / 192 | 50 / 128 | 20 / 256 |
| lr / warmup | 1e-3 / 15 ep | 5e-4 / 5 ep | 5e-4 / 5 ep |
| VICReg $\lambda_v/\lambda_c$ | 1.0 / 0.04 | 1.0 / 0.04 | 1.0 / 0.04 |
| EMA $\tau$ | 0.996→1.0 | 0.996→1.0 | 0.996→1.0 |

## D. Command-line examples
```bash
# satellite
python scripts/run_matrix.py --config configs/model/tjepa_8gb.yaml --data configs/data/pastis.yaml \
       --device cuda:0 --max-cells 5 --knn --resume
# finance
python scripts/download_finance.py
python scripts/run_finance_matrix.py --config configs/model/fjepa.yaml --data configs/data/finance.yaml --device cuda:0
python scripts/aggregate_finance.py
# industrial
python scripts/download_cmapss.py            # or --zip CMAPSSData.zip
python scripts/run_cmapss_matrix.py --config configs/model/cjepa.yaml --data configs/data/cmapss.yaml --device cuda:0
python scripts/aggregate_cmapss.py
# correctness
pytest -q                                     # 47 pass / 3 skip
python scripts/cmapss_smoketest.py --device cuda:0   # M1 gate
```

## E. Hardware, training time, GPU memory
Development card: single **RTX 4060 Laptop (8 GB)**. Satellite "8 GB" config peaks ~6 GB (batch 16 ×
grad-accum 12); finance/industrial models are ~1.8 M params and peak < 3 GB (BYOL forced to gradient-
checkpoint its backbones to fit). Per-cell GPU-hours are logged in every results CSV. Finance matrix
(9 cells, 50 ep): tens of minutes/cell, ~1–2 h total. C-MAPSS matrix (7 cells × 4 subsets + 3
ablations, 20 ep): a few minutes/JEPA-cell (BYOL slower), ~1.5–2.5 h total. Satellite (P8, 100 ep) is
GPU-hours per cell and was run on a server card.

## F. Dataset preprocessing (precise)
- *PASTIS:* per-band normalization from train folds only; variable-length collate front-packs real
  frames + builds the pad mask; DOY in [1,366]; labels sanitized (out-of-range/void → ignore).
- *Finance:* causal per-asset features; per-feature z-score from train windows; out-of-time split +
  purge gap; regime/anomaly/vol/forecast labels from the index, never fed to the encoder.
- *C-MAPSS:* condition-KMeans (k=6) + per-regime z-score (train-only); drop ≈0-variance sensors;
  per-sensor [value, Δ, rolling-mean]; RUL cap 125; health thresholds (100,50,20); anomaly RUL≤20;
  standard-protocol last-cycle set vs RUL.txt (short engines excluded, counts logged).

## G. Evaluation scripts
`eval/linear_probe.py` (dense mIoU), `eval/knn.py` (parcel kNN), `eval/fewshot.py`,
`eval/finance_tasks.py` (5 tasks + raw-feature floor), `eval/cmapss_tasks.py` (5 tasks, PHM08,
healthy-reference anomaly + raw floor), `scripts/mechanistic.py` (decode acquisition time from frozen
spatial features).

## H. Mathematical proofs / derivations referenced
Scaled-dot-product scale (§3.3); conditional-expectation as MMSE and law of total variance (§3.1);
L2-latent ↔ MI lower bound under Gaussian residual (§3.2); VICReg variance hinge gradient and the
covariance-decorrelation argument (§3.8); effective rank as $\exp$ of spectral entropy (§3.9);
predictability ⇒ benefit claim and the trajectory-$R^2$ criterion (§18).

## I. Additional experiments (coded, compute-pending)
Satellite: VICReg coefficient grid, predictor-width/depth and embed-dim grids, 5-fold CV, multi-seed
Wilcoxon (needs n≥6), t-SNE/UMAP feature figure. Finance/industrial: multi-seed error bars (`--seed`),
500-name finance panel, FD-specific ablations, distributional vol predictor.

## J. Glossary
**JEPA** joint-embedding predictive architecture; **EMA** exponential moving average; **VICReg**
variance-invariance-covariance regularization; **RUL** remaining useful life; **PHM08** the NASA
asymmetric RUL score; **mIoU** mean intersection-over-union; **DOY** day-of-year; **SITS** satellite
image time series; **IC** information coefficient (rank correlation of prediction vs target);
**erank** effective rank; **martingale** $\mathbb E[X_{t+1}\mid\mathcal F_t]=X_t$ (best forecast is
the present); **stop-grad** stop-gradient; **frozen probe** train only a head on a frozen encoder.

## K. Notation table
See the **Notation** section at the top.

## L. Bibliography (selected)
Assran et al., *I-JEPA*, CVPR 2023 (2301.08243) · Bardes et al., *V-JEPA*, 2024 (2404.08471) · Bardes
et al., *VICReg*, ICLR 2022 (2105.04906) · He et al., *MAE*, CVPR 2022 (2111.06377) · Grill et al.,
*BYOL*, NeurIPS 2020 (2006.07733) · Chen et al., *SimCLR*, ICML 2020 (2002.05709) · Chen & He,
*SimSiam*, CVPR 2021 · Tian et al., *Understanding self-supervised learning dynamics*, ICML 2021 · Oord
et al., *CPC / InfoNCE*, 2018 (1807.03748) · Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 ·
Ha & Schmidhuber, *World Models*, 2018 · Hafner et al., *PlaNet/Dreamer*, 2019–2020 · Littman et al.,
*Predictive State Representations*, NeurIPS 2001 · Tishby & Zaslavsky, *Information Bottleneck*, 2015 ·
Roy & Vetterli, *Effective rank*, EUSIPCO 2007 · Loshchilov & Hutter, *AdamW*, ICLR 2019 · Yue et al.,
*TS2Vec*, AAAI 2022 · Cong et al., *SatMAE*, NeurIPS 2022 · Wang et al., *SSL4EO*, 2023 · Garnot &
Landrieu, *PASTIS / U-TAE*, ICCV 2021 (2107.07933) · Saxena & Goebel, *C-MAPSS / PHM08*, NASA PCoE 2008.

---

*End of monograph. Per-domain sources of record: [report.md](report.md), [report_finance.md](report_finance.md),
[report_cmapss.md](report_cmapss.md). All numbers reproduce from `runs/{matrix,finance,cmapss}_results*.csv`.*
