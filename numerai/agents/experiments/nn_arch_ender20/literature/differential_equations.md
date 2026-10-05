# Literature notes: differential equations in ML and quant finance

Collected 2026-10-04 by a literature-search agent; every URL was checked against its abstract page.

## A. Training as a continuous-time process (most relevant)

| paper | key idea | relevance |
|---|---|---|
| Ali, Kolter, Tibshirani (AISTATS 2019), *A Continuous-Time View of Early Stopping for Least Squares*, https://arxiv.org/abs/1810.10082 | Gradient flow stopped at time t ≈ ridge with λ = 1/t; risk within 1.69× of ridge | High: theoretical basis for "training amount is the regularizer" |
| Barrett, Dherin (ICLR 2021), *Implicit Gradient Regularization*, https://arxiv.org/abs/2009.11162 | Finite-step GD follows a modified flow penalizing ‖∇L‖²; lr × steps sets its strength | Explains the lr ↔ epochs trade-off |
| Smith, Dherin, Barrett, De (ICLR 2021), *On the Origin of Implicit Regularization in SGD*, https://arxiv.org/abs/2101.12176 | SGD's implicit penalty scales with lr / batch size | Era subsampling is an unexplored knob |
| Li, Tai, E (ICML 2017), *Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms*, https://arxiv.org/abs/1511.06251 | SGD ≈ an SDE; schedules as optimal control | Background |
| Li, Grandvalet, Davoine (ICML 2018), *Explicit Inductive Bias for Transfer Learning* (L2-SP), https://arxiv.org/abs/1802.01483 | Penalize ‖θ − θ₀‖² instead of ‖θ‖² | Explicit counterpart of early stopping |

## B. Noise injection (SDE and heat-equation views)

| paper | key idea |
|---|---|
| Bishop (Neural Computation 1995), *Training with Noise is Equivalent to Tikhonov Regularization*, https://doi.org/10.1162/neco.1995.7.1.108 | Input noise ≈ penalty on input derivatives |
| Camuto et al. (NeurIPS 2020), *Explicit Regularisation in Gaussian Noise Injections*, https://arxiv.org/abs/2007.07368 | Noise penalizes high-frequency components, most near the output |
| Campbell, Finlay, Oberman (2020), *Deterministic Gaussian Averaged Neural Networks*, https://arxiv.org/abs/2006.06061 | Gaussian-averaged network = heat-equation solution |
| Wang, Bao, Shi (2024), *Convection-Diffusion Equation: A Theoretically Certified Framework for Neural Networks*, https://arxiv.org/abs/2403.15726 | Noise, dropout and smoothing as the diffusion term of a PDE |

## C. Neural ODEs and continuous-depth networks

| paper | relevance |
|---|---|
| Chen, Rubanova, Bettencourt, Duvenaud (NeurIPS 2018), *Neural Ordinary Differential Equations*, https://arxiv.org/abs/1806.07366 | On static features ≈ weight-tied ResNet; residual MLPs already lost |
| Dupont, Doucet, Teh (NeurIPS 2019), *Augmented Neural ODEs*, https://arxiv.org/abs/1904.01681 | Low |
| Finlay, Jacobsen, Nurbekyan, Oberman (ICML 2020), *How to Train Your Neural ODE*, https://arxiv.org/abs/2002.02798 | Kinetic/Jacobian penalties; low-medium |
| Kelly, Bettencourt, Johnson, Duvenaud (NeurIPS 2020), *Learning Differential Equations that are Easy to Solve*, https://arxiv.org/abs/2007.04504 | Low |
| Haber, Ruthotto (Inverse Problems 2017), *Stable Architectures for Deep Neural Networks*, https://arxiv.org/abs/1705.03341 | Low (our nets are shallow) |
| Ruthotto, Haber (JMIV 2020), *Deep Neural Networks Motivated by PDEs*, https://arxiv.org/abs/1804.04272 | Low |

## D. Neural SDEs/CDEs and stochastic-calculus finance

| paper | relevance |
|---|---|
| Liu et al. (2019), *Neural SDE: Stabilizing Neural ODE Networks with Stochastic Noise*, https://arxiv.org/abs/1906.02355 | ≈ noise injection in a residual block |
| Li, Wong, Chen, Duvenaud (AISTATS 2020), *Scalable Gradients for SDEs*, https://arxiv.org/abs/2001.01328 | Infrastructure only |
| Kidger, Morrill, Foster, Lyons (NeurIPS 2020), *Neural CDEs for Irregular Time Series*, https://arxiv.org/abs/2005.08926 | Needs per-stock history; we have snapshots |
| Gierjatowicz et al. (2020), *Robust Pricing and Hedging via Neural SDEs*, https://arxiv.org/abs/2007.04154 | Pricing, not ranking |
| Yang et al. (2021), *Neural Network SDE Models with Applications to Financial Data Forecasting*, https://arxiv.org/abs/2111.13164 | Univariate series |
| Black, Scholes (JPE 1973), *The Pricing of Options and Corporate Liabilities*, https://doi.org/10.1086/260062 | Background |

**Gap.** No verified paper applies neural ODEs/SDEs to static cross-sectional stock ranking.
The useful material is the training-dynamics and noise theory (A, B), not the architectures.

## Suggested experiments (ranked)
1. Post-hoc weight interpolation θ(α) = θ₀ + α(θ_T − θ₀), α ∈ {0.3 … 1.15}: a free, continuous
   "training amount" dial from one run.
2. L2-SP (decay toward init) with longer training: same peak, flatter plateau, less sensitivity.
3. Test-time Gaussian input smoothing (heat-equation averaging); judge on BMC, not corr.
4. Training noise: input jitter, or subsampling 25–50% of stocks per era per step.
5. Fixed-T ODE block with kinetic penalty; only if 1–4 show leverage.
