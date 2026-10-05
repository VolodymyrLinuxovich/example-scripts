# Literature notes: number theory and tabular deep learning

Collected 2026-10-04 by a literature-search agent; every URL was opened and its title, authors
and venue checked.

## A. Number theory and related discrete math

| paper | key idea | applies here? |
|---|---|---|
| Bousquet, Gelly, Kurach, Teytaud, Vincent (2017), *Critical Hyper-Parameters: No Random, No Cry*, https://arxiv.org/abs/1706.03200 | Low-discrepancy sequences beat random search for hyperparameters | Yes: Sobol' search over lr × epochs × width × regularization |
| Roy, Owen, Balandat, Haberland (2023), *Quasi-Monte Carlo Methods in Python*, JOSS 8(84):5309, https://doi.org/10.21105/joss.05309 | `scipy.stats.qmc` (scrambled Sobol', Halton, LHS) | Tooling for the above |
| Rahimi & Recht (2007), *Random Features for Large-Scale Kernel Machines*, NIPS 20, https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines | Random Fourier features approximate kernels | Indirectly, via periodic embeddings |
| Avron, Sindhwani, Yang, Mahoney (2016), *Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels*, https://arxiv.org/abs/1412.8293 | Low-discrepancy frequencies approximate kernels better | Second-order at best |
| Weinberger et al. (2009), *Feature Hashing for Large Scale Multitask Learning*, https://arxiv.org/abs/0902.2206 | Hashing trick for huge sparse feature spaces | No: features are dense and few |

**Verdict.** Number theory offers no model of stock returns and no way into obfuscated features.
Its one real use here is quasi-Monte Carlo (Sobol'/Halton) hyperparameter search: fewer runs for
the same coverage. Number-theoretic transforms, p-adic and modular methods have no documented
use on tabular or financial prediction.

## B. Tabular deep learning

| paper | key idea | relevance |
|---|---|---|
| Gorishniy et al. (NeurIPS 2021), *Revisiting Deep Learning Models for Tabular Data*, https://arxiv.org/abs/2106.11959 | ResNet and FT-Transformer baselines | FT-Transformer is O(780²) per row; too costly here |
| Gorishniy, Rubachev, Babenko (NeurIPS 2022), *On Embeddings for Numerical Features*, https://arxiv.org/abs/2203.05556 | Piecewise-linear and periodic embeddings | With 5 bins, piecewise-linear = one-hot, which explains our overfitting embeddings |
| Gorishniy, Kotelnikov, Babenko (ICLR 2025), *TabM*, https://arxiv.org/abs/2410.24210 | Parameter-efficient BatchEnsemble MLPs | Lost in our setting |
| Holzmüller, Grinsztajn, Steinwart (NeurIPS 2024), *Better by Default* (RealMLP), https://arxiv.org/abs/2407.04491 | Diagonal input scaling, β₂=0.95, periodic embeddings, scheduled regularization | Several one-line changes to test |
| Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022 D&B), *Why do tree-based models still outperform deep learning on tabular data?*, https://arxiv.org/abs/2207.08815 | NNs struggle with uninformative features and irregular targets | Matches 780 weak features; NN–tree differences are what BMC rewards |
| Kadra, Lindauer, Hutter, Grabocka (NeurIPS 2021), *Well-tuned Simple Nets Excel on Tabular Datasets*, https://arxiv.org/abs/2106.11189 | Regularized plain MLPs beat specialized nets | Matches our finding |
| Somepalli et al. (2021), *SAINT*, https://arxiv.org/abs/2106.01342 | Row + column attention, contrastive pretraining | Low: costly and behind MLPs |
| Hollmann et al. (Nature 2025), *TabPFN v2*, https://www.nature.com/articles/s41586-024-08328-6 (also TabPFN-2.5 https://arxiv.org/abs/2511.08667, TabPFN-3 https://arxiv.org/abs/2605.13986) | In-context tabular foundation models | Do not scale to millions of rows × 780 features |
| Izmailov et al. (UAI 2018), *Averaging Weights Leads to Wider Optima* (SWA), https://arxiv.org/abs/1803.05407 | Weight averaging finds flatter optima | Nearly free to test |
| Lakshminarayanan, Pritzel, Blundell (NIPS 2017), *Deep Ensembles*, https://arxiv.org/abs/1612.01474 | Independent seeds ensemble well | Seed averaging, already used for confirmation |
| Wortsman et al. (ICML 2022), *Model soups*, https://arxiv.org/abs/2203.05482 | Averaging weights across hyperparameters | Needs a shared start; checkpoint averaging only |

## Suggested experiments (ranked)
1. EMA or SWA of weights over the last epochs; more seeds.
2. RealMLP tricks: learnable diagonal input scaling (optional L1), AdamW β₂=0.95.
3. Feature dropout (set 10–30% of inputs to the neutral bin), swept with weight decay.
4. Sobol'-sampled joint search over lr, epochs, width, dropout, weight decay.
5. Shared-frequency periodic embeddings (very few parameters); stop if overfitting appears.
