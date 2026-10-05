# Literature notes: ML for cross-sectional return prediction

Collected 2026-10-04 by a literature-search agent; every URL was opened.

**Key fact (verified against https://docs.numer.ai/numerai-tournament/scoring on 2026-10-04):**
since round 1343, CORR, MMC and BMC are scored on the **60-day** Ender target, and BMC
neutralizes against the **stake-weighted benchmark models**. This study's BMC (ender20 target,
single `v53_lgbm_ender20` benchmark) is a proxy, not the payout metric.

## A. Asset-pricing ML
| source | key idea | relevance |
|---|---|---|
| Gu, Kelly, Xiu (RFS 2020), *Empirical Asset Pricing via Machine Learning*, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3159577 | NNs/trees win via interactions; shallow NN3 beats deeper; seed ensembles | Matches "shallow MLP wins" |
| Gu, Kelly, Xiu (J. Econometrics 2021), *Autoencoder Asset Pricing Models*, https://www.semanticscholar.org/paper/Autoencoder-Asset-Pricing-Models-Gu-Kelly/5953bcc383c3ceb866b8ef7f7c16d166e5f2ab4d | Nonlinear factor exposures | Factor-structure regularizer idea |
| Chen, Pelger, Zhu (Mgmt Sci 2023/24), *Deep Learning in Asset Pricing*, https://arxiv.org/abs/1904.00745 | Adversarial SDF via moment conditions | Precedent for portfolio-moment losses |
| Avramov, Cheng, Metzker (Mgmt Sci 2023), *Machine Learning vs. Economic Restrictions*, https://pubsonline.informs.org/doi/10.1287/mnsc.2022.4449 | DL profits concentrate in hard-to-arbitrage stocks | Unique signal is the valuable part |

## B. The complexity debate
| source | key idea |
|---|---|
| Kelly, Malamud, Zhou (J. Finance 2024), *The Virtue of Complexity in Return Prediction*, https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13298 | Overparameterized ridge models raise OOS R² |
| Nagel (NBER w34104, 2025), *Seemingly Virtuous Complexity in Return Prediction*, https://www.nber.org/papers/w34104 | Those models reduce to volatility-timed momentum |
| Fallahgoul (2025), *High-Dimensional Learning in Finance*, https://arxiv.org/abs/2506.03780 | Standardization and information limits bound what is learnable |

## C. Loss functions
| source | key idea |
|---|---|
| Poh, Lim, Zohren, Roberts (2020), *Building Cross-Sectional Systematic Strategies by Learning to Rank*, https://arxiv.org/abs/2012.07149 | Listwise ranking ~3× Sharpe vs regression |
| Lin, Su, Yang (2026), *LambdaRankIC*, https://arxiv.org/abs/2605.00501 | Directly optimizes rank IC |
| Bai, Pukthuanthong, *ML Classification and Portfolio Construction*, https://arxiv.org/abs/2108.02283 | Cross-entropy on return classes beats MSE |
| Zhang, Zohren, Roberts (2020), *Deep Learning for Portfolio Optimization*, https://arxiv.org/abs/2005.13665 | End-to-end Sharpe loss |
| mdo (Numerai forum 2020), *Custom loss functions for XGBoost using PyTorch*, https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960 | Sharpe-of-corr loss; ratio losses can degenerate |

## D. Neutralization
| source | key idea |
|---|---|
| Daniel, Mota, Rottke, Santos (RFS 2020), *The Cross-Section of Risk and Returns*, https://academic.oup.com/rfs/article-abstract/33/5/1927/5803086 | Hedging unpriced risk nearly doubles squared Sharpe |
| bobyfisch (forum 2022), *An introduction to feature neutralization / exposure*, https://forum.numer.ai/t/an-introduction-to-feature-neutralization-exposure/4955 | Partial / riskiest-feature neutralization |
| by256 (forum 2024), *Feature Neutralization Increases Bias and Reduces Variance*, https://forum.numer.ai/t/feature-neutralization-increases-bias-and-reduces-variance/7486 | Best proportion ~0.5; NNs like higher |
| master_key (forum 2023), *Benchmark Models*, https://forum.numer.ai/t/benchmark-models/6754 | BMC/MMC definitions |

## E. Ensembling and auxiliary targets
| source | key idea |
|---|---|
| Rapach, Strauss, Zhou (RFS 2010), *Out-of-Sample Equity Premium Prediction: Combination Forecasts*, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1257858 | Equal-weight combinations beat single models |
| master_key (forum 2022), *New Targets for the Tournament*, https://forum.numer.ai/t/new-targets-for-the-tournament/5842 | Auxiliary-target ensembles improve Sharpe |
| Wong, Barahona (2022/23), *Online learning techniques for temporal tabular datasets with regime changes*, https://arxiv.org/abs/2301.00790 | Numerai data; dynamic ensembling under regime shift |
| richai (forum 2020), *Era Boosted Models*, https://forum.numer.ai/t/era-boosted-models/189 | Weak OOS gains |

## F. Overfitting and shrinkage
| source | key idea |
|---|---|
| Bailey, López de Prado (JPM 2014), *The Deflated Sharpe Ratio*, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | Deflate Sharpe for the number of trials |
| Bailey, Borwein, López de Prado, Zhu (J. Comp. Finance 2017), *The Probability of Backtest Overfitting*, https://escholarship.org/uc/item/4w1110bb | PBO → 1 as configurations multiply |
| Arian, Norouzi, Seco (KBS 2024), *Backtest Overfitting in the Machine Learning Era*, https://www.ssrn.com/abstract=4778909 | Combinatorial purged CV lowers PBO |
| Kozak, Nagel, Santosh (JFE 2020), *Shrinking the Cross-Section*, https://econpapers.repec.org/RePEc:eee:jfinec:v:135:y:2020:i:2:p:271-292 | Shrinkage toward PCs |

## Suggested experiments (ranked)
1. Benchmark-aware loss: −corr(p, y) + λ·corr(p, b)². Leakage risk: benchmark predictions on
   training eras may be in-sample fits; check their per-era corr on train vs validation eras first.
2. Multi-task heads on ender60 and low-correlation auxiliary targets (≥13-era embargo for 60d).
3. Target ensemble + seed and checkpoint ensembles; fit weights on validation folds only.
4. Partial feature neutralization (p ≈ 0.5) of predictions.
5. Sharpe-of-corr or soft era-weighting loss (keep ε; degenerate-solution risk).
6. Listwise / soft-Spearman / 5-class cross-entropy losses.
