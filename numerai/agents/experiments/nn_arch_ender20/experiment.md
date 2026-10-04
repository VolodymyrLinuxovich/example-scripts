# Neural-network architecture search for target ender20

Started 2026-10-04. Work in progress.

## Goal
Find the neural-network architecture that best predicts `target` (= `target_ender_20`, v5.3)
as measured by BMC (Benchmark Model Contribution vs official `v53_lgbm_ender20`).

## Setup
- **Data (scout phase):** `v5.3/downsampled_full.parquet` (every 4th era of train+validation,
  307 eras, 1.7M rows), `medium` feature set (780 int8 features). Chosen over `all`
  (3,555 features) to fit a 16 GB MacBook Air.
- **CV:** 5 expanding era folds, 13-era embargo; fold 0 has no training data, so OOF covers
  246 eras from 4 trained folds.
- **Model:** `TorchNNRegressor` (`agents/code/modeling/models/torch_nn.py`), trained on Apple MPS.
  Fixed epoch count, no early stopping (no leakage into OOF scores).
- **Primary metric:** `bmc_last_200_eras`, tie-break `bmc` mean. Sanity: `corr` and correlation
  with the benchmark.
- **Baseline:** repo `small_lgbm_ender20_baseline` on the same data/features.

## Round 1: architecture and loss scout
Base: MLP 780→256→128→1, SiLU, dropout 0.1, MSE, 4 epochs, batch 4096, AdamW lr 1e-3, cosine.
Each variant changes one thing.

| model | change | corr | bmc | bmc_last_200 | corr_with_bench |
|---|---|---|---|---|---|
| r1_mlp_corr | per-era Pearson loss, 4-era batches, 20 epochs | 0.0184 | 0.0055 | **0.0059** | 0.258 |
| r1_mlp_mse | base | 0.0249 | 0.0053 | 0.0052 | 0.402 |
| r0_baseline_small_lgbm | LightGBM baseline | 0.0277 | 0.0042 | 0.0042 | 0.485 |
| r1_resmlp_mse | residual MLP, width 256, 2 blocks | 0.0187 | 0.0036 | 0.0037 | 0.319 |
| r1_tabm_mse | TabM BatchEnsemble, k=8, 256×256 | 0.0213 | 0.0026 | 0.0026 | 0.393 |
| r0_smoke_mlp | base, 1 epoch | 0.0188 | 0.0024 | 0.0022 | 0.345 |
| r1_mlp_embed_mse | learned per-bin embeddings (d=4) | 0.0157 | 0.0012 | 0.0006 | 0.312 |
| r1_mlp_wide_mse | 1024→512→256, dropout 0.2 | pending | | | |

**Findings so far**
- Plain MLPs beat the LightGBM baseline on BMC, despite lower raw corr: their predictions
  are less correlated with the benchmark, so they add more unique signal.
- Correlation loss gives the best BMC but overfits hard (train corr 0.24 vs OOF 0.018).
- More elaborate architectures (residual MLP, TabM) and learned embeddings all did worse.
  Scalar inputs ((x−2)/2) are clearly better than embeddings.

**Next (round 2):** regularize the correlation-loss MLP (dropout, weight decay, fewer epochs,
correlation + MSE hybrid) and test whether the MSE MLP is under-trained.
