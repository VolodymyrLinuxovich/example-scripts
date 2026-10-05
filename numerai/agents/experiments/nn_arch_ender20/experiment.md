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
| r1_mlp_wide_mse | 1024→512→256, dropout 0.2 | 0.0259 | 0.0061 | **0.0062** | 0.404 |
| r1_mlp_corr | per-era Pearson loss, 4-era batches, 20 epochs | 0.0184 | 0.0055 | 0.0059 | 0.258 |
| r1_mlp_mse | base | 0.0249 | 0.0053 | 0.0052 | 0.402 |
| r0_baseline_small_lgbm | LightGBM baseline | 0.0277 | 0.0042 | 0.0042 | 0.485 |
| r1_resmlp_mse | residual MLP, width 256, 2 blocks | 0.0187 | 0.0036 | 0.0037 | 0.319 |
| r1_tabm_mse | TabM BatchEnsemble, k=8, 256×256 | 0.0213 | 0.0026 | 0.0026 | 0.393 |
| r0_smoke_mlp | base, 1 epoch | 0.0188 | 0.0024 | 0.0022 | 0.345 |
| r1_mlp_embed_mse | learned per-bin embeddings (d=4) | 0.0157 | 0.0012 | 0.0006 | 0.312 |

**Findings so far**
- Width helps: the wide MLP (1024→512→256) is the round-1 winner on corr and both BMC metrics.
- Plain MLPs beat the LightGBM baseline on BMC, despite lower raw corr: their predictions
  are less correlated with the benchmark, so they add more unique signal.
- Correlation loss at 20 epochs overfits hard (train corr 0.24 vs OOF 0.018).
- More elaborate architectures (residual MLP, TabM) and learned embeddings all did worse.
  Scalar inputs ((x−2)/2) are clearly better than embeddings.

## Round 2: training length, regularization, width
Base for the correlation-loss variants: r1_mlp_corr (256→128, 4-era batches).

| model | change | corr | bmc | bmc_last_200 | corr_with_bench |
|---|---|---|---|---|---|
| r2_corr_ep8 | corr loss, 8 epochs (was 20) | 0.0253 | 0.0083 | **0.0085** | 0.340 |
| r2_corr_epb1 | corr loss, 1 era per batch, 5 epochs | 0.0254 | 0.0078 | 0.0080 | 0.350 |
| r2_mse_ep8 | MSE MLP, 8 epochs (was 4) | 0.0244 | 0.0071 | 0.0074 | 0.340 |
| r2_corr_reg | corr loss, dropout 0.3, weight decay 1e-3 | 0.0230 | 0.0068 | 0.0070 | 0.319 |
| r2_corr_mse | corr + 0.1·MSE hybrid, 20 epochs | 0.0184 | 0.0052 | 0.0057 | 0.265 |
| r2_wide_corr | wide MLP + corr loss, 20 epochs | 0.0183 | 0.0059 | 0.0059 | 0.247 |
| r2_wider_mse | 2048→1024→512, dropout 0.3 | 0.0255 | 0.0057 | 0.0059 | 0.404 |
| r2_wide_ep8 | wide MLP, MSE, 8 epochs (was 4) | 0.0229 | 0.0073 | 0.0076 | 0.304 |

**Findings so far**
- Training length is the dominant lever. Correlation loss at 20 epochs overfits, at 8 it is the
  best model so far (+37% BMC over the round-1 best). MSE at 4 epochs was under-trained; 8 epochs
  lifts it past the wide MLP.
- Both losses converge on about 8 epochs for this data size.
- Width has plateaued for MSE: 2048-wide is no better than 1024-wide. The wide MLP also gains
  from 8 epochs (0.0062 → 0.0076), but still trails the narrow correlation-loss model.
- Wide + correlation loss at 20 epochs overfits even faster (train corr 0.28).
- The MSE term in the hybrid is too weak at weight 0.1 to change anything; dropped.

## Round 3: refine around the leader (r2_corr_ep8)
One change each: correlation-loss epochs 5 and 12 (map the peak), wide network + correlation
loss at 6 epochs, dropout 0.3 at 8 epochs, and a 3-seed average (also measures seed noise).

| model | change | corr | bmc | bmc_last_200 | corr_with_bench |
|---|---|---|---|---|---|
| r2_corr_ep8 | leader (round 2) | 0.0253 | 0.0083 | **0.0085** | 0.340 |
| r3_corr_ep8_seeds3 | 3-seed average | 0.0261 | 0.0082 | 0.0083 | 0.357 |
| r3_wide_corr_ep6 | wide MLP, corr loss, 6 epochs | 0.0261 | 0.0079 | 0.0077 | 0.360 |
| r3_corr_ep12 | 12 epochs | 0.0225 | 0.0073 | 0.0077 | 0.301 |
| r3_corr_ep5 | 5 epochs | 0.0253 | 0.0067 | 0.0069 | 0.379 |
| r3_corr_ep8_drop03 | dropout 0.3 | 0.0260 | 0.0067 | 0.0068 | 0.390 |

**Findings**
- No improvement over the leader (first non-improving round).
- 8 epochs is the peak: longer training keeps lowering correlation with the benchmark, but past
  8 epochs raw corr drops faster than uniqueness helps.
- Seed noise is about ±0.0002 BMC: the 3-seed average matches the leader and raises corr.
  Its predictions correlate 0.96 with the single-seed run.
- Width works with correlation loss once training is short (6 epochs), close to but below the
  narrow leader. Extra dropout at 8 epochs pulls predictions toward the benchmark and lowers BMC.

## Round 4: untested dimensions
One change each from the leader: learning rate 5e-4 and 2e-3, 8 eras per batch, depth
(512→256→128), input dropout 0.1.
