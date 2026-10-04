"""Shared settings for the nn_arch_ender20 experiment (downsampled scout phase)."""
import copy

BASE = {
    "data": {
        "data_version": "v5.3",
        "embargo_eras": 13,
        "era_col": "era",
        "feature_set": "medium",
        "target_col": "target",  # == target_ender_20
        "id_col": "id",
        "full_data_path": "v5.3/downsampled_full.parquet",
        "benchmark_data_path": "v5.3/downsampled_full_benchmark_models.parquet",
    },
    "model": {
        "type": "TorchNNRegressor",
        "x_groups": ["features", "era", "benchmark_models"],
        "params": {
            "arch": "mlp",
            "hidden": [256, 128],
            "dropout": 0.1,
            "loss": "mse",
            "batching": "random",
            "batch_size": 4096,
            "epochs": 4,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "random_state": 1337,
        },
    },
    "output": {},
    "preprocessing": {"missing_value": 2.0, "nan_missing_all_twos": False},
    "training": {"cv": {"embargo": 13, "enabled": True, "min_train_size": 0,
                        "mode": "expanding", "n_splits": 5}},
}


def make(name, **params):
    cfg = copy.deepcopy(BASE)
    cfg["model"]["params"].update(params)
    cfg["output"]["results_name"] = name
    return cfg
