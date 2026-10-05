"""Test numerai/nn_ender20.pkl in a clean process, as Numerai's container would run it.

Run with the container-matched venv, single-threaded like Numerai's 1-CPU machine:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    .venv_numerai312/bin/python numerai/agents/experiments/nn_arch_ender20/deploy/test_pkl.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NUMERAI = Path(__file__).parents[4]
PKL = NUMERAI / "nn_ender20.pkl"
LIMIT_SECONDS, LIMIT_GB = 600, 4.0

t0 = time.time()
model = pd.read_pickle(PKL)  # exactly how numerai-predict loads it
t_load = time.time() - t0
live = pd.read_parquet(NUMERAI / "v5.3/live.parquet")
bench = pd.read_parquet(NUMERAI / "v5.3/live_benchmark_models.parquet")
t1 = time.time()
pred = model(live, bench)
t_pred = time.time() - t1
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS

checks = {
    "returns a DataFrame": isinstance(pred, pd.DataFrame),
    "single 'prediction' column": list(pred.columns) == ["prediction"],
    "one row per live id": len(pred) == len(live) and pred.index.equals(live.index) and pred.index.is_unique,
    "no NaN": not pred.isna().any().any(),
    "all values in [0, 1]": bool(pred["prediction"].between(0, 1).all()),
    f"predict time < {LIMIT_SECONDS}s": t_pred < LIMIT_SECONDS,
    f"peak memory < {LIMIT_GB} GB": peak_gb < LIMIT_GB,
}
ex_path = NUMERAI / "v5.3/live_example_preds.parquet"
example = pd.read_parquet(ex_path)["prediction"].reindex(pred.index) if ex_path.exists() else None
stats = {
    "rows": len(pred), "live_ids": len(live), "eras": live["era"].unique().tolist(),
    "load_seconds": round(t_load, 2), "predict_seconds": round(t_pred, 2), "peak_rss_gb": round(peak_gb, 3),
    "prediction_min": float(pred["prediction"].min()), "prediction_max": float(pred["prediction"].max()),
    "spearman_vs_example_model": None if example is None else round(float(pred["prediction"].corr(example, method="spearman")), 4),
    "spearman_vs_benchmark_ender60": round(float(pred["prediction"].corr(
        bench["v53_lgbm_ender60"].reindex(pred.index), method="spearman")), 4),
    "spearman_vs_benchmark_ender20": round(float(pred["prediction"].corr(
        bench["v53_lgbm_ender20"].reindex(pred.index), method="spearman")), 4),
    "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
}
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
print(json.dumps(stats, indent=1))
sys.exit(0 if all(checks.values()) else 1)
