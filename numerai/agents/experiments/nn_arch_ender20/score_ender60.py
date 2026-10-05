"""Re-score saved OOF predictions on the payout target (target_ender_60 vs v53_lgbm_ender60).

Usage (repo root):  PYTHONPATH=numerai .venv/bin/python numerai/agents/experiments/nn_arch_ender20/score_ender60.py run1 run2 ...
Writes results_ender60/<run>.json with the same metric layout as results/.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
NUMERAI = HERE.parents[2]
sys.path.insert(0, str(NUMERAI))
from agents.code.metrics import numerai_metrics as nm  # noqa: E402

DATA = {"scout": ("v5.3/downsampled_full.parquet", "v5.3/downsampled_full_benchmark_models.parquet"),
        "scale": ("v5.3/half_odd_full.parquet", "v5.3/half_odd_full_benchmark_models.parquet")}


def score(run):
    data_path, bench_path = DATA["scale" if run.startswith("s") else "scout"]
    preds = pd.read_parquet(HERE / "predictions" / f"{run}.parquet", columns=["id", "era", "prediction"])
    t60 = pd.read_parquet(NUMERAI / data_path, columns=["id", "target_ender_60"])
    df = preds.merge(t60, on="id").dropna(subset=["target_ender_60"])
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        df.to_parquet(tmp.name, index=False)
        s = nm.summarize_prediction_file_with_bmc(
            tmp.name, ["prediction"], "target_ender_60", "v5.3", benchmark_model="v53_lgbm_ender60",
            benchmark_data_path=bench_path, era_col="era", id_col="id")
    metrics = {k: s[k].loc["prediction"].to_dict() for k in ("corr", "bmc", "bmc_last_200_eras")}
    out = HERE / "results_ender60"
    out.mkdir(exist_ok=True)
    (out / f"{run}.json").write_text(json.dumps({"run": run, "target": "target_ender_60",
                                                  "benchmark": "v53_lgbm_ender60", "metrics": metrics}, indent=2))
    m = metrics
    print(f"{run:32s} corr60 {m['corr']['mean']:.4f}  bmc60 {m['bmc']['mean']:.4f}  "
          f"bmc60_l200 {m['bmc_last_200_eras']['mean']:.4f}  w/bench60 {m['bmc']['avg_corr_with_benchmark']:.3f}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        score(r)
