# Repo small baseline (LGBM, medium features, downsampled) for apples-to-apples comparison.
import runpy; from pathlib import Path
CONFIG = runpy.run_path(str(Path(__file__).parents[3] / "baselines/configs/small_lgbm_ender20_baseline.py"))["CONFIG"]
CONFIG["model"]["params"].update({"device_type": "cpu", "n_jobs": 10})
CONFIG["data"]["id_col"] = "id"
CONFIG["output"] = {"results_name": "r0_baseline_small_lgbm"}
