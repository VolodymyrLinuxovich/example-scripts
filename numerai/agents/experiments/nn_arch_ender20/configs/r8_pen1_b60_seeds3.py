import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r8_pen1_b60_seeds3", loss="corr_bench", batching="era", eras_per_batch=4, lr=2e-3, hidden=[1024, 512, 256], dropout=0.2, n_seeds=3, epochs=6, bench_col="v53_lgbm_ender60", bench_penalty=1.0)
CONFIG["data"]["benchmark_model"] = "v53_lgbm_ender60"  # target == target_ender_60 in v5.3
