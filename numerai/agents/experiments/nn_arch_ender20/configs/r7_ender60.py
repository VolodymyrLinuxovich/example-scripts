import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r7_ender60", loss="corr", batching="era", eras_per_batch=4, lr=2e-3, epochs=6, hidden=[1024, 512, 256], dropout=0.2)
# Train and score on the 60-day target that Numerai pays on, against the ender60 benchmark.
CONFIG["data"]["target_col"] = "target_ender_60"
CONFIG["data"]["benchmark_model"] = "v53_lgbm_ender60"
