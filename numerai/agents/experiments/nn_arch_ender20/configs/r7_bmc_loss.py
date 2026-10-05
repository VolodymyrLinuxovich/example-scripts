import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r7_bmc_loss", loss="bmc", batching="era", eras_per_batch=4, lr=2e-3, epochs=6, hidden=[1024, 512, 256], dropout=0.2, bench_col="v53_lgbm_ender20")

