import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r2_corr_mse", loss="corr_mse", mse_weight=0.1, batching="era", eras_per_batch=4, epochs=20)
