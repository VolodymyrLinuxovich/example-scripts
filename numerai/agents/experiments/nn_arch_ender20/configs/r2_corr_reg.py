import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r2_corr_reg", loss="corr", batching="era", eras_per_batch=4, epochs=20, dropout=0.3, weight_decay=1e-3)
