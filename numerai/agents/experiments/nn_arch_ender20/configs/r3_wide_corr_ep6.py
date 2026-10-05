import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r3_wide_corr_ep6", loss="corr", batching="era", eras_per_batch=4, epochs=6, hidden=[1024, 512, 256], dropout=0.2)
