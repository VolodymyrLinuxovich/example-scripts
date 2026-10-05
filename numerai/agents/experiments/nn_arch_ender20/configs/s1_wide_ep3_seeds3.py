import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _scale import make
CONFIG = make("s1_wide_ep3_seeds3", loss="corr", batching="era", eras_per_batch=4, lr=2e-3, n_seeds=3, epochs=3, hidden=[1024, 512, 256], dropout=0.2)
