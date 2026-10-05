import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _scale import make
CONFIG = make("s1_narrow_ep6_seeds3", loss="corr", batching="era", eras_per_batch=4, lr=2e-3, n_seeds=3, epochs=6)
