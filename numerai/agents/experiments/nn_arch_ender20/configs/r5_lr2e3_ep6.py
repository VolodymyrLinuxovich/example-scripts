import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r5_lr2e3_ep6", loss="corr", batching="era", eras_per_batch=4, epochs=6, lr=2e-3)
