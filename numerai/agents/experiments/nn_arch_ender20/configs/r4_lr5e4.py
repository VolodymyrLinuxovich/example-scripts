import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r4_lr5e4", loss="corr", batching="era", eras_per_batch=4, epochs=8, lr=5e-4)
