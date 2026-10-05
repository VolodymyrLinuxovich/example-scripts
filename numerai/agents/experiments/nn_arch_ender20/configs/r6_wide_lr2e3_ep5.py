import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r6_wide_lr2e3_ep5", loss="corr", batching="era", eras_per_batch=4, lr=2e-3, epochs=5, hidden=[1024, 512, 256], dropout=0.2)
