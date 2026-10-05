import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r3_corr_ep8_drop03", loss="corr", batching="era", eras_per_batch=4, epochs=8, dropout=0.3)
