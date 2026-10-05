import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r4_indrop01", loss="corr", batching="era", eras_per_batch=4, epochs=8, input_dropout=0.1)
