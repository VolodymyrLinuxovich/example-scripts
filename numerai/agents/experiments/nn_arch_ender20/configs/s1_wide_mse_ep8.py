import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _scale import make
CONFIG = make("s1_wide_mse_ep8", hidden=[1024, 512, 256], dropout=0.2, epochs=8)
