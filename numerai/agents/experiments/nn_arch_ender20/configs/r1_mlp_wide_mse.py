import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r1_mlp_wide_mse", hidden=[1024, 512, 256], dropout=0.2)
