import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r2_wider_mse", hidden=[2048, 1024, 512], dropout=0.3)
