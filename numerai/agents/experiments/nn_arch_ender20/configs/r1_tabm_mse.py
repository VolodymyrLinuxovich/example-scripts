import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r1_tabm_mse", arch="tabm", hidden=[256, 256], k=8)
