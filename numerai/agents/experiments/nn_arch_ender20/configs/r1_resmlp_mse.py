import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r1_resmlp_mse", arch="resmlp", hidden=256, n_blocks=2)
