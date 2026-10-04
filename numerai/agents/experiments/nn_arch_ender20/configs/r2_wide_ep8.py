import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r2_wide_ep8", hidden=[1024, 512, 256], dropout=0.2, epochs=8)
