import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make
CONFIG = make("r1_mlp_embed_mse", input_encoding="embed", embed_dim=4)
