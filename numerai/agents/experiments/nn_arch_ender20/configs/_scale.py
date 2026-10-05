"""Scale/confirmation phase: every 2nd era (offset 1), disjoint from the scout eras."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from _common import make as _make


def make(name, **params):
    cfg = _make(name, **params)
    cfg["data"]["full_data_path"] = "v5.3/half_odd_full.parquet"
    cfg["data"]["benchmark_data_path"] = "v5.3/half_odd_full_benchmark_models.parquet"
    return cfg
