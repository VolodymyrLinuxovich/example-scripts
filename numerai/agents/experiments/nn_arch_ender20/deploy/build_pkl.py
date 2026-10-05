"""Build the Numerai upload pickle from deploy/model_weights.npz + model_meta.json.

Run ONLY with the venv that matches Numerai's default compute image
(Python 3.12, numpy 2.1.3, pandas 2.3.1, cloudpickle 3.1.1):
  .venv_numerai312/bin/python numerai/agents/experiments/nn_arch_ender20/deploy/build_pkl.py
Writes numerai/nn_ender20.pkl. The pickled function imports only numpy and pandas.
"""
import argparse
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np

HERE = Path(__file__).parent
OUT = HERE.parents[3] / "nn_ender20.pkl"

p = argparse.ArgumentParser()
p.add_argument("--blend-weight", type=float, default=0.0,
               help="weight on the benchmark's live ranks (0 = pure NN)")
p.add_argument("--blend-benchmark", default="v53_lgbm_ender60")
args = p.parse_args()

assert sys.version_info[:2] == (3, 12), f"build with Python 3.12, not {sys.version}"
assert np.__version__ == "2.1.3", f"numpy {np.__version__} != container's 2.1.3"

meta = json.loads((HERE / "model_meta.json").read_text())
w = np.load(HERE / "model_weights.npz")
# Weights travel as (bytes, shape) so the pickle holds no numpy objects at all.
networks = [[(w[f"s{s}_w{i}"].astype("<f4").tobytes(), w[f"s{s}_w{i}"].shape,
              w[f"s{s}_b{i}"].astype("<f4").tobytes(), w[f"s{s}_b{i}"].shape)
             for i in range(meta["n_layers"])] for s in range(meta["seeds"])]


def make_predict(features, networks, blend_weight, blend_benchmark):
    """Return a self-contained predict(live_features, live_benchmark_models) closure."""

    def predict(live_features, live_benchmark_models=None):
        import numpy as np
        import pandas as pd

        mats = [[(np.frombuffer(wb, dtype="<f4").reshape(ws), np.frombuffer(bb, dtype="<f4").reshape(bs))
                 for wb, ws, bb, bs in net] for net in networks]
        x = live_features.reindex(columns=features).fillna(2).to_numpy(dtype=np.float32)
        x = (x - 2.0) / 2.0
        raw = np.zeros(len(x), dtype=np.float64)
        # numpy 2.x on macOS Accelerate emits spurious FP warnings in float32 matmul; results
        # are exact (checked against float64), and non-finite outputs are caught below.
        with np.errstate(all="ignore"):
            for layers in mats:
                h = x
                for i, (wt, bias) in enumerate(layers):
                    h = h @ wt.T + bias
                    if i < len(layers) - 1:
                        h = h / (1.0 + np.exp(-h))  # SiLU
                raw += h[:, 0]
        if not np.isfinite(raw).all():
            raise ValueError("non-finite network output")
        score = pd.Series(raw / len(mats), index=live_features.index)
        group = live_features["era"] if "era" in live_features.columns else None

        def rank(s):
            return s.groupby(group).rank(pct=True) if group is not None else s.rank(pct=True)

        pred = rank(score)
        if blend_weight > 0 and live_benchmark_models is not None \
                and blend_benchmark in live_benchmark_models.columns:
            bench = live_benchmark_models[blend_benchmark].reindex(live_features.index)
            if bench.notna().all():
                pred = rank((1 - blend_weight) * pred + blend_weight * rank(bench))
        return pred.clip(0.0, 1.0).to_frame("prediction")

    return predict


predict = make_predict(meta["features"], networks, args.blend_weight, args.blend_benchmark)
with open(OUT, "wb") as f:
    cloudpickle.dump(predict, f)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB), Python {sys.version.split()[0]}, "
      f"numpy {np.__version__}, cloudpickle {cloudpickle.__version__}, blend_weight={args.blend_weight}")
