"""Train the deployment model on train + validation eras and export plain numpy weights.

Same algorithm as the study's TorchNNRegressor (wide MLP, per-era correlation loss with an
optional benchmark penalty, AdamW + warm-up/cosine), but with the training amount fixed in
optimizer steps, the study's main finding.

Run from repo root (needs torch):
  .venv/bin/python numerai/agents/experiments/nn_arch_ender20/deploy/train_final.py --penalty 0.3
Writes deploy/model_weights.npz and deploy/model_meta.json.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).parent
DATA = HERE.parents[3] / "v5.3"

p = argparse.ArgumentParser()
p.add_argument("--penalty", type=float, default=0.3, help="benchmark penalty λ (0 = plain correlation loss)")
p.add_argument("--benchmark", default="v53_lgbm_ender60")
p.add_argument("--steps", type=int, default=350, help="optimizer steps (each = ERAS_PER_BATCH eras)")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--era-stride", type=int, default=2, help="use every Nth era (memory)")
args = p.parse_args()

HIDDEN, DROPOUT, LR, WD, ERAS_PER_BATCH, WARMUP = [1024, 512, 256], 0.2, 2e-3, 1e-5, 4, 0.05
TARGET = "target"  # == target_ender_60 in v5.3
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
features = json.load(open(DATA / "features.json"))["feature_sets"]["medium"]

# --- data: every Nth era of train + validation, newest era always included -----------------
eras = sorted(set(pq.read_table(DATA / "full.parquet", columns=["era"])["era"].to_pylist()), key=int)
keep = eras[::-1][::args.era_stride][::-1]
t0 = time.time()
tbl = pq.read_table(DATA / "full.parquet", columns=["id", "era", TARGET] + features,
                    filters=pc.field("era").isin(keep))
df = tbl.to_pandas()
del tbl
df = df.dropna(subset=[TARGET])
bench = pd.read_parquet(DATA / "full_benchmark_models.parquet", columns=[args.benchmark])
df[args.benchmark] = bench[args.benchmark].reindex(df["id"].to_numpy()).to_numpy()
df = df.sort_values("era", kind="stable")
print(f"data: {len(df):,} rows, {df.era.nunique()} eras ({df.era.iloc[0]}..{df.era.iloc[-1]}), "
      f"loaded in {time.time() - t0:.0f}s, device {DEVICE}", flush=True)

x = torch.from_numpy(df[features].to_numpy(dtype=np.int8)).to(DEVICE)
y = torch.from_numpy(df[TARGET].to_numpy(dtype=np.float32) - 0.5).to(DEVICE)
b = torch.from_numpy(df[args.benchmark].fillna(0.5).to_numpy(dtype=np.float32)).to(DEVICE)
codes = pd.factorize(df["era"])[0]
bounds = np.flatnonzero(np.diff(codes)) + 1
starts, ends = np.r_[0, bounds], np.r_[bounds, len(codes)]
train_eras = (str(df.era.iloc[0]), str(df.era.iloc[-1]), int(df.era.nunique()))
del df


def era_corr(p, t):
    p = p - p.mean()
    t = t - t.mean()
    return (p * t).sum() / (p.norm() * t.norm() + 1e-8)


def build():
    layers, d = [], len(features)
    for h in HIDDEN:
        layers += [torch.nn.Linear(d, h), torch.nn.SiLU(), torch.nn.Dropout(DROPOUT)]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


arrays = {}
for seed in range(args.seeds):
    torch.manual_seed(1337 + seed)
    rng = np.random.default_rng(1337 + seed)
    net = build().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    warm = max(1, int(args.steps * WARMUP))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    order, pos = rng.permutation(len(starts)), 0
    net.train()
    for step in range(args.steps):
        if pos + ERAS_PER_BATCH > len(order):
            order, pos = rng.permutation(len(starts)), 0
        corrs, pens = [], []
        for e in order[pos:pos + ERAS_PER_BATCH]:
            s, t = int(starts[e]), int(ends[e])
            pred = net((x[s:t].float() - 2.0) / 2.0).squeeze(-1)
            corrs.append(era_corr(pred, y[s:t]))
            pens.append(era_corr(pred, b[s:t]) ** 2)
        pos += ERAS_PER_BATCH
        loss = -torch.stack(corrs).mean() + args.penalty * torch.stack(pens).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            print(f"  seed {seed} step {step + 1}/{args.steps} loss {float(loss):+.4f}", flush=True)
    net.eval().cpu()
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(linears):
        arrays[f"s{seed}_w{i}"] = m.weight.detach().numpy().astype(np.float32)
        arrays[f"s{seed}_b{i}"] = m.bias.detach().numpy().astype(np.float32)

np.savez(HERE / "model_weights.npz", **arrays)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                        cwd=HERE).stdout.strip()
meta = {
    "features": features, "input_scaling": "(x - 2) / 2, missing -> 2", "activation": "silu",
    "hidden": HIDDEN, "seeds": args.seeds, "n_layers": len(HIDDEN) + 1,
    "loss": "per-era corr" + (f" + {args.penalty} * corr(pred, {args.benchmark})^2" if args.penalty else ""),
    "target": "target (== target_ender_60)", "steps": args.steps, "eras_per_batch": ERAS_PER_BATCH,
    "lr": LR, "dropout": DROPOUT, "weight_decay": WD,
    "train_eras": {"first": train_eras[0], "last": train_eras[1], "count": train_eras[2], "stride": args.era_stride},
    "git_commit": commit, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
}
(HERE / "model_meta.json").write_text(json.dumps(meta, indent=1))
print(f"saved weights ({sum(a.size for a in arrays.values()):,} params) and meta; {time.time() - t0:.0f}s total")
