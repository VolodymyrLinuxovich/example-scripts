from __future__ import annotations

import math

import numpy as np


class TorchNNRegressor:
    """PyTorch neural-net regressor for Numerai int8 features.

    Params:
      arch: "mlp" | "resmlp" | "tabm" (BatchEnsemble MLP with k members)
      input_encoding: "scalar" ((x-2)/2) | "embed" (learned per-bin embedding of size embed_dim)
      hidden: list[int] for mlp/tabm, or width for resmlp (with n_blocks)
      loss: "mse" | "corr" (per-era Pearson) | "corr_mse" (corr + mse_weight * mse)
      batching: "random" (row batches) | "era" (each batch is eras_per_batch whole eras)
    Training runs a fixed number of epochs (no early stopping, no validation leakage).
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        *,
        arch: str = "mlp",
        hidden: list[int] | int = (256, 128),
        n_blocks: int = 2,
        k: int = 8,
        input_encoding: str = "scalar",
        embed_dim: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
        loss: str = "mse",
        mse_weight: float = 0.1,
        batching: str = "random",
        batch_size: int = 4096,
        eras_per_batch: int = 1,
        epochs: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_schedule: str = "cosine",
        warmup_frac: float = 0.05,
        input_dropout: float = 0.0,
        n_seeds: int = 1,
        random_state: int = 1337,
        device: str | None = None,
        era_col: str = "era",
        verbose: bool = True,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for TorchNNRegressor. Install with `.venv/bin/pip install torch`."
            ) from exc
        self._torch = torch
        self.feature_cols = feature_cols
        self.params = dict(
            arch=arch,
            hidden=list(hidden) if isinstance(hidden, (list, tuple)) else int(hidden),
            n_blocks=n_blocks,
            k=k,
            input_encoding=input_encoding,
            embed_dim=embed_dim,
            dropout=dropout,
            activation=activation,
            loss=loss,
            mse_weight=mse_weight,
            batching=batching,
            batch_size=batch_size,
            eras_per_batch=eras_per_batch,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            warmup_frac=warmup_frac,
            input_dropout=input_dropout,
        )
        self.n_seeds = int(n_seeds)
        self.random_state = int(random_state)
        self.era_col = era_col
        self.verbose = verbose
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.models_: list = []

    # ------------------------------------------------------------------ data
    def _features(self, X) -> np.ndarray:
        if self.feature_cols and hasattr(X, "columns"):
            X = X[self.feature_cols]
        if hasattr(X, "dtypes") and all(str(dt) == "int8" for dt in X.dtypes):
            return X.to_numpy(dtype=np.int8)
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=2.0)
        return np.clip(np.rint(arr), 0, 4).astype(np.int8)

    # ----------------------------------------------------------------- model
    def _build(self, n_features: int):
        torch = self._torch
        nn = torch.nn
        p = self.params
        act = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[p["activation"]]

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = p["input_encoding"]
                if self.mode == "embed":
                    # One embedding table per feature: (n_features, 5, d)
                    d = p["embed_dim"]
                    self.emb = nn.Parameter(torch.randn(n_features, 5, d) * 0.1)
                    self.out_dim = n_features * d
                else:
                    self.out_dim = n_features
                self.in_drop = nn.Dropout(p["input_dropout"])

            def forward(self, x_int):
                if self.mode == "embed":
                    idx = x_int.long()  # (B, F)
                    f = torch.arange(idx.shape[1], device=idx.device)
                    out = self.emb[f.unsqueeze(0), idx]  # (B, F, d)
                    out = out.reshape(idx.shape[0], -1)
                else:
                    out = (x_int.float() - 2.0) / 2.0
                return self.in_drop(out)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Encoder()
                layers, d_in = [], self.enc.out_dim
                for h in p["hidden"]:
                    layers += [nn.Linear(d_in, h), act(), nn.Dropout(p["dropout"])]
                    d_in = h
                layers.append(nn.Linear(d_in, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(self.enc(x)).squeeze(-1)

        class ResBlock(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.ff = nn.Sequential(
                    nn.Linear(d, 2 * d), act(), nn.Dropout(p["dropout"]), nn.Linear(2 * d, d)
                )

            def forward(self, x):
                return x + self.ff(self.norm(x))

        class ResMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Encoder()
                d = int(p["hidden"] if isinstance(p["hidden"], int) else p["hidden"][0])
                self.inp = nn.Linear(self.enc.out_dim, d)
                self.blocks = nn.Sequential(*[ResBlock(d) for _ in range(p["n_blocks"])])
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

            def forward(self, x):
                return self.head(self.blocks(self.inp(self.enc(x)))).squeeze(-1)

        class BELinear(nn.Module):
            """BatchEnsemble linear: shared W with per-member rank-1 scaling (TabM)."""

            def __init__(self, d_in, d_out, k):
                super().__init__()
                self.lin = nn.Linear(d_in, d_out, bias=False)
                self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
                self.s = nn.Parameter(torch.ones(k, d_out))
                self.b = nn.Parameter(torch.zeros(k, d_out))

            def forward(self, x):  # x: (B, k, d_in)
                return self.lin(x * self.r) * self.s + self.b

        class TabM(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Encoder()
                self.k = p["k"]
                layers, d_in = [], self.enc.out_dim
                for h in p["hidden"]:
                    layers += [BELinear(d_in, h, self.k), act(), nn.Dropout(p["dropout"])]
                    d_in = h
                self.body = nn.Sequential(*layers)
                self.head_w = nn.Parameter(torch.randn(self.k, d_in) / math.sqrt(d_in))
                self.head_b = nn.Parameter(torch.zeros(self.k))

            def forward(self, x):
                z = self.enc(x).unsqueeze(1).expand(-1, self.k, -1)
                z = self.body(z)  # (B, k, d)
                out = (z * self.head_w).sum(-1) + self.head_b  # (B, k)
                return out  # members kept separate for training; averaged at predict

        arch = p["arch"]
        if arch == "mlp":
            return MLP()
        if arch == "resmlp":
            return ResMLP()
        if arch == "tabm":
            return TabM()
        raise ValueError(f"Unknown arch '{arch}'")

    # ------------------------------------------------------------------ loss
    def _loss(self, pred, y, groups):
        """pred: (B,) or (B, k); y: (B,); groups: list of (start, end) era slices."""
        torch = self._torch
        p = self.params
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        yk = y.unsqueeze(-1).expand_as(pred)
        mse = ((pred - yk) ** 2).mean()
        if p["loss"] == "mse":
            return mse
        corrs = []
        for s, e in groups:
            pp = pred[s:e] - pred[s:e].mean(0, keepdim=True)
            yy = yk[s:e] - yk[s:e].mean(0, keepdim=True)
            c = (pp * yy).sum(0) / (pp.norm(dim=0) * yy.norm(dim=0) + 1e-8)
            corrs.append(c.mean())
        corr_loss = -torch.stack(corrs).mean()
        if p["loss"] == "corr":
            return corr_loss
        if p["loss"] == "corr_mse":
            return corr_loss + p["mse_weight"] * mse * 100.0
        raise ValueError(f"Unknown loss '{p['loss']}'")

    # ------------------------------------------------------------------- fit
    def fit(self, X, y, **kwargs):
        torch = self._torch
        p = self.params
        if p["loss"] != "mse" and p["batching"] != "era":
            raise ValueError("Correlation losses require batching='era'.")
        eras = X[self.era_col].to_numpy() if self.era_col in getattr(X, "columns", []) else None
        x_np = self._features(X)
        y_np = np.asarray(y, dtype=np.float32) - 0.5
        y_np = np.nan_to_num(y_np, nan=0.0)

        # sort by era so era batches are contiguous slices
        if eras is not None:
            era_codes, _ = pd_factorize_sorted(eras)
            order = np.argsort(era_codes, kind="stable")
            x_np, y_np, era_codes = x_np[order], y_np[order], era_codes[order]
            bounds = np.flatnonzero(np.diff(era_codes)) + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [len(era_codes)]])
        else:
            if p["batching"] == "era":
                raise ValueError("batching='era' requires the era column in X.")
            starts = ends = None

        dev = self.device
        x_t = torch.from_numpy(x_np).to(dev)
        y_t = torch.from_numpy(y_np).to(dev)
        n = len(y_np)

        self.models_ = []
        for seed_i in range(self.n_seeds):
            seed = self.random_state + seed_i
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = self._build(x_np.shape[1]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

            if p["batching"] == "era":
                n_era = len(starts)
                steps_per_epoch = math.ceil(n_era / p["eras_per_batch"])
            else:
                steps_per_epoch = math.ceil(n / p["batch_size"])
            total = steps_per_epoch * p["epochs"]
            warm = max(1, int(total * p["warmup_frac"]))

            def lr_at(step):
                if step < warm:
                    return (step + 1) / warm
                if p["lr_schedule"] == "cosine":
                    t = (step - warm) / max(1, total - warm)
                    return 0.5 * (1 + math.cos(math.pi * t))
                return 1.0

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
            model.train()
            step = 0
            for ep in range(p["epochs"]):
                ep_loss = 0.0
                if p["batching"] == "era":
                    era_perm = rng.permutation(len(starts))
                    for b in range(0, len(era_perm), p["eras_per_batch"]):
                        sel = era_perm[b : b + p["eras_per_batch"]]
                        idx_parts, groups, off = [], [], 0
                        for ei in sel:
                            s, e = int(starts[ei]), int(ends[ei])
                            idx_parts.append(torch.arange(s, e, device=dev))
                            groups.append((off, off + e - s))
                            off += e - s
                        idx = torch.cat(idx_parts)
                        loss = self._loss(model(x_t[idx]), y_t[idx], groups)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        sched.step()
                        ep_loss += float(loss.detach())
                        step += 1
                else:
                    perm = torch.from_numpy(rng.permutation(n)).to(dev)
                    for b in range(0, n, p["batch_size"]):
                        idx = perm[b : b + p["batch_size"]]
                        loss = self._loss(model(x_t[idx]), y_t[idx], [(0, len(idx))])
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        sched.step()
                        ep_loss += float(loss.detach())
                        step += 1
                if self.verbose:
                    print(
                        f"  [seed {seed}] epoch {ep + 1}/{p['epochs']} loss={ep_loss / steps_per_epoch:.6f}",
                        flush=True,
                    )
            model.eval()
            self.models_.append(model.cpu())
        del x_t, y_t
        if dev.type == "mps":
            torch.mps.empty_cache()
        return self

    # --------------------------------------------------------------- predict
    def predict(self, X, batch_size: int = 32768):
        torch = self._torch
        x_np = self._features(X)
        out = np.zeros(len(x_np), dtype=np.float64)
        dev = self.device
        for model in self.models_:
            model.to(dev).eval()
            preds = []
            with torch.no_grad():
                for b in range(0, len(x_np), batch_size):
                    xb = torch.from_numpy(x_np[b : b + batch_size]).to(dev)
                    pb = model(xb)
                    if pb.dim() == 2:
                        pb = pb.mean(-1)
                    preds.append(pb.float().cpu().numpy())
            model.cpu()
            out += np.concatenate(preds)
        return out / max(1, len(self.models_)) + 0.5


def pd_factorize_sorted(values):
    import pandas as pd

    return pd.factorize(pd.Series(values), sort=True)
