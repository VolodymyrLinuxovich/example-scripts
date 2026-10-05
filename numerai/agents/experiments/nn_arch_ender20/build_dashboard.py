"""Build the research-tracking dashboard (dashboard.html) from results/ and predictions/.

Run from the repo root:  PYTHONPATH=numerai .venv/bin/python numerai/agents/experiments/nn_arch_ender20/build_dashboard.py
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
NUMERAI = HERE.parents[2]
sys.path.insert(0, str(NUMERAI))
from agents.code.metrics import numerai_metrics as nm  # noqa: E402

ROUND_NAMES = {
    "r0": "Baselines", "r1": "Round 1 · architecture", "r2": "Round 2 · training length",
    "r3": "Round 3 · refine", "r4": "Round 4 · new knobs", "r5": "Round 5 · lr 2e-3",
    "r6": "Round 6 · seed check", "s1": "Scale · unseen eras",
}
# Models whose cumulative BMC is drawn (scout data, same eras for all).
CURVES = [
    ("LightGBM baseline", "r0_baseline_small_lgbm", "base"),
    ("MLP, MSE (round 1)", "r1_mlp_mse", "s1"),
    ("Wide MLP, corr loss (best)", "r6_wide_lr2e3_ep6_seeds3", "s2"),
]


def describe(name, model):
    if model["type"] != "TorchNNRegressor":
        return "LightGBM, 2,000 trees"
    p = model["params"]
    hidden = p.get("hidden", [256, 128])
    arch = {"mlp": "MLP", "resmlp": "Residual MLP", "tabm": "TabM"}[p.get("arch", "mlp")]
    width = "→".join(map(str, hidden)) if isinstance(hidden, list) else f"w{hidden}"
    parts = [f"{arch} {width}", "corr loss" if p.get("loss", "mse") != "mse" else "MSE",
             f"{p.get('epochs')} ep", f"lr {p.get('lr', 1e-3):g}"]
    if p.get("loss") == "corr_mse":
        parts[1] = "corr+MSE"
    if p.get("input_encoding") == "embed":
        parts.append("embeddings")
    if p.get("dropout", 0.1) != 0.1:
        parts.append(f"drop {p['dropout']}")
    if p.get("weight_decay", 1e-5) != 1e-5:
        parts.append(f"wd {p['weight_decay']:g}")
    if p.get("input_dropout", 0):
        parts.append(f"in-drop {p['input_dropout']}")
    if p.get("eras_per_batch", 4) != 4 and p.get("batching") == "era":
        parts.append(f"{p['eras_per_batch']} era/batch")
    if p.get("n_seeds", 1) > 1:
        parts.append(f"{p['n_seeds']} seeds")
    return " · ".join(parts)


def load_runs():
    runs = []
    for f in sorted((HERE / "results").glob("*.json")):
        r = json.loads(f.read_text())
        m, model = r["metrics"], r["model"]
        rnd = f.stem.split("_")[0]
        loss = "lgbm" if model["type"] != "TorchNNRegressor" else (
            "mse" if model["params"].get("loss", "mse") == "mse" else "corr")
        runs.append({
            "name": f.stem, "round": rnd, "roundName": ROUND_NAMES.get(rnd, rnd),
            "desc": describe(f.stem, model), "loss": loss,
            "seeds": model["params"].get("n_seeds", 1) if model["type"] == "TorchNNRegressor" else 1,
            "data": "scale" if rnd == "s1" else "scout",
            "corr": m["corr"]["mean"], "corrSharpe": m["corr"]["sharpe"],
            "bmc": m["bmc"]["mean"], "bmc200": m["bmc_last_200_eras"]["mean"],
            "bmcSharpe": m["bmc"]["sharpe"], "benchCorr": m["bmc"]["avg_corr_with_benchmark"],
            "maxDD": m["bmc"]["max_drawdown"],
        })
    return runs


def cumulative_curves():
    bench = pd.read_parquet(NUMERAI / "v5.3/downsampled_full_benchmark_models.parquet",
                            columns=["v53_lgbm_ender20"])
    out = []
    for label, name, slot in CURVES:
        path = HERE / "predictions" / f"{name}.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path, columns=["id", "era", "target", "prediction"]).set_index("id")
        p = p.join(bench, how="inner")
        per_era = nm.per_era_bmc(p, ["prediction"], "v53_lgbm_ender20", "target", "era")["prediction"]
        per_era = per_era.sort_index()
        out.append({"label": label, "slot": slot, "eras": [int(e) for e in per_era.index],
                    "cum": per_era.cumsum().round(5).tolist()})
    return out


def literature():
    groups = []
    for f in sorted((HERE / "literature").glob("*.md")):
        text = f.read_text()
        title = text.splitlines()[0].lstrip("# ").replace("Literature notes: ", "")
        items = []
        for line in text.splitlines():
            if not line.startswith("| ") or line.startswith("| paper") or line.startswith("| source") or line.startswith("|---"):
                continue
            cell = line.split("|")[1].strip()
            url = re.search(r"https?://\S+", cell)
            label = re.sub(r",?\s*https?://\S+", "", cell).strip().rstrip(",")
            items.append({"label": label, "url": url.group(0) if url else None,
                          "note": line.split("|")[-2].strip()})
        ideas = re.findall(r"^\d+\. (.+)$", text.split("## Suggested experiments")[-1], re.M)
        groups.append({"title": title, "items": items, "ideas": ideas})
    return groups


def main():
    runs = load_runs()
    data = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "runs": runs, "curves": cumulative_curves(), "literature": literature(),
    }
    done = {r["name"] for r in runs}
    data["running"] = [c.stem for c in sorted((HERE / "configs").glob("s*.py"))
                       if not c.stem.startswith("_") and c.stem not in done]
    template = (HERE / "dashboard_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (HERE / "dashboard.html").write_text(html)
    print(f"dashboard.html: {len(runs)} runs, {sum(len(g['items']) for g in data['literature'])} sources, "
          f"running={data['running']}")


if __name__ == "__main__":
    main()
