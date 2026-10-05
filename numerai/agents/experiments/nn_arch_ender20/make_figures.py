"""Paper figures for nn_arch_ender20 (run from anywhere with the repo venv)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
PAPER_FIGS = HERE / "paper" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)


def save(fig, stem):
    """PNG for the markdown write-up, vector PDF for the LaTeX paper."""
    fig.savefig(OUT / f"{stem}.png", dpi=160)
    fig.savefig(PAPER_FIGS / f"{stem}.pdf")
    plt.close(fig)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def metric(name, key="bmc_last_200_eras", field="mean"):
    m = json.loads((HERE / "results" / f"{name}.json").read_text())["metrics"]
    return m[key][field]


def fig_architectures():
    """Round-1 architecture scout: BMC (last 200 eras), same 4-epoch MSE recipe except where noted."""
    rows = [
        ("LightGBM baseline", "r0_baseline_small_lgbm"),
        ("MLP 256→128", "r1_mlp_mse"),
        ("Wide MLP 1024→512→256", "r1_mlp_wide_mse"),
        ("Residual MLP", "r1_resmlp_mse"),
        ("TabM (BatchEnsemble, k=8)", "r1_tabm_mse"),
        ("MLP + per-bin embeddings", "r1_mlp_embed_mse"),
        ("Final: wide MLP, corr loss", "r6_wide_lr2e3_ep6_seeds3"),
    ]
    rows = sorted(rows, key=lambda r: metric(r[1]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (label, name) in enumerate(rows):
        v = metric(name)
        color = GRAY if "LightGBM" in label else (ORANGE if label.startswith("Final") else BLUE)
        ax.barh(i, v, color=color, height=0.6)
        ax.text(v + 0.0001, i, f"{v:.4f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("BMC, last 200 eras (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(metric(r[1]) for r in rows) * 1.18)
    fig.tight_layout()
    save(fig, "fig1_architectures")


def fig_training_amount():
    """Correlation-loss MLP (256→128): BMC and benchmark correlation vs epochs, two learning rates."""
    series = {
        "lr 1e-3": (BLUE, [(5, "r3_corr_ep5"), (8, "r2_corr_ep8"), (12, "r3_corr_ep12"), (20, "r1_mlp_corr")]),
        "lr 2e-3": (ORANGE, [(5, "r6_lr2e3_ep5"), (6, "r5_lr2e3_ep6"), (8, "r4_lr2e3"), (10, "r5_lr2e3_ep10")]),
    }
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for label, (color, pts) in series.items():
        xs = [p[0] for p in pts]
        a.plot(xs, [metric(n) for _, n in pts], color=color, lw=2, marker="o", ms=6,
               markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        b.plot(xs, [metric(n, "bmc", "avg_corr_with_benchmark") for _, n in pts], color=color, lw=2,
               marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        a.annotate(label, (xs[-1], metric(pts[-1][1])), xytext=(6, 0), textcoords="offset points",
                   va="center", color=INK2, fontsize=9)
    a.set_title("a. BMC, last 200 eras", loc="left", color=INK)
    b.set_title("b. Correlation with benchmark", loc="left", color=INK)
    for ax in (a, b):
        ax.set_xlabel("Training epochs")
        ax.set_xticks([5, 6, 8, 10, 12, 20])
    a.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig2_training_amount")


def fig_tradeoff():
    """All scout runs: correlation with the benchmark vs BMC, coloured by loss."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    groups = {"LightGBM": (GRAY, []), "MSE loss": (BLUE, []), "Correlation loss": (ORANGE, [])}
    for f in sorted((HERE / "results").glob("*.json")):
        if f.stem.startswith("s1"):
            continue
        r = json.loads(f.read_text())
        model = r["model"]
        key = "LightGBM" if model["type"] != "TorchNNRegressor" else (
            "MSE loss" if model["params"].get("loss", "mse") == "mse" else "Correlation loss")
        groups[key][1].append((r["metrics"]["bmc"]["avg_corr_with_benchmark"],
                               r["metrics"]["bmc_last_200_eras"]["mean"]))
    for label, (color, pts) in groups.items():
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=color,
                   edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
    ax.set_xlabel("Correlation with benchmark predictions")
    ax.set_ylabel("BMC, last 200 eras")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig3_tradeoff")


def fig_cumulative():
    """Cumulative per-era BMC on the same 246 scout OOF eras."""
    import sys
    import pandas as pd
    sys.path.insert(0, str(HERE.parents[2]))
    from agents.code.metrics import numerai_metrics as nm

    bench = pd.read_parquet(HERE.parents[2] / "v5.3/downsampled_full_benchmark_models.parquet",
                            columns=["v53_lgbm_ender20"])
    curves = [("LightGBM baseline", "r0_baseline_small_lgbm", GRAY),
              ("MLP, MSE (round 1)", "r1_mlp_mse", BLUE),
              ("Wide MLP, corr loss (final)", "r6_wide_lr2e3_ep6_seeds3", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for label, name, color in curves:
        p = pd.read_parquet(HERE / "predictions" / f"{name}.parquet",
                            columns=["id", "era", "target", "prediction"]).set_index("id").join(bench, how="inner")
        per_era = nm.per_era_bmc(p, ["prediction"], "v53_lgbm_ender20", "target", "era")["prediction"].sort_index()
        eras = [int(e) for e in per_era.index]
        cum = per_era.cumsum().to_numpy()
        ax.plot(eras, cum, color=color, lw=2)
        ax.annotate(label, (eras[-1], cum[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_xlabel("Era")
    ax.set_ylabel("Cumulative BMC")
    ax.set_xlim(right=ax.get_xlim()[1] + 230)
    fig.tight_layout()
    save(fig, "fig4_cumulative_bmc")


def fig_steps():
    """Scout vs scale data: BMC against training amount in optimizer steps (scout-epoch units)."""
    scout = [(5, "r6_wide_lr2e3_ep5"), (6, "r5_wide_lr2e3_ep6")]
    scale = [(6, "s1_wide_ep3_seeds3"), (8, "s1_wide_ep4_seeds3"), (12, "s1_wide_ep6_seeds3")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for label, color, pts in [("Scout data (307 eras)", BLUE, scout), ("Scale data (613 unseen eras)", ORANGE, scale)]:
        xs = [x for x, _ in pts]
        for ax, key, field in [(a, "bmc_last_200_eras", "mean"), (b, "bmc", "avg_corr_with_benchmark")]:
            ax.plot(xs, [metric(n, key, field) for _, n in pts], color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    a.set_title("a. BMC, last 200 eras", loc="left", color=INK)
    b.set_title("b. Correlation with benchmark", loc="left", color=INK)
    for ax in (a, b):
        ax.set_xlabel("Optimizer steps (scout-epoch equivalents)")
        ax.set_xticks([5, 6, 8, 12])
    a.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig5_steps")


if __name__ == "__main__":
    fig_architectures()
    fig_training_amount()
    fig_tradeoff()
    fig_cumulative()
    fig_steps()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
