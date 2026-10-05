"""Paper figures for nn_arch_ender20 (run from anywhere with the repo venv)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

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
    ax.set_title("Architecture comparison (downsampled scout data)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_architectures.png", dpi=160)


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
    fig.savefig(OUT / "fig2_training_amount.png", dpi=160)


if __name__ == "__main__":
    fig_architectures()
    fig_training_amount()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
