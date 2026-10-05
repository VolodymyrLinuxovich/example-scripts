"""Write appendix_tables.tex (one table per round, every run) from ../results/*.json."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from build_dashboard import ROUND_NAMES, load_runs  # noqa: E402


def tex(s):
    return s.replace("_", r"\_").replace("→", "--").replace("·", r"$\cdot$")


def main():
    runs = load_runs()
    out = ["Every configuration that ran, by round. Single seeds unless the configuration says "
           "otherwise; bold marks each round's best \\bmc{} over the last 200 eras.\n"]
    for rnd in dict.fromkeys(r["round"] for r in runs):
        rows = sorted((r for r in runs if r["round"] == rnd), key=lambda r: -r["bmc200"])
        best = rows[0]["bmc200"]
        out.append(f"\\subsection*{{{tex(ROUND_NAMES.get(rnd, rnd))}}}")
        out.append("\\begin{center}\\scriptsize\\begin{tabular}{p{3.2cm}p{6.0cm}cccc}\\toprule")
        out.append("Run & Configuration & corr & \\bmc{} & last 200 & w/ bench. \\\\\\midrule")
        for r in rows:
            b200 = f"{r['bmc200']:.4f}"
            if r["bmc200"] == best:
                b200 = f"\\textbf{{{b200}}}"
            out.append(f"\\path{{{r['name']}}} & {tex(r['desc'])} & {r['corr']:.4f} & "
                       f"{r['bmc']:.4f} & {b200} & {r['benchCorr']:.3f} \\\\")
        out.append("\\bottomrule\\end{tabular}\\end{center}\n")
    (HERE / "appendix_tables.tex").write_text("\n".join(out))
    print(f"appendix_tables.tex: {len(runs)} runs")


if __name__ == "__main__":
    main()
