import json, sys
from pathlib import Path
rows = []
for f in sorted((Path(__file__).parent / "results").glob("*.json")):
    if len(sys.argv) > 1 and not any(f.stem.startswith(p) for p in sys.argv[1:]):
        continue
    m = json.loads(f.read_text())["metrics"]
    rows.append((f.stem, m["corr"]["mean"], m["corr"]["sharpe"], m["bmc"]["mean"],
                 m["bmc_last_200_eras"]["mean"], m["bmc"]["sharpe"], m["bmc"]["avg_corr_with_benchmark"]))
rows.sort(key=lambda r: -r[4])
print(f"| {'model':40s} | corr | corr_sh | bmc | bmc_l200 | bmc_sh | corr_w_bench |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]:40s} | {r[1]:.4f} | {r[2]:.2f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.2f} | {r[6]:.3f} |")
