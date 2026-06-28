#!/usr/bin/env python3
"""Cross-solver robustness figure: per-decision work reproduces across solvers.

Combines the cross-solver runs (instrument/cross_solver*.csv) and plots median
propagations per decision per family for MiniSat, CaDiCaL, and Kissat, over the
families where all three solvers branch (modern solvers preprocess the easiest
families away). The point: the structural rise in per-decision work with instance
scale is not an artifact of any one solver.

Usage:
    python instrument/plot_cross_solver.py --out figures/cross_solver.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

SOLVERS = ["minisat", "cadical", "kissat"]
LABELS = {"minisat": "MiniSat", "cadical": "CaDiCaL", "kissat": "Kissat"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csvs", nargs="+", type=Path,
                    default=[Path("instrument/cross_solver.csv"),
                             Path("instrument/cross_solver_big.csv")])
    ap.add_argument("--out", type=Path, default=Path("figures/cross_solver.png"))
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs if c.exists()], ignore_index=True)
    ok = df[df.status == "ok"]

    # keep families where all three solvers produced usable ppd
    counts = ok.groupby(["family", "solver"]).size().unstack()
    full = counts.dropna(subset=SOLVERS).index.tolist()
    full = sorted(full, key=lambda f: int("".join(ch for ch in f.split("-")[0] if ch.isdigit())))

    med = ok[ok.family.isin(full)].groupby(["family", "solver"])["props_per_decision"].median().unstack()
    med = med.reindex(full)

    x = np.arange(len(full))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, s in enumerate(SOLVERS):
        ax.bar(x + (i - 1) * width, med[s].values, width, label=LABELS[s])
    ax.set_xticks(x)
    ax.set_xticklabels(full, rotation=20, ha="right")
    ax.set_ylabel("Median propagations per decision")
    ax.set_xlabel("Benchmark family (increasing size)")
    ax.set_title("Per-decision work rises with scale across three independent solvers")
    ax.legend(title="Solver")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("families with full cross-solver coverage:", full)
    print(med.round(1).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
