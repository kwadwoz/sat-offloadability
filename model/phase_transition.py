#!/usr/bin/env python3
"""E8: is propagations-per-decision distinct from the clause/variable ratio?

The clause-to-variable ratio m/n is the classic structural predictor of SAT
hardness (the phase transition: Cheeseman, Kanefsky & Taylor 1991; Mitchell,
Selman & Levesque 1992). This checks whether our offloadability signal,
propagations per decision (ppd), is merely a proxy for m/n or carries independent
structural information. Pure offline analysis: no solving, no hardware.

Variables n and clauses m come from the `p cnf n m` header, already recorded per
instance in props.csv by E4.

Usage:
    python model/phase_transition.py --props instrument/props.csv \
        --figures figures/ --out model/phase_transition.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def regime(rho: float) -> str:
    a = abs(rho)
    if a >= 0.9:
        return "proxy (ppd essentially tracks the ratio)"
    if a >= 0.4:
        return "related but distinct (ppd carries information beyond the ratio)"
    return "independent (ppd is largely unrelated to the ratio)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--props", required=True, type=Path)
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/phase_transition.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.props)
    df = df[(df.status == "ok") & (df.variables > 0)].copy()
    df["ratio"] = df["clauses"] / df["variables"]
    ppd = "props_per_decision"

    # overall correlation
    rho, rho_p = spearmanr(df["ratio"], df[ppd])
    r, r_p = pearsonr(df["ratio"], df[ppd])
    print(f"OVERALL (n={len(df)} instances)")
    print(f"  Spearman rho = {rho:+.3f}  (p={rho_p:.1e})")
    print(f"  Pearson  r   = {r:+.3f}  (p={r_p:.1e})")
    print(f"  regime: {regime(rho)}\n")

    # per-family: ratio is near-constant within a family, so any ppd spread there
    # is variation that the ratio cannot explain.
    rows = []
    for fam, g in df.groupby("family"):
        # within-family rank correlation only meaningful if ratio varies at all
        if g["ratio"].nunique() > 1:
            wrho, _ = spearmanr(g["ratio"], g[ppd])
        else:
            wrho = np.nan
        rows.append({
            "family": fam, "n": len(g),
            "mean_ratio": round(g["ratio"].mean(), 3),
            "ratio_spread": round(g["ratio"].std(), 3),
            "mean_ppd": round(g[ppd].mean(), 2),
            "ppd_std": round(g[ppd].std(), 2),
            "within_rho": round(wrho, 3) if wrho == wrho else "",
        })
    table = pd.DataFrame(rows).sort_values("mean_ratio")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))

    # the headline: families at near-identical ratio still spread widely in ppd
    print(f"\nWithin a fixed family the clause/variable ratio is essentially "
          f"constant (spread <= {table['ratio_spread'].max():.3f}), yet ppd still "
          f"varies (std up to {table['ppd_std'].max():.1f} propagations/decision), "
          f"so ppd is not determined by the ratio.")

    # figure
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fams = sorted(df["family"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(fams)))
    for fam, c in zip(fams, cmap):
        g = df[df.family == fam]
        ax.scatter(g["ratio"], g[ppd], s=8, alpha=0.35, color=c, label=fam,
                   edgecolors="none")
    ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:+.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax.set_xlabel("Clause-to-variable ratio  (m / n)")
    ax.set_ylabel("Propagations per decision")
    ax.set_title("Propagation work vs. the phase-transition ratio")
    leg = ax.legend(title="Family", fontsize=9, title_fontsize=9.5,
                    markerscale=2, loc="upper right")
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figures / "e8_ppd_vs_ratio.png")
    plt.close(fig)
    print(f"\nwrote {args.out} and e8_ppd_vs_ratio.png")


if __name__ == "__main__":
    main()
