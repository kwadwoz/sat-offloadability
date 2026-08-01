#!/usr/bin/env python3
"""E10: is propagations-per-decision just a stand-in for formula size?

E8 showed ppd is not a proxy for the clause/variable ratio. The other obvious
structural explanation is sheer size: a bigger formula has more clauses to
propagate through, so maybe ppd only measures n (or m) in disguise. Three
checks, from weakest to strongest:

  1. Global correlation of ppd with n and with m across all instances.
  2. The uniform-random ladder (uf20..uf250) holds m/n fixed at ~4.26 and varies
     only n, so it isolates the size effect. If ppd is size-driven it must rise
     monotonically along that ladder and account for most of the variance.
  3. Within a single family n and m are *constant*, so any remaining ppd spread
     is variation size cannot explain. Reported as the fraction of total ppd
     variance left after removing family means, and as a size-controlled partial
     Spearman correlation between ppd and the ratio.

Pure offline analysis: no solving, no hardware.

Usage:
    python model/size_effect.py --props instrument/props.csv \
        --figures figures/ --out model/size_effect.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
        return "proxy (ppd essentially tracks size)"
    if a >= 0.4:
        return "related but distinct (ppd carries information beyond size)"
    return "independent (ppd is largely unrelated to size)"


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing what z linearly explains
    of each (on ranks)."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy(float) for v in (x, y, z))
    def resid(a):
        A = np.column_stack([np.ones_like(rz), rz])
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return spearmanr(resid(rx), resid(ry))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--props", required=True, type=Path)
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/size_effect.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.props)
    df = df[(df.status == "ok") & (df.variables > 0)].copy()
    df["ratio"] = df["clauses"] / df["variables"]
    ppd = "props_per_decision"

    # ---- 1. global correlations -------------------------------------------
    rho_n, p_n = spearmanr(df["variables"], df[ppd])
    rho_m, p_m = spearmanr(df["clauses"], df[ppd])
    print(f"OVERALL (n={len(df)} instances)")
    print(f"  ppd vs variables n: Spearman rho = {rho_n:+.3f}  (p={p_n:.1e})")
    print(f"  ppd vs clauses   m: Spearman rho = {rho_m:+.3f}  (p={p_m:.1e})")
    print(f"  regime (vs n): {regime(rho_n)}\n")

    # ---- 2. the fixed-ratio size ladder ------------------------------------
    lad = df[df.family.str.match(r"^uf\d+-\d+$")].copy()
    print(f"FIXED-RATIO LADDER (uf*, m/n = {lad['ratio'].mean():.2f} "
          f"+/- {lad['ratio'].std():.2f}, {len(lad)} instances)")
    lad_rho, lad_p = spearmanr(lad["variables"], lad[ppd])
    print(f"  ppd vs n along the ladder: Spearman rho = {lad_rho:+.3f} (p={lad_p:.1e})")
    # how much of ppd does size explain here, at best? R^2 of ppd on log n.
    lx = np.log(lad["variables"].to_numpy(float))
    A = np.column_stack([np.ones_like(lx), lx])
    yv = lad[ppd].to_numpy(float)
    fit = A @ np.linalg.lstsq(A, yv, rcond=None)[0]
    r2 = 1 - ((yv - fit) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    print(f"  R^2 of ppd on log n:      {r2:.3f}  "
          f"(so {(1 - r2) * 100:.0f}% of ppd variance is not size)\n")

    # ---- 3. within-family (size exactly constant) --------------------------
    rows = []
    for fam, g in df.groupby("family"):
        rows.append({
            "family": fam, "n_instances": len(g),
            "variables": int(g["variables"].median()),
            "clauses": int(g["clauses"].median()),
            "size_spread_n": round(g["variables"].std(), 3),
            "mean_ppd": round(g[ppd].mean(), 2),
            "ppd_std": round(g[ppd].std(), 2),
            "ppd_min": round(g[ppd].min(), 2),
            "ppd_max": round(g[ppd].max(), 2),
        })
    table = pd.DataFrame(rows).sort_values("variables")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))

    within = df.groupby("family")[ppd].transform("mean")
    frac = ((df[ppd] - within) ** 2).sum() / ((df[ppd] - df[ppd].mean()) ** 2).sum()
    print(f"\nWithin a family n and m are fixed exactly (spread "
          f"{table['size_spread_n'].max():.3f}), yet ppd still ranges up to "
          f"{(table['ppd_max'] - table['ppd_min']).max():.1f} propagations/decision. "
          f"{frac * 100:.0f}% of all ppd variance survives removing family means, "
          f"i.e. is not attributable to formula size.")

    # families that sit at (near-)identical size but differ in ppd are the
    # cleanest counterexample to "ppd is size".
    print("\nSAME SIZE, DIFFERENT ppd:")
    for lo, hi in [(85, 105), (145, 155)]:
        band = table[(table.variables >= lo) & (table.variables <= hi)]
        if len(band) > 1:
            hi_r = band.loc[band.mean_ppd.idxmax()]
            lo_r = band.loc[band.mean_ppd.idxmin()]
            print(f"  n ~ {lo}-{hi}: {hi_r.family} ppd {hi_r.mean_ppd:.1f} vs "
                  f"{lo_r.family} ppd {lo_r.mean_ppd:.1f} "
                  f"({hi_r.mean_ppd / lo_r.mean_ppd:.1f}x at the same size)")

    pr_rho, pr_p = partial_spearman(df["ratio"], df[ppd], df["variables"])
    print(f"\nppd vs clause/variable ratio, controlling for n: partial Spearman "
          f"rho = {pr_rho:+.3f} (p={pr_p:.1e})")

    # ---- figure -------------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.3))
    fams = sorted(df["family"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(fams)))
    for fam, c in zip(fams, cmap):
        g = df[df.family == fam]
        ax1.scatter(g["variables"], g[ppd], s=8, alpha=0.35, color=c, label=fam,
                    edgecolors="none")
    ax1.set_xscale("log")
    ax1.text(0.03, 0.97, f"Spearman $\\rho$ = {rho_n:+.2f}", transform=ax1.transAxes,
             va="top", ha="left", fontsize=12,
             bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax1.set_xlabel("Variables  n  (log scale)")
    ax1.set_ylabel("Propagations per decision")
    ax1.set_title("Propagation work vs. formula size")
    leg = ax1.legend(title="Family", fontsize=8.5, title_fontsize=9, markerscale=2,
                     loc="upper left", bbox_to_anchor=(0.0, 0.93))
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax1.grid(True, alpha=0.3)

    order = sorted(lad["variables"].unique())
    ax2.boxplot([lad[lad.variables == v][ppd] for v in order],
                labels=[str(v) for v in order], showfliers=False)
    ax2.set_xlabel(f"Variables  n   (uniform-random, m/n = "
                   f"{lad['ratio'].mean():.2f} $\\pm$ {lad['ratio'].std():.2f})")
    ax2.set_ylabel("Propagations per decision")
    ax2.set_title("Size ladder at constant ratio")
    ax2.text(0.03, 0.97, f"$R^2$ on $\\log n$ = {r2:.2f}", transform=ax2.transAxes,
             va="top", ha="left", fontsize=12,
             bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.figures / "e10_ppd_vs_size.png")
    plt.close(fig)
    print(f"\nwrote {args.out} and e10_ppd_vs_size.png")


if __name__ == "__main__":
    main()
