#!/usr/bin/env python3
"""A1 offload map: per family, how much MiniSat solve time per-call BCP offload
could save at each accelerator round-trip time (upper bound, step 2 data).

Cell value = median over a family's instances of
    sum over search-phase BCP calls of max(0, t - L)  /  total traced wall time
i.e. an infinitely fast accelerator with no state-synchronization cost.

    python model/a1_offload_map.py   ->  figures/a1_offload_map.{png,pdf}
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ACM rejects Type 3 fonts; matplotlib's PDF default is Type 3. 42 = TrueType.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/a1-step2"
OUT = ROOT / "figures/a1_offload_map.png"

COLUMNS = [100, 250, 500, 1_000, 2_000, 5_000, 10_000, 100_000, 790_000,
           10_000_000, 100_000_000, 2_000_000_000]
# Savings classes: neutral gray below 1% (not worth it), then one blue hue, light -> dark.
EDGES = [0, 0.01, 0.10, 0.25, 0.50, 1.0001]
COLORS = ["#f0efec", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
LABELS = ["< 1%  don't offload", "1–10%  marginal", "10–25%", "25–50%", "≥ 50%"]
OFFLOAD_AT = 0.10                     # outline encloses cells at or above this

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def fmt_latency(ns):
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= scale:
            v = ns / scale
            return f"{v:g} {unit}"
    return f"{ns} ns"


def load():
    s = pd.read_csv(RES / "summary.csv")
    g = pd.read_csv(RES / "bound_grid.csv")
    g = g[(g.granularity == "call") & (g.speedup.astype(float) == np.inf) & g.latency_ns.isin(COLUMNS)]
    g = g.merge(s[["hash", "family", "wall_traced_ns"]], on="hash")
    g["frac"] = g.savings_ns / g.wall_traced_ns
    # Instances solved before search have no rows: they save nothing, so count them as 0, not missing.
    per = g.pivot_table(index="hash", columns="latency_ns", values="frac").reindex(s.hash).fillna(0.0)
    grid = per.groupby(s.set_index("hash").family).median()[COLUMNS]
    call_us = (s.search_bcp_kernel_ns / s.calls_search / 1e3).groupby(s.family).median()   # instances with search calls
    search = (s.search_ns / s.wall_traced_ns).groupby(s.family).median()
    order = grid[5_000].sort_values(ascending=False).index      # most benefit at PCIe-class latency first
    return grid.loc[order], call_us.loc[order], search.loc[order], s.groupby("family").size().loc[order]


def main():
    grid, call_us, search, n = load()
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor="white")
    ax.set_facecolor(SURFACE)
    cmap, norm = ListedColormap(COLORS), BoundaryNorm(EDGES, len(COLORS))
    ax.pcolormesh(np.arange(cols + 1), np.arange(rows + 1), grid.values, cmap=cmap, norm=norm,
                  edgecolors="white", linewidth=1.5)

    # Value in every cell: identity never depends on color alone.
    for i in range(rows):
        for j in range(cols):
            v = grid.values[i, j]
            text = "<1" if v < 0.01 else f"{100 * v:.0f}"
            dark = v >= 0.10
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=6.6,
                    color="white" if dark else (MUTED if v < 0.01 else INK))

    # Outline the "offload" region (>= 10% saved) cell by cell.
    on = grid.values >= OFFLOAD_AT
    for i in range(rows):
        for j in range(cols):
            if not on[i, j]:
                continue
            for di, dj, seg in ((-1, 0, ([j, j + 1], [i, i])), (1, 0, ([j, j + 1], [i + 1, i + 1])),
                                (0, -1, ([j, j], [i, i + 1])), (0, 1, ([j + 1, j + 1], [i, i + 1]))):
                ii, jj = i + di, j + dj
                if not (0 <= ii < rows and 0 <= jj < cols and on[ii, jj]):
                    ax.plot(*seg, color=INK, linewidth=1.4, solid_capstyle="round", zorder=3)

    ax.set_xlim(0, cols); ax.set_ylim(rows, 0)
    ax.set_xticks(np.arange(cols) + 0.5, [fmt_latency(c) for c in COLUMNS], fontsize=7, color=INK2, rotation=40,
                  ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(rows) + 0.5, grid.index, fontsize=7.5, color=INK)
    ax.tick_params(length=0, pad=3)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Accelerator round-trip time per propagation call (columns not to scale)", fontsize=8, color=INK2, labelpad=6)

    # Right-hand columns: why a family can or cannot benefit.
    for i, fam in enumerate(grid.index):
        ax.text(cols + 0.25, i + 0.5, f"{call_us[fam]:.1f} µs", ha="left", va="center", fontsize=6.8, color=INK2)
        ax.text(cols + 2.05, i + 0.5, f"{100 * search[fam]:.0f}%", ha="right", va="center", fontsize=6.8, color=INK2)
    ax.text(cols + 0.25, -0.35, "median\ncall", ha="left", va="bottom", fontsize=6.8, color=MUTED, linespacing=1.0)
    ax.text(cols + 2.05, -0.35, "time in\nsearch", ha="right", va="bottom", fontsize=6.8, color=MUTED, linespacing=1.0)

    # Interconnect reference brackets above the grid.
    def bracket(j0, j1, label):
        y = -0.35
        ax.plot([j0 + 0.12, j1 + 0.88], [y, y], color=MUTED, linewidth=1.0, clip_on=False)
        for x in (j0 + 0.12, j1 + 0.88):
            ax.plot([x, x], [y, y + 0.18], color=MUTED, linewidth=1.0, clip_on=False)
        ax.text((j0 + j1 + 1) / 2, y - 0.12, label, ha="center", va="bottom", fontsize=6.8, color=INK2, clip_on=False)
    col = {c: k for k, c in enumerate(COLUMNS)}
    bracket(col[100], col[500], "on-chip / coherent")
    bracket(col[1_000], col[5_000], "PCIe-class")
    bracket(col[790_000], col[790_000], "ECP5 Ethernet\n(measured)")
    bracket(col[2_000_000_000], col[2_000_000_000], "mrg cloud\nload")

    fig.suptitle("When does offloading a propagation call pay off?", x=0.02, y=1.0, ha="left",
                 fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.945,
             "Best-case % of total MiniSat solve time saved by sending every propagation call to an accelerator.\n"
             f"Infinitely fast device, no synchronization cost (upper bound). Median of {int(n.median())} instances per family; "
             "outlined: ≥ 10% saved.",
             ha="left", va="top", fontsize=7.2, color=INK2, linespacing=1.35)

    handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(COLORS, LABELS)]
    handles.append(Line2D([], [], color=INK, linewidth=1.4, label="offload (≥ 10%)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.045),
               handlelength=1.2, handleheight=1.0, columnspacing=1.4, title="Time saved", title_fontsize=7)
    fig.subplots_adjust(left=0.2, right=0.84, top=0.79, bottom=0.19)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    print(OUT, OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
