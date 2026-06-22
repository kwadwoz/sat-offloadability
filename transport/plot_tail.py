#!/usr/bin/env python3
"""E2: latency-tail CDF from the measured RTT samples.

Reads the per-sample RTTs dumped by measure_rtt.py and plots the round-trip
latency CDF for a few representative payload sizes, with p50/p99/p99.9 marked.
The heavy upper tail (USB-Ethernet adapter polling + TCP retransmits) is the
point: the fixed cost alpha used in E3 is the robust median, while the tail
shows the worst-case exposure an offload path must tolerate.

Usage:
    python transport/plot_tail.py --raw transport/tcp_raw.csv \
        --out figures/e2_latency_cdf.png
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
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12.5,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

SHOW_SIZES = [8, 128, 512, 1400]   # representative payloads to draw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("transport/tcp_raw.csv"))
    ap.add_argument("--out", type=Path, default=Path("figures/e2_latency_cdf.png"))
    ap.add_argument("--transport", default="TCP")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sizes = [s for s in SHOW_SIZES if s in set(df["size"])] or sorted(df["size"].unique())[:4]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for s in sizes:
        rtt = np.sort(df.loc[df["size"] == s, "rtt_us"].values)
        if rtt.size == 0:
            continue
        cdf = np.arange(1, rtt.size + 1) / rtt.size
        ax.plot(rtt, cdf, lw=1.9,
                label=f"{s} B (median {np.percentile(rtt,50):.0f} µs)")

    for q, name in [(0.50, "p50"), (0.99, "p99"), (0.999, "p99.9")]:
        ax.axhline(q, color="grey", ls=":", lw=0.8)
        ax.text(ax.get_xlim()[1], q, f" {name}", va="center", fontsize=9,
                color="grey", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel("Round-trip latency (µs)")
    ax.set_ylabel("Cumulative fraction of trips")
    ax.set_title("Round-trip latency distribution, measured on ECP5", fontsize=13)
    ax.legend(title="Message size", fontsize=9.5, title_fontsize=9.5,
              loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
