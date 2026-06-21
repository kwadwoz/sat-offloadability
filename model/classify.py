#!/usr/bin/env python3
"""E3: roofline / offloadability classification (the headline result).

Combines the E4 propagation distributions with the affine transport model to
compute the required batching factor and classify SAT instance families by how
hard they are to offload.

Model
-----
Round-trip transport time is affine in payload bytes s:

    T(s) = alpha + beta * s            (alpha = fixed per-trip latency)

End-to-end offload throughput with hardware solve rate R_hw and W work units
shipped per round trip:

    P(W) = W / (alpha + beta*s + W / R_hw)

In the small-message, latency-dominated regime this beats a CPU running at
R_cpu propagations/sec exactly when the per-trip work clears

    W_min  ~=  alpha * R_cpu

i.e. an offload must carry at least the work the CPU would finish during one
round-trip latency. R_hw enters only the roofline curve, never W_min, so the
classification is independent of any particular accelerator (R_hw is swept).

Work unit and the batching factor (framing B)
---------------------------------------------
One propagation. From E4, props_per_decision is the average propagation work a
formula batches per branching decision -- its natural granule of work. A single
granule (5-24 propagations) is far below W_min over a real link, so a useful
offload must aggregate several decisions' worth of propagation per round trip.
The structural quantity we report is the REQUIRED BATCHING FACTOR

    B_req(alpha) = W_min(alpha) / props_per_decision

= how many decisions'-worth of propagation a formula must present per round trip
to clear break-even. B_req = 1 means a single decision already pays off; large
B_req means the formula must aggregate a lot of work to be worth offloading.
High-propagation families need a smaller B_req and are easier to offload; this
ranking is a structural property of the formula, independent of R_hw.

(B_req is an amortization requirement on per-trip payload, not a claim that a
CDCL solver runs future decisions in parallel -- decisions remain sequential.)

Usage
-----
    python model/classify.py --props instrument/props.csv --figures figures/ \
        --out model/classification.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Representative transport parameters. Reported as MEASURED-style swept values,
# not universal constants. alpha spans coupling regimes from tight on-chip /
# PCIe-class links up to a LiteEth/VexRiscv UDP echo over Ethernet.
ALPHA_REGIMES = {
    "on-chip (1 us)": 1e-6,
    "PCIe-class (10 us)": 1e-5,
    "tuned-link (50 us)": 5e-5,
    "ECP5 UDP (200 us)": 2e-4,
}
DEFAULT_R_CPU = 1.0e7   # propagations/sec; MiniSat steady-state, swept-able
BETA = 8e-8             # s/byte (~100 Mbit/s inverse bandwidth)


def w_min(alpha: float, r_cpu: float) -> float:
    return alpha * r_cpu


def classify(df: pd.DataFrame, r_cpu: float) -> pd.DataFrame:
    """Annotate each instance with required batching factor per alpha regime."""
    out = df.copy()
    for label, alpha in ALPHA_REGIMES.items():
        out[f"B_req@{label}"] = w_min(alpha, r_cpu) / out["props_per_decision"]
    return out


def fig_distributions(df: pd.DataFrame, r_cpu: float, path: Path) -> None:
    """E4 figure: per-family props-per-decision distributions."""
    fams = sorted(df["family"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df.loc[df.family == f, "props_per_decision"].values for f in fams]
    ax.boxplot(data, tick_labels=fams, showfliers=False)
    ax.set_ylabel("propagations per decision (work batch / decision)")
    ax.set_xlabel("benchmark family")
    ax.set_title("E4: per-decision work-batch distributions")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_batching_factor(df: pd.DataFrame, r_cpu: float, path: Path) -> None:
    """E3 HEADLINE: required batching factor B_req per family vs link latency.

    For each family, median B_req(alpha) as alpha sweeps. Lower = easier to
    offload (less work must be aggregated per round trip). The break-even line
    B_req = 1 marks where a single decision already pays off.
    """
    fams = sorted(df["family"].unique())
    alphas = np.logspace(-6.5, -3.5, 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in fams:
        ppd = np.median(df.loc[df.family == f, "props_per_decision"].values)
        b_req = w_min(alphas, r_cpu) / ppd
        ax.plot(alphas * 1e6, b_req, label=f"{f} (ppd~{ppd:.0f})", lw=1.5)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0] * 1.1, 1.0, " B_req = 1 (single decision pays off)",
            va="bottom", fontsize=8)
    for label, alpha in ALPHA_REGIMES.items():
        ax.axvline(alpha * 1e6, color="grey", ls=":", lw=0.7, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fixed round-trip latency alpha (us)")
    ax.set_ylabel("required batching factor B_req (decisions / round trip)")
    ax.set_title(f"E3: work that must be aggregated to break even "
                 f"(R_cpu = {r_cpu:.0e})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_roofline(r_cpu: float, path: Path, payload_bytes: int = 64) -> None:
    """E3 roofline: throughput P(W) for swept R_hw, with CPU and W_min lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    W = np.logspace(0, 6, 400)  # work units per trip
    alpha = ALPHA_REGIMES["tuned-link (50 us)"]
    s = payload_bytes
    for r_hw in [1e6, 1e7, 1e8, 1e9]:
        P = W / (alpha + BETA * s + W / r_hw)
        ax.loglog(W, P, label=f"R_hw = {r_hw:.0e} prop/s")
    ax.axhline(r_cpu, color="k", ls="-", lw=1.5, label=f"R_cpu = {r_cpu:.0e}")
    wm = w_min(alpha, r_cpu)
    ax.axvline(wm, color="r", ls="--", lw=1.2, label=f"W_min = {wm:.0f}")
    ax.set_xlabel("work shipped per round trip W (propagations)")
    ax.set_ylabel("end-to-end throughput P (prop/s)")
    ax.set_title(f"E3: offload roofline (alpha = {alpha*1e6:.0f} us, s = {s} B)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--props", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("model/classification.csv"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--r-cpu", type=float, default=DEFAULT_R_CPU,
                    help="CPU propagation rate (props/sec); swept parameter")
    args = ap.parse_args()

    df = pd.read_csv(args.props)
    df = df[df.status == "ok"].copy()
    if df.empty:
        raise SystemExit("no usable rows in props.csv")

    args.figures.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    classified = classify(df, args.r_cpu)
    classified.to_csv(args.out, index=False)

    fig_distributions(df, args.r_cpu, args.figures / "e4_distributions.png")
    fig_batching_factor(df, args.r_cpu, args.figures / "e3_batching_factor.png")
    fig_roofline(args.r_cpu, args.figures / "e3_roofline.png")

    # console summary: median required batching factor per family x alpha
    print(f"instances classified: {len(df)}    R_cpu = {args.r_cpu:.2e} prop/s\n")
    fams = sorted(df["family"].unique())
    header = f"{'family':<14}{'ppd':>7}" + "".join(f"{lbl.split()[0]:>12}" for lbl in ALPHA_REGIMES)
    print(header)
    print(f"{'':<14}{'':>7}" + "".join(f"{a*1e6:>11.0f}u" for a in ALPHA_REGIMES.values()))
    for f in fams:
        ppd = np.median(df.loc[df.family == f, "props_per_decision"].values)
        cells = "".join(f"{w_min(a, args.r_cpu)/ppd:>12.0f}" for a in ALPHA_REGIMES.values())
        print(f"{f:<14}{ppd:>7.1f}{cells}")
    print("\n(values = required batching factor B_req; 1 = single decision already pays off)")
    print(f"\nwrote {args.out} and 3 figures to {args.figures}/")


if __name__ == "__main__":
    main()
