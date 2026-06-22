#!/usr/bin/env python3
"""E7c: portfolio latency-hiding (overlap source S1, the clean win).

The fixed round-trip latency alpha (=790 us measured on ECP5) is dead time only
if the CPU sits idle waiting for the accelerator. With a portfolio of Q
*independent* SAT instances, the CPU works on the other instances while one
instance's propagation burst is in flight, so up to Q round trips overlap in the
same wall-clock window. The per-instance effective latency is then

    alpha_eff(Q) = alpha / Q                      (Q independent trips in flight)
    W_min_eff(Q) = alpha_eff(Q) * R_cpu = alpha * R_cpu / Q

At Q = 1 this reproduces the no-overlap break-even W_min = alpha * R_cpu (cross
check). As Q grows the dead time is amortized away and the work a formula must
present per trip to break even shrinks as 1/Q. Combined with the per-decision
work batch ppd (E4), the required batching factor becomes

    B_req(Q) = W_min_eff(Q) / ppd = (alpha * R_cpu) / (Q * ppd)

so portfolio depth and per-trip batching trade off directly. K_family is the
portfolio depth at which a single decision already pays off (B_req = 1).

Independence claim: distinct instances do not depend on each other, so this
overlap needs no speculation -- it is the conservative, defensible headline.
We MEASURE available overlap; realizing it needs a solver restructured for
in-flight offload (future work).

Usage:
    python model/overlap.py --props instrument/props.csv \
        --measured transport/tcp_params.json --figures figures/
"""
from __future__ import annotations

import argparse
import json
import math
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

DEFAULT_ALPHA = 2e-4      # s; placeholder until --measured supplies the real one
DEFAULT_R_CPU = 9.7e6     # propagations/sec; MEASURED (see classify.py)


def w_min_eff(alpha: float, r_cpu: float, Q):
    """Effective break-even work per trip at portfolio depth Q."""
    return alpha * r_cpu / np.asarray(Q, dtype=float)


def fig_wmin_vs_Q(alpha, r_cpu, ppd_by_family, path: Path) -> None:
    """F7b: W_min_eff vs portfolio depth Q, with family chunk sizes overlaid."""
    Q = np.logspace(0, 4, 400)
    wm = w_min_eff(alpha, r_cpu, Q)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(Q, wm, color="k", lw=2.4, label="break-even work (∝ 1/parallel)")

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ppd_by_family)))
    for (fam, ppd), c in zip(sorted(ppd_by_family.items()), cmap):
        K = alpha * r_cpu / ppd                 # parallel solves where one decision pays off
        ax.axhline(ppd, color=c, ls=":", lw=1.3)
        ax.plot([K], [ppd], "o", color=c, ms=8,
                label=f"{fam} ({ppd:.0f}/dec, K={K:,.0f})")

    ax.axhline(1, color="grey", lw=0.6)
    ax.set_xlabel("Independent solves in parallel")
    ax.set_ylabel("Break-even work per round trip (propagations)")
    ax.set_title(f"Parallel solves hide the round-trip latency "
                 f"(link latency {alpha*1e6:.0f} µs)", fontsize=13)
    ax.legend(fontsize=8.5, loc="upper right",
              title="Family (props/dec, break-even depth K)", title_fontsize=8.5,
              ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(1, Q[-1])
    ax.set_ylim(1, w_min_eff(alpha, r_cpu, 1) * 1.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_wall_shrinks(alpha, r_cpu, ppd_by_family, path: Path,
                     depths=(1, 10, 100)) -> None:
    """F7c: required batching factor per family, shrinking with portfolio depth."""
    fams = sorted(ppd_by_family)
    x = np.arange(len(fams))
    width = 0.8 / len(depths)
    labels = {1: "1 (no overlap)", 10: "10 parallel", 100: "100 parallel"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, Q in enumerate(depths):
        vals = [alpha * r_cpu / (Q * ppd_by_family[f]) for f in fams]
        ax.bar(x + i * width, vals, width, label=labels.get(Q, f"{Q} parallel"))
    ax.set_yscale("log")
    ax.axhline(1, color="k", ls="--", lw=1.2, label="break-even")
    ax.set_xticks(x + width * (len(depths) - 1) / 2)
    ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("Decisions' work per round trip to break even")
    ax.set_xlabel("Benchmark family")
    ax.set_title(f"Parallel solves shrink the offload requirement "
                 f"(link latency {alpha*1e6:.0f} µs)", fontsize=13)
    ax.legend(title="Solves in parallel", fontsize=9.5, title_fontsize=9.5)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--props", required=True, type=Path)
    ap.add_argument("--measured", type=Path, default=None,
                    help="transport params.json; supplies measured alpha")
    ap.add_argument("--alpha-us", type=float, default=None,
                    help="override alpha in microseconds")
    ap.add_argument("--r-cpu", type=float, default=DEFAULT_R_CPU)
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/overlap.csv"))
    args = ap.parse_args()

    if args.alpha_us is not None:
        alpha = args.alpha_us * 1e-6
    elif args.measured:
        alpha = json.loads(args.measured.read_text())["alpha_us"] * 1e-6
    else:
        alpha = DEFAULT_ALPHA

    df = pd.read_csv(args.props)
    df = df[df.status == "ok"]
    ppd_by_family = df.groupby("family")["props_per_decision"].median().to_dict()

    args.figures.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # cross-check required by definition of done
    wmin_q1 = w_min_eff(alpha, args.r_cpu, 1)
    print(f"alpha = {alpha*1e6:.1f} us   R_cpu = {args.r_cpu:.1e} prop/s")
    print(f"cross-check: W_min_eff(Q=1) = {float(wmin_q1):,.0f}  "
          f"(should equal no-overlap W_min)\n")

    rows = []
    print(f"{'family':<14}{'ppd':>7}{'K (single-decision)':>22}{'Q for B_req<=10':>18}")
    for fam in sorted(ppd_by_family):
        ppd = ppd_by_family[fam]
        K = alpha * args.r_cpu / ppd            # depth where one decision pays off
        Q10 = K / 10                            # depth where 10-decision batch pays off
        rows.append({"family": fam, "ppd_median": round(ppd, 2),
                     "K_single_decision": math.ceil(K), "Q_batch10": math.ceil(Q10)})
        print(f"{fam:<14}{ppd:>7.1f}{math.ceil(K):>22,}{math.ceil(Q10):>18,}")
    pd.DataFrame(rows).to_csv(args.out, index=False)

    fig_wmin_vs_Q(alpha, args.r_cpu, ppd_by_family, args.figures / "e7_wmin_vs_Q.png")
    fig_wall_shrinks(alpha, args.r_cpu, ppd_by_family, args.figures / "e7_wall_shrinks.png")

    # headline
    Kmax = max(math.ceil(alpha * args.r_cpu / p) for p in ppd_by_family.values())
    print(f"\nHEADLINE: with a portfolio of Q >= {Kmax:,} independent instances, the "
          f"{alpha*1e6:.0f} us round trip is fully hidden and EVERY family becomes "
          f"offloadable at single-decision granularity.")
    print(f"(With a modest 50-decision batch per trip, that drops to "
          f"Q >= {math.ceil(Kmax/50):,}.)")
    print(f"wrote {args.out}, e7_wmin_vs_Q.png, e7_wall_shrinks.png")


if __name__ == "__main__":
    main()
