#!/usr/bin/env python3
"""E15: what granularity of PPD does the E3 offloading decision actually need?

E14 showed instance-level PPD is largely not a property of the instance (ICC
0.06-0.31 under semantics-preserving perturbation), so no formula-derived
predictor can resolve it. That is only fatal to the project if the E3 decision
NEEDS instance-level resolution. E3's decision variable is

    B_req(alpha) = W_min(alpha) / props_per_decision,      W_min = alpha * R_cpu

and PPD is its only instance-dependent input, so B_req inherits PPD's structure
exactly (B_req is proportional to 1/PPD). The question is therefore concrete:

  Q1  How much of the decision-relevant variation in PPD is BETWEEN families
      (which E9/E13 show is stable and reproducible) versus WITHIN a family
      (which E14 shows is mostly unpredictable noise)?

  Q2  If you estimate an instance's B_req by its FAMILY's median PPD -- i.e. you
      give up on instance-level prediction entirely -- how wrong are you?

  Q3  How wrong is a PERFECT instance-level predictor, given that the target
      itself moves under rewriting? This is the irreducible error floor from
      E14's repeated runs.

If the family-proxy error (Q2) is comparable to the irreducible floor (Q3), then
instance-level prediction buys essentially nothing that is achievable, and
family-level PPD -- which is stable -- is the right granularity for the paper's
claim.

Usage:
    python model/granularity_check.py --props instrument/props.csv \
        --stability model/ppd_stability.csv --figures figures/ \
        --out model/granularity_check.csv
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

# same regimes and constants as classify.py (E3)
ALPHA_REGIMES = {
    "on-chip (1 us)": 1e-6,
    "PCIe-class (10 us)": 1e-5,
    "tuned-link (50 us)": 5e-5,
    "ECP5 TCP (measured 790 us)": 7.90e-4,
}
R_CPU = 9.7e6
PERTURB_MODE = "all"      # E14 mode used for the irreducible floor


def b_req(ppd: np.ndarray | float, alpha: float) -> np.ndarray | float:
    return (alpha * R_CPU) / ppd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--props", type=Path, default=Path("instrument/props.csv"))
    ap.add_argument("--stability", type=Path,
                    default=Path("model/ppd_stability.csv"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/granularity_check.csv"))
    args = ap.parse_args()

    props = pd.read_csv(args.props)
    props = props[(props.status == "ok") & props.props_per_decision.notna()].copy()
    props["log_ppd"] = np.log(props.props_per_decision)
    pd.set_option("display.width", 240)

    # ---- Q1: variance decomposition of log PPD ------------------------------
    grand = props.log_ppd.mean()
    fam_mean = props.groupby("family").log_ppd.transform("mean")
    ss_total = ((props.log_ppd - grand) ** 2).sum()
    ss_between = ((fam_mean - grand) ** 2).sum()
    ss_within = ((props.log_ppd - fam_mean) ** 2).sum()
    frac_between = ss_between / ss_total

    fam = (props.groupby("family")
                .agg(n=("props_per_decision", "size"),
                     ppd_med=("props_per_decision", "median"),
                     ppd_mean=("props_per_decision", "mean"),
                     ppd_cv=("props_per_decision", lambda s: s.std() / s.mean()))
                .reset_index())
    fam = fam.sort_values("ppd_med")

    print("=== Q1: where does PPD variation live? ===")
    print(f"log-PPD variance BETWEEN families : {frac_between:6.1%}")
    print(f"log-PPD variance WITHIN  families : {1 - frac_between:6.1%}")
    print(f"family median PPD spans {fam.ppd_med.min():.2f} -> "
          f"{fam.ppd_med.max():.2f}  "
          f"({fam.ppd_med.max() / fam.ppd_med.min():.2f}x)")
    print()
    print(fam.round(3).to_string(index=False))

    # ---- Q2/Q3: error of the family proxy vs the irreducible floor ----------
    med_map = dict(zip(fam.family, fam.ppd_med))
    props["fam_med_ppd"] = props.family.map(med_map)
    # B_req is proportional to 1/PPD at every alpha, so the RELATIVE error of
    # B_req is alpha-independent; compute it once from PPD directly.
    props["fam_proxy_relerr"] = (
        (1 / props.fam_med_ppd - 1 / props.props_per_decision).abs()
        * props.props_per_decision)

    stab = pd.read_csv(args.stability)
    stab = stab[stab["mode"] == PERTURB_MODE]
    # irreducible: for each instance, spread of 1/PPD across repeats relative to
    # its own mean -- the error a PERFECT instance predictor still suffers
    irr = []
    for (famname, inst), g in stab.groupby(["family", "instance"]):
        inv = 1.0 / g.ppd.to_numpy(dtype=float)
        inv = inv[np.isfinite(inv)]
        if len(inv) < 2:
            continue
        irr.append({"family": famname, "instance": inst,
                    "irreducible_relerr": float(np.mean(np.abs(inv - inv.mean()))
                                                / inv.mean())})
    irr = pd.DataFrame(irr)

    rows = []
    for f in fam.family:
        p = props[props.family == f]
        i = irr[irr.family == f]
        rows.append({
            "family": f,
            "n_instances": len(p),
            "ppd_med": round(med_map[f], 3),
            "fam_proxy_relerr_med": round(p.fam_proxy_relerr.median(), 4),
            "fam_proxy_relerr_p90": round(p.fam_proxy_relerr.quantile(.90), 4),
            "irreducible_relerr_med": (round(i.irreducible_relerr.median(), 4)
                                       if len(i) else np.nan),
            "irreducible_relerr_p90": (round(i.irreducible_relerr.quantile(.90), 4)
                                       if len(i) else np.nan),
        })
    tab = pd.DataFrame(rows)
    tab["headroom"] = (tab.fam_proxy_relerr_med
                       - tab.irreducible_relerr_med).round(4)

    print("\n=== Q2 vs Q3: cost of giving up instance-level resolution ===")
    print("fam_proxy   = error from using the FAMILY median PPD for every instance")
    print("irreducible = error a PERFECT instance predictor still suffers, "
          "because the\n              target itself moves under rewriting (E14, "
          f"mode '{PERTURB_MODE}')")
    print("headroom    = what instance-level prediction could win, at best\n")
    print(tab.to_string(index=False))

    # ---- decision impact: B_req at each alpha -------------------------------
    print("\n=== Decision impact: B_req by family (median PPD) ===")
    brows = []
    for name, alpha in ALPHA_REGIMES.items():
        rec = {"alpha_regime": name, "alpha_s": alpha,
               "W_min": round(alpha * R_CPU, 1)}
        for f in fam.family:
            rec[f] = round(float(b_req(med_map[f], alpha)), 1)
        brows.append(rec)
    breq = pd.DataFrame(brows)
    print(breq.to_string(index=False))

    spread = fam.ppd_med.max() / fam.ppd_med.min()
    print(f"\nAcross families B_req varies by {spread:.2f}x (inverse of the PPD "
          f"span).")
    print("Within a family, the achievable extra resolution is 'headroom' above "
          "-- compare\nthat to the "
          f"{spread:.2f}x family separation to see which granularity carries "
          "the decision.")

    # family ranking stability under perturbation (E14) vs props.csv
    fam_stab = (stab.groupby("family").ppd.mean()
                    .reindex(fam.family).to_numpy())
    fam_orig = fam.ppd_med.to_numpy()
    ok = np.isfinite(fam_stab) & np.isfinite(fam_orig)
    if ok.sum() >= 3:
        rho = spearmanr(fam_stab[ok], fam_orig[ok]).statistic
        print(f"\nFamily-level ranking survives perturbation: Spearman "
              f"{rho:.3f} over {ok.sum()} families\n(perturbed mean PPD vs "
              "props.csv median PPD).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out, index=False)
    breq.to_csv(args.out.with_name("granularity_breq.csv"), index=False)

    # ---- figure -------------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # (0) where the variance lives
    ax = axes[0]
    ax.bar(["between\nfamilies", "within\nfamilies"],
           [frac_between, 1 - frac_between],
           color=["#4c72b0", "#c44e52"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of log-PPD variance")
    ax.set_title("Q1: where does PPD variation live?", fontsize=12)
    for i, v in enumerate([frac_between, 1 - frac_between]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (1) family proxy error vs irreducible floor
    ax = axes[1]
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab.fam_proxy_relerr_med, 0.38,
           label="error using family median", color="#4c72b0")
    ax.bar(x + 0.2, tab.irreducible_relerr_med, 0.38,
           label="irreducible (perfect predictor)", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(tab.family, rotation=30, ha="right")
    ax.set_ylabel("Median relative error in $B_{req}$")
    ax.set_title("Q2 vs Q3: what instance-level prediction\ncould actually win",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (2) B_req separation between families, measured ECP5 alpha
    ax = axes[2]
    alpha = ALPHA_REGIMES["ECP5 TCP (measured 790 us)"]
    order = fam.sort_values("ppd_med")
    vals = [float(b_req(med_map[f], alpha)) for f in order.family]
    lo = [float(b_req(med_map[f] * (1 + tab.set_index('family')
                                    .loc[f, 'irreducible_relerr_med']), alpha))
          for f in order.family]
    hi = [float(b_req(med_map[f] * (1 - tab.set_index('family')
                                    .loc[f, 'irreducible_relerr_med']), alpha))
          for f in order.family]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color="#4c72b0", alpha=0.85)
    ax.hlines(yy, lo, hi, color="black", lw=2.2)
    ax.set_yticks(yy)
    ax.set_yticklabels(order.family)
    ax.set_xlabel("$B_{req}$ (decisions'-worth per round trip)")
    ax.set_title("Decision impact at measured ECP5 $\\alpha$=790 us\n"
                 "(bars = family medians, whiskers = irreducible spread)",
                 fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(args.figures / "e15_granularity.png")
    plt.close(fig)
    print(f"\nwrote {args.out}, granularity_breq.csv and e15_granularity.png")


if __name__ == "__main__":
    main()
