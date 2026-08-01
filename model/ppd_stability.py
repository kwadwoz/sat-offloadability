#!/usr/bin/env python3
"""E14: is instance-level PPD a stable property of the formula at all?

E13's consistency gate failed for an unexpected reason: two MiniSat variants
agreed on each family's PPD *distribution* (same mean, same spread) but ranked
individual instances almost independently -- uf150-645 cross-engine Spearman was
0.064. That raises the possibility that instance-level PPD is a property of the
(instance, solver-configuration) pair rather than of the instance, in which case
no formula-derived feature can predict it and both G0 and E13 fail for a reason
that has nothing to do with the features chosen.

This measures that directly. Same engine (PySAT Minisat22), same instance,
repeated under semantics-preserving perturbations:

    identical       byte-identical input, repeated -- a control. Any variance
                    here would mean the measurement itself is noisy.
    clause_shuffle  clause order permuted
    var_rename      variables renamed by a random permutation
    phases          random initial polarity via set_phases (formula untouched)
    all             all three at once

None of these change the formula's satisfiability or its structure up to
isomorphism, so a PPD that is a property of the *instance* should be invariant.

Key quantities, per (family, mode):

    within_cv     median over instances of the CV of PPD across repeats
                  -- how much PPD moves for one fixed instance
    between_cv    CV across instances of their mean PPD
                  -- how much real instance-to-instance signal exists
    ICC           between-instance variance / total variance, one-way ANOVA.
                  The reliability of a single PPD measurement.
    split_half    correlate mean PPD from two disjoint halves of the repeats,
                  across instances (median over random splits). This is the
                  empirical ceiling: no predictor of instance-level PPD can
                  correlate better than this with a fresh measurement.
    ceiling_1run  sqrt(ICC) -- ceiling for predicting a SINGLE run's PPD, which
                  is what props.csv records.

Usage:
    python model/ppd_stability.py --benchmarks benchmarks/ \
        --props instrument/props.csv --figures figures/ \
        --out model/ppd_stability.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysat.solvers import Minisat22

from up_engine import parse_dimacs   # SATLIB-tolerant DIMACS parser

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

MODES = ["identical", "clause_shuffle", "var_rename", "phases", "all"]
REPEATS = 12
N_SPLITS = 200      # random split-half draws
MIN_N = 10          # below this an aggregate cell is uninformative
SEARCHING = ["uf100-430", "uf150-645"]
LOGN_BASELINE = 0.73


def perturb(clauses: list[list[int]], n_vars: int, mode: str,
            rng: random.Random) -> tuple[list[list[int]], list[int] | None]:
    """Return a semantics-preserving variant of the formula, plus initial phases."""
    cl = [list(c) for c in clauses]
    phases = None
    if mode in ("var_rename", "all"):
        perm = list(range(1, n_vars + 1))
        rng.shuffle(perm)
        mp = {v: perm[v - 1] for v in range(1, n_vars + 1)}
        cl = [[(1 if l > 0 else -1) * mp[abs(l)] for l in c] for c in cl]
    if mode in ("clause_shuffle", "all"):
        rng.shuffle(cl)
    if mode in ("phases", "all"):
        phases = [v if rng.random() < 0.5 else -v
                  for v in range(1, n_vars + 1)]
    return cl, phases


def run_once(clauses: list[list[int]], phases: list[int] | None) -> dict:
    s = Minisat22(bootstrap_with=clauses)
    if phases:
        s.set_phases(phases)
    sat = s.solve()
    st = s.accum_stats()
    s.delete()
    dec = st["decisions"]
    return {"sat": bool(sat), "decisions": dec,
            "propagations": st["propagations"], "conflicts": st["conflicts"],
            "ppd": st["propagations"] / dec if dec else np.nan}


def icc_oneway(arr: np.ndarray) -> float:
    """One-way ICC: between-instance variance / total variance.

    arr is (instances x repeats), equal repeats per instance, so the standard
    mean-squares estimator applies. Returns nan if not estimable; a negative
    variance-component estimate is clipped to 0, meaning no detectable
    between-instance signal.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return np.nan
    if not np.isfinite(arr).all():
        return np.nan
    k, n = arr.shape
    grand = arr.mean()
    ms_between = n * ((arr.mean(axis=1) - grand) ** 2).sum() / (k - 1)
    ms_within = ((arr - arr.mean(axis=1, keepdims=True)) ** 2).sum() / (k * (n - 1))
    if ms_within <= 0:
        # perfectly repeatable measurement: all variance is between instances
        return 1.0 if ms_between > 0 else np.nan
    var_b = (ms_between - ms_within) / n
    if var_b <= 0:
        return 0.0
    return float(var_b / (var_b + ms_within))


def split_half(arr: np.ndarray, rng: np.random.Generator) -> float:
    """Median Spearman between per-instance means of two disjoint halves of the
    repeats. arr is (instances x repeats)."""
    k, n = arr.shape
    if k < MIN_N or n < 4:
        return np.nan
    half = n // 2
    rs = []
    for _ in range(N_SPLITS):
        idx = rng.permutation(n)
        a = arr[:, idx[:half]].mean(axis=1)
        b = arr[:, idx[half:2 * half]].mean(axis=1)
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            continue
        rs.append(spearmanr(a, b).statistic)
    return float(np.median(rs)) if rs else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    ap.add_argument("--props", type=Path, default=Path("instrument/props.csv"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/ppd_stability.csv"))
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--families", nargs="+", default=None,
                    help="restrict to these families (default: all)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()

    props = pd.read_csv(args.props)
    props = props[(props.status == "ok") & props.props_per_decision.notna()]
    props = (props.groupby(["family", "instance"], as_index=False)
                  .props_per_decision.mean()
                  .rename(columns={"props_per_decision": "ppd_orig"}))
    if args.families:
        props = props[props.family.isin(args.families)]
        if props.empty:
            raise SystemExit(f"no instances for families {args.families}")

    cnf_index: dict[tuple[str, str], Path] = {}
    for p in args.benchmarks.rglob("*.cnf"):
        cnf_index.setdefault((p.relative_to(args.benchmarks).parts[0], p.name), p)

    rows = []
    for fam, g in props.groupby("family"):
        g = g.sort_values("instance")
        if args.per_family:
            g = g.iloc[:args.per_family]
        print(f"{fam}: {len(g)} instances x {len(MODES)} modes x "
              f"{args.repeats} repeats", flush=True)
        for r in g.itertuples():
            path = cnf_index.get((fam, r.instance))
            if path is None:
                continue
            f = parse_dimacs(path)
            base = [list(c) for c in f.clauses]
            for mode in MODES:
                for rep in range(args.repeats):
                    rng = random.Random(hash((r.instance, mode, rep))
                                        ^ args.seed)
                    cl, ph = perturb(base, f.n_vars, mode, rng)
                    out = run_once(cl, ph)
                    rows.append({"family": fam, "instance": r.instance,
                                 "n_vars": f.n_vars, "ppd_orig": r.ppd_orig,
                                 "mode": mode, "rep": rep, **out})

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 240)

    # ---- per (family, mode) stability ---------------------------------------
    gen = np.random.default_rng(args.seed)
    recs = []
    for (fam, mode), g in df.groupby(["family", "mode"]):
        piv = g.pivot_table(index="instance", columns="rep", values="ppd")
        arr = piv.to_numpy(dtype=float)
        per_inst_mean = np.nanmean(arr, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            inst_cv = np.nanstd(arr, axis=1) / per_inst_mean
        icc = icc_oneway(arr)
        sh = split_half(arr, gen)
        rec = {
            "family": fam, "mode": mode, "n_instances": arr.shape[0],
            "repeats": arr.shape[1],
            "mean_ppd": round(float(np.nanmean(arr)), 3),
            "within_cv": round(float(np.nanmedian(inst_cv)), 4),
            "between_cv": round(float(np.nanstd(per_inst_mean)
                                      / np.nanmean(per_inst_mean)), 4),
            "icc": round(icc, 4) if np.isfinite(icc) else np.nan,
            "split_half_rho": round(sh, 4) if np.isfinite(sh) else np.nan,
            "ceiling_1run": (round(float(np.sqrt(icc)), 4)
                             if np.isfinite(icc) and icc >= 0 else np.nan),
            "informative": arr.shape[0] >= MIN_N,
        }
        # does the perturbed mean PPD still track the original props.csv value?
        om = g.groupby("instance").ppd_orig.first().reindex(piv.index).to_numpy()
        if arr.shape[0] >= MIN_N and len(np.unique(om)) > 1:
            rec["rho_vs_props_csv"] = round(
                spearmanr(per_inst_mean, om).statistic, 4)
        else:
            rec["rho_vs_props_csv"] = np.nan
        recs.append(rec)
    tab = pd.DataFrame(recs).sort_values(["family", "mode"])
    tab.to_csv(args.out.with_name("ppd_stability_summary.csv"), index=False)

    print("\n=== E14: PPD stability under semantics-preserving perturbation ===")
    print(tab[["family", "mode", "n_instances", "mean_ppd", "within_cv",
               "between_cv", "icc", "split_half_rho", "ceiling_1run",
               "rho_vs_props_csv"]].to_string(index=False))

    # ---- control check ------------------------------------------------------
    ctrl = tab[tab["mode"] == "identical"]
    bad = ctrl[ctrl.within_cv > 1e-9]
    print("\n--- control: 'identical' mode should have within_cv == 0 ---")
    if bad.empty:
        print("OK: byte-identical repeats give identical PPD in every family, "
              "so the solver is deterministic and all variance below comes from "
              "the perturbations, not measurement noise.")
    else:
        print("WARNING: nonzero variance with identical input:")
        print(bad[["family", "within_cv"]].to_string(index=False))

    # ---- verdict -----------------------------------------------------------
    print("\n=== VERDICT ===")
    real = tab[(tab["mode"] != "identical") & tab.informative]
    for fam in sorted(real.family.unique()):
        t = real[real.family == fam]
        tag = " (searching)" if fam in SEARCHING else ""
        print(f"\n{fam}{tag}")
        for r in t.itertuples():
            ratio = (r.within_cv / r.between_cv
                     if r.between_cv and np.isfinite(r.between_cv) else np.nan)
            print(f"  {r.mode:14s} within_cv={r.within_cv:.3f} "
                  f"between_cv={r.between_cv:.3f} "
                  f"noise/signal={ratio:5.2f} ICC={r.icc:.3f} "
                  f"split-half={r.split_half_rho:.3f} "
                  f"ceiling(1 run)={r.ceiling_1run:.3f}")

    worst = real[real["mode"] == "all"]
    if not worst.empty:
        below = worst[worst.ceiling_1run < LOGN_BASELINE]
        print(f"\nFamilies whose ceiling(1 run) falls below the log n baseline "
              f"({LOGN_BASELINE}), under full perturbation:")
        if below.empty:
            print("  none -- instance-level PPD is stable enough to be a target")
        else:
            print(below[["family", "icc", "ceiling_1run"]].to_string(index=False))

    print("\nInterpretation: ICC is the fraction of PPD variance that is a real "
          "property of the instance.\nceiling_1run = sqrt(ICC) bounds how well "
          "ANY formula-derived predictor could correlate\nwith a single measured "
          "PPD such as the props.csv values.")

    # ---- figure ------------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fams = sorted(df.family.unique())
    modes = [m for m in MODES if m != "identical"]
    colors = dict(zip(fams, plt.cm.tab10(np.linspace(0, 1, len(fams)))))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))

    # (0,0) within vs between CV, mode 'all'
    ax = axes[0][0]
    t = tab[tab["mode"] == "all"]
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.within_cv, 0.38, label="within-instance CV (noise)",
           color="#c44e52")
    ax.bar(x + 0.2, t.between_cv, 0.38, label="between-instance CV (signal)",
           color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(t.family, rotation=30, ha="right")
    ax.set_ylabel("Coefficient of variation of PPD")
    ax.set_title("Noise vs signal, all perturbations\n"
                 "(bars equal => nothing instance-specific to predict)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (0,1) ICC by family and mode
    ax = axes[0][1]
    w = 0.8 / len(modes)
    xs = np.arange(len(fams))
    for i, mode in enumerate(modes):
        vals = [tab[(tab.family == f) & (tab["mode"] == mode)].icc.iloc[0]
                if not tab[(tab.family == f) & (tab["mode"] == mode)].empty
                else np.nan for f in fams]
        ax.bar(xs + (i - (len(modes) - 1) / 2) * w, vals, w, label=mode)
    ax.set_xticks(xs)
    ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ICC (fraction of variance that is instance-intrinsic)")
    ax.set_title("How much of PPD is a property of the instance?", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0) the ceiling
    ax = axes[1][0]
    for i, mode in enumerate(modes):
        vals = [tab[(tab.family == f) & (tab["mode"] == mode)].ceiling_1run.iloc[0]
                if not tab[(tab.family == f) & (tab["mode"] == mode)].empty
                else np.nan for f in fams]
        ax.bar(xs + (i - (len(modes) - 1) / 2) * w, vals, w, label=mode)
    ax.axhline(LOGN_BASELINE, color="crimson", ls="--", lw=1.3,
               label=f"log n baseline $\\rho$={LOGN_BASELINE} (pooled)")
    ax.set_xticks(xs)
    ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("$\\sqrt{ICC}$: max achievable $\\rho$ vs one measured PPD")
    ax.set_title("Ceiling on ANY predictor of single-run instance PPD",
                 fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # (1,1) per-instance spread, searching families, mode 'all'
    ax = axes[1][1]
    for fam in SEARCHING:
        g = df[(df.family == fam) & (df["mode"] == "all")]
        if g.empty:
            continue
        piv = g.pivot_table(index="instance", columns="rep", values="ppd")
        order = piv.mean(axis=1).sort_values().index
        piv = piv.loc[order]
        lo = piv.min(axis=1).to_numpy()
        hi = piv.max(axis=1).to_numpy()
        mid = piv.mean(axis=1).to_numpy()
        xx = np.arange(len(mid))
        ax.fill_between(xx, lo, hi, alpha=0.3, color=colors[fam],
                        label=f"{fam} min-max over {args.repeats} runs")
        ax.plot(xx, mid, "-", color=colors[fam], lw=1.6,
                label=f"{fam} mean")
    ax.set_xlabel("Instance (sorted by mean PPD within family)")
    ax.set_ylabel("PPD")
    ax.set_title("Per-instance PPD range vs the family's spread\n"
                 "(overlapping bands => instance ranking is not recoverable)",
                 fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.figures / "e14_ppd_stability.png")
    plt.close(fig)
    print(f"\nwrote {args.out}, ppd_stability_summary.csv "
          f"and e14_ppd_stability.png")


if __name__ == "__main__":
    main()
