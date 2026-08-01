#!/usr/bin/env python3
"""E13: does PPD measured from a short prefix of a real solver run predict the
full-run PPD?

This is reviewer 19A's "short initial execution phase" suggestion. After G0 came
back negative (offline UP cascade aggregates add nothing to log n -- see
bridge_check.py), sampling the actual solver early is the surviving route to
pre-solve PPD estimation.

Engine: PySAT's Minisat22, used for BOTH sides of every comparison, so prefix and
full run come from the identical propagation counter. `conf_budget(k)` plus
`solve_limited()` gives a controlled prefix; `accum_stats()` returns decisions and
propagations, so prefix PPD = propagations / decisions at that budget.
solve_limited() returns None exactly when the budget is exhausted, which is the
solved_within_prefix signal.

Method:
  Step 1  consistency gate -- full-run PySAT PPD vs the PPD recorded in
          props.csv, per family and pooled. Below 0.9 pooled Spearman the two
          engines count differently and everything downstream is unsound, so the
          script stops.
  Step 2  prefix sweep over conflict budgets, recording PPD, decisions,
          propagations, wall-clock and whether the solve finished early.
  Step 3  ceiling control -- instances SOLVED within the prefix have
          prefix == full by construction and would inflate every correlation, so
          every statistic is computed twice: all instances, and unsolved-only.
          The unsolved-only version is the honest one. Cells with fewer than
          MIN_N unsolved instances are marked uninformative rather than reported;
          in the G0 work two n<10 correlations were the entire false positive.
  Step 4  Spearman per family (n constant, so size cannot hide there) and
          pooled, partial correlation controlling log n, and the relative error
          from using prefix PPD directly as the estimate of full PPD.
  Step 5  cost accounting -- median prefix wall-clock as a fraction of median
          full solve time; a prefix costing >10% of the solve is not "short".

Usage:
    python model/prefix_ppd.py --benchmarks benchmarks/ \
        --props instrument/props.csv --figures figures/ \
        --out model/prefix_ppd.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysat.solvers import Minisat22

from up_engine import parse_dimacs          # SATLIB-tolerant DIMACS parser
from bridge_check import partial_spearman   # rank-residual partial correlation

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

BUDGETS = [10, 30, 100, 300, 1000]
MIN_N = 10          # below this, a correlation cell is uninformative
RHO_GATE = 0.8      # step 6: within-family gate for the searching families
ERR_GATE = 0.20     # step 6: median relative error gate
COST_GATE = 0.10    # step 5: prefix stops being "cheap" above this
SEARCHING = ["uf100-430", "uf150-645"]
CONSISTENCY_GATE = 0.9


def solve_full(clauses: list[list[int]]) -> dict:
    s = Minisat22(bootstrap_with=clauses)
    t0 = time.perf_counter()
    res = s.solve()
    wall = time.perf_counter() - t0
    st = s.accum_stats()
    s.delete()
    dec = st["decisions"]
    return {"full_sat": bool(res), "full_wall": wall,
            "full_decisions": dec, "full_props": st["propagations"],
            "full_conflicts": st["conflicts"],
            "full_ppd": st["propagations"] / dec if dec else np.nan}


def solve_prefix(clauses: list[list[int]], budget: int) -> dict:
    """Fresh solver, conflict-budgeted run. res is None iff the budget was
    exhausted before the instance was decided."""
    s = Minisat22(bootstrap_with=clauses)
    s.conf_budget(budget)
    t0 = time.perf_counter()
    res = s.solve_limited()
    wall = time.perf_counter() - t0
    st = s.accum_stats()
    s.delete()
    dec = st["decisions"]
    return {"budget": budget, "pre_wall": wall,
            "pre_decisions": dec, "pre_props": st["propagations"],
            "pre_conflicts": st["conflicts"],
            "pre_ppd": st["propagations"] / dec if dec else np.nan,
            "solved_within_prefix": res is not None}


def rho_or_none(x: pd.Series, y: pd.Series) -> tuple[float, float, bool]:
    """Spearman, plus whether the cell is informative at all."""
    if len(x) < MIN_N or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan, False
    r, p = spearmanr(x, y)
    return r, p, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    ap.add_argument("--props", type=Path, default=Path("instrument/props.csv"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/prefix_ppd.csv"))
    ap.add_argument("--per-family", type=int, default=60,
                    help="instances per family (0 = all with measured PPD)")
    args = ap.parse_args()

    props = pd.read_csv(args.props)
    props = props[(props.status == "ok") & props.props_per_decision.notna()]
    props = (props.groupby(["family", "instance"], as_index=False)
                  .props_per_decision.mean()
                  .rename(columns={"props_per_decision": "ppd_orig"}))

    # families may nest their CNFs in subdirectories (see bridge_check.py)
    cnf_index: dict[tuple[str, str], Path] = {}
    for p in args.benchmarks.rglob("*.cnf"):
        cnf_index.setdefault((p.relative_to(args.benchmarks).parts[0], p.name), p)

    # ---- steps 1 + 2: full runs and prefix sweep ----------------------------
    rows, missing = [], 0
    for fam, g in props.groupby("family"):
        g = g.sort_values("instance")
        if args.per_family:
            g = g.iloc[:args.per_family]
        print(f"{fam}: {len(g)} instances", flush=True)
        for r in g.itertuples():
            path = cnf_index.get((fam, r.instance))
            if path is None:
                missing += 1
                continue
            # PySAT's CNF reader rejects SATLIB's trailing '%'/'0' footer, so
            # reuse the tolerant parser this repo already has.
            f = parse_dimacs(path)
            clauses = [list(c) for c in f.clauses]
            full = solve_full(clauses)
            for b in BUDGETS:
                pre = solve_prefix(clauses, b)
                rows.append({"family": fam, "instance": r.instance,
                             "ppd_orig": r.ppd_orig, "n_vars": f.n_vars,
                             "n_clauses": f.n_clauses, **full, **pre})

    df = pd.DataFrame(rows)
    df["log_n"] = np.log(df.n_vars)
    df["rel_err"] = (df.pre_ppd - df.full_ppd).abs() / df.full_ppd
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 240)

    # ---- step 1: consistency gate -------------------------------------------
    one = df[df.budget == BUDGETS[0]].copy()   # one row per instance
    cons = []
    for label, g in list(one.groupby("family")) + [("POOLED", one)]:
        r, p, ok = rho_or_none(g.full_ppd, g.ppd_orig)
        cons.append({"scope": label, "n": len(g), "rho": r, "p": p,
                     "informative": ok,
                     "mean_pysat_ppd": g.full_ppd.mean(),
                     "mean_orig_ppd": g.ppd_orig.mean()})
    cons = pd.DataFrame(cons).round(4)
    cons.to_csv(args.out.with_name("prefix_ppd_consistency.csv"), index=False)
    print("\n=== STEP 1: consistency, PySAT full-run PPD vs props.csv PPD ===")
    print(cons.to_string(index=False))
    pooled_rho = cons.loc[cons.scope == "POOLED", "rho"].iloc[0]
    if not (pooled_rho >= CONSISTENCY_GATE):
        print(f"\nSTOP -- pooled Spearman {pooled_rho:.3f} < {CONSISTENCY_GATE}. "
              "The PySAT engine's propagation counting differs from the original "
              "MiniSat runs; prefix results would rest on a mismatch.")
        return
    print(f"\nconsistency gate passed (pooled rho = {pooled_rho:.3f} >= "
          f"{CONSISTENCY_GATE}); using PySAT full-run PPD as ground truth.")

    # ---- steps 3 + 4 + 5: per (family, budget) ------------------------------
    recs = []
    for (fam, b), g in df.groupby(["family", "budget"]):
        uns = g[~g.solved_within_prefix]
        rec = {"family": fam, "budget": b, "n": len(g),
               "frac_solved_in_prefix": round(g.solved_within_prefix.mean(), 3),
               "n_unsolved": len(uns)}
        # (a) all instances -- inflated by solved-within-prefix rows
        r_all, _, ok_all = rho_or_none(g.pre_ppd, g.full_ppd)
        rec["rho_all"] = r_all
        rec["rho_all_informative"] = ok_all
        # (b) unsolved only -- the honest version
        r_u, p_u, ok_u = rho_or_none(uns.pre_ppd, uns.full_ppd)
        rec["rho_unsolved"] = r_u
        rec["p_unsolved"] = p_u
        rec["informative"] = ok_u
        # step 4: prediction error, unsolved subset
        if len(uns) >= MIN_N:
            rec["med_rel_err"] = round(uns.rel_err.median(), 4)
            rec["p90_rel_err"] = round(uns.rel_err.quantile(0.90), 4)
        else:
            rec["med_rel_err"] = np.nan
            rec["p90_rel_err"] = np.nan
        # step 5: cost, over all instances at this budget
        med_pre, med_full = g.pre_wall.median(), g.full_wall.median()
        rec["med_pre_wall"] = med_pre
        rec["med_full_wall"] = med_full
        rec["cost_frac"] = round(med_pre / med_full, 4) if med_full > 0 else np.nan
        rec["med_pre_decisions"] = uns.pre_decisions.median() if len(uns) else np.nan
        recs.append(rec)
    tab = pd.DataFrame(recs).sort_values(["family", "budget"])

    # pooled, plus the partial correlation controlling log n
    prec = []
    for b, g in df.groupby("budget"):
        uns = g[~g.solved_within_prefix]
        rec = {"budget": b, "n_unsolved": len(uns),
               "n_vars_distinct": uns.n_vars.nunique()}
        r_u, p_u, ok = rho_or_none(uns.pre_ppd, uns.full_ppd)
        rec["rho_pooled"] = r_u
        rec["p_pooled"] = p_u
        rec["informative"] = ok
        if ok and uns.n_vars.nunique() > 1:
            pr, pp = partial_spearman(uns.pre_ppd.values, uns.full_ppd.values,
                                      uns.log_n.values)
            rn, _ = spearmanr(uns.log_n, uns.full_ppd)
        else:
            pr, pp, rn = np.nan, np.nan, np.nan
        rec["partial_rho_ctrl_logn"] = pr
        rec["partial_p"] = pp
        rec["rho_logn_baseline"] = rn
        if len(uns) >= MIN_N:
            rec["med_rel_err"] = round(uns.rel_err.median(), 4)
            rec["p90_rel_err"] = round(uns.rel_err.quantile(0.90), 4)
        prec.append(rec)
    pooled = pd.DataFrame(prec).round(4)

    tab.to_csv(args.out.with_name("prefix_ppd_by_family.csv"), index=False)
    pooled.to_csv(args.out.with_name("prefix_ppd_pooled.csv"), index=False)

    print("\n=== STEP 3: ceiling control + STEP 4/5 per family and budget ===")
    show = tab.copy()
    for c in ("rho_all", "rho_unsolved"):
        show[c] = show[c].round(3)
    show.loc[~show.informative, "rho_unsolved"] = np.nan
    print(show[["family", "budget", "n", "frac_solved_in_prefix", "n_unsolved",
                "rho_all", "rho_unsolved", "informative", "med_rel_err",
                "p90_rel_err", "cost_frac"]].to_string(index=False))
    print(f"\n(cells with n_unsolved < {MIN_N} are marked "
          f"informative=False and their rho suppressed)")

    print("\n=== STEP 4: pooled, with log n control ===")
    print(pooled.to_string(index=False))

    # ---- step 6: verdict ----------------------------------------------------
    print("\n=== STEP 6: verdict ===")
    inf = tab[tab.informative]
    srch = inf[inf.family.isin(SEARCHING)]
    hits = srch[srch.rho_unsolved > RHO_GATE]
    if hits.empty:
        if srch.empty:
            print(f"Q1 within-family rho > {RHO_GATE} for searching families: "
                  "NO informative cell at any budget.")
        else:
            best = srch.loc[srch.rho_unsolved.idxmax()]
            print(f"Q1 within-family rho > {RHO_GATE} for searching families: "
                  f"NEVER. Best informative = {best.rho_unsolved:.3f} "
                  f"({best.family}, budget {int(best.budget)}).")
    else:
        print(f"Q1 within-family rho > {RHO_GATE} for searching families, "
              "cheapest first:")
        print(hits[["family", "budget", "n_unsolved",
                    "rho_unsolved"]].to_string(index=False))

    pi = pooled[pooled.informative]
    if pi.empty:
        print("Q2 vs log n: no informative pooled cell.")
    else:
        for r in pi.itertuples():
            print(f"Q2 budget {int(r.budget):5d}: pooled rho="
                  f"{r.rho_pooled:.3f}, partial(ctrl log n)="
                  f"{r.partial_rho_ctrl_logn:.3f}, "
                  f"log n baseline={r.rho_logn_baseline:.3f}")

    cheap = inf[inf.med_rel_err < ERR_GATE].sort_values("budget")
    if cheap.empty:
        print(f"Q3 cheapest budget with median relative error < "
              f"{ERR_GATE:.0%}: NONE at any budget/family.")
    else:
        print(f"Q3 median relative error < {ERR_GATE:.0%} at:")
        print(cheap[["family", "budget", "n_unsolved", "med_rel_err",
                     "cost_frac"]].to_string(index=False))

    exp = tab[tab.cost_frac > COST_GATE]
    if not exp.empty:
        print(f"\nStep 5: prefix exceeds {COST_GATE:.0%} of full solve time at:")
        print(exp[["family", "budget", "cost_frac"]].to_string(index=False))
    if missing:
        print(f"\n({missing} instances skipped: CNF not found)")

    # ---- figure -------------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fams = sorted(df.family.unique())
    colors = dict(zip(fams, plt.cm.tab10(np.linspace(0, 1, len(fams)))))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (0,0) scatter at the largest budget, unsolved only
    ax = axes[0][0]
    bmax = BUDGETS[-1]
    sub = df[(df.budget == bmax) & (~df.solved_within_prefix)]
    for fam in fams:
        g = sub[sub.family == fam]
        if g.empty:
            continue
        ax.scatter(g.pre_ppd, g.full_ppd, s=26, alpha=0.8, color=colors[fam],
                   label=f"{fam} (n={len(g)})", edgecolor="none")
    if not sub.empty:
        lo = min(sub.pre_ppd.min(), sub.full_ppd.min())
        hi = max(sub.pre_ppd.max(), sub.full_ppd.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="prefix = full")
    pr = pooled.loc[pooled.budget == bmax, "rho_pooled"]
    ax.set_xlabel(f"Prefix PPD (budget {bmax} conflicts)")
    ax.set_ylabel("Full-run PPD")
    ax.set_title(f"Prefix vs full PPD, unsolved-in-prefix only\n"
                 f"budget {bmax}, pooled $\\rho$="
                 f"{pr.iloc[0] if len(pr) else float('nan'):.2f}", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) within-family rho vs budget
    ax = axes[0][1]
    for fam in fams:
        t = tab[(tab.family == fam) & tab.informative].sort_values("budget")
        if t.empty:
            continue
        style = "o-" if fam in SEARCHING else "s:"
        ax.plot(t.budget, t.rho_unsolved, style, color=colors[fam], label=fam,
                lw=2.2 if fam in SEARCHING else 1.3)
    ax.axhline(RHO_GATE, color="crimson", ls="--", lw=1.2,
               label=f"gate {RHO_GATE}")
    ax.set_xscale("log")
    ax.set_xlabel("Conflict budget")
    ax.set_ylabel("Spearman $\\rho$ (prefix vs full PPD)")
    ax.set_title("Within-family correlation, unsolved subset\n"
                 f"(only cells with n$\\geq${MIN_N} plotted)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) relative error vs budget
    ax = axes[1][0]
    for fam in fams:
        t = tab[(tab.family == fam) & tab.informative].sort_values("budget")
        if t.empty:
            continue
        ax.plot(t.budget, t.med_rel_err, "o-", color=colors[fam], label=fam)
    ax.axhline(ERR_GATE, color="crimson", ls="--", lw=1.2,
               label=f"gate {ERR_GATE:.0%}")
    ax.set_xscale("log")
    ax.set_xlabel("Conflict budget")
    ax.set_ylabel("Median relative error of prefix PPD")
    ax.set_title("Using prefix PPD directly as the estimate", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) cost
    ax = axes[1][1]
    for fam in fams:
        t = tab[tab.family == fam].sort_values("budget")
        ax.plot(t.budget, t.cost_frac, "o-", color=colors[fam], label=fam)
    ax.axhline(COST_GATE, color="crimson", ls="--", lw=1.2,
               label=f'"cheap" limit {COST_GATE:.0%}')
    ax.axhline(1.0, color="grey", ls=":", lw=1, label="= full solve")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Conflict budget")
    ax.set_ylabel("Median prefix wall / median full solve wall")
    ax.set_title("Step 5: is the prefix actually cheap?", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.figures / "e13_prefix_ppd.png")
    plt.close(fig)
    print(f"\nwrote {args.out}, prefix_ppd_by_family.csv, "
          f"prefix_ppd_pooled.csv, prefix_ppd_consistency.csv "
          f"and e13_prefix_ppd.png")


if __name__ == "__main__":
    main()
