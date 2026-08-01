#!/usr/bin/env python3
"""G0 (gate 2): does the offline UP cascade statistic connect to the measured
propagations-per-decision (PPD) that the hardware framing rests on?

E0 shows cascade size varies a lot across candidate decisions inside one
formula. That only matters for this project if the *aggregate* of those
cascades tracks what a real solver actually does. So: for every instance in
instrument/props.csv with a measured PPD, sample candidate decisions at a fixed
decision depth, take mean and max UP cascade size, and correlate those
aggregates against measured PPD -- Spearman, per family and pooled.

log n alone already reaches R^2 = 0.81 on the uf ladder (see E10), so a raw
pooled correlation proves nothing: most of it could be size. The partial
Spearman controlling for log n is the number that decides the gate.

Method per instance:
  * simulated descent to decision depth d (default 20) with backtracking on
    conflict, exactly as in cascade_spread.py.
  * probe up to --max-lits randomly chosen candidate literals (both polarities
    of sampled unassigned variables), record cascade size, restore the trail.
  * TRAILS independent descents; aggregates pooled over all of them.

Instances whose descents all fail (small formulas conflict out before reaching
depth d) are reported as uncovered rather than silently dropped.

Usage:
    python model/bridge_check.py --benchmarks benchmarks/ \
        --props instrument/props.csv --figures figures/ \
        --out model/bridge_check.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from up_engine import (Formula, descend_bt, free_vars, parse_dimacs,
                       propagate, undo)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

TRAILS = 3
AGGS = ["up_mean", "up_max"]


def probe_sample(f: Formula, assign: list[int], rng: random.Random,
                 max_lits: int) -> list[int]:
    """Cascade size for a random sample of candidate next decisions."""
    free = free_vars(f, assign)
    rng.shuffle(free)
    sizes = []
    for var in free[:max(1, max_lits // 2)]:
        for lit in (var, -var):
            r = propagate(f, assign, lit)
            undo(assign, r.assigned)
            sizes.append(r.size)
    return sizes


def resolve_depth(spec: float, n_vars: int) -> int:
    """A spec >= 1 is an absolute decision depth; a fraction below 1 scales with
    formula size, so small families are reachable at all."""
    return int(spec) if spec >= 1 else max(1, int(round(spec * n_vars)))


def up_aggregates(path: Path, spec: float, seed: int,
                  max_lits: int) -> dict | None:
    f = parse_dimacs(path)
    depth = resolve_depth(spec, f.n_vars)
    if depth >= f.n_vars:
        return None
    sizes: list[int] = []
    ok_trails = 0
    for t in range(TRAILS):
        rng = random.Random(seed + 7919 * t)
        got = descend_bt(f, depth, rng)
        if got is None:
            continue
        ok_trails += 1
        sizes.extend(probe_sample(f, got[0], rng, max_lits))
    if not sizes:
        return None
    a = np.array(sizes, dtype=float)
    return {"n_vars": f.n_vars, "n_clauses": f.n_clauses, "depth": depth,
            "ok_trails": ok_trails, "n_probes": len(a),
            "up_mean": a.mean(), "up_max": a.max(),
            "up_p90": float(np.percentile(a, 90))}


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     z: np.ndarray) -> tuple[float, float]:
    """Spearman between x and y controlling for z: correlate the residuals of
    the rank-transformed variables after regressing each on ranked z."""
    if len(x) < 5:
        return np.nan, np.nan
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(A, r, rcond=None)
        res.append(r - A @ beta)
    if np.std(res[0]) < 1e-12 or np.std(res[1]) < 1e-12:
        return np.nan, np.nan
    rho, p = spearmanr(res[0], res[1])
    return rho, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    ap.add_argument("--props", type=Path, default=Path("instrument/props.csv"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/bridge_check.csv"))
    ap.add_argument("--depths", type=float, nargs="+", default=[20, 0.2],
                    help="decision depths to report; a value >= 1 is absolute, "
                         "a fraction scales with n_vars. Default: 20 (the "
                         "headline spec) and 0.2n (covers small families).")
    ap.add_argument("--max-lits", type=int, default=120)
    ap.add_argument("--per-family", type=int, default=25,
                    help="instances per family (0 = all with measured PPD)")
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args()

    props = pd.read_csv(args.props)
    props = props[(props.status == "ok") & props.props_per_decision.notna()]
    props = (props.groupby(["family", "instance"], as_index=False)
                  .props_per_decision.mean())

    # some families nest their CNFs in subdirectories, so index by basename
    cnf_index: dict[tuple[str, str], Path] = {}
    for p in args.benchmarks.rglob("*.cnf"):
        rel = p.relative_to(args.benchmarks).parts[0]
        cnf_index.setdefault((rel, p.name), p)

    def label_of(spec: float) -> str:
        return f"d={int(spec)}" if spec >= 1 else f"d={spec:g}n"

    all_rows, all_unc = [], []
    for spec in args.depths:
        dl = label_of(spec)
        for fam, g in props.groupby("family"):
            g = g.sort_values("instance")
            if args.per_family:
                g = g.iloc[:args.per_family]
            print(f"[{dl}] {fam}: {len(g)} instances", flush=True)
            for i, r in enumerate(g.itertuples()):
                path = cnf_index.get((fam, r.instance))
                if path is None:
                    all_unc.append((dl, fam, r.instance, "cnf missing"))
                    continue
                agg = up_aggregates(path, spec,
                                    args.seed + 131 * i + hash(fam) % 1000,
                                    args.max_lits)
                if agg is None:
                    all_unc.append((dl, fam, r.instance,
                                    "no descent reached depth"))
                    continue
                all_rows.append({"depth_mode": dl, "family": fam,
                                 "instance": r.instance,
                                 "ppd": r.props_per_decision, **agg})

    df = pd.DataFrame(all_rows)
    df["log_n"] = np.log(df.n_vars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- correlations, per depth mode ---------------------------------------
    out = []
    for dl, dg in df.groupby("depth_mode"):
        for label, g in list(dg.groupby("family")) + [("POOLED", dg)]:
            rec = {"depth_mode": dl, "scope": label, "n": len(g),
                   "n_vars_distinct": g.n_vars.nunique()}
            for a in AGGS:
                rho, p = (spearmanr(g[a], g.ppd) if len(g) >= 5
                          else (np.nan, np.nan))
                rec[f"rho_{a}"] = rho
                rec[f"p_{a}"] = p
                # within one family n is constant, so partialling log n is moot
                if g.n_vars.nunique() > 1:
                    pr, pp = partial_spearman(g[a].values, g.ppd.values,
                                              g.log_n.values)
                else:
                    pr, pp = np.nan, np.nan
                rec[f"partial_rho_{a}"] = pr
                rec[f"partial_p_{a}"] = pp
            rho_n, _ = (spearmanr(g.log_n, g.ppd) if g.n_vars.nunique() > 1
                        else (np.nan, np.nan))
            rec["rho_log_n"] = rho_n
            out.append(rec)
    corr = pd.DataFrame(out).round(4)
    corr.to_csv(args.out.with_name("bridge_check_corr.csv"), index=False)

    pd.set_option("display.width", 240)
    for dl in corr.depth_mode.unique():
        sub = df[df.depth_mode == dl]
        print(f"\nG0 -- UP cascade aggregate vs measured PPD  [{dl}]  "
              f"({len(sub)} instances, {sub.family.nunique()} families, "
              f"{sub.n_vars.nunique()} distinct sizes)")
        print(corr[corr.depth_mode == dl][
            ["scope", "n", "n_vars_distinct", "rho_up_mean", "p_up_mean",
             "rho_up_max", "p_up_max", "rho_log_n", "partial_rho_up_mean",
             "partial_rho_up_max"]].to_string(index=False))
    if all_unc:
        u = pd.DataFrame(all_unc, columns=["depth_mode", "family", "instance",
                                           "reason"])
        print(f"\n{len(u)} instance-runs uncovered:")
        print(u.groupby(["depth_mode", "family", "reason"]).size().to_string())

    # ---- figure -------------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fams = sorted(df.family.unique())
    colors = dict(zip(fams, plt.cm.tab20(np.linspace(0, 1, len(fams)))))
    modes = list(corr.depth_mode.unique())
    fig, axes = plt.subplots(len(modes), 3, figsize=(16.5, 5.2 * len(modes)),
                             squeeze=False)

    for row, dl in enumerate(modes):
        dg = df[df.depth_mode == dl]
        cg = corr[corr.depth_mode == dl]
        pooled = cg[cg.scope == "POOLED"]

        for ax, a in zip(axes[row][:2], AGGS):
            for fam in fams:
                g = dg[dg.family == fam]
                if g.empty:
                    continue
                ax.scatter(g[a], g.ppd, s=26, alpha=0.8, color=colors[fam],
                           label=fam, edgecolor="none")
            pr = pooled[f"rho_{a}"].iloc[0]
            pp = pooled[f"partial_rho_{a}"].iloc[0]
            ax.set_xlabel(f"UP cascade {a.split('_')[1]} (offline, {dl})")
            ax.set_ylabel("Measured propagations / decision")
            ax.set_title(f"[{dl}] {a}\npooled $\\rho$={pr:.2f},  "
                         f"partial (log n) $\\rho$={pp:.2f}", fontsize=12)
            ax.grid(True, alpha=0.3)
        axes[row][0].legend(fontsize=7.5, ncol=2)

        ax = axes[row][2]
        x = np.arange(len(AGGS))
        raw = [pooled[f"rho_{a}"].iloc[0] for a in AGGS]
        par = [pooled[f"partial_rho_{a}"].iloc[0] for a in AGGS]
        ax.bar(x - 0.19, np.abs(raw), 0.36, label="pooled $|\\rho|$",
               color="#4c72b0")
        ax.bar(x + 0.19, np.abs(np.nan_to_num(par)), 0.36,
               label="partial, controls log n", color="#dd8452")
        ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="gate ~0.5")
        ax.set_xticks(x)
        ax.set_xticklabels(AGGS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("|Spearman $\\rho$| with measured PPD")
        ax.set_title(f"[{dl}] G0: does UP add anything\nbeyond size?", fontsize=12)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.figures / "e12_bridge_check.png")
    plt.close(fig)
    print(f"\nwrote {args.out}, bridge_check_corr.csv and e12_bridge_check.png")


if __name__ == "__main__":
    main()
