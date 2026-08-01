#!/usr/bin/env python3
"""E0 (gate 1): how much do unit-propagation cascades vary within one formula?

If every literal at a given point in the search propagates about the same amount,
there is nothing for a per-decision predictor to predict. This measures the
spread of cascade size across all candidate next decisions at a fixed point in a
simulated descent, and checks whether that spread is explained by four cheap
"shortcut" features -- if one of them already tracks cascade size, a lookup table
beats any learned model.

Method, per instance and per decision depth d:
  * simulated descent: pick a uniformly random unassigned variable, assign a
    random polarity, propagate, repeat to depth d; restart on conflict.
  * at depth d, try both polarities of every still-unassigned variable as the
    next decision, propagate from the current trail, record cascade size, depth
    and conflict, then restore the trail.
  * 5 independent trails per (instance, depth) so nothing is an artifact of one
    descent path.

Shortcut features for a trial literal l (step 4):
  occ_l         |occurrences of l|
  occ_neg_l     |occurrences of ~l|      (clauses l's negation can shrink)
  near_unit     unsatisfied clauses containing ~l with exactly 2 unassigned
                literals -- one more falsification away from going unit
  two_hop       distinct literals sharing a clause with l (2-hop neighbourhood
                in the literal-clause bipartite graph)

Pure offline analysis: no solving, no hardware.

Usage:
    python model/cascade_spread.py --benchmarks benchmarks/ \
        --figures figures/ --out model/cascade_spread.csv
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

from up_engine import (Formula, descend_bt, free_vars, parse_dimacs, propagate,
                       undo, value)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

FAMILIES = ["flat30-60", "flat50-115", "uf100-430"]
DEPTHS = [5, 10, 20, 40]
TRAILS = 5
N_INSTANCES = 20
SHORTCUTS = ["occ_l", "occ_neg_l", "near_unit", "two_hop"]


def two_hop_sizes(f: Formula) -> dict[int, int]:
    """Distinct literals sharing a clause with each literal (excluding itself)."""
    out = {}
    for v in range(1, f.n_vars + 1):
        for lit in (v, -v):
            nb = set()
            for ci in f.occ[lit]:
                nb.update(f.clauses[ci])
            nb.discard(lit)
            out[lit] = len(nb)
    return out


def near_unit_count(f: Formula, assign: list[int], lit: int) -> int:
    """Unsatisfied clauses containing ~lit that currently have exactly two
    unassigned literals."""
    n = 0
    for ci in f.occ[-lit]:
        unassigned = 0
        satisfied = False
        for cand in f.clauses[ci]:
            v = value(assign, cand)
            if v > 0:
                satisfied = True
                break
            if v == 0:
                unassigned += 1
        if not satisfied and unassigned == 2:
            n += 1
    return n


def probe(f: Formula, assign: list[int], twohop: dict[int, int]) -> list[dict]:
    """Try both polarities of every unassigned variable as the next decision."""
    rows = []
    for var in free_vars(f, assign):
        for lit in (var, -var):
            nu = near_unit_count(f, assign, lit)
            r = propagate(f, assign, lit)
            undo(assign, r.assigned)
            rows.append({
                "lit": lit,
                "cascade_size": r.size,
                "cascade_depth": r.depth,
                "conflict": int(r.conflict),
                "occ_l": len(f.occ[lit]),
                "occ_neg_l": len(f.occ[-lit]),
                "near_unit": nu,
                "two_hop": twohop[lit],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    ap.add_argument("--out", type=Path, default=Path("model/cascade_spread.csv"))
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    records: list[pd.DataFrame] = []
    per_instance: list[dict] = []
    failed_descents = 0

    for fam in FAMILIES:
        paths = sorted((args.benchmarks / fam).glob("*.cnf"))[:N_INSTANCES]
        print(f"{fam}: {len(paths)} instances")
        for pi, path in enumerate(paths):
            f = parse_dimacs(path)
            twohop = two_hop_sizes(f)
            for d in DEPTHS:
                if d >= f.n_vars:
                    continue
                trial_rows = []
                reached = 0
                for t in range(TRAILS):
                    rng = random.Random(args.seed + 1000 * pi + 17 * d + t)
                    got = descend_bt(f, d, rng)
                    if got is None:
                        failed_descents += 1
                        continue
                    reached += 1
                    assign, _ = got
                    rows = probe(f, assign, twohop)
                    for r in rows:
                        r["trail"] = t
                    trial_rows.extend(rows)
                if not trial_rows:
                    continue
                g = pd.DataFrame(trial_rows)
                g["family"] = fam
                g["instance"] = path.name
                g["depth"] = d
                records.append(g)

                # step 4: shortcut correlations within this (instance, depth)
                rec = {"family": fam, "instance": path.name, "depth": d,
                       "trails_reached": reached, "trails_attempted": TRAILS,
                       "n_trials": len(g), "mean": g.cascade_size.mean(),
                       "std": g.cascade_size.std(),
                       "cv": g.cascade_size.std() / g.cascade_size.mean()
                             if g.cascade_size.mean() > 0 else np.nan}
                for feat in SHORTCUTS:
                    if g[feat].nunique() > 1 and g.cascade_size.nunique() > 1:
                        rho, _ = spearmanr(g[feat], g.cascade_size)
                    else:
                        rho = np.nan
                    rec[f"rho_{feat}"] = rho
                per_instance.append(rec)
            print(f"  [{pi + 1}/{len(paths)}] {path.name}", flush=True)

    df = pd.concat(records, ignore_index=True)
    pins = pd.DataFrame(per_instance)

    # ---- step 3 table: distribution of cascade size per (family, depth) ------
    rows = []
    for (fam, d), g in df.groupby(["family", "depth"]):
        cs = g.cascade_size
        p50, p90, p99 = np.percentile(cs, [50, 90, 99])
        sc = pins[(pins.family == fam) & (pins.depth == d)]
        rec = {
            "family": fam, "depth": d,
            "n_instances": g.instance.nunique(), "n_trials": len(g),
            "reach_rate": round(sc.trails_reached.sum()
                                / max(1, sc.trails_attempted.sum()), 3),
            "mean": round(cs.mean(), 3), "median": round(cs.median(), 3),
            "std": round(cs.std(), 3),
            "cv": round(cs.std() / cs.mean(), 3) if cs.mean() > 0 else np.nan,
            "min": int(cs.min()), "max": int(cs.max()),
            "p50": round(p50, 2), "p90": round(p90, 2), "p99": round(p99, 2),
            "p90_over_p50": round(p90 / p50, 2) if p50 > 0 else np.inf,
            "frac_zero": round((cs == 0).mean(), 3),
            "frac_conflict": round(g.conflict.mean(), 3),
            "mean_cascade_depth": round(g.cascade_depth.mean(), 2),
        }
        for feat in SHORTCUTS:
            rec[f"rho_{feat}"] = round(sc[f"rho_{feat}"].median(), 3)
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values(["family", "depth"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    pins.to_csv(args.out.with_name("cascade_spread_per_instance.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nE0 -- cascade size distribution (pooled over instances and trails)")
    print(table[["family", "depth", "n_instances", "reach_rate", "n_trials",
                 "mean", "median", "std", "cv", "min", "max", "p50", "p90",
                 "p99", "p90_over_p50", "frac_zero",
                 "frac_conflict"]].to_string(index=False))
    print("\nStep 4 -- shortcut correlations with cascade size "
          "(Spearman, median over instances within family+depth)")
    print(table[["family", "depth"] + [f"rho_{s}" for s in SHORTCUTS]
                ].to_string(index=False))
    if failed_descents:
        print(f"\n({failed_descents} descents abandoned after max restarts)")

    # ---- figure -------------------------------------------------------------
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
    colors = dict(zip(FAMILIES, plt.cm.tab10([0, 1, 3])))

    ax = axes[0]
    width = 0.26
    for i, fam in enumerate(FAMILIES):
        pos = [j + (i - 1) * width for j, _ in enumerate(DEPTHS)]
        data = [df[(df.family == fam) & (df.depth == d)].cascade_size.values
                for d in DEPTHS]
        data = [x if len(x) else np.array([np.nan]) for x in data]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set(facecolor=colors[fam], alpha=0.65, edgecolor="black")
        ax.plot([], [], color=colors[fam], lw=6, alpha=0.65, label=fam)
    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels([str(d) for d in DEPTHS])
    ax.set_xlabel("Decision depth  d")
    ax.set_ylabel("Cascade size (literals forced)")
    ax.set_title("Spread of cascade size across candidate decisions")
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for fam in FAMILIES:
        t = table[table.family == fam].sort_values("depth")
        ax.plot(t.depth, t.cv, "o-", color=colors[fam], label=f"{fam} CV")
        ax.plot(t.depth, t.p90_over_p50, "s--", color=colors[fam], alpha=0.6,
                label=f"{fam} p90/p50")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.axhline(2.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("Decision depth  d")
    ax.set_ylabel("Dispersion")
    ax.set_title("Is the spread wide?  (gates: CV > 0.5, p90/p50 > 2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    x = np.arange(len(SHORTCUTS))
    for i, fam in enumerate(FAMILIES):
        t = table[(table.family == fam) & (table.depth == 20)]
        if t.empty:
            continue
        vals = [abs(t[f"rho_{s}"].iloc[0]) for s in SHORTCUTS]
        ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=colors[fam],
               alpha=0.8, label=fam)
    ax.axhline(0.9, color="crimson", ls="--", lw=1.2, label="lookup-table threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORTCUTS, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("|Spearman $\\rho$| with cascade size")
    ax.set_title("Do cheap features already explain it?  (d = 20)")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.figures / "e11_cascade_spread.png")
    plt.close(fig)
    print(f"\nwrote {args.out}, cascade_spread_per_instance.csv "
          f"and e11_cascade_spread.png")


if __name__ == "__main__":
    main()
