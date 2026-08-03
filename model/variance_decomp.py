#!/usr/bin/env python3
"""E19: how much of the variation in ppd is family, how much is solver, how much
is leftover instance noise?

The method rests on using a family median as the a priori estimate of ppd --
provenance is free at deployment time, so if family explains most of the
variation the estimate is sound and no per-instance prediction is needed. E14
already showed the instance-level signal is weak (ICC 0.06-0.31); this asks the
complementary question and puts a number on the part that does work.

Variance is decomposed on log10(ppd), because ppd spans orders of magnitude and
the offloading model divides by it -- proportional error is what matters, not
absolute. Reported as eta^2, the share of total variance each factor explains.

Two datasets, deliberately kept separate:
  * SATLIB random/colouring families (props.csv), MiniSat only -- one solver, so
    only the family term is identifiable.
  * SATLIB industrial families (industrial_fullrun.csv), three solvers -- family
    and solver are both identifiable, but coverage is uneven because CaDiCaL and
    Kissat dispose of much of the set in preprocessing, so the solver term is
    computed on the complete-cases subset only and flagged as such.

Usage:
    python model/variance_decomp.py --props instrument/props.csv \
        --industrial instrument/industrial_fullrun.csv --out model/variance_decomp.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def eta_sq(groups: dict[str, list[float]]) -> tuple[float, int, int]:
    """Share of total variance explained by group membership (one-way eta^2)."""
    vals = [v for g in groups.values() for v in g]
    n = len(vals)
    if n < 2 or len(groups) < 2:
        return float("nan"), len(groups), n
    grand = sum(vals) / n
    ss_total = sum((v - grand) ** 2 for v in vals)
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values() if g)
    return (ss_between / ss_total if ss_total else float("nan")), len(groups), n


def spread(groups: dict[str, list[float]]) -> float:
    """Ratio of largest to smallest family median, in linear ppd units."""
    meds = []
    for g in groups.values():
        s = sorted(g)
        meds.append(10 ** s[len(s) // 2])
    return max(meds) / min(meds) if meds and min(meds) > 0 else float("nan")


def load_props(path: Path) -> dict[str, list[float]]:
    fam = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("status") == "ok" and r.get("props_per_decision"):
            v = float(r["props_per_decision"])
            if v > 0:
                fam[r["family"]].append(math.log10(v))
    return fam


def load_industrial(path: Path):
    """rows keyed by (solver, family, instance) -> log10 ppd"""
    rows = {}
    for r in csv.DictReader(open(path)):
        if r.get("status") == "ok" and r.get("props_per_decision"):
            v = float(r["props_per_decision"])
            if v > 0:
                rows[(r["solver"], r["family"], r["instance"])] = math.log10(v)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", type=Path, default=Path("instrument/props.csv"))
    ap.add_argument("--industrial", type=Path,
                    default=Path("instrument/industrial_fullrun.csv"))
    ap.add_argument("--out", type=Path, default=Path("model/variance_decomp.csv"))
    args = ap.parse_args()

    out = []

    # ---- SATLIB random/colouring, MiniSat only -------------------------------
    fam = load_props(args.props)
    e, k, n = eta_sq(fam)
    print(f"SATLIB random+colouring (minisat)   families={k:2}  n={n}")
    print(f"  family explains {e*100:5.1f}% of variance in log10 ppd"
          f"   (family median spread {spread(fam):.1f}x)")
    out.append(dict(dataset="satlib_random", factor="family", eta_sq=f"{e:.4f}",
                    groups=k, n=n))

    # ---- SATLIB industrial, three solvers ------------------------------------
    rows = load_industrial(args.industrial)
    ms = {(f, i): v for (s, f, i), v in rows.items() if s == "minisat"}
    fam_i = defaultdict(list)
    for (f, i), v in ms.items():
        fam_i[f].append(v)
    e, k, n = eta_sq(fam_i)
    print(f"\nSATLIB industrial (minisat)         families={k:2}  n={n}")
    print(f"  family explains {e*100:5.1f}% of variance in log10 ppd"
          f"   (family median spread {spread(fam_i):.1f}x)")
    out.append(dict(dataset="satlib_industrial", factor="family", eta_sq=f"{e:.4f}",
                    groups=k, n=n))

    # complete cases only: instances every solver produced a usable ppd for
    inst = defaultdict(dict)
    for (s, f, i), v in rows.items():
        inst[(f, i)][s] = v
    solvers = sorted({s for (s, _, _) in rows})
    complete = {k2: d for k2, d in inst.items() if len(d) == len(solvers)}
    print(f"\ncomplete cases (all {len(solvers)} solvers usable): "
          f"{len(complete)} of {len(inst)} instances")
    if len(complete) >= 10:
        by_solver = defaultdict(list)
        by_family = defaultdict(list)
        for (f, i), d in complete.items():
            for s, v in d.items():
                by_solver[s].append(v)
                by_family[f].append(v)
        es, ks, ns = eta_sq(by_solver)
        ef, kf, nf = eta_sq(by_family)
        print(f"  solver explains {es*100:5.1f}% of variance   (groups={ks}, n={ns})")
        print(f"  family explains {ef*100:5.1f}% of variance   (groups={kf}, n={nf})")
        out.append(dict(dataset="industrial_complete", factor="solver",
                        eta_sq=f"{es:.4f}", groups=ks, n=ns))
        out.append(dict(dataset="industrial_complete", factor="family",
                        eta_sq=f"{ef:.4f}", groups=kf, n=nf))
    else:
        print("  too few complete cases to separate solver from family")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "factor", "eta_sq", "groups", "n"])
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
