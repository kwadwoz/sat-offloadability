#!/usr/bin/env python3
"""E21: does VIG community modularity predict the denoised PPD target?

E20 restored the instance-level ceiling to ~0.9 (mean PPD over clause
shuffles, variable numbering and phases fixed) and ruled out UP-cascade
aggregates. This tests the literature's leading structural candidate:
community modularity Q of the variable incidence graph (Ansotegui et al.),
computed statically -- no solving.

VIG: one node per variable, clauses contribute edges between every pair of
variables they mention, weighted 1/C(|c|,2) so each clause distributes unit
weight. Q from Louvain (networkx), best of a few seeds since Louvain is
stochastic. Degree CV is recorded as a cheap secondary feature.

Correlated within family (log n constant by construction) against
target_redefinition.csv's ppd_clean_mean, with the old single-run ppd_orig
alongside for the same contrast E20 drew.

Known risk, stated up front: within a homogeneous random 3-SAT family Q may
simply not vary enough to correlate with anything. A flat result here bounds
what structure can do *within* these families; it says nothing about
industrial instances, where Q genuinely varies.

Output: model/modularity_check.csv + printed correlation table.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

import networkx as nx
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / "benchmarks"


def locate(fam: str, name: str) -> Path | None:
    # some family dirs nest (uf150-645/ai/hoos/...), so glob rather than join
    hits = list((BENCH / fam).rglob(name))
    return hits[0] if hits else None


def read_cnf(path: Path) -> list[list[int]]:
    clauses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line[0] in "cp%0":
            continue
        lits = [int(x) for x in line.split() if x != "0"]
        if lits:
            clauses.append(lits)
    return clauses


def vig_features(clauses: list[list[int]]) -> dict:
    g = nx.Graph()
    for cl in clauses:
        vs = sorted({abs(l) for l in cl})
        if len(vs) < 2:
            continue
        w = 1.0 / (len(vs) * (len(vs) - 1) / 2)
        for a, b in combinations(vs, 2):
            g.add_edge(a, b, weight=g.get_edge_data(a, b, {"weight": 0.0})["weight"] + w)
    best_q = -1.0
    for seed in (0, 1, 2):  # Louvain is stochastic; keep the best partition
        comms = nx.community.louvain_communities(g, weight="weight", seed=seed)
        q = nx.community.modularity(g, comms, weight="weight")
        best_q = max(best_q, q)
    degs = [d for _, d in g.degree(weight="weight")]
    return {"modularity": best_q,
            "n_edges": g.number_of_edges(),
            "degree_cv": stdev(degs) / mean(degs)}


def main() -> None:
    targets = list(csv.DictReader(open(ROOT / "target_redefinition.csv")))
    # only instances G0 featurised; keeps the comparison set identical to E20
    bridge_insts = {r["instance"] for r in csv.DictReader(open(ROOT / "bridge_check.csv"))}

    rows = []
    for r in targets:
        if r["instance"] not in bridge_insts:
            continue
        p = locate(r["family"], r["instance"])
        if p is None:
            print(f"MISSING: {r['family']}/{r['instance']}")
            continue
        f = vig_features(read_cnf(p))
        rows.append({"family": r["family"], "instance": r["instance"],
                     "ppd_orig": float(r["ppd_orig"]),
                     "ppd_clean": float(r["ppd_clean_mean"]), **f})

    with open(ROOT / "modularity_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} instances featurised\n")
    print("== VIG features vs PPD, within family (Spearman)")
    print(f"{'family':<12} {'n':>3} {'Q_range':>15} {'feature':<10} "
          f"{'rho_orig':>9} {'rho_clean':>10} {'p_clean':>8}")
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam in sorted(by_fam):
        rs = by_fam[fam]
        if len(rs) < 8:
            continue
        qs = [r["modularity"] for r in rs]
        qrange = f"{min(qs):.3f}-{max(qs):.3f}"
        for feat in ("modularity", "degree_cv"):
            x = [r[feat] for r in rs]
            ro, _ = spearmanr(x, [r["ppd_orig"] for r in rs])
            rc, pc = spearmanr(x, [r["ppd_clean"] for r in rs])
            mark = " *" if pc < 0.05 else ""
            print(f"{fam:<12} {len(rs):>3} {qrange:>15} {feat:<10} "
                  f"{ro:9.3f} {rc:10.3f} {pc:8.4f}{mark}")


if __name__ == "__main__":
    main()
