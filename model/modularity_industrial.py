#!/usr/bin/env python3
"""E24: modularity vs full-run ppd on the E23 industrial set.

E21/E22 killed modularity within random 3-SAT, where Q barely varies
(uf100 range 0.167-0.183). This is the test it was actually proposed for:
the E23 families span real structural variety, so Q should range widely.
Two questions, in order of importance:

  1. between-family: does family-median Q track family-median ppd?  If Q is
     the mechanism behind the 265x family spread, provenance could be
     replaced (or backed up) by a computed feature when provenance is
     unavailable.
  2. within-family: any per-instance signal left after E22's negative?

CNFs were deleted after E23 measurement, so this re-downloads by hash
(same GBD host). Guards for industrial scale: clauses longer than MAX_CLAUSE
are skipped in the VIG (pairwise expansion is quadratic), files larger than
MAX_MB are dropped and recorded. Rows stream to the CSV; resume keys on hash.

Output: model/modularity_industrial.csv + printed correlations.
"""
from __future__ import annotations

import csv
import random
import signal
import sys
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev

import igraph as ig
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instrument"))
from run_midrange import fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent
MAX_CLAUSE = 50      # skip longer clauses in the VIG (quadratic expansion)
MAX_MB = 64          # skip files larger than this, recorded as dropped
MAX_EDGES = 10_000_000  # drop instances whose VIG exceeds this (Louvain cost)
FEAT_TIMEOUT = 120   # wall seconds per featurisation; structure caps cannot
                     # predict Louvain's pathological cases (one planning
                     # instance under the edge cap ran 4h before this existed)
SOLVER = "minisat"   # ppd target: MiniSat full-run values from E23


def read_cnf(path: Path) -> list[list[int]]:
    clauses = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "cp%0":
                continue
            lits = [int(x) for x in line.split() if x != "0"]
            if lits:
                clauses.append(lits)
    return clauses


def vig_q(clauses: list[list[int]]) -> tuple[float, float, int] | None:
    """VIG modularity via igraph's C Louvain, or None past MAX_EDGES."""
    weights: dict[tuple[int, int], float] = {}
    skipped = 0
    for cl in clauses:
        vs = sorted({abs(l) for l in cl})
        if len(vs) < 2:
            continue
        if len(vs) > MAX_CLAUSE:
            skipped += 1
            continue
        w = 1.0 / (len(vs) * (len(vs) - 1) / 2)
        for e in combinations(vs, 2):
            weights[e] = weights.get(e, 0.0) + w
        if len(weights) > MAX_EDGES:
            return None
    ids = {v: i for i, v in enumerate({v for e in weights for v in e})}
    g = ig.Graph(n=len(ids),
                 edges=[(ids[a], ids[b]) for a, b in weights],
                 edge_attrs={"weight": list(weights.values())})
    best_q = -1.0
    for seed in (0, 1, 2):
        ig.set_random_number_generator(random.Random(seed))
        part = g.community_multilevel(weights="weight")
        best_q = max(best_q, g.modularity(part, weights="weight"))
    degs = g.strength(weights="weight")
    dcv = stdev(degs) / mean(degs) if len(degs) > 1 else 0.0
    return best_q, dcv, skipped


def main() -> None:
    # (hash, family, ppd) for completed MiniSat rows of E23
    targets = {}
    with open(ROOT.parent / "instrument" / "industrial_families.csv") as fh:
        for r in csv.DictReader(fh):
            if (r["solver"] == SOLVER and r["completed"] == "True"
                    and r["props_per_decision"]):
                targets[r["hash"]] = (r["family"], float(r["props_per_decision"]))

    out_path = ROOT / "modularity_industrial.csv"
    done = set()
    if out_path.exists():
        done = {r["hash"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done)} done", file=sys.stderr, flush=True)

    fields = ["hash", "family", "ppd", "modularity", "degree_cv",
              "n_clauses", "skipped_long"]
    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        with tempfile.TemporaryDirectory() as td:
            for i, (h, (fam, ppd)) in enumerate(sorted(targets.items()), 1):
                if h in done:
                    continue
                try:
                    cnf = fetch(f"https://benchmark-database.de/file/{h}",
                                Path(td), timeout=180.0, retries=4, backoff=5.0)
                except Exception as e:
                    print(f"[{i}] {fam}: FETCH FAIL {e}", file=sys.stderr, flush=True)
                    continue
                if cnf.stat().st_size > MAX_MB * 1024 * 1024:
                    print(f"[{i}] {fam}/{cnf.name[:40]}: DROP size",
                          file=sys.stderr, flush=True)
                    cnf.unlink()
                    continue
                clauses = read_cnf(cnf)
                cnf.unlink()
                signal.signal(signal.SIGALRM,
                              lambda *_: (_ for _ in ()).throw(TimeoutError))
                signal.alarm(FEAT_TIMEOUT)
                try:
                    res = vig_q(clauses)
                except TimeoutError:
                    res = None
                finally:
                    signal.alarm(0)
                if res is None:
                    print(f"[{i}] {fam}: DROP feat (edges>{MAX_EDGES} or "
                          f">{FEAT_TIMEOUT}s)", file=sys.stderr, flush=True)
                    continue
                q, dcv, skipped = res
                w.writerow({"hash": h, "family": fam, "ppd": ppd,
                            "modularity": f"{q:.4f}", "degree_cv": f"{dcv:.4f}",
                            "n_clauses": len(clauses), "skipped_long": skipped})
                fh.flush()
                print(f"[{i}/{len(targets)}] {fam}/{cnf.name[:36]}: Q={q:.3f}",
                      file=sys.stderr, flush=True)

    # -- analysis ------------------------------------------------------------
    rows = list(csv.DictReader(open(out_path)))
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append((float(r["modularity"]), float(r["ppd"])))

    print("\n== 1. between-family: median Q vs median ppd")
    fams = sorted(by_fam)
    fq = [median(q for q, _ in by_fam[f]) for f in fams]
    fp = [median(p for _, p in by_fam[f]) for f in fams]
    rho, p = spearmanr(fq, fp)
    for f, q, pp in sorted(zip(fams, fq, fp), key=lambda t: -t[2]):
        print(f"  {f:<24} Q={q:.3f}  ppd={pp:9.1f}")
    print(f"  rho={rho:+.3f} p={p:.4f} (n={len(fams)} families)")

    print("\n== 2. within-family: Q vs ppd (families with n>=10)")
    for f in fams:
        pts = by_fam[f]
        if len(pts) < 10:
            continue
        rho, p = spearmanr([q for q, _ in pts], [pp for _, pp in pts])
        mark = " *" if p < 0.05 else ""
        print(f"  {f:<24} n={len(pts):>2} rho={rho:+.3f} p={p:.4f}{mark}")


if __name__ == "__main__":
    main()
