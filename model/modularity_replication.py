#!/usr/bin/env python3
"""E22: replicate E21's uf100 modularity hit on fresh instances.

E21 found VIG modularity vs denoised PPD at rho=-0.52 (p=0.008, n=25) in
uf100-430 -- right sign for the community-boundary story, strengthened by the
E20 target -- but uf150-645 showed nothing (0.02), and n=25 cannot separate a
false positive from an n-dependent effect. Preregistered call, made before
running: replication means uf100 rho <= -0.3 on a fresh sample of 100; a
collapse toward 0 closes the within-family door.

Method is E14/E20's exactly: per fresh instance, 12 clause_shuffle repeats
(PySAT Minisat22, variable numbering and phases untouched), target is the
mean PPD across repeats. Feature is E21's Louvain Q, unchanged. Instances
already used by E21 are excluded, so this is out-of-sample in the strict
sense. uf150 is rerun alongside at the same n to give the n-dependence
question a fair answer.

Each solve carries a conflict budget (CONF_BUDGET); an instance with any
capped repeat is dropped entirely, since a truncated PPD is a different
quantity from a full-run one. The first unbounded attempt ran >4 CPU-hours
on a pathological draw before being killed -- threshold uf150 run times are
heavy-tailed, and the drop rule trades a few instances for a bounded,
reportable protocol. Rows stream to the CSV as they complete.

Output: model/modularity_replication.csv + printed verdict.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

from modularity_check import BENCH, locate, read_cnf, vig_features
from ppd_stability import perturb

from pysat.solvers import Minisat22

FAMILIES = ["uf100-430", "uf150-645"]
N_FRESH = 100
REPEATS = 12
SEED = 20260803
CONF_BUDGET = 1_000_000
ROOT = Path(__file__).resolve().parent


def run_capped(clauses: list[list[int]]) -> float | None:
    """One budgeted solve; PPD on completion, None if the budget was hit."""
    s = Minisat22(bootstrap_with=clauses)
    s.conf_budget(CONF_BUDGET)
    res = s.solve_limited()
    st = s.accum_stats()
    s.delete()
    if res is None or not st["decisions"]:
        return None
    return st["propagations"] / st["decisions"]


def fresh_sample(fam: str, used: set[str], rng: random.Random) -> list[Path]:
    pool = sorted(p for p in (BENCH / fam).rglob("*.cnf") if p.name not in used)
    return rng.sample(pool, min(N_FRESH, len(pool)))


def main() -> None:
    used = {r["instance"] for r in csv.DictReader(open(ROOT / "modularity_check.csv"))}
    rng = random.Random(SEED)

    fields = ["family", "instance", "ppd_clean", "n_reps",
              "modularity", "n_edges", "degree_cv"]
    rows = []
    dropped = 0
    with open(ROOT / "modularity_replication.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for fam in FAMILIES:
            paths = fresh_sample(fam, used, rng)
            print(f"{fam}: {len(paths)} fresh instances", file=sys.stderr, flush=True)
            for k, p in enumerate(paths):
                clauses = read_cnf(p)
                n_vars = max(abs(l) for c in clauses for l in c)
                ppds = []
                for rep in range(REPEATS):
                    cl, _ = perturb(clauses, n_vars, "clause_shuffle", rng)
                    ppd = run_capped(cl)
                    if ppd is None:   # budget hit: drop the whole instance
                        ppds = None
                        break
                    ppds.append(ppd)
                if ppds is None:
                    dropped += 1
                    print(f"  DROP {p.name} (budget)", file=sys.stderr, flush=True)
                    continue
                f = vig_features(clauses)
                row = {"family": fam, "instance": p.name,
                       "ppd_clean": mean(ppds), "n_reps": len(ppds), **f}
                rows.append(row)
                w.writerow(row)
                fh.flush()
                if (k + 1) % 10 == 0:
                    print(f"  {k + 1}/{len(paths)}", file=sys.stderr, flush=True)
    print(f"dropped {dropped} instances at budget", file=sys.stderr, flush=True)

    print("\n== replication verdict (preregistered: uf100 rho <= -0.3 replicates)")
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        for feat in ("modularity", "degree_cv"):
            rho, p = spearmanr([r[feat] for r in rs], [r["ppd_clean"] for r in rs])
            print(f"{fam:<12} n={len(rs):>3} {feat:<10} rho={rho:+.3f} p={p:.4f}")


if __name__ == "__main__":
    main()
