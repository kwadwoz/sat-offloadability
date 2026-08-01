#!/usr/bin/env python3
"""Step 2: validate the UP engine three ways before anything downstream uses it.

(a) hand-checkable chain      (~x1 | x2) & (~x2 | x3), x1=true -> size 2, depth 2
(b) expected-zero  on uf100-430: pure 3-CNF, empty trail, every cascade size 0
(c) expected-nonzero on flat30-60: has binary clauses, some cascades nonzero

Exits nonzero and reports if any check fails.

Usage:
    python model/test_up_engine.py --benchmarks benchmarks/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from up_engine import (Formula, new_assignment, parse_dimacs, propagate, undo)


def check_hand_example() -> list[str]:
    f = Formula(n_vars=3, clauses=[(-1, 2), (-2, 3)])
    f.build_occ()
    assign = new_assignment(f)
    r = propagate(f, assign, 1)
    fails = []
    if r.size != 2:
        fails.append(f"(a) cascade size {r.size}, expected 2")
    if r.depth != 2:
        fails.append(f"(a) cascade depth {r.depth}, expected 2")
    if r.conflict:
        fails.append("(a) reported a conflict, expected none")
    if sorted(r.assigned) != [1, 2, 3]:
        fails.append(f"(a) assigned {r.assigned}, expected x1,x2,x3 all true")
    print(f"(a) hand example: size={r.size} depth={r.depth} "
          f"conflict={r.conflict} assigned={r.assigned}")
    return fails


def sweep_all_literals(f: Formula) -> list[int]:
    """Cascade size from every literal, on an empty trail."""
    sizes = []
    assign = new_assignment(f)
    for v in range(1, f.n_vars + 1):
        for lit in (v, -v):
            r = propagate(f, assign, lit)
            sizes.append(r.size)
            undo(assign, r.assigned)
    return sizes


def min_clause_len(f: Formula) -> int:
    return min(len(c) for c in f.clauses)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", type=Path, default=Path("benchmarks"))
    args = ap.parse_args()

    fails = check_hand_example()

    # (b) pure 3-CNF: no clause can go unit until two of its three literals are
    # falsified, so a single decision on an empty trail forces nothing.
    p = sorted((args.benchmarks / "uf100-430").glob("*.cnf"))[0]
    f = parse_dimacs(p)
    sizes = sweep_all_literals(f)
    nz = [s for s in sizes if s != 0]
    print(f"(b) {p.name}: {f.n_vars} vars, {f.n_clauses} clauses, "
          f"min clause len {min_clause_len(f)}; swept {len(sizes)} literals, "
          f"{len(nz)} nonzero cascades")
    if nz:
        fails.append(f"(b) expected all-zero cascades, got {len(nz)} nonzero "
                     f"(max {max(nz)}) -- engine forces literals it should not")

    # (c) graph-colouring encoding contains binary clauses, so cascades exist.
    p = sorted((args.benchmarks / "flat30-60").glob("*.cnf"))[0]
    f = parse_dimacs(p)
    sizes = sweep_all_literals(f)
    nz = [s for s in sizes if s != 0]
    print(f"(c) {p.name}: {f.n_vars} vars, {f.n_clauses} clauses, "
          f"min clause len {min_clause_len(f)}; swept {len(sizes)} literals, "
          f"{len(nz)} nonzero cascades, max size {max(sizes)}")
    if not nz:
        fails.append("(c) expected some nonzero cascades, got all zero -- "
                     "engine is failing to propagate")

    print()
    if fails:
        print("FAIL -- do not proceed:")
        for m in fails:
            print(f"  {m}")
        return 1
    print("all three validation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
