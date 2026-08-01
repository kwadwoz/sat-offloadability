#!/usr/bin/env python3
"""Unit propagation engine over DIMACS CNF, with cascade instrumentation.

Given a formula and a partial assignment (the trail), deciding one more literal
triggers a cascade of forced assignments. This module runs that cascade to
fixpoint and reports its shape:

    size     number of literals forced, EXCLUDING the triggering decision
    depth    number of propagation rounds needed to reach fixpoint (the
             decision itself is round 0, so the hand example
             (~x1 | x2) & (~x2 | x3) with x1=true gives depth 2)
    conflict whether some clause became fully falsified

Correctness over speed: plain occurrence lists, a clause is re-examined whenever
any of its literals is falsified. No watched literals, no clause deletion.

Literals are DIMACS integers (+v true, -v false). The assignment is an array
indexed by variable holding +1 (true), -1 (false) or 0 (unassigned).

Used by cascade_spread.py (E0) and bridge_check.py (G0).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Formula:
    n_vars: int
    clauses: list[tuple[int, ...]]
    occ: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def build_occ(self) -> None:
        self.occ = {l: [] for v in range(1, self.n_vars + 1) for l in (v, -v)}
        for ci, cl in enumerate(self.clauses):
            for lit in cl:
                self.occ[lit].append(ci)


def parse_dimacs(path: str | Path) -> Formula:
    """Read a DIMACS CNF. Tolerates comments, blank lines, and clauses that run
    across multiple lines."""
    n_vars = 0
    clauses: list[tuple[int, ...]] = []
    buf: list[int] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "c%":
                continue
            if line[0] == "p":
                parts = line.split()
                n_vars = int(parts[2])
                continue
            for tok in line.split():
                try:
                    lit = int(tok)
                except ValueError:
                    continue  # trailing junk such as a lone '0' marker line
                if lit == 0:
                    if buf:
                        clauses.append(tuple(buf))
                        buf = []
                else:
                    buf.append(lit)
    if buf:
        clauses.append(tuple(buf))
    if n_vars == 0:
        n_vars = max((abs(l) for cl in clauses for l in cl), default=0)
    f = Formula(n_vars=n_vars, clauses=clauses)
    f.build_occ()
    return f


def new_assignment(f: Formula) -> list[int]:
    return [0] * (f.n_vars + 1)


def value(assign: list[int], lit: int) -> int:
    """+1 if lit is true, -1 if false, 0 if unassigned."""
    v = assign[abs(lit)]
    return v if lit > 0 else -v


@dataclass
class UPResult:
    assigned: list[int]      # every literal set, decision first
    size: int                # len(assigned) - 1, i.e. forced only
    depth: int               # propagation rounds after the decision
    conflict: bool
    conflict_clause: int | None = None


def undo(assign: list[int], assigned: list[int]) -> None:
    """Restore the trail after a trial decision."""
    for lit in assigned:
        assign[abs(lit)] = 0


def propagate(f: Formula, assign: list[int], decision_lit: int) -> UPResult:
    """Assign decision_lit on top of `assign` and propagate to fixpoint.

    Mutates `assign`; the caller restores it with undo(assign, result.assigned).
    """
    if value(assign, decision_lit) != 0:
        raise ValueError(f"literal {decision_lit} is already assigned")

    assign[abs(decision_lit)] = 1 if decision_lit > 0 else -1
    assigned = [decision_lit]
    frontier = [decision_lit]
    depth = 0
    conflict = False
    conflict_clause: int | None = None

    while frontier and not conflict:
        nxt: list[int] = []
        # A clause can only become unit or falsified when one of its literals is
        # falsified, i.e. when the negation of a newly assigned literal is in it.
        for lit in frontier:
            for ci in f.occ[-lit]:
                cl = f.clauses[ci]
                unassigned = 0
                last = 0
                satisfied = False
                for cand in cl:
                    v = value(assign, cand)
                    if v > 0:
                        satisfied = True
                        break
                    if v == 0:
                        unassigned += 1
                        last = cand
                        if unassigned > 1:
                            break
                if satisfied or unassigned > 1:
                    continue
                if unassigned == 0:
                    conflict = True
                    conflict_clause = ci
                    break
                # unit: `last` is forced
                assign[abs(last)] = 1 if last > 0 else -1
                assigned.append(last)
                nxt.append(last)
            if conflict:
                break
        if nxt:
            depth += 1
        frontier = nxt

    return UPResult(assigned=assigned, size=len(assigned) - 1, depth=depth,
                    conflict=conflict, conflict_clause=conflict_clause)


def descend(f: Formula, depth: int, rng: random.Random,
            max_restarts: int = 200) -> tuple[list[int], list[int]] | None:
    """Simulated descent: repeatedly pick a uniformly random unassigned variable,
    give it a random polarity, and propagate, until `depth` decisions have been
    made without a conflict.

    On conflict the whole descent restarts from an empty trail. Returns
    (assignment array, list of decision literals) or None if `max_restarts`
    descents all hit a conflict or ran out of free variables.
    """
    for _ in range(max_restarts):
        assign = new_assignment(f)
        free = list(range(1, f.n_vars + 1))
        rng.shuffle(free)
        decisions: list[int] = []
        ok = True
        while len(decisions) < depth:
            while free and assign[free[-1]] != 0:
                free.pop()
            if not free:
                ok = False
                break
            var = free.pop()
            lit = var if rng.random() < 0.5 else -var
            res = propagate(f, assign, lit)
            if res.conflict:
                ok = False
                break
            decisions.append(lit)
        if ok and len(decisions) == depth:
            return assign, decisions
    return None


def descend_bt(f: Formula, depth: int, rng: random.Random,
               max_conflicts: int = 20000
               ) -> tuple[list[int], list[int]] | None:
    """Simulated descent with chronological backtracking.

    `descend` throws the whole trail away on conflict, which cannot reach deep
    trails in families with a high per-decision conflict rate: at a 40% conflict
    rate, surviving 20 independent decisions has probability 0.6**20 ~ 4e-5.
    Here a conflict instead undoes just the offending decision and tries another
    candidate literal at the same level, backtracking a level only once every
    candidate at that level is exhausted.

    This still samples a random descent path -- candidates are shuffled -- but
    conditions it on reaching `depth` decisions conflict-free, which is exactly
    the point in the search a per-decision predictor would be asked about.

    Returns (assignment array, decision literals) or None if the search
    exhausts its conflict budget or the formula genuinely cannot support
    `depth` free decisions (variables run out).
    """
    assign = new_assignment(f)
    # each frame: (decision lit taken, literals it assigned, untried candidates)
    stack: list[tuple[int, list[int], list[int]]] = []
    conflicts = 0

    def candidates() -> list[int]:
        cs = [l for v in range(1, f.n_vars + 1) if assign[v] == 0
              for l in (v, -v)]
        rng.shuffle(cs)
        return cs

    todo = candidates()
    while len(stack) < depth:
        if conflicts > max_conflicts:
            return None
        # find a candidate that is still unassigned and does not conflict
        placed = False
        while todo:
            lit = todo.pop()
            if value(assign, lit) != 0:
                continue
            res = propagate(f, assign, lit)
            if res.conflict:
                undo(assign, res.assigned)
                conflicts += 1
                if conflicts > max_conflicts:
                    return None
                continue
            stack.append((lit, res.assigned, todo))
            todo = candidates()
            placed = True
            break
        if placed:
            continue
        # every candidate at this level failed -- back up one decision
        if not stack:
            return None
        _, assigned, todo = stack.pop()
        undo(assign, assigned)
        conflicts += 1

    return assign, [lit for lit, _, _ in stack]


def free_vars(f: Formula, assign: list[int]) -> list[int]:
    return [v for v in range(1, f.n_vars + 1) if assign[v] == 0]
