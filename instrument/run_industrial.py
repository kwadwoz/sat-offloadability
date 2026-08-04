#!/usr/bin/env python3
"""E23: family-labeled industrial ppd -- the set E18's track could not be.

The 2018 competition track turned out to be ~80% singleton families, useless
for family-level decomposition. This selects by family instead, from the GBD
metadata database (downloaded once as sqlite): application families with
minisat1m=yes -- GBD's own precomputed "MiniSat solves it inside a minute",
i.e. E18's gate, already run for us -- sampled K per family.

Measures MiniSat, CaDiCaL, Kissat at --cpu-lim seconds each (gated instances
mostly complete), streaming rows with the ground-truth family label. Resume
is keyed on (hash, solver): rerunning the same command skips finished work.

Usage:
    python instrument/run_industrial.py --meta scratchpad/meta.db \
        --out instrument/industrial_families.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_midrange import SOLVERS, run, fetch  # noqa: E402

FAMILIES = [
    "hardware-verification", "planning", "coloring", "cryptography",
    "quasigroup-completion", "miter", "bitvector", "design-debugging",
    "scheduling", "software-verification", "prime-factoring", "fpga-routing",
    "diagnosis", "testpattern-generation",
]
PER_FAMILY = 15
SEED = 20260803
FIELDS = ["hash", "family", "instance", "solver", "result", "completed",
          "variables", "clauses", "decisions", "propagations",
          "props_per_decision", "cpu_time", "status"]


def sample(meta: Path) -> list[tuple[str, str, str]]:
    con = sqlite3.connect(meta)
    rng = random.Random(SEED)
    picks = []
    for fam in FAMILIES:
        rows = con.execute(
            "SELECT hash, filename FROM features "
            "WHERE family=? AND minisat1m='yes'", (fam,)).fetchall()
        for h, fn in rng.sample(rows, min(PER_FAMILY, len(rows))):
            picks.append((h, fam, fn))
    con.close()
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cpu-lim", type=int, default=60)
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS),
                    choices=list(SOLVERS))
    args = ap.parse_args()

    done: set[tuple[str, str]] = set()
    if args.out.exists():
        with open(args.out) as fh:
            done = {(r["hash"], r["solver"]) for r in csv.DictReader(fh)}
        print(f"resuming: {len(done)} rows already measured", file=sys.stderr)

    picks = sample(args.meta)
    new = args.out.stat().st_size == 0 if args.out.exists() else True
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        with tempfile.TemporaryDirectory() as td:
            for i, (h, fam, fn) in enumerate(picks, 1):
                todo = [s for s in args.solvers if (h, s) not in done]
                if not todo:
                    continue
                url = f"https://benchmark-database.de/file/{h}"
                try:
                    cnf = fetch(url, Path(td), timeout=180.0,
                                retries=4, backoff=5.0)
                except Exception as e:
                    print(f"[{i}/{len(picks)}] {fam}/{fn}: FETCH FAIL {e}",
                          file=sys.stderr, flush=True)
                    continue
                for s in todo:
                    row = run(s, cnf, args.cpu_lim)
                    row.update(hash=h, family=fam, instance=cnf.name)
                    w.writerow(row)
                    fh.flush()
                cnf.unlink(missing_ok=True)
                print(f"[{i}/{len(picks)}] {fam}/{cnf.name[:40]}: done",
                      file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
