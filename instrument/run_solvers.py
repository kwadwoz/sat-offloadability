#!/usr/bin/env python3
"""Cross-solver ppd: measure propagations-per-decision on several CDCL solvers.

Robustness check for E4/E8: if the fivefold per-family spread in propagations
per decision reproduces across independent solvers (MiniSat, CaDiCaL, Kissat),
it is a property of the formulas, not an artifact of one solver.

Each solver reports total propagations and decisions in its own format; we parse
both and compute ppd = propagations / decisions. Instances a solver disposes of
with zero decisions (pure preprocessing) have undefined ppd and are skipped.

Usage:
    python instrument/run_solvers.py --benchmarks benchmarks/ \
        --solvers minisat cadical kissat --sample 120 --out instrument/cross_solver.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Per-solver: command builder and regexes for the propagation / decision counters.
SOLVERS = {
    "minisat": {
        "cmd": lambda f: ["minisat", f],
        "prop": re.compile(r"^propagations\s*:\s*(\d+)", re.M),
        "dec": re.compile(r"^decisions\s*:\s*(\d+)", re.M),
    },
    "cadical": {
        "cmd": lambda f: ["cadical", f],
        "prop": re.compile(r"^c propagations:\s*(\d+)", re.M),
        "dec": re.compile(r"^c decisions:\s*(\d+)", re.M),
    },
    "kissat": {
        "cmd": lambda f: ["kissat", "--statistics", f],
        "prop": re.compile(r"^c propagations:\s*(\d+)", re.M),
        "dec": re.compile(r"^c decisions:\s*(\d+)", re.M),
    },
}

FIELDS = ["solver", "instance", "family", "result", "propagations", "decisions",
          "props_per_decision", "status"]


def sanitize(cnf: Path, tmpdir: str) -> Path:
    """Strip the SATLIB '%' footer that several parsers reject."""
    dst = Path(tmpdir) / cnf.name
    with open(cnf) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.lstrip().startswith("%"):
                break
            fout.write(line)
    return dst


def run_one(solver: str, cnf: Path, family: str, timeout: float, tmpdir: str) -> dict:
    spec = SOLVERS[solver]
    row = {k: "" for k in FIELDS}
    row.update(solver=solver, instance=cnf.name, family=family)
    clean = sanitize(cnf, tmpdir)
    try:
        proc = subprocess.run(spec["cmd"](str(clean)), capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        return row
    except FileNotFoundError:
        sys.exit(f"error: solver '{solver}' not on PATH")

    out = proc.stdout
    row["result"] = {10: "SAT", 20: "UNSAT"}.get(proc.returncode, f"rc{proc.returncode}")
    mp, md = spec["prop"].search(out), spec["dec"].search(out)
    if not (mp and md):
        row["status"] = "parse_error"
        return row
    prop, dec = int(mp.group(1)), int(md.group(1))
    row["propagations"], row["decisions"] = prop, dec
    if dec > 0:
        row["props_per_decision"] = f"{prop / dec:.4f}"
        row["status"] = "ok"
    else:
        row["status"] = "zero_decisions"   # solved by preprocessing; ppd undefined
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", required=True, type=Path)
    ap.add_argument("--solvers", nargs="+", default=["minisat", "cadical", "kissat"],
                    choices=list(SOLVERS))
    ap.add_argument("--sample", type=int, default=120,
                    help="instances per family (sampled, same set across solvers)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # one fixed sample per family, shared by every solver for a fair comparison
    by_family: dict[str, list[Path]] = {}
    for p in sorted(args.benchmarks.rglob("*.cnf")):
        fam = p.relative_to(args.benchmarks).parts[0]
        by_family.setdefault(fam, []).append(p)
    sample = []
    for fam, files in by_family.items():
        rng.shuffle(files)
        sample += [(fam, f) for f in files[:args.sample]]
    print(f"{len(sample)} instances/solver across {len(by_family)} families", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for solver in args.solvers:
            ok = 0
            for fam, cnf in sample:
                row = run_one(solver, cnf, fam, args.timeout, tmp)
                w.writerow(row)
                ok += row["status"] == "ok"
            fh.flush()
            print(f"{solver}: {ok}/{len(sample)} usable", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
