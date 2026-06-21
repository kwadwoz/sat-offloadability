#!/usr/bin/env python3
"""E4: run MiniSat with statistics across SAT benchmark families.

For each CNF instance we record the solver's reported counters and derive
propagations-per-decision -- the average amount of propagation work the solver
batches per branching decision. This per-decision work-batch is the structural
quantity the offloadability model compares against W_min.

Output: one CSV row per instance.

Usage:
    python instrument/run_stats.py --benchmarks benchmarks/ --out instrument/props.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# MiniSat reports "name : value (...)". Capture the first integer after the colon.
_PATTERNS = {
    "restarts": re.compile(r"^restarts\s*:\s*(\d+)"),
    "conflicts": re.compile(r"^conflicts\s*:\s*(\d+)"),
    "decisions": re.compile(r"^decisions\s*:\s*(\d+)"),
    "propagations": re.compile(r"^propagations\s*:\s*(\d+)"),
    "cpu_time": re.compile(r"^CPU time\s*:\s*([\d.]+)"),
}
_VARS = re.compile(r"Number of variables:\s*(\d+)")
_CLAUSES = re.compile(r"Number of clauses:\s*(\d+)")

# MiniSat exit codes: 10 = SAT, 20 = UNSAT, 0 = indeterminate (e.g. timeout).
_RESULT = {10: "SAT", 20: "UNSAT", 0: "INDET"}

FIELDS = [
    "instance", "family", "status", "result",
    "variables", "clauses",
    "decisions", "propagations", "conflicts", "restarts",
    "props_per_decision", "cpu_time",
]


def _opener(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r")


def _prepare(path: Path, tmpdir: str) -> Path:
    """Return a sanitized plain-text CNF in tmpdir.

    Decompresses .gz and strips the SATLIB DIMACS footer: many SATLIB files end
    with a "%" line followed by a stray "0", which MiniSat's parser rejects.
    We truncate at the first line that begins with "%".
    """
    dst = Path(tmpdir) / (path.with_suffix("").name if path.suffix == ".gz" else path.name)
    with _opener(path) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            if line.lstrip().startswith("%"):
                break
            f_out.write(line)
    return dst


def run_one(cnf: Path, family: str, minisat: str, timeout: float, tmpdir: str) -> dict:
    row = {k: "" for k in FIELDS}
    row["instance"] = cnf.name
    row["family"] = family
    try:
        plain = _prepare(cnf, tmpdir)
    except OSError as e:
        row["status"] = f"prepare_error:{e}"
        return row

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [minisat, str(plain)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["result"] = "TIMEOUT"
        row["cpu_time"] = f"{time.perf_counter() - start:.3f}"
        return row
    except FileNotFoundError:
        sys.exit(f"error: solver '{minisat}' not found on PATH")

    out = proc.stdout
    stats = {}
    for line in out.splitlines():
        line = line.strip()
        for key, pat in _PATTERNS.items():
            m = pat.match(line)
            if m:
                stats[key] = m.group(1)
    mv, mc = _VARS.search(out), _CLAUSES.search(out)

    row["result"] = _RESULT.get(proc.returncode, f"rc{proc.returncode}")
    row["variables"] = mv.group(1) if mv else ""
    row["clauses"] = mc.group(1) if mc else ""
    row["decisions"] = stats.get("decisions", "")
    row["propagations"] = stats.get("propagations", "")
    row["conflicts"] = stats.get("conflicts", "")
    row["restarts"] = stats.get("restarts", "")
    row["cpu_time"] = stats.get("cpu_time", "")

    try:
        dec = int(stats["decisions"])
        prop = int(stats["propagations"])
        row["props_per_decision"] = f"{prop / dec:.4f}" if dec > 0 else ""
        row["status"] = "ok"
    except (KeyError, ValueError):
        row["status"] = "parse_error"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", required=True, type=Path,
                    help="directory of CNF families; immediate subdirs are family names")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--minisat", default="minisat")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-instance wall-clock timeout in seconds")
    args = ap.parse_args()

    if not args.benchmarks.is_dir():
        sys.exit(f"error: --benchmarks {args.benchmarks} is not a directory")

    cnfs = sorted(
        p for p in args.benchmarks.rglob("*")
        if p.is_file() and (p.suffix == ".cnf" or p.name.endswith(".cnf.gz"))
    )
    if not cnfs:
        sys.exit(f"error: no .cnf/.cnf.gz files under {args.benchmarks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"found {len(cnfs)} instances; timeout={args.timeout}s", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir, open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, cnf in enumerate(cnfs, 1):
            # family = first path component below the benchmarks root, else "root"
            rel = cnf.relative_to(args.benchmarks).parts
            family = rel[0] if len(rel) > 1 else "root"
            row = run_one(cnf, family, args.minisat, args.timeout, tmpdir)
            writer.writerow(row)
            f.flush()
            print(f"[{i}/{len(cnfs)}] {family}/{cnf.name}: "
                  f"{row['status']} ppd={row['props_per_decision']}", file=sys.stderr)

    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
