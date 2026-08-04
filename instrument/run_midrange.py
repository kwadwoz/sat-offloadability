#!/usr/bin/env python3
"""E18: find a middle-difficulty instance set and measure full-run ppd on it.

E17 hit a squeeze. On SATLIB industrial families MiniSat searches happily but
CaDiCaL and Kissat dispose of ~half the set in preprocessing, so ppd is
undefined for them. On SAT Competition 2023 the reverse: modern solvers search,
MiniSat finishes almost nothing inside any affordable budget. Neither set
supports a cross-solver comparison of *full-run* industrial ppd.

The set we need satisfies both ends simultaneously:

  * MiniSat completes it within --gate-lim seconds  (so ppd is a full run)
  * every solver makes at least one decision        (so ppd is defined)

This walks an older competition track -- 2018 is the oldest GBD serves, and is
markedly easier for MiniSat than 2023 -- downloading one instance at a time and
using MiniSat itself as the gate. Instances MiniSat cannot finish are deleted
without running the other solvers, so the expensive case costs one download and
--gate-lim seconds rather than three solver runs.

Every row records which gate an instance failed, so the yield is auditable
rather than a silent filter.

Usage:
    python instrument/run_midrange.py --uris t_main_2018.uri \
        --out instrument/midrange.csv --gate-lim 30
"""
from __future__ import annotations

import argparse
import csv
import gzip
import lzma
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SOLVERS = {
    "minisat": {
        "cmd": lambda f, lim: ["minisat", f"-cpu-lim={lim}", f],
        "prop": re.compile(r"^propagations\s*:\s*(\d+)", re.M),
        "dec": re.compile(r"^decisions\s*:\s*(\d+)", re.M),
        "time": re.compile(r"^CPU time\s*:\s*([\d.]+)", re.M),
    },
    "cadical": {
        "cmd": lambda f, lim: ["cadical", "-t", str(lim), f],
        "prop": re.compile(r"^c propagations:\s*(\d+)", re.M),
        "dec": re.compile(r"^c decisions:\s*(\d+)", re.M),
        "time": re.compile(r"^c total process time since initialization:\s*([\d.]+)", re.M),
    },
    "kissat": {
        "cmd": lambda f, lim: ["kissat", "--statistics", f"--time={lim}", f],
        "prop": re.compile(r"^c propagations:\s*(\d+)", re.M),
        "dec": re.compile(r"^c decisions:\s*(\d+)", re.M),
        # kissat prints a rounded magnitude first ("2m", "5s") and the real
        # value last -- anchor on "seconds" or we capture the wrong field
        "time": re.compile(r"^c process-time:.*?([\d.]+)\s+seconds", re.M),
    },
}
_RESULT = {10: "SAT", 20: "UNSAT"}

FIELDS = ["instance", "solver", "result", "completed", "variables", "clauses",
          "decisions", "propagations", "props_per_decision", "cpu_time",
          "status", "gate"]

_VARS = re.compile(r"Number of variables:\s*(\d+)")
_CLAUSES = re.compile(r"Number of clauses:\s*(\d+)")


def url_id_of(s: str) -> str:
    return s.rsplit("/", 1)[-1].split("-", 1)[0]


def fetch(url: str, dest: Path, timeout: float, retries: int, backoff: float):
    req = urllib.request.Request(url, headers={"User-Agent": "sat-offloadability/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                disp = resp.headers.get("content-disposition", "")
                m = re.search(r'filename="?([^";]+)"?', disp)
                name = m.group(1) if m else url.rsplit("/", 1)[-1] + ".cnf.xz"
                blob = resp.read()
            break
        except Exception as e:                                   # noqa: BLE001
            if attempt == retries:
                raise
            wait = backoff * (2 ** attempt)
            print(f"    {type(e).__name__} -- retry {attempt+1}/{retries} in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    raw = dest / name
    raw.write_bytes(blob)
    if name.endswith((".xz", ".gz")):
        opener = lzma.open if name.endswith(".xz") else gzip.open
        out = dest / name.rsplit(".", 1)[0]
        with opener(raw, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        raw.unlink()
        return out
    return raw


def run(solver: str, cnf: Path, lim: int) -> dict:
    """One solver on one instance. `completed` means it answered within the cap,
    which is what makes the ppd a full-run value rather than a prefix."""
    spec = SOLVERS[solver]
    row = {k: "" for k in FIELDS}
    row["solver"] = solver
    try:
        p = subprocess.run(spec["cmd"](str(cnf), lim), capture_output=True,
                           text=True, timeout=lim + 120)
    except subprocess.TimeoutExpired:
        row.update(status="hard_timeout", completed=False)
        return row
    out = p.stdout
    row["result"] = _RESULT.get(p.returncode, f"rc{p.returncode}")
    row["completed"] = row["result"] in ("SAT", "UNSAT")
    mv, mc = _VARS.search(out), _CLAUSES.search(out)
    if mv:
        row["variables"] = mv.group(1)
    if mc:
        row["clauses"] = mc.group(1)
    mt = spec["time"].search(out)
    if mt:
        # needed for this solver's own R_cpu: B_req divides that solver's ppd by
        # that solver's propagation rate, never by another's
        row["cpu_time"] = mt.group(1)
    mp, md = spec["prop"].search(out), spec["dec"].search(out)
    if not (mp and md):
        # CaDiCaL omits the decisions line entirely at zero decisions, which is
        # the same condition Kissat reports as a literal 0 -- not a parse bug.
        row["status"] = "no_decisions"
        return row
    prop, dec = int(mp.group(1)), int(md.group(1))
    row["propagations"], row["decisions"] = prop, dec
    if dec == 0:
        row["status"] = "no_decisions"
    else:
        row["props_per_decision"] = f"{prop/dec:.4f}"
        row["status"] = "ok"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uris", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gate-lim", type=int, default=30,
                    help="seconds each solver gets")
    ap.add_argument("--solvers", nargs="+", default=["minisat", "cadical", "kissat"],
                    choices=list(SOLVERS))
    ap.add_argument("--gate", default="minisat",
                    help="solver that must complete before the others are run, "
                         "or 'none' to run every solver on every instance "
                         "(use 'none' when the gate solver cannot finish the set)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--net-timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--backoff", type=float, default=15.0)
    ap.add_argument("--give-up-after", type=int, default=10)
    args = ap.parse_args()

    urls = [l.strip() for l in args.uris.read_text().splitlines() if l.strip()]
    if args.limit:
        urls = urls[:args.limit]

    done: set[str] = set()
    if args.out.exists():
        with open(args.out) as fh:
            done = {url_id_of(r["instance"]) for r in csv.DictReader(fh) if r.get("instance")}
        print(f"resuming: {len(done)} instances already seen", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    fails = 0
    kept = 0
    t0 = time.perf_counter()

    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
            fh.flush()
        for i, url in enumerate(urls, 1):
            uid = url_id_of(url)
            if uid in done:
                continue
            with tempfile.TemporaryDirectory() as td:
                try:
                    cnf = fetch(url, Path(td), args.net_timeout, args.retries, args.backoff)
                except Exception as e:                            # noqa: BLE001
                    fails += 1
                    print(f"[{i}/{len(urls)}] download failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    if args.give_up_after and fails >= args.give_up_after:
                        print("aborting: too many consecutive download failures -- rerun to resume",
                              file=sys.stderr)
                        break
                    continue
                fails = 0

                rows = []
                if args.gate != "none":
                    gate = run(args.gate, cnf, args.gate_lim)
                    gate["instance"] = cnf.name
                    if gate["status"] != "ok" or not gate["completed"]:
                        # failed the gate: too hard for it, or no search at all
                        gate["gate"] = (f"{args.gate}_incomplete"
                                        if not gate["completed"] else gate["status"])
                        w.writerow(gate)
                        fh.flush()
                        done.add(uid)
                        print(f"[{i}/{len(urls)}] {cnf.name[:48]}: DROP ({gate['gate']})",
                              file=sys.stderr)
                        continue
                    gate["gate"] = "kept"
                    rows.append(gate)

                for s in args.solvers:
                    if args.gate != "none" and s == args.gate:
                        continue
                    r = run(s, cnf, args.gate_lim)
                    r["instance"] = cnf.name
                    r["gate"] = "kept"
                    rows.append(r)
                for r in rows:
                    w.writerow(r)
                fh.flush()
                done.add(uid)
                kept += 1
                ppds = " ".join(f"{r['solver'][:3]}={r['props_per_decision'] or '-'}"
                                for r in rows)
                print(f"[{i}/{len(urls)}] {cnf.name[:40]}: KEEP {ppds} "
                      f"({kept} kept, {(time.perf_counter()-t0)/60:.0f} min)", file=sys.stderr)

    print(f"wrote {args.out}: {kept} instances kept", file=sys.stderr)


if __name__ == "__main__":
    main()
