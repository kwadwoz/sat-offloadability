#!/usr/bin/env python3
"""E16: measure propagations-per-decision on SAT Competition 2023 main track.

Everything in props.csv is SATLIB random 3-SAT and graph colouring, n <= 250,
solving in under half a second. The obvious reviewer question is whether any of
it holds on real industrial instances. This measures the same quantity on the
399-instance SAT Competition 2023 main track.

Two things make this practical:

  * MiniSat's own -cpu-lim prints the FULL statistics block and exits 0
    (INDETERMINATE) when the budget runs out. So propagations and decisions --
    and therefore ppd -- are recorded for every instance, finished or not.
    Verified: ibm-2004-03-k70 gives ppd 118.0 complete (1.36s) vs 108.5
    truncated at 1s.
  * Instances are served individually, so we download -> decompress -> measure
    -> delete one at a time and never hold more than one instance on disk.
    Necessary here: the benchmark set does not fit in available space.

`completed` marks instances that finished inside the budget; their ppd is a true
full-run value. For the rest ppd is "ppd over the first --cpu-lim seconds of
search", which is a different (still well-defined) quantity -- keep the two
apart when analysing, and use the completed subset to calibrate how far
truncated ppd sits from full ppd.

Resumable: rows already present in --out are skipped, so the sweep can be
interrupted and restarted.

Usage:
    curl -L "https://benchmark-database.de/getinstances?track=main_2023" -o main_2023.uri
    python instrument/run_competition.py --uris main_2023.uri \
        --out instrument/props_sc2023.csv --cpu-lim 10
"""
from __future__ import annotations

import argparse
import csv
import lzma
import gzip
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# MiniSat's stats block, same patterns the SATLIB harness uses.
_PATTERNS = {
    "restarts": re.compile(r"^restarts\s*:\s*(\d+)"),
    "conflicts": re.compile(r"^conflicts\s*:\s*(\d+)"),
    "decisions": re.compile(r"^decisions\s*:\s*(\d+)"),
    "propagations": re.compile(r"^propagations\s*:\s*(\d+)"),
    "cpu_time": re.compile(r"^CPU time\s*:\s*([\d.]+)"),
}
_VARS = re.compile(r"Number of variables:\s*(\d+)")
_CLAUSES = re.compile(r"Number of clauses:\s*(\d+)")
_RESULT = {10: "SAT", 20: "UNSAT", 0: "INDET"}

FIELDS = [
    "instance", "family", "status", "result", "completed",
    "variables", "clauses",
    "decisions", "propagations", "conflicts", "restarts",
    "props_per_decision", "cpu_time", "download_bytes", "wall_total",
    "error",
]


def url_id_of(s: str) -> str:
    """The database's id for an instance.

    Served filenames are '<id>-<original name>.cnf', so the id can be recovered
    from either a URL or an already-recorded instance name. That is what makes
    resume cheap: without it the id is only known after downloading."""
    return s.rsplit("/", 1)[-1].split("-", 1)[0]


def fetch(url: str, dest_dir: Path, timeout: float,
          retries: int = 4, backoff: float = 15.0) -> tuple[Path, int]:
    """Download one instance, returning the decompressed CNF path and the
    compressed size in bytes.

    The server hands out 504s under load, and a bare failure here silently
    drops the instance from the sweep. Retry with exponential backoff so a
    transient outage costs time rather than coverage."""
    req = urllib.request.Request(url, headers={"User-Agent": "sat-offloadability/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                disp = resp.headers.get("content-disposition", "")
                m = re.search(r'filename="?([^";]+)"?', disp)
                name = m.group(1) if m else url.rsplit("/", 1)[-1] + ".cnf.xz"
                blob = resp.read()
            break
        except Exception as e:                              # noqa: BLE001
            if attempt == retries:
                raise
            wait = backoff * (2 ** attempt)
            print(f"    {type(e).__name__}: {e} -- retry {attempt + 1}/{retries} "
                  f"in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raw = dest_dir / name
    raw.write_bytes(blob)

    if name.endswith(".xz"):
        out = dest_dir / name[:-3]
        with lzma.open(raw, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        raw.unlink()
    elif name.endswith(".gz"):
        out = dest_dir / name[:-3]
        with gzip.open(raw, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        raw.unlink()
    else:
        out = raw
    return out, len(blob)


def measure(cnf: Path, minisat: str, cpu_lim: int) -> dict:
    """Run MiniSat under its own CPU limit and parse the stats block."""
    row: dict = {k: "" for k in FIELDS}
    proc = subprocess.run(
        [minisat, f"-cpu-lim={cpu_lim}", str(cnf)],
        capture_output=True, text=True,
        # generous backstop: parsing a multi-GB CNF can outlast the CPU limit
        timeout=cpu_lim + 300,
    )
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
    row["completed"] = row["result"] in ("SAT", "UNSAT")
    row["variables"] = mv.group(1) if mv else ""
    row["clauses"] = mc.group(1) if mc else ""
    for k in ("decisions", "propagations", "conflicts", "restarts", "cpu_time"):
        row[k] = stats.get(k, "")
    try:
        dec, prop = int(stats["decisions"]), int(stats["propagations"])
        row["props_per_decision"] = f"{prop / dec:.4f}" if dec > 0 else ""
        row["status"] = "ok" if dec > 0 else "no_decisions"
    except (KeyError, ValueError):
        # No stats block: usually the solver died before printing one (out of
        # memory on the multi-million-variable instances). Keep its last words,
        # otherwise the cause is unrecoverable from the CSV alone.
        row["status"] = "parse_error"
        tail = " ".join((proc.stderr or out).split())[-200:]
        row["error"] = tail
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uris", required=True, type=Path,
                    help="file of instance URLs, one per line")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cpu-lim", type=int, default=10,
                    help="MiniSat CPU seconds per instance")
    ap.add_argument("--minisat", default="minisat")
    ap.add_argument("--family", default="sc2023-main")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N instances (0 = all)")
    ap.add_argument("--net-timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=4,
                    help="download attempts per instance beyond the first")
    ap.add_argument("--backoff", type=float, default=15.0,
                    help="seconds before the first retry, doubling thereafter")
    ap.add_argument("--give-up-after", type=int, default=10,
                    help="abort once this many instances fail back to back "
                         "(0 = never); the sweep is resumable, so stopping "
                         "beats spinning through the rest of the list")
    args = ap.parse_args()

    urls = [l.strip() for l in args.uris.read_text().splitlines() if l.strip()]
    if args.limit:
        urls = urls[:args.limit]

    done: set[str] = set()
    if args.out.exists():
        with open(args.out) as fh:
            done = {url_id_of(r["instance"]) for r in csv.DictReader(fh)
                    if r.get("instance")}
        print(f"resuming: {len(done)} instances already measured", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    t_start = time.perf_counter()
    consecutive_failures = 0

    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()
        for i, url in enumerate(urls, 1):
            uid = url_id_of(url)
            if uid in done:
                print(f"[{i}/{len(urls)}] {uid}: skip (done)", file=sys.stderr)
                continue
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                t0 = time.perf_counter()
                try:
                    cnf, nbytes = fetch(url, tdp, args.net_timeout,
                                        args.retries, args.backoff)
                except Exception as e:                      # noqa: BLE001
                    consecutive_failures += 1
                    print(f"[{i}/{len(urls)}] download failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    if args.give_up_after and consecutive_failures >= args.give_up_after:
                        print(f"aborting: {consecutive_failures} consecutive download "
                              f"failures -- rerun to resume", file=sys.stderr)
                        break
                    continue
                consecutive_failures = 0
                name = cnf.name
                try:
                    row = measure(cnf, args.minisat, args.cpu_lim)
                except subprocess.TimeoutExpired:
                    row = {k: "" for k in FIELDS}
                    row["status"] = "hard_timeout"
                    row["result"] = "TIMEOUT"
                    row["completed"] = False
                except FileNotFoundError:
                    sys.exit(f"error: solver '{args.minisat}' not found on PATH")
                row["instance"] = name
                row["family"] = args.family
                row["download_bytes"] = nbytes
                row["wall_total"] = f"{time.perf_counter() - t0:.2f}"
                writer.writerow(row)
                fh.flush()
                done.add(uid)
                el = time.perf_counter() - t_start
                print(f"[{i}/{len(urls)}] {name[:60]}: {row['status']} "
                      f"{row['result']} ppd={row['props_per_decision']} "
                      f"({el/60:.1f} min elapsed)", file=sys.stderr)

    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
