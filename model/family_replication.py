#!/usr/bin/env python3
"""E26: replicate the remaining E24 within-family modularity slopes.

E25 validated bitvector (rho=-0.87 at n=15 -> -0.64 at n=59 out-of-sample).
The other E24 hits get the same treatment: fresh GBD instances the earlier
experiments never touched, full-run MiniSat ppd, the same igraph featuriser.

Preregistered bars, fixed here before any data, sign taken from E24:

    prime-factoring  E24 rho=+0.75  -> replicates iff rho >= +0.4, p < 0.05
    miter            E24 rho=+0.71  -> replicates iff rho >= +0.4, p < 0.05
    planning         E24 rho=+0.68  -> replicates iff rho >= +0.4, p < 0.05
    scheduling       E24 rho=+0.55  -> replicates iff rho >= +0.4, p < 0.05

Up to N_FRESH per family, bounded by what GBD holds (prime-factoring has 69
minisat1m instances total, 14 used). Rows stream; resume on (family, hash).

Usage:
    python model/family_replication.py --meta <meta.db path>
"""
from __future__ import annotations

import argparse
import csv
import random
import signal
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instrument"))
from run_midrange import fetch, run  # noqa: E402
from modularity_industrial import (read_cnf, vig_q, MAX_MB,  # noqa: E402
                                   FEAT_TIMEOUT)

BARS = {  # family -> (required sign, |rho| threshold)
    "prime-factoring": +1,
    "miter": +1,
    "planning": +1,
    "scheduling": +1,
}
RHO_MIN = 0.4
N_FRESH = 50
SEED = 20260804
CPU_LIM = 60
ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True, type=Path)
    args = ap.parse_args()

    used = {r["hash"] for r in
            csv.DictReader(open(ROOT.parent / "instrument" / "industrial_families.csv"))}
    con = sqlite3.connect(args.meta)
    rng = random.Random(SEED)
    picks: list[tuple[str, str]] = []
    for fam in BARS:
        pool = [h for (h,) in con.execute(
            "SELECT hash FROM features WHERE family=? AND minisat1m='yes'",
            (fam,)) if h not in used]
        picks += [(fam, h) for h in rng.sample(pool, min(N_FRESH, len(pool)))]
    con.close()

    out_path = ROOT / "family_replication.csv"
    done = set()
    if out_path.exists():
        done = {r["hash"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done)} done", file=sys.stderr, flush=True)

    fields = ["family", "hash", "ppd", "modularity", "degree_cv", "n_clauses"]
    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        with tempfile.TemporaryDirectory() as td:
            for i, (fam, h) in enumerate(picks, 1):
                if h in done:
                    continue
                try:
                    cnf = fetch(f"https://benchmark-database.de/file/{h}",
                                Path(td), timeout=180.0, retries=4, backoff=5.0)
                except Exception as e:
                    print(f"[{i}] {fam}: FETCH FAIL {e}", file=sys.stderr, flush=True)
                    continue
                if cnf.stat().st_size > MAX_MB * 1024 * 1024:
                    print(f"[{i}] {fam}: DROP size", file=sys.stderr, flush=True)
                    cnf.unlink()
                    continue
                r = run("minisat", cnf, CPU_LIM)
                if not (r["status"] == "ok" and r["completed"]):
                    print(f"[{i}] {fam}: DROP solve ({r['status']})",
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
                    print(f"[{i}] {fam}: DROP feat", file=sys.stderr, flush=True)
                    continue
                q, dcv, _ = res
                w.writerow({"family": fam, "hash": h,
                            "ppd": r["props_per_decision"],
                            "modularity": f"{q:.4f}", "degree_cv": f"{dcv:.4f}",
                            "n_clauses": len(clauses)})
                fh.flush()
                print(f"[{i}/{len(picks)}] {fam}: Q={q:.3f} "
                      f"ppd={r['props_per_decision']}", file=sys.stderr, flush=True)

    rows = list(csv.DictReader(open(out_path)))
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append((float(r["modularity"]), float(r["ppd"])))
    print(f"\n== E26 verdicts (prereg: sign as E24, |rho| >= {RHO_MIN}, p < 0.05)")
    for fam, sign in BARS.items():
        pts = by_fam.get(fam, [])
        if len(pts) < 10:
            print(f"{fam:<16} n={len(pts)} -- insufficient")
            continue
        rho, p = spearmanr([q for q, _ in pts], [pp for _, pp in pts])
        ok = (rho * sign >= RHO_MIN) and p < 0.05
        print(f"{fam:<16} n={len(pts):>3} rho={rho:+.3f} p={p:.4f} -> "
              f"{'REPLICATED' if ok else 'NOT REPLICATED'}")


if __name__ == "__main__":
    main()
