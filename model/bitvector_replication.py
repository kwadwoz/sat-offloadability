#!/usr/bin/env python3
"""E25: replicate E24's bitvector modularity slope on fresh instances.

E24's strongest within-family hit was bitvector: Q vs full-run MiniSat ppd
at rho=-0.87 (p<1e-4) -- but at n=15, and E22 watched a p=0.008 hit die on
out-of-sample replication. Preregistered call, fixed before running:
replication means rho <= -0.4 with p < 0.05 on fresh bitvector instances.

Fresh sample: every GBD bitvector instance with minisat1m=yes that E23 did
not use (126 total, 15 used, up to 60 drawn seeded). Per instance: one
MiniSat run at --cpu-lim for full-run ppd (drop if incomplete), then the
E24 featuriser (igraph Louvain, same guards). Rows stream; resume on hash.

Usage:
    python model/bitvector_replication.py --meta <meta.db path>
"""
from __future__ import annotations

import argparse
import csv
import random
import signal
import sqlite3
import sys
import tempfile
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instrument"))
from run_midrange import fetch, run  # noqa: E402
from modularity_industrial import (read_cnf, vig_q, MAX_MB,  # noqa: E402
                                   FEAT_TIMEOUT)

FAMILY = "bitvector"
N_FRESH = 60
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
    pool = [h for (h,) in con.execute(
        "SELECT hash FROM features WHERE family=? AND minisat1m='yes'",
        (FAMILY,)) if h not in used]
    con.close()
    picks = random.Random(SEED).sample(pool, min(N_FRESH, len(pool)))

    out_path = ROOT / "bitvector_replication.csv"
    done = set()
    if out_path.exists():
        done = {r["hash"] for r in csv.DictReader(open(out_path))}
        print(f"resuming: {len(done)} done", file=sys.stderr, flush=True)

    fields = ["hash", "ppd", "modularity", "degree_cv", "n_clauses"]
    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        with tempfile.TemporaryDirectory() as td:
            for i, h in enumerate(picks, 1):
                if h in done:
                    continue
                try:
                    cnf = fetch(f"https://benchmark-database.de/file/{h}",
                                Path(td), timeout=180.0, retries=4, backoff=5.0)
                except Exception as e:
                    print(f"[{i}] FETCH FAIL {e}", file=sys.stderr, flush=True)
                    continue
                if cnf.stat().st_size > MAX_MB * 1024 * 1024:
                    print(f"[{i}] DROP size", file=sys.stderr, flush=True)
                    cnf.unlink()
                    continue
                r = run("minisat", cnf, CPU_LIM)
                if not (r["status"] == "ok" and r["completed"]):
                    print(f"[{i}] DROP solve ({r['status']})",
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
                    print(f"[{i}] DROP feat", file=sys.stderr, flush=True)
                    continue
                q, dcv, _ = res
                w.writerow({"hash": h, "ppd": r["props_per_decision"],
                            "modularity": f"{q:.4f}", "degree_cv": f"{dcv:.4f}",
                            "n_clauses": len(clauses)})
                fh.flush()
                print(f"[{i}/{len(picks)}] Q={q:.3f} ppd={r['props_per_decision']}",
                      file=sys.stderr, flush=True)

    rows = list(csv.DictReader(open(out_path)))
    qs = [float(r["modularity"]) for r in rows]
    ps = [float(r["ppd"]) for r in rows]
    rho, p = spearmanr(qs, ps)
    print(f"\n== E25 verdict (prereg: rho <= -0.4, p < 0.05)")
    print(f"n={len(rows)} rho={rho:+.3f} p={p:.4f} -> "
          f"{'REPLICATED' if rho <= -0.4 and p < 0.05 else 'NOT REPLICATED'}")


if __name__ == "__main__":
    main()
