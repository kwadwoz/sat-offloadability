#!/usr/bin/env python3
"""A1 step 2: per-call BCP traces of the E23 industrial set, reduced to summaries.

For each (hash, family) in industrial_families.csv: download the CNF, run the
instrumented MiniSat with the buffered binary log and the original MiniSat
without it (seeded random order), then reduce the trace to

  summary.csv     one row per instance: answers, wall times, phase durations,
                  call/conflict/restart counts, search BCP kernel time
  segments.csv    per instance and granularity, a log-binned histogram
                  (10 bins/decade) of segment durations: count and sum
  bound_grid.csv  per instance, granularity, interface latency L and device
                  speedup S: sum over segments of max(0, t - L - t/S)

Granularities (search phase only): call (BCP kernel time per call), conflict
(wall time between the first calls of consecutive conflict intervals), restart
(same per restart interval), search (whole search phase). Coarser than call
means offloading the CDCL loop, not only BCP. The bound ignores state
synchronization and is an upper bound. No overhead correction is applied here;
wall_traced_ns and wall_untraced_ns are kept for that.

Large files stay outside the repo: the CNF and trace are deleted after
summarizing, except one trace per family (first whose binary log is at most
--keep-max-mb) gzipped under --data-dir/kept. Resumable: instances already in
summary.csv are skipped.
"""
import argparse, csv, gzip, json, math, pathlib, random, shutil, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'integration' / 'minisat'))
sys.path.insert(0, str(ROOT / 'instrument'))
import bcplog  # noqa: E402
from run_midrange import fetch  # noqa: E402

SEED = 20261012
L_NS = [50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 100_000, 790_000, 1_000_000,
        10_000_000, 100_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000]
SPEEDUPS = [2, 5, 10, 100, math.inf]
BINS_PER_DECADE = 10
SUMMARY_FIELDS = ['hash', 'family', 'instance', 'cnf_sha256', 'status_traced', 'status_untraced',
                  'wall_traced_ns', 'wall_untraced_ns', 'parsing_ns', 'preprocessing_ns', 'search_ns',
                  'calls_total', 'calls_search', 'conflicts', 'search_starts', 'search_bcp_kernel_ns',
                  'trace_complete', 'kept_trace']


def run(solver, cnf, out, timeout, logging):
    cmd = [sys.executable, str(ROOT / 'integration/minisat/run.py'), '--solver', str(solver), '--cnf', str(cnf),
           '--out', str(out), '--timeout', str(timeout)]
    if logging:
        cmd += ['--logging', '--no-decode']
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.loads((out / 'metadata.json').read_text())


def segments(recs, phases):
    """Search-phase segment durations (ns) per granularity."""
    names = [p['phase'] for p in phases]
    starts = [int(p['start_ns']) for p in phases]
    if 'search' not in names:
        return {}, 0
    k = names.index('search')
    search_begin, search_end = starts[k], starts[k + 1]
    r = recs[recs['phase_seq'] == k]
    out = {'call': r['kernel_ns'].astype(np.int64), 'search': np.array([search_end - search_begin], dtype=np.int64)}
    s = r['start_ns'].astype(np.int64)
    for gran, key in (('conflict', 'conflicts_before'), ('restart', 'restarts_before')):
        if len(r) == 0:
            out[gran] = np.zeros(0, dtype=np.int64); continue
        first = np.flatnonzero(np.r_[True, np.diff(r[key].astype(np.int64)) != 0])
        bounds = np.r_[s[first], search_end]
        out[gran] = np.diff(bounds)
    return out, int(r['kernel_ns'].sum())


def bound_rows(h, segs):
    for gran, t in segs.items():
        t = t.astype(np.float64)
        for L in L_NS:
            for S in SPEEDUPS:
                saved = np.maximum(0.0, t - L - (0.0 if math.isinf(S) else t / S)).sum()
                yield {'hash': h, 'granularity': gran, 'segments': len(t), 'total_ns': int(t.sum()),
                       'latency_ns': L, 'speedup': 'inf' if math.isinf(S) else S, 'savings_ns': int(saved)}


def histogram_rows(h, segs):
    for gran, t in segs.items():
        b = np.where(t > 0, np.floor(np.log10(np.maximum(t, 1)) * BINS_PER_DECADE), -1).astype(np.int64)
        for bin_, count, total in zip(*np.unique(b, return_counts=True), [t[b == x].sum() for x in np.unique(b)]):
            lo = 0 if bin_ < 0 else 10 ** (bin_ / BINS_PER_DECADE)
            yield {'hash': h, 'granularity': gran, 'bin': int(bin_), 'bin_lo_ns': round(lo, 3),
                   'count': int(count), 'sum_ns': int(total)}


def append(path, fields, rows):
    new = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--builds', required=True, type=pathlib.Path, help='External root from build.py')
    ap.add_argument('--out', type=pathlib.Path, default=ROOT / 'results/a1-step2')
    ap.add_argument('--data-dir', required=True, type=pathlib.Path, help='Outside the repo; work files and kept traces')
    ap.add_argument('--timeout', type=float, default=60)
    ap.add_argument('--keep-max-mb', type=float, default=200)
    ap.add_argument('--limit', type=int, help='Only the first N instances (smoke test)')
    a = ap.parse_args()
    data = a.data_dir.expanduser().resolve()
    if data == ROOT or ROOT in data.parents:
        ap.error('--data-dir must be outside the repository')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'runs').mkdir(exist_ok=True)
    traced_bin, plain_bin = a.builds / 'build-instrumented/minisat', a.builds / 'build-original/minisat'

    with open(ROOT / 'instrument/industrial_families.csv') as f:
        picks = [(r['hash'], r['family'], r['instance']) for r in csv.DictReader(f) if r['solver'] == 'minisat']
    picks = picks[:a.limit] if a.limit else picks
    summary = a.out / 'summary.csv'
    done, kept_families = set(), set()
    if summary.exists():
        with summary.open() as f:
            for r in csv.DictReader(f):
                done.add(r['hash'])
                if r['kept_trace'] == 'True':
                    kept_families.add(r['family'])
    rng = random.Random(SEED)
    for i, (h, fam, name) in enumerate(picks, 1):
        traced_first = rng.random() < 0.5   # drawn for every pick so resuming keeps the same order
        if h in done:
            continue
        work = data / 'work' / h
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        try:
            cnf = fetch(f'https://benchmark-database.de/file/{h}', work, timeout=180.0, retries=4, backoff=5.0)
        except Exception as e:  # noqa: BLE001
            print(f'[{i}/{len(picks)}] {fam}/{name}: FETCH FAIL {e}', file=sys.stderr, flush=True)
            shutil.rmtree(work)
            continue
        order = [True, False] if traced_first else [False, True]
        meta = {}
        for logging in order:
            meta[logging] = run(traced_bin if logging else plain_bin, cnf, work / ('traced' if logging else 'untraced'),
                                a.timeout, logging)
        for logging, m in meta.items():
            (a.out / 'runs' / f'{h}.{"traced" if logging else "untraced"}.json').write_text(json.dumps(m, indent=2) + '\n')
        t, u = meta[True], meta[False]
        log = work / 'traced' / 'calls.bin'
        row = {'hash': h, 'family': fam, 'instance': name, 'cnf_sha256': t['formula_sha256'],
               'status_traced': t['status'], 'status_untraced': u['status'],
               'wall_traced_ns': t['process_wall_ns'], 'wall_untraced_ns': u['process_wall_ns'],
               'trace_complete': False, 'kept_trace': False}
        if t['status'] in ('SAT', 'UNSAT'):
            phases = bcplog.read_phases(log)
            recs = bcplog.read_records(log)
            names, starts = [p['phase'] for p in phases], [int(p['start_ns']) for p in phases]
            for ph in ('parsing', 'preprocessing', 'search'):
                row[ph + '_ns'] = sum(starts[j + 1] - starts[j] for j in range(len(names) - 1) if names[j] == ph)
            segs, kernel = segments(recs, phases)
            row.update(calls_total=len(recs), calls_search=len(segs.get('call', [])),
                       # MiniSat counts a conflict after propagate() returns it, so add the last call's.
                       conflicts=int(recs['conflicts_before'][-1]) + int(recs['conflict'][-1]) if len(recs) else 0,
                       search_starts=int(recs['restarts_before'][-1]) if len(recs) else 0,  # MiniSat's printed 'restarts'
                       search_bcp_kernel_ns=kernel, trace_complete=True)
            append(a.out / 'bound_grid.csv', ['hash', 'granularity', 'segments', 'total_ns', 'latency_ns', 'speedup', 'savings_ns'],
                   bound_rows(h, segs))
            append(a.out / 'segments.csv', ['hash', 'granularity', 'bin', 'bin_lo_ns', 'count', 'sum_ns'], histogram_rows(h, segs))
            if fam not in kept_families and log.stat().st_size <= a.keep_max_mb * 1e6:
                dest = data / 'kept' / fam / h
                dest.mkdir(parents=True, exist_ok=True)
                with log.open('rb') as fi, gzip.open(dest / 'calls.bin.gz', 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                shutil.copy(str(log) + '.phases.csv', dest / 'calls.bin.phases.csv')
                shutil.copy(work / 'traced' / 'metadata.json', dest / 'metadata.json')
                kept_families.add(fam)
                row['kept_trace'] = True
        append(summary, SUMMARY_FIELDS, [row])
        shutil.rmtree(work)
        print(f'[{i}/{len(picks)}] {fam}/{name[:40]}: {t["status"]}/{u["status"]} '
              f'traced {t["process_wall_ns"]/1e9:.2f}s untraced {u["process_wall_ns"]/1e9:.2f}s', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
