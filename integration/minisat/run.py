#!/usr/bin/env python3
"""Run CPU MiniSat with optional BCP logging and provenance; preprocessing enabled."""
import argparse, csv, gzip, hashlib, json, pathlib, subprocess, time, uuid
from collections import deque
import bcplog

ROOT = pathlib.Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solver', required=True, type=pathlib.Path)
    ap.add_argument('--cnf', required=True, type=pathlib.Path)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--logging', action='store_true')
    ap.add_argument('--timeout', type=float, default=60)
    ap.add_argument('--history-window', type=int, default=64)
    ap.add_argument('--min-decisions', type=int, default=1)
    args = ap.parse_args()
    if args.timeout <= 0 or args.history_window <= 0 or args.min_decisions <= 0:
        ap.error('timeout, history-window and min-decisions must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    solver, cnf = args.solver.resolve(), args.cnf.resolve()
    cmd = [str(solver), '-pre', '-rnd-seed=91648253', '-verb=1']
    if args.logging:
        cmd.append('-bcp-log=' + str((args.out / 'calls.bin').resolve()))
    # Preserve the existing harness's handling of SATLIB '%' footers.
    prepared = (args.out / 'input.cnf').resolve()
    opener = gzip.open if cnf.suffix == '.gz' else open
    with opener(cnf, 'rt') as source, prepared.open('w') as dest:
        for line in source:
            if line.lstrip().startswith('%'):
                break
            dest.write(line)
    cmd.extend([str(prepared), str((args.out / 'model.txt').resolve())])
    begin = time.perf_counter_ns()
    with (args.out / 'stdout.txt').open('w') as stdout, (args.out / 'stderr.txt').open('w') as stderr:
        try:
            proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=args.timeout)
            rc, status = proc.returncode, {10:'SAT', 20:'UNSAT', 0:'INDET'}.get(proc.returncode, 'ERROR')
        except subprocess.TimeoutExpired:
            rc, status = None, 'TIMEOUT'
    elapsed = time.perf_counter_ns() - begin
    metadata = {'run_id':str(uuid.uuid4()), 'formula_sha256':sha(cnf), 'prepared_formula_sha256':sha(prepared), 'solver_sha256':sha(solver),
                'source_version':'2.2.1', 'source_sha256':'432985833596653fcd698ab439588471cc0f2437617d0df2bb191a0252ba423d',
                'patch_sha256':{p:sha(ROOT/'integration/minisat'/p) for p in ('bcp-instrumentation.patch','bcp-timestamps.patch','bcp-buffered-log.patch')},
                'configuration_sha256':hashlib.sha256(json.dumps({'preprocessing':True,'seed':91648253,'logging':args.logging,'history_window':args.history_window,'min_decisions':args.min_decisions,'timeout':args.timeout},sort_keys=True).encode()).hexdigest(),
                'command':cmd, 'preprocessing':True, 'logging':args.logging, 'exit_code':rc,
                'status':status, 'process_wall_ns':elapsed, 'timeout_seconds':args.timeout}
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.logging and status in ('SAT','UNSAT'):
        bcplog.write_csv(args.out/'calls.bin', args.out/'calls.csv')
    # Offline history uses only earlier completed calls. It is not an online policy.
    if args.logging and status in ('SAT','UNSAT'):
        history = deque(maxlen=args.history_window)
        with (args.out/'calls.csv').open() as source, (args.out/'features.csv').open('w') as dest:
            reader = csv.DictReader(source)
            fields = reader.fieldnames + ['decisions_since_previous_call','recent_pr','recent_w','history_calls','history_valid','pr_valid']
            writer = csv.DictWriter(dest, fieldnames=fields); writer.writeheader()
            previous_decisions = 0
            for row in reader:
                work = sum(int(r['processed_entries']) for r in history)
                decisions = sum(int(r['decisions_since_previous_call']) for r in history)
                row['decisions_since_previous_call'] = int(row['decisions_before']) - previous_decisions
                previous_decisions = int(row['decisions_before'])
                row.update(recent_pr=work/decisions if decisions>=args.min_decisions else '',
                           recent_w=work/len(history) if history else '',
                           history_calls=len(history), history_valid=bool(history), pr_valid=decisions>=args.min_decisions)
                writer.writerow(row); history.append(row)
    print(json.dumps(metadata))
    if status not in ('SAT','UNSAT','INDET','TIMEOUT'):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
