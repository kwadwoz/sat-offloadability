#!/usr/bin/env python3
"""Differential BCP logging check with exhaustive small-formula answers."""
import argparse, csv, itertools, json, pathlib, random, re, subprocess, sys, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'integration'/'minisat'))
import bcplog


def check_timeline(rows, phases, name):
    """Timestamps, conflict/restart counters and the phase log are mutually consistent."""
    assert phases[-1]['phase']=='closed',(name,'phase log not closed')
    starts=[int(p['start_ns']) for p in phases]
    assert starts==sorted(starts),(name,'phase times decrease')
    for prev,cur in zip(rows,rows[1:]):
        # Calls do not overlap: the next call starts after the previous call's observed interval.
        assert int(cur['start_ns'])>=int(prev['start_ns'])+int(prev['kernel_ns']),(name,'overlap',cur['call_id'])
        assert int(cur['conflicts_before'])>=int(prev['conflicts_before'])
        assert int(cur['restarts_before'])>=int(prev['restarts_before'])
        if prev['phase']==cur['phase']=='search':
            # MiniSat counts a conflict right after propagate() returns one, before the next call.
            assert int(cur['conflicts_before'])-int(prev['conflicts_before'])==(prev['status']=='CONFLICT'),(name,'conflict count',cur['call_id'])
    if rows:
        assert int(rows[-1]['start_ns'])+int(rows[-1]['observed_ns_before_write'])<=starts[-1],(name,'call after close')
    # Every call's phase label matches the phase log interval it falls into.
    for r in rows:
        i=max(k for k,s in enumerate(starts) if s<=int(r['start_ns']))
        assert phases[i]['phase']==r['phase'],(name,'phase interval',r['call_id'])


def check_trace(calls_bin, calls_csv):
    """Apply the timeline checks to an existing trace, e.g. a large benchmark run."""
    with open(calls_csv) as f: rows=list(csv.DictReader(f))
    check_timeline(rows, bcplog.read_phases(calls_bin), calls_bin)
    return len(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original', required=True)
    ap.add_argument('--instrumented', required=True)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    a=ap.parse_args(); rng=random.Random(20261011)
    cases=[('chain',3,[[1],[-1,2],[-2,3]]),('root_conflict',1,[[1],[-1]]),
           ('empty_clause',1,[[]]),('no_clauses',3,[]),
           ('branch_unsat',2,[[1,2],[1,-2],[-1,2],[-1,-2]])]
    for i in range(100):
        n=rng.randint(1,8); cs=[]
        for _ in range(rng.randint(0,32)):
            vs=rng.sample(range(1,n+1),rng.randint(1,min(4,n)))
            cs.append([v if rng.randrange(2) else -v for v in vs])
        cases.append((f'random_{i}',n,cs))
    phases=set(); rows_total=0
    with tempfile.TemporaryDirectory(prefix='minisat-log-check-') as tmp:
        d=pathlib.Path(tmp)
        for name,n,cs in cases:
            expected=10 if any(all(any(bits[abs(l)-1]==(l>0) for l in c) for c in cs) for bits in itertools.product([False,True],repeat=n)) else 20
            cnf=d/'input.cnf'; cnf.write_text(f'p cnf {n} {len(cs)}\n'+''.join(' '.join(map(str,c))+' 0\n' for c in cs))
            outputs=[]
            for label, binary, logging in [('original',a.original,False),('disabled',a.instrumented,False),('enabled',a.instrumented,True)]:
                cmd=[binary,'-pre','-verb=1']; log=d/'calls.bin'; table=d/'calls.csv'; model=d/'model.txt'
                if logging: cmd.append('-bcp-log='+str(log))
                p=subprocess.run(cmd+[str(cnf),str(model)],capture_output=True,text=True,timeout=10)
                assert p.returncode==expected,(name,label,p.returncode)
                if expected==10:
                    vals={abs(int(t)):int(t)>0 for t in model.read_text().split()[1:] if int(t)}
                    assert all(any(vals.get(abs(l))==(l>0) for l in c) for c in cs),(name,label,'model')
                counters={k:int(re.search(r'^'+k+r'\s*:\s*(\d+)',p.stdout,re.M).group(1)) for k in ['propagations','decisions','conflicts']}
                outputs.append((model.read_text(),counters))
                if logging:
                    bcplog.write_csv(log, table)
                    with table.open() as f: rows=list(csv.DictReader(f))
                    rows_total+=len(rows)
                    assert sum(int(r['processed_entries']) for r in rows)==counters['propagations']
                    for i,r in enumerate(rows,1):
                        phases.add(r['phase']); assert int(r['call_id'])==i
                        assert int(r['processed_entries'])==int(r['propagations_after'])-int(r['propagations_before'])
                        assert int(r['implied_assignments'])==int(r['trail_after'])-int(r['trail_before'])
                        assert int(r['observed_ns_before_write'])>=int(r['kernel_ns'])>=0
                        assert r['status'] in ('CONFLICT','FIXED_POINT')
                    check_timeline(rows, bcplog.read_phases(log), name)
            assert outputs[0]==outputs[1]==outputs[2],(name,'logging changed behavior')
        bad=subprocess.run([a.instrumented,'-bcp-log='+str(d/'missing'/'log.csv'),str(cnf)],capture_output=True)
        assert bad.returncode==1
    report={'seed':20261011,'formulas':len(cases),'solver_runs':len(cases)*3,'all_passed':True,'logged_calls':rows_total,'phases_seen':sorted(phases),'invalid_log_path_rejected':True}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))

if __name__=='__main__': main()
