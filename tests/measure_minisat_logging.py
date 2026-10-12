#!/usr/bin/env python3
"""Repeated, randomized-order process timing of three matched CPU modes."""
import argparse,csv,hashlib,json,pathlib,random,statistics,subprocess,tempfile,time

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original',required=True); ap.add_argument('--instrumented',required=True)
    ap.add_argument('--benchmarks',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--repetitions',type=int,default=7)
    ap.add_argument('--instances',type=int,default=3); a=ap.parse_args()
    paths=sorted(a.benchmarks.rglob('*.cnf'))[:a.instances]; assert paths
    rng=random.Random(20261011); rows=[]
    modes={'original':a.original,'disabled':a.instrumented,'enabled':a.instrumented}
    with tempfile.TemporaryDirectory(prefix='minisat-overhead-') as tmp:
        d=pathlib.Path(tmp)
        for cnf in paths:
            # Match the existing harness's SATLIB footer sanitization.
            lines=cnf.read_text().splitlines(True); end=next((i for i,l in enumerate(lines) if l.lstrip().startswith('%')),len(lines))
            inp=d/'input.cnf'; inp.write_text(''.join(lines[:end]))
            expected=None
            for rep in range(-1,a.repetitions):
                order=list(modes); rng.shuffle(order)
                for mode in order:
                    cmd=[modes[mode],'-pre','-verb=0','-rnd-seed=91648253']
                    if mode=='enabled': cmd.append('-bcp-log='+str(d/'calls.csv'))
                    start=time.perf_counter_ns(); p=subprocess.run(cmd+[str(inp)],capture_output=True,timeout=30); elapsed=time.perf_counter_ns()-start
                    assert p.returncode in (10,20)
                    if expected is None: expected=p.returncode
                    assert expected==p.returncode
                    if rep>=0: rows.append({'instance':str(cnf),'formula_sha256':hashlib.sha256(cnf.read_bytes()).hexdigest(),'mode':mode,'repetition':rep,'process_wall_ns':elapsed,'exit_code':p.returncode})
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for cnf in paths:
        med={m:statistics.median(r['process_wall_ns'] for r in rows if r['instance']==str(cnf) and r['mode']==m) for m in modes}
        summary.append({'instance':str(cnf),'median_process_wall_ns':med,'enabled_over_original':med['enabled']/med['original'],'disabled_over_original':med['disabled']/med['original']})
    report={'seed':20261011,'repetitions':a.repetitions,'warmups_per_mode':1,'preprocessing':True,'timing':'Host monotonic process wall time including launch, parsing, preprocessing, solve and CSV writes/close; temporary file storage.',
            'binary_sha256':{m:hashlib.sha256(pathlib.Path(b).read_bytes()).hexdigest() for m,b in modes.items()},'instances':summary}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))

if __name__=='__main__': main()
