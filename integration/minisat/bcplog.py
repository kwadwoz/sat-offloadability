#!/usr/bin/env python3
"""Decode the instrumented MiniSat binary BCP log into the per-call CSV schema."""
import argparse, csv, pathlib
import numpy as np

MAGIC, VERSION = 0x47504342, 1
# Must match Solver::BcpRecord in bcp-buffered-log.patch (native little-endian).
RECORD = np.dtype([(n, '<u8') for n in ('decisions_before', 'propagations_before', 'decisions_after', 'propagations_after',
                                        'kernel_ns', 'observed_ns', 'start_ns', 'conflicts_before', 'restarts_before')]
                  + [(n, '<i4') for n in ('pending_entries', 'decision_level', 'trail_before', 'trail_after')]
                  + [('phase_seq', '<u4'), ('conflict', '<u4')])
COLUMNS = ['call_id', 'phase', 'pending_entries', 'decision_level', 'decisions_before', 'propagations_before', 'trail_before',
           'status', 'processed_entries', 'implied_assignments', 'decisions_after', 'propagations_after', 'trail_after',
           'kernel_ns', 'observed_ns_before_write', 'start_ns', 'conflicts_before', 'restarts_before']


def read_phases(log):
    with pathlib.Path(str(log) + '.phases.csv').open() as f:
        return list(csv.DictReader(f))


def read_records(log):
    """Return the structured record array; validates the header and record size."""
    with open(log, 'rb') as f:
        magic, version, size, _ = np.fromfile(f, dtype='<u4', count=4)
        if magic != MAGIC or version != VERSION or size != RECORD.itemsize:
            raise ValueError(f'{log}: not a version {VERSION} BCP log (magic {magic:#x}, record size {size})')
        data = f.read()
    if len(data) % RECORD.itemsize:
        raise ValueError(f'{log}: truncated record (killed before close?)')
    return np.frombuffer(data, dtype=RECORD)


def write_csv(log, out):
    recs, phases = read_records(log), [p['phase'] for p in read_phases(log)]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLUMNS)
        for i, r in enumerate(recs.tolist(), 1):
            (db, pb, da, pa, kns, ons, sns, cb, rb, pend, lvl, tb, ta, seq, confl) = r
            w.writerow([i, phases[seq], pend, lvl, db, pb, tb, 'CONFLICT' if confl else 'FIXED_POINT',
                        pa - pb, ta - tb, da, pa, ta, kns, ons, sns, cb, rb])
    return len(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log', type=pathlib.Path); ap.add_argument('csv', type=pathlib.Path)
    a = ap.parse_args()
    print(write_csv(a.log, a.csv))


if __name__ == '__main__':
    main()
