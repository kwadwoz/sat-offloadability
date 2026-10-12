# A1 step 2: industrial per-call traces and offload upper bound

Status: complete; awaiting user review. Nothing committed.

## What ran

`instrument/trace_industrial.py` over the 210 E23 instances (14 GBD families
× 15, `instrument/industrial_families.csv`), re-downloaded from
benchmark-database.de. Per instance: instrumented MiniSat with the buffered
binary log (step 1b) and the original MiniSat without it, seeded random
order, 60 s timeout, preprocessing on. `run.py` gained `--no-decode` so the
binary trace is summarized directly instead of converted to CSV.

```sh
python3 integration/minisat/build.py --external-root /private/tmp/sat-bcp-p1c --archive /private/tmp/sat-bcp-minisat-2.2.1.tar.gz
python3 instrument/trace_industrial.py --builds /private/tmp/sat-bcp-p1c --data-dir ~/sat-bcp-data/a1-step2 --out results/a1-step2
```

Outputs (`results/a1-step2/`): `summary.csv` (per instance), `segments.csv`
(log-binned segment-duration histograms), `bound_grid.csv` (bound per
instance × granularity × latency × speedup), `runs/` (provenance of all 420
runs). One gzipped raw trace per family is kept outside the repo in
`~/sat-bcp-data/a1-step2/kept/`.

## Validation

- 210/210 instances completed (93 SAT, 117 UNSAT); traced and untraced runs
  agree on every answer. No timeouts or fetch failures.
- On one industrial instance, decoded trace totals equal MiniSat's printed
  conflicts (15,049), restarts (62) and propagations (9,521,419), and the
  timeline checks pass.
- The script's bound equals an independent plain-Python recomputation from
  the decoded CSV of the same run at all 320 grid points (max difference
  1 ns, integer rounding).
- Logging overhead on industrial: traced/untraced wall median 1.006
  (p10 0.962, p90 1.112). No overhead correction applied.

## Definitions

Savings bound = Σ over segments max(0, t − L − t/S), with interface round
trip L and device speedup S. It ignores state synchronization and result
import, so it is an upper bound. Granularities, search phase only: `call`
(t = BCP kernel time), `conflict` and `restart` (t = wall time of the
interval), `search` (whole search phase). Anything coarser than `call`
means offloading the whole CDCL loop, where S is a whole-solver speedup.

## Results

Per-call BCP time is much longer than on uf250 (~1 µs): family medians run
from 0.7 µs (testpattern-generation) to 4–12 µs (bitvector, cryptography,
prime-factoring, diagnosis) and 69 µs (design-debugging). BCP is 35–80% of
search time.

Per-call offload, S = ∞, upper bound as a fraction of total solve time
(all instances, time-weighted):

| Round trip L | 100 ns | 1 µs | 5 µs | 790 µs (measured ECP5 link) |
| --- | ---: | ---: | ---: | ---: |
| Savings bound | 0.636 | 0.568 | 0.423 | 0.017 |

Per family (median over all 15 instances; instances solved before search
count as 0 saved), four families keep 34–55% at L = 5 µs (diagnosis,
cryptography, prime-factoring, bitvector), miter and hardware-verification
keep 13–20%, and the rest fall below 5%. At 790 µs no family median exceeds
1%. Three families are mostly solved during preprocessing, before search
(design-debugging 11/15 instances, software-verification 10/15,
testpattern-generation 6/15), so search-only offload saves them nothing at the
median. Figure: `figures/a1_offload_map.{png,pdf}` (`model/a1_offload_map.py`).

Correction: an earlier version of this analysis took medians over only the
instances that reached search, which overstated software-verification and
design-debugging. `bound_grid.csv` has no rows for instances without search
calls; analyses must treat them as zero savings. The all-instance
time-weighted totals above were unaffected.

Coarser granularities look near-total at low L, but only if the device runs
the whole CDCL loop S times faster than MiniSat. PigWeedSAT on the cloud
ECP5 is ~0.25× MiniSat on its Table 3 instances (its README), so with that
device every coarse-granularity saving is zero. At the measured mrg load
cost (~2 s) only whole-search offload saves anything, and only on long
bitvector/cryptography/diagnosis/software-verification/prime-factoring runs,
again only with S > 1.

## Limitations

- Upper bound: no synchronization, transfer or import cost; S is swept, not
  measured. Measuring a realistic per-call S and sync cost is the next step.
- Median-of-instances per column; columns are not jointly from one instance.
- One machine (Apple M3), one solver (MiniSat 2.2.1), one seed; 15 instances
  per family, filtered to MiniSat-solvable within a minute (GBD minisat1m).
