# A1 step 1: call timestamps, conflict/restart counters, phase log

Status: implemented and validated; awaiting user acceptance. Nothing committed.

## Purpose

The A1 limit study needs true elapsed time for segments coarser than one
propagation call (conflict interval, restart interval, whole solve). The P1
log recorded only per-call durations, so time spent between calls (branching,
conflict analysis, backtracking) was not observable.

## What changed

- `integration/minisat/bcp-timestamps.patch` (new), applied after the
  unchanged P1 `bcp-instrumentation.patch`, so P1's patch hash stays valid.
  - Three columns appended to `calls.csv` (existing columns unchanged):
    `start_ns` (monotonic call entry time, relative to log open),
    `conflicts_before`, `restarts_before` (native `conflicts`/`starts`
    counters at call entry).
  - A sidecar `<log>.phases.csv` (`phase,start_ns`) records every phase label
    change on the same clock, plus a final `closed` row.
- `integration/minisat/build.py` applies both patches in order.
- `integration/minisat/run.py` records the hashes of both patches.
- `tests/check_minisat_logging.py` adds timeline checks; `check_trace()`
  applies them to any existing trace.

Counter semantics: MiniSat increments `conflicts` in `search()` right after
`propagate()` returns a conflict, so between consecutive search calls
`conflicts_before` rises by exactly 1 after a CONFLICT call and 0 otherwise.
`starts` counts `search()` invocations, i.e. restarts + 1 per solve.
Preprocessing propagation does not touch either counter.

## Reproduce

```sh
python3 integration/minisat/build.py --external-root /private/tmp/sat-bcp-p1b --archive /private/tmp/sat-bcp-minisat-2.2.1.tar.gz
python3 tests/check_minisat_logging.py --original /private/tmp/sat-bcp-p1b/build-original/minisat --instrumented /private/tmp/sat-bcp-p1b/build-instrumented/minisat --out results/a1-step1/correctness.json
python3 tests/measure_minisat_logging.py --original /private/tmp/sat-bcp-p1b/build-original/minisat --instrumented /private/tmp/sat-bcp-p1b/build-instrumented/minisat --benchmarks benchmarks/uf250-1065 --instances 3 --repetitions 5 --out results/a1-step1/overhead
```

## Evidence

- Correctness suite: 105 formulas × 3 modes = 315 runs passed. Answers,
  models and native counters are identical across original, disabled and
  enabled modes; the new timeline checks pass on all 319 logged calls
  (`results/a1-step1/correctness.json`).
- Library API check passes; the phase log is opened, labeled `api`, and closed
  by the destructor.
- Real traces: timeline checks pass on uf250-01 (85,540 calls) and uf250-0100
  (1,012,989 calls). Final `conflicts_before`/`restarts_before` equal
  MiniSat's reported totals (38,133/126 and 457,926/1,022).
- Overhead (`results/a1-step1/overhead/`, 1 warmup + 5 runs, randomized order):

| Instance | Original ms | Disabled ms | Logging ms | Logging / original |
| --- | ---: | ---: | ---: | ---: |
| uf250-01 | 169.7 | 169.2 | 230.5 | 1.358 |
| uf250-010 | 183.0 | 183.3 | 264.4 | 1.445 |
| uf250-0100 | 2317.9 | 2315.1 | 3203.6 | 1.382 |

P1 measured 1.339–1.351 on the same instances. The added fields cost a few
points; uf250-010's larger jump is within what five repetitions can resolve.
Disabled mode remains indistinguishable from the original build.

## Limitations

- Gaps between consecutive calls include the cost of formatting and writing
  the previous row, so traced non-BCP time is inflated by logging overhead.
  The limit study must correct for this (e.g. scale against untraced wall
  time) or bound its effect; that is a step 2 design choice.
- Timeout or forced termination leaves no `closed` row.
- Validated on uf250 and small random formulas only; industrial instances are
  not yet traced (step 2).

# Step 1b: buffered binary log (user chose option B)

Status: implemented and validated; awaiting user acceptance. Nothing committed.
Step 1 above was accepted by the user.

## Purpose

Step 1's logging slowed MiniSat by 36–45%, almost entirely from formatting a
CSV row after every call. That inflates the time between calls, which the
coarse-granularity limit study depends on.

## What changed

- `integration/minisat/bcp-buffered-log.patch` (new, applied third). The hook
  fills a fixed 96-byte record per call into a 65,536-record (~6 MB) memory
  buffer and writes it with one `fwrite` when full and at close. No text
  formatting happens during the solve. Memory stays bounded for runs with
  tens of millions of calls. The file starts with a header (magic, version,
  record size). The phase is stored as an index into the `.phases.csv` log.
- `integration/minisat/bcplog.py` (new) decodes the binary log into the exact
  step 1 CSV schema, rejecting wrong headers and truncated files.
- `run.py` logs to `calls.bin` and decodes to `calls.csv` after SAT/UNSAT
  completion; the tests decode before checking. `build.py` applies three
  patches; run metadata records all three hashes.

## Evidence (build in `/private/tmp/sat-bcp-p1c`)

- Correctness suite: 315 runs passed, including the timeline checks
  (`results/a1-step1b/correctness.json`). The API check passes.
- On uf250-01 (85,540 calls) and uf250-0100 (1,012,989 calls), every
  non-timing column decoded from the binary log is identical to step 1's CSV
  log, and the timeline checks pass.
- Overhead (`results/a1-step1b/overhead/`, same protocol as step 1):

| Instance | Original ms | Disabled ms | Logging ms | Logging / original | Step 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uf250-01 | 170.8 | 171.6 | 180.4 | 1.056 | 1.358 |
| uf250-010 | 183.5 | 183.5 | 195.3 | 1.065 | 1.445 |
| uf250-0100 | 2352.0 | 2351.3 | 2481.3 | 1.055 | 1.382 |

The remaining ~6% is likely dominated by the three clock reads per ~1 µs
call; this was not profiled.

## Limitations

- A run killed by timeout loses up to the last unflushed buffer, and the
  decoder rejects a partially written record; such traces are incomplete.
- The binary format is native-endian; the decoder assumes little-endian
  (arm64/x86) and checks the header.
- The ~6% still inflates between-call time slightly; step 2 can quantify it
  per instance with a logging-off run where needed.
