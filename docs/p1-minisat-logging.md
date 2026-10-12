# P1: optional CPU propagation logging

Status: the approved all-call logging scope is implemented and its initial
validation passed. Full P1 research evaluation is still pending; no dispatcher,
selection policy, mock accelerator, or hardware backend has been implemented.
Nothing has been committed or pushed.

## Approved choices

MiniSat 2.2.1; preprocessing enabled; external solver source; all propagation
calls logged when explicitly enabled. Repository contents are our patch, build
and run scripts, checks, and measurement evidence. Upstream source and builds
remain under `/private/tmp/sat-bcp-p1/`, not in this repository. These temporary
paths can be replaced by a user-chosen external directory.

## What changed

- `integration/minisat/build.py` verifies the pinned source archive, creates
  separate external original/instrumented trees, applies the patch, and builds
  both with matching Release/C++11 settings. It rejects an external root inside
  this repository and refuses to overwrite an existing nonempty directory.
- `integration/minisat/bcp-instrumentation.patch` wraps the existing propagation
  routine. Every caller passes through the wrapper. Without a log, it immediately
  calls the original routine. The separately built original executable has no
  instrumentation changes at all.
- `integration/minisat/run.py` enables preprocessing and a fixed seed, handles
  plain/gzipped input and SATLIB footers, and writes logs with provenance. It
  derives recent work/PR offline using only previously completed call records.
  This is trace analysis, not an online policy.
- `tests/check_minisat_logging.py` checks original, disabled, and enabled modes
  against exhaustive small-formula answers and validates SAT models and counters.
- `tests/minisat_api_logging.cc` checks library implications, empty work, the API
  phase, and destructor cleanup.
- `tests/measure_minisat_logging.py` measures randomized-order repeated runs.

## Reproduce

Use Python 3.12+ (the extraction safety filter is required), CMake, C++, patch,
and zlib. From this repository, choose an empty external directory:

```sh
python3 integration/minisat/build.py --external-root /private/tmp/my-minisat-bcp
```

An already downloaded archive can be supplied with
`--archive /private/tmp/sat-bcp-minisat-2.2.1.tar.gz`. Source SHA-256:
`432985833596653fcd698ab439588471cc0f2437617d0df2bb191a0252ba423d`.

Commands used for the final build and checks:

```sh
python3 integration/minisat/build.py --external-root /private/tmp/sat-bcp-p1 --archive /private/tmp/sat-bcp-minisat-2.2.1.tar.gz
python3 tests/check_minisat_logging.py --original /private/tmp/sat-bcp-p1/build-original/minisat --instrumented /private/tmp/sat-bcp-p1/build-instrumented/minisat --out results/p1/correctness.json
c++ -std=c++11 -I /private/tmp/sat-bcp-p1/minisat-instrumented tests/minisat_api_logging.cc /private/tmp/sat-bcp-p1/build-instrumented/libminisat.a -lz -o /private/tmp/sat-bcp-p1/api-check
/private/tmp/sat-bcp-p1/api-check /private/tmp/sat-bcp-p1/api.csv
python3 tests/measure_minisat_logging.py --original /private/tmp/sat-bcp-p1/build-original/minisat --instrumented /private/tmp/sat-bcp-p1/build-instrumented/minisat --benchmarks benchmarks/uf250-1065 --instances 3 --repetitions 5 --out results/p1/overhead-verified
```

Output directories for runs/timing experiments must be new. Example run:

```sh
python3 integration/minisat/run.py --solver /private/tmp/sat-bcp-p1/build-instrumented/minisat --cnf benchmarks/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100/uf250-01.cnf --out /private/tmp/bcp-example --logging
```

Omit `--logging` to disable the logger; use `build-original/minisat` for the
original baseline. Defaults: 60-second timeout, seed 91648253, history window
64 calls, minimum one native decision for PR. History settings can be changed
with `--history-window` and `--min-decisions`.

## Schema and counter definitions

`calls.csv` contains one row per completed propagation invocation:

| Group | Fields and meaning |
| --- | --- |
| Identity | `call_id`, starting at 1; join with this run's metadata sidecar |
| Pre-call | `phase`, `pending_entries`, `decision_level`, `decisions_before`, `propagations_before`, `trail_before` |
| Outcome | `status` (FIXED_POINT or CONFLICT), `processed_entries`, `implied_assignments`, `decisions_after`, `propagations_after`, `trail_after` |
| Timing | `kernel_ns`, `observed_ns_before_write`, from a monotonic host clock |

The executable labels parsing, preprocessing, and search. Direct library usage
starts with phase `api`; callers can set a phase explicitly. Labels describe the
surrounding execution phase, not the reason for an individual call. Root search
simplification stays labeled search. All direct propagation callers share the
same wrapper, including optional preprocessing paths and `implies()`.

Processed entries use the native propagation counter delta; this includes
processed decision literals. On conflict, qhead jumps to the trail end, so its
delta is not the work count. Implied assignments count newly appended trail
entries during the call; they can include entries not processed before conflict.
Native decisions count branch-selection attempts, including the final attempt
that finds no branch. PR retains that convention and is not wall-clock throughput.
A contradictory unit or empty clause may establish UNSAT outside propagation;
we do not invent a propagation call or conflict row for it.

`features.csv` adds `decisions_since_previous_call`, `recent_pr`, `recent_w`,
`history_calls`, `history_valid`, and `pr_valid`. The decision delta is captured
from native counter snapshots at successive call entries; the first entry is
relative to zero. For a window of earlier completed calls, W is their summed
processed work divided by call count, and PR is that work divided by their
summed decision deltas. Current-call work and current-call decision delta do not
enter that row's recent features. PR stays empty below the minimum decision
count, including root-only windows. This is a provisional trace definition to
review before any online selector. No signal-quality claim is made yet.

`metadata.json` includes run ID, raw/prepared formula hashes, executable hash,
source version/archive hash, patch hash, configuration hash, exact command,
result/exit code, timeout, and process wall time. Configuration hash covers the
fixed seed, preprocessing, logging, history settings, and timeout. Each run
preserves the prepared formula, returned model, stdout, and stderr. The source
version fields assume binaries built with this helper; arbitrary user-supplied
binaries are not independently identified from their contents.

Kernel timing covers the original propagation routine. Observed timing also
includes pre-call snapshots and timestamp work up to row formatting. Neither
includes formatting/writing that row, final flush, or its own remaining timing
bookkeeping. Do not sum these nested intervals. Host process wall time includes
launch, input parsing, preprocessing, search, output, and log flush/close. It is
not a separately measured search-only time. Short calls may measure zero ns.
Input preparation and offline feature extraction occur outside process timing.
Timeout/error traces can be incomplete; features are generated only for normal
SAT/UNSAT completion. Normal log open/write failures are explicit errors;
forced termination may lose buffered rows.

## Evidence and interpretation

Correctness: 105 formulas × three modes = 315 runs passed, with exhaustive
expected SAT/UNSAT answers and original-formula SAT model checks. Outputs and
native counters match exactly across modes. Logged work sums match MiniSat's
native totals. The 319 logged calls cover parsing, preprocessing, and search.
The separate API check passed the implication and empty-work cases and verified
flushing on destruction. Derived features were also checked across 85,540 calls
on uf250-01: all windows use only preceding rows, with root PR unavailable.

Final overhead evidence is ONLY `results/p1/overhead-verified/`. One warmup and
five measured runs per mode per instance, randomized order, on three uf250
instances, with preprocessing enabled:

| Instance | Original median ms | Disabled median ms | Logging median ms | Logging overhead |
| --- | ---: | ---: | ---: | ---: |
| uf250-01 | 173.08 | 172.57 | 231.80 | 33.9% |
| uf250-010 | 184.31 | 184.86 | 248.97 | 35.1% |
| uf250-0100 | 2343.36 | 2334.99 | 3161.57 | 34.9% |

These are local process measurements, not general performance claims. Disabled
medians vary from about -0.36% to +0.30%; the sample is too small to establish a
precise disabled overhead. CSV observation is materially costly and suitable
for trace collection; a production selector needs separately measured counters
and logging settings. No FPGA acceleration or selection benefit is claimed.
Earlier experiments/examples carry EXCLUDED.txt and are not final evidence.

## Remaining work and next decision

The next useful step is to measure whether recent PR/W predicts upcoming work
and time on held-out traces, before adding a dispatcher. Alternatively, reduce
trace-writing overhead first. This choice needs user approval. Broader dataset
coverage, grouped data splits, separate run-phase wall timers, explicit solver
revision IDs beyond the release/archive pin, proof support investigation, and
hardware feasibility remain open. No later implementation stage is approved.
