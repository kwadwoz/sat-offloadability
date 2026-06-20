# sat-offloadability

Predicting, from a SAT formula's structure alone, whether offloading it to an
FPGA accelerator can ever beat solving it on a CPU, *before touching any
hardware.*

> **Status:** work in progress. Targeting a 2-page abstract for the ACM Student
> Research Competition at SPLASH/ISSTA 2026.

---

## The question

Hardware acceleration of SAT solving has an under-examined cost: every
host↔accelerator round trip pays a **fixed latency**, independent of how much
work is sent. Offloading only pays off when each round trip carries *enough*
work to amortize that fixed cost.

This project argues that whether a given instance clears that bar is a
**structural property of the formula**, driven by how much propagation work the
solver does per decision, and not a property of the accelerator. Some formulas
do large work-batches per round trip and are *offloadable*; others do tiny
batches and can *never* be worth offloading, regardless of how fast the
accelerator is.

The practical upshot: we can classify SAT instances as offloadable or not from
software-measurable structure, and validate that classification against a
measured link-latency model.

## The model

Round-trip transport time is affine in payload size `s`:

```
T(s) = alpha + beta * s
```

where `alpha` is the fixed per-round-trip latency and `beta` is inverse
bandwidth. With a hardware solve rate `R_hw` and `W` units of work offloaded per
trip, end-to-end throughput is:

```
P = W / (alpha + beta*s + W/R_hw)
```

In the small-message, latency-dominated regime, the break-even point against a
CPU solving at rate `R_cpu` reduces to a clean threshold:

```
W_min  ≈  alpha * R_cpu
```

i.e. an offload must carry at least as much work as the CPU could complete during
one round-trip latency. Instances whose per-trip work falls below `W_min` cannot
be accelerated by offload at any `R_hw`.

`R_hw` is treated as a **swept parameter** throughout, so the classification
result is independent of any particular accelerator implementation.

## Approach

| Step | What | Output |
|------|------|--------|
| Solver instrumentation | Run a mature solver (MiniSat) with statistics across SAT benchmark families | propagations-per-decision distributions |
| Transport measurement | RTT vs payload over UDP and TCP on an existing LiteX/VexRiscv ECP5 Ethernet SoC | `alpha`, `beta` per transport |
| Tail characterization | Round-trip latency distributions (p50/p99/p99.9), UDP vs TCP | latency CDFs |
| Classification | Combine the above with the model; compute `W_min`; classify instances | roofline + offloadability map |
| Soundness argument | Propagation is idempotent, deterministic, and monotone, so an unreliable transport preserves correctness (a lost packet costs time, never a wrong answer) | correctness rationale |

Latency parameters are reported as measured on a specific platform
(ECP5 + LiteEth + VexRiscv), not as universal constants. The classification
result holds across a swept range of `R_hw` and does not depend on any single
measured latency value.

## Repository layout

```
benchmarks/    SAT benchmark families (SATLIB / SAT-Comp) + reactive-synthesis CNFs
instrument/    run solver with stats; parse propagation-per-decision distributions -> CSV
transport/     UDP/TCP round-trip measurement harness for the Ethernet SoC
model/          W_min / roofline computation and instance classification
figures/        generated plots (RTT vs payload, latency CDF, roofline, distributions)
paper/          acmart abstract source
```

## Reproducing the figures

```bash
# 1. instrument the solver across benchmark families
python instrument/run_stats.py --benchmarks benchmarks/ --out instrument/props.csv

# 2. measure transport parameters (requires the SoC reachable on the LAN)
python transport/measure_rtt.py --transport udp --out transport/udp.csv
python transport/measure_rtt.py --transport tcp --out transport/tcp.csv

# 3. fit the model, classify instances, render figures
python model/classify.py --props instrument/props.csv \
                         --transport transport/ \
                         --figures figures/
```

*(Scripts land as the experiments are built out; this is the intended interface.)*

## Context

This work characterizes *which* SAT instances are worth offloading. The
achievable accelerator throughput `R_hw` (how a parallel solver datapath scales
with replication and area) is the subject of companion work and enters this
analysis only as a swept parameter.

## License

TBD.

