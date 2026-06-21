#!/usr/bin/env python3
"""E1/E2: round-trip latency sweep against the ECP5 echo SoC (UDP or TCP).

Sweeps payload size, measures round-trip time per packet, and:
  E1  fits the affine transport model  RTT(s) = alpha + beta * s
      (alpha = fixed per-trip latency [s], beta = inverse bandwidth [s/byte])
  E2  dumps every RTT sample for latency-tail (p50/p99/p99.9) analysis.

Echo path: Mac -> FPGA -> echo firmware -> Mac. Measured on
ECP5 + LiteEth + VexRiscv; alpha/beta are reported as measured on that
platform, not as universal constants.

The board's UDP firmware echoes datagrams; the TCP firmware (lwIP) echoes a
byte stream on a persistent connection. Pick the one currently flashed.

Outputs (TAG = udp/tcp):
  transport/<TAG>_raw.csv      one row per (size, sample) RTT
  transport/<TAG>_summary.csv  per-size min/mean/p50/p99/p99.9
  transport/<TAG>_params.json  fitted alpha, beta, R^2

Usage:
    python transport/measure_rtt.py --transport tcp --ip 192.168.2.101
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SIZES = [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1400]


def make_payload(size: int) -> bytes:
    pattern = bytes(range(256))
    return (pattern * (size // 256 + 1))[:size]


def measure_udp(ip, port, src_port, size, samples, warmup, timeout):
    """One UDP socket; one sendto/recvfrom per sample. RTTs in microseconds."""
    payload = make_payload(size)
    rtts = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", src_port))
    sock.settimeout(timeout)
    try:
        for i in range(samples + warmup):
            t0 = time.perf_counter()
            sock.sendto(payload, (ip, port))
            try:
                data, _ = sock.recvfrom(4096)
                t1 = time.perf_counter()
            except socket.timeout:
                continue
            if data != payload:
                continue
            if i >= warmup:
                rtts.append((t1 - t0) * 1e6)
    finally:
        sock.close()
    return rtts


def _recv_exact(sock, n):
    """Read exactly n bytes from a stream socket (echo reply)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-reply")
        buf += chunk
    return bytes(buf)


def measure_tcp(ip, port, size, samples, warmup, timeout):
    """Persistent TCP connection; send payload, read it echoed back per sample."""
    payload = make_payload(size)
    rtts = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((ip, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # don't batch
    try:
        for i in range(samples + warmup):
            t0 = time.perf_counter()
            sock.sendall(payload)
            try:
                data = _recv_exact(sock, size)
                t1 = time.perf_counter()
            except (socket.timeout, ConnectionError):
                continue
            if data != payload:
                continue
            if i >= warmup:
                rtts.append((t1 - t0) * 1e6)
    finally:
        sock.close()
    return rtts


def pct(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", choices=["udp", "tcp"], default="tcp")
    ap.add_argument("--ip", default="192.168.2.101")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--src-port", type=int, default=5000, help="UDP only")
    ap.add_argument("--samples", type=int, default=200,
                    help="RTTs per payload size (more = better tail stats)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--outdir", type=Path, default=Path("transport"))
    args = ap.parse_args()
    tag = args.transport

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"sweeping {tag.upper()} {args.ip}:{args.port}  ({args.samples} samples/size)")
    print(f"{'size':>6}{'min':>9}{'mean':>9}{'p50':>9}{'p99':>9}{'p99.9':>10}  (us)")

    raw_rows, summary_rows = [], []
    sizes_fit, medians_fit = [], []
    for size in SIZES:
        try:
            if tag == "udp":
                rtts = measure_udp(args.ip, args.port, args.src_port, size,
                                   args.samples, args.warmup, args.timeout)
            else:
                rtts = measure_tcp(args.ip, args.port, size,
                                   args.samples, args.warmup, args.timeout)
        except OSError as e:
            print(f"{size:>6}   socket error: {e}")
            continue
        if not rtts:
            print(f"{size:>6}   no replies")
            continue
        mean = statistics.mean(rtts)
        row = {
            "size": size, "n": len(rtts),
            "min": min(rtts), "mean": mean,
            "p50": pct(rtts, 50), "p99": pct(rtts, 99), "p99_9": pct(rtts, 99.9),
            "max": max(rtts),
            "stdev": statistics.stdev(rtts) if len(rtts) > 1 else 0.0,
        }
        summary_rows.append(row)
        raw_rows.extend({"size": size, "rtt_us": r} for r in rtts)
        sizes_fit.append(size)
        medians_fit.append(row["p50"])
        print(f"{size:>6}{row['min']:>9.1f}{mean:>9.1f}{row['p50']:>9.1f}"
              f"{row['p99']:>9.1f}{row['p99_9']:>10.1f}")

    if not sizes_fit:
        sys.exit("no replies received -- check IP, that the right firmware "
                 "(udp/tcp) is flashed, and the static ARP/route on en5")

    # E1: least-squares fit RTT[s] = alpha + beta * size[bytes].
    # Fit on the per-size median (p50): robust to the heavy USB/TCP-retransmit
    # tail that otherwise inflates the mean at large payloads.
    x = np.array(sizes_fit, dtype=float)
    y = np.array(medians_fit, dtype=float) * 1e-6  # us -> s
    beta, alpha = np.polyfit(x, y, 1)            # slope, intercept
    resid = y - (beta * x + alpha)
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    params = {
        "transport": tag,
        "alpha_s": float(alpha), "alpha_us": float(alpha * 1e6),
        "beta_s_per_byte": float(beta), "beta_ns_per_byte": float(beta * 1e9),
        "r_squared": r2, "fit_basis": "per-size median (p50)",
        "platform": "ECP5 + LiteEth + VexRiscv",
    }

    raw_path = args.outdir / f"{tag}_raw.csv"
    sum_path = args.outdir / f"{tag}_summary.csv"
    par_path = args.outdir / f"{tag}_params.json"
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size", "rtt_us"]); w.writeheader(); w.writerows(raw_rows)
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)
    par_path.write_text(json.dumps(params, indent=2))

    print(f"\nE1 fit:  alpha = {alpha*1e6:.2f} us   beta = {beta*1e9:.3f} ns/byte   R^2 = {r2:.4f}")
    print(f"wrote {raw_path}, {sum_path}, {par_path}")


if __name__ == "__main__":
    main()
