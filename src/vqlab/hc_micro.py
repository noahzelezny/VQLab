#!/usr/bin/env python
"""hc-micro — the hyper-connection chain in isolation: dtype x batch.

WHY THIS EXISTS. F131 measured the hyper-connections at 44.4% of Flash decode
for 12.9% of the bytes. F133 killed the mx.compile lever and left one
hypothesis: at batch 1 those 10240x320 skinny GEMVs may be SLOWER in affine
8-bit than in bf16, because the dequant amortizes over no batch. And the
mirror question -- prefill runs the same chain at thousands of rows, where
dequant amortizes and 8-bit should WIN -- decides whether a dtype change is a
win or a trade.

THE DEFECT THIS REPLACES (F133). The first version ran one `mx.eval` per op
and reported parts summing to 1667 us against a 410 us whole, with a
projection to FOUR outputs slower than the entire chain. It was timing
per-eval round-trip latency, not the ops. The fix is to build a DEPENDENT
CHAIN of k calls and eval once, which both amortizes the round trip and
matches how the model actually runs these: 97 modules back to back, each
waiting on the last.
"""

from __future__ import annotations

import argparse
import time

import json
import subprocess

import mlx.core as mx
import mlx.nn as nn


def gpu_watts() -> float:
    """Instantaneous GPU package watts, or -1 if macmon is unavailable.

    Utilization is NOT usable here: WindowServer and WebKit peg gpu_usage to
    60-95% while drawing 2-4 W, so a utilization gate both never opens and
    reads 'busy' on an idle box. Power separates compositing from compute.
    """
    try:
        out = subprocess.run(
            ["/opt/homebrew/bin/macmon", "pipe", "--interval", "700", "-s", "1"],
            capture_output=True, text=True, timeout=6).stdout.splitlines()
        return float(json.loads(out[0])["gpu_power"])
    except Exception:
        return -1.0

HC, D, LOWRANK = 4, 2560, 320
HCDIM = HC * D


class HCChain(nn.Module):
    """mlx_lm's GatedResidual, shapes taken from the shipped qwen4_exp config."""

    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(HCDIM)
        self.down = nn.Linear(HCDIM, LOWRANK, bias=False)
        self.up = nn.Linear(LOWRANK, HCDIM, bias=False)
        self.inject = nn.Linear(HCDIM, HC, bias=False)

    def __call__(self, h):
        normed = self.norm(h)
        w = nn.silu(self.down(normed) / HC)
        w = mx.sigmoid(self.up(w))
        w = w.reshape(*w.shape[:-1], HC, D)
        mixed = (w * normed.reshape(*normed.shape[:-1], HC, D)).mean(axis=-2)
        inj = 2 * mx.sigmoid(self.inject(normed) / HC)
        return mixed, h, inj


def bench(fn, x, chain: int, reps: int) -> float:
    """us per CALL, from a dependent chain of `chain` calls evaluated once.

    The chain is dependent on purpose: each call consumes the previous call's
    output (tiled back to hc width, the same op the model's trunk performs),
    so the GPU cannot overlap them and the number means what the model pays.
    """
    def once(h):
        mixed, _, _ = fn(h)
        return mx.tile(mixed, (1, 1, HC))      # restore hc width for the next link

    for _ in range(3):                          # warm caches + any compile trace
        h = x
        for _ in range(chain):
            h = once(h)
        mx.eval(h)
    mx.synchronize()

    best = float("inf")
    for _ in range(reps):
        t0 = time.time()
        h = x
        for _ in range(chain):
            h = once(h)
        mx.eval(h)
        mx.synchronize()
        best = min(best, time.time() - t0)
    return best / chain * 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 512, 4096],
                    help="rows per call: 1 is decode, the rest are prefill")
    ap.add_argument("--chain", type=int, default=48,
                    help="dependent calls per eval (Flash runs 97 per token)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-watts", type=float, default=25.0,
                    help="refuse to measure, and to REPORT, if GPU package "
                         "power exceeds this before or after the run")
    ap.add_argument("--dtype", choices=("bf16", "affine-8", "affine-4"),
                    default=None,
                    help="measure ONE arm and exit. RULE III: one process per "
                         "arm. Running all three in one process made the FIRST "
                         "arm absorb allocator/page-in warmup and produced a "
                         "5.7x swing on bf16 at batch 1 between runs -- larger "
                         "than every effect being measured.")
    a = ap.parse_args()

    # CONTENTION GATE. This bench times ~5 ms of work per rep, which makes it
    # far more fragile than a 10 s model run: a foreign GPU job moved bf16 at
    # batch 1 from 72 us to 1116 us, a 15x swing across identical invocations,
    # and silently inverted every dtype verdict. Numbers taken while the box
    # serves are garbage (F47), so refuse to produce them.
    w0 = gpu_watts()
    if w0 > a.max_watts:
        raise SystemExit(
            f"REFUSING: GPU at {w0:.1f} W (limit {a.max_watts:.0f} W). "
            "Another job is on this GPU; any number taken now is void.")

    print(f"\nGatedResidual chain  hc={HC} d={D} lowrank={LOWRANK} "
          f"(hc_dim={HCDIM})   chain={a.chain}, best-of-{a.reps}")

    arms = [("bf16", None), ("affine-8", 8), ("affine-4", 4)]
    if a.dtype:
        arms = [t for t in arms if t[0] == a.dtype]

    for b in a.batch:
        x = mx.random.normal((1, b, HCDIM)).astype(mx.float16)
        row = {}
        for tag, bits in arms:
            m = HCChain()
            if bits:
                nn.quantize(m, group_size=64, bits=bits)
            mx.eval(m.parameters())
            row[tag] = bench(m, x, a.chain, a.reps)
        regime = "DECODE" if b == 1 else "prefill"
        print(f"\n  batch {b:<5} ({regime})")
        for tag, us in row.items():
            # No cross-arm verdict here ON PURPOSE. Arms run in SEPARATE
            # PROCESSES (rule III, and --dtype exists for exactly that), so
            # one process holds one arm and a ratio computed here would
            # either KeyError or silently compare against a same-process
            # arm that absorbed allocator warmup. Compare across the log.
            print(f"    {tag:<9} {us:9.1f} us/call  {us/b:8.3f} us/row",
                  flush=True)
    # Re-check AFTER: a job that landed mid-run is exactly the case the
    # pre-gate cannot catch, and it is what voided this bench's first results.
    w1 = gpu_watts()
    if w1 > a.max_watts:
        raise SystemExit(
            f"\nVOID: GPU ended at {w1:.1f} W (started {w0:.1f} W, limit "
            f"{a.max_watts:.0f} W). A foreign job landed mid-run; discard "
            "every number above.")
    print(f"\n  contention gate PASSED: {w0:.1f} W before, {w1:.1f} W after"
          f" (limit {a.max_watts:.0f} W)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
