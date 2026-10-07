"""Is this Mac quiet enough to time something? (structural feedback #5)

Instrument rule III says never time on a contended box, and it was a rule
nobody's tool enforced: on 2026-10-02 a tensor-split head A/B tracked the
M3's load average at r = -0.82 (19-21 tok/s at load ~30-35, 29-31 at ~5-7)
regardless of the head under test. The CLI now runs this before every timed
command and refuses a busy box; VQLAB_ALLOW_BUSY=1 overrides, and the state
is printed either way so it lands in the run log next to the numbers.
"""
from __future__ import annotations

import json
import os
import subprocess

# Commands whose OUTPUT is a timing. Everything else is free to run anywhere.
TIMED = {"speed-pair", "speed-pair-knurlogic", "prefill-bench", "decode-timeline",
         "prefill-timeline", "decode-ladder", "mtp-bench", "hc-micro",
         "gpu-capture"}
# Processes that own the GPU for minutes to hours.
HEAVY = ("fit-moe", "fit-dense", "fit moe", "fit dense", "geo-build", "fit_moe",
         "vq_397b_codes", "fit_dense_vq",
         "stream-score", "stream_score", "kl-ladder", "knurlogic serve", "exo ")
LOAD_FRACTION = 0.25          # load1 above a quarter of the cores is busy


def state() -> dict:
    ncpu = os.cpu_count() or 1
    load1 = os.getloadavg()[0]
    me = os.getpid()
    try:
        ps = subprocess.run(["ps", "-Ao", "pid=,args="], capture_output=True, text=True).stdout
    except OSError:
        ps = ""
    heavy = []
    for ln in ps.splitlines():
        pid, _, args = ln.strip().partition(" ")
        if pid.isdigit() and int(pid) != me and any(h in args for h in HEAVY) \
                and "box_quiet" not in args:
            heavy.append(args[:160])
    holder = None
    if not os.environ.get("VQLAB_QUEUE"):        # a queue running US holds the lease
        try:
            from vqlab.agents import mcp_server
            holder = mcp_server._lease_holder()
        except Exception:  # noqa: BLE001
            holder = None
    busy = []
    if load1 > LOAD_FRACTION * ncpu:
        busy.append(f"load average {load1:.1f} on {ncpu} cores (limit {LOAD_FRACTION * ncpu:.0f})")
    if heavy:
        busy.append(f"{len(heavy)} GPU-heavy process(es): {heavy[0]}")
    if holder:
        busy.append(f"GPU lease held by {holder.get('holder', holder)}")
    return {"load1": round(load1, 2), "ncpu": ncpu, "heavy": heavy,
            "lease_holder": holder, "busy": busy}


def guard(cmd: str) -> None:
    """Refuse a timed command on a busy box (SystemExit), unless overridden."""
    if cmd not in TIMED:
        return
    s = state()
    print(f"[box] {json.dumps({k: s[k] for k in ('load1', 'ncpu', 'busy')})}", flush=True)
    if s["busy"] and os.environ.get("VQLAB_ALLOW_BUSY") != "1":
        raise SystemExit(
            f"REFUSED: `{cmd}` produces timings and this Mac is busy:\n  - "
            + "\n  - ".join(s["busy"])
            + "\nA number measured now describes the contention, not the build (rule III). "
              "Wait, or set VQLAB_ALLOW_BUSY=1 and label the result contended.")
