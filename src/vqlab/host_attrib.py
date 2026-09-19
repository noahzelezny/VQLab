#!/usr/bin/env python3
"""vqlab host-attrib -- separate F41's unattributed host cost.

F41 measured the VQ module at 44% of prefill, the kernel body at 30, and left
~12-13 points named but never separated: "the per-call numpy tile build, the
mx.array uploads, kernel dispatch, and the broadcast/cast in __call__". Four
rounds of kernel work could not reach that 12-13%, and it is about the size of
the whole VQ-vs-affine parity gap.

ATTRIBUTION BEFORE DELETION, deliberately. F41 already carries an INVALID ARM
from the other order: stubbing np.argsort to an identity permutation ran
SLOWER than baseline, because handing the kernel unsorted expert order changes
its memory access pattern. That arm attributed nothing. A profiler does not
change semantics, so it cannot make that mistake.

SCOPE: host time only, which is the right scope. MLX is lazy -- wall time
around mx.array() is enqueue cost, not GPU work -- and the GPU side is already
attributed by F38/F39/F41. Anything that blocks on the device (notably
np.array(idx_flat) forcing a sync) will be MIS-attributed by any Python
profiler; those rows are marked and must not be read as compute cost.

    vqlab host-attrib <artifact> [--tokens 2048] [--top 25]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time


# Call sites that block on the device. Their self-time is WAIT, not work.
_SYNC_SUSPECT = ("np.array", "__array__", "tolist", "item", "eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab host-attrib",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--tokens", type=int, default=2048,
                    help="prompt length; the host cost scales with routed "
                         "rows, so a short prompt understates it")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=1,
                    help="un-profiled forward passes first: the first one "
                         "pays kernel compilation and memo misses, which are "
                         "startup, not steady state")
    a = ap.parse_args(argv)

    import mlx.core as mx
    from mlx_lm.utils import load

    print(f"loading {a.artifact} ...", flush=True)
    import inspect
    kw = {}
    if "trust_remote_code" in inspect.signature(load).parameters:
        kw["trust_remote_code"] = True
    model, tokenizer = load(a.artifact, **kw)

    ids = mx.array([[1] * a.tokens])

    def forward():
        out = model(ids)
        mx.eval(out)
        return out

    for i in range(a.warmup):
        t0 = time.perf_counter()
        forward()
        print(f"  warmup {i + 1}: {time.perf_counter() - t0:.3f}s", flush=True)

    t0 = time.perf_counter()
    forward()
    wall = time.perf_counter() - t0
    print(f"\nunprofiled steady-state forward: {wall:.3f}s "
          f"({a.tokens / wall:.0f} tok/s)")

    pr = cProfile.Profile()
    pr.enable()
    forward()
    pr.disable()

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats("tottime")
    st.print_stats(a.top)
    text = s.getvalue()

    prof_total = st.total_tt
    print(f"\nprofiled forward: {prof_total:.3f}s "
          f"(profiler overhead {prof_total / wall:.2f}x -- compare SHARES, "
          f"never absolutes)\n")

    print("HOST SELF-TIME, share of the profiled forward")
    print("  (rows marked SYNC block on the GPU; their time is WAIT, not "
          "work -- do not read them as compute)\n")
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6 or not parts[0].replace("/", "").isdigit():
            continue
        try:
            tottime = float(parts[1])
        except ValueError:
            continue
        where = parts[5]
        share = 100.0 * tottime / prof_total if prof_total else 0.0
        if share < 0.3:
            continue
        mark = "SYNC" if any(k in where for k in _SYNC_SUSPECT) else "    "
        print(f"  {share:5.1f}%  {tottime:7.3f}s  {mark}  {where[:96]}")

    print("\nReminder: shares are of HOST time under a profiler. To convert "
          "any row into a real prefill win you must build the change and A/B "
          "it unprofiled, one process per arm, ratio not absolute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
