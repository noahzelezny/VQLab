#!/usr/bin/env python
"""gpu-capture — record ONE decode step's Metal work as an Xcode .gputrace.

The timelines (decode-timeline, prefill-timeline, stage-bandwidth) say
which STAGE is slow. They cannot say WHY a kernel is slow: whether it
waits on memory, runs too few threads to hide that wait, spills
registers, or sits idle between launches. That is the GPU's own
performance counters, and on Apple Silicon the way to read them is Xcode's
Metal debugger over a GPU capture: per-kernel duration, occupancy, memory
throughput, limiter percentages, and the gaps between dispatches.

This records the capture; Xcode reads it. It loads the artifact, prefills
a short context outside the capture, then captures exactly `--steps`
decode steps, so the trace holds the thing being asked about and nothing
else (a capture of a whole generation is gigabytes and unreadable).

Metal only records a capture when the process STARTED with
MTL_CAPTURE_ENABLED=1; the tool refuses without it rather than writing an
empty file.

    MTL_CAPTURE_ENABLED=1 vqlab gpu-capture --art <dir> --out step.gputrace
    open step.gputrace        # Xcode: Performance -> Counters / Timeline

Pin and smoke the artifact first (AGENTS.md "Two agents, one artifact
root"); a capture of a mid-rebundle model is a capture of the wrong code.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--family", default="qwen3_5")
    ap.add_argument("--out", required=True, help="a path ending .gputrace")
    ap.add_argument("--context", type=int, default=64,
                    help="prompt tokens prefilled OUTSIDE the capture")
    ap.add_argument("--steps", type=int, default=1,
                    help="decode steps inside the capture (keep it small)")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    if out.suffix != ".gputrace":
        raise SystemExit("--out must end in .gputrace (Xcode opens it by name)")
    if out.exists():
        raise SystemExit(f"{out} exists; Metal will not overwrite a capture")
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise SystemExit("REFUSING: start the process with MTL_CAPTURE_ENABLED=1;"
                         " without it Metal records nothing")

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_tokenizer

    from vqlab import _layout  # noqa: F401  one module object per name
    from vqlab import runtime_load

    model, _cfg = runtime_load.load_for_family(a.family, a.art, lazy=True)
    tok = load_tokenizer(pathlib.Path(a.art))
    text = "the quick brown fox jumps over the lazy dog "
    ids = tok.encode(text * (a.context // 9 + 2))[: a.context]

    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    nxt = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(nxt)
    # one uncaptured step first: first-call kernel compilation and lazy
    # allocations land here, not in the trace
    logits = model(nxt[:, None], cache=cache)
    nxt = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(nxt)
    mx.synchronize()

    mx.metal.start_capture(str(out))
    for _ in range(a.steps):
        logits = model(nxt[:, None], cache=cache)
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(nxt)
    mx.synchronize()
    mx.metal.stop_capture()

    print(f"\n  {a.steps} decode step(s) of {pathlib.Path(a.art).name} at "
          f"context {a.context} -> {out}")
    print("  open it in Xcode (`open` the file). In the Metal debugger:\n"
          "    Performance -> Timeline   dispatch order and the GAPS between "
          "kernels (launch/sync overhead)\n"
          "    Performance -> Counters   per kernel: duration, occupancy, "
          "memory bandwidth, ALU and memory limiter %\n"
          "  Sort kernels by duration; for the top few, a high memory "
          "limiter with low bandwidth means scattered access, low occupancy "
          "means registers or threadgroup memory cap the threads in flight.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
