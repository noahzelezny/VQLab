#!/usr/bin/env python3
"""Warmed PREFILL tok/s for ONE artifact (run once per side; never two loaded).

The decode twin (bench_decode_ab.py) times first-token-to-last and so
EXCLUDES prefill by construction. This measures the other half: wall time
from submit to the FIRST emitted token over a long prompt, which is the
prefill pass plus one decode step.

Same methodology as the decode harness, for the same reasons:
  * load(..., lazy=False) so weights are resident before timing
  * a full THROWAWAY generation first (the first run after a load pages
    weights in off SSD and is not a measurement)
  * greedy; three timed repeats; report each and the median
  * ONE process per arm -- never two artifacts/flag-sets in one process

Quote a RATIO between arms from the same session, never an absolute
(quantlab III: the decode instrument is bimodal at ~100 GiB).

    python scripts/bench_prefill_ab.py <artifact-dir> [--prompt-tokens 8192]
"""
import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("artifact")
ap.add_argument("--prompt-tokens", type=int, default=8192)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out")
a = ap.parse_args()

t0 = time.time()
model, tok = load(a.artifact, lazy=False)
print(f"loaded in {time.time() - t0:.1f}s   resident "
      f"{mx.get_active_memory() / 2**30:.2f} GiB", flush=True)

# A deterministic filler prompt of the requested token length. Content is
# irrelevant to prefill cost; LENGTH is the variable, so build it by tokens.
unit = ("The quick brown fox jumps over the lazy dog. Numbers: "
        "0 1 2 3 4 5 6 7 8 9. ")
ids = tok.encode(unit)
reps = max(1, a.prompt_tokens // max(1, len(ids)) + 1)
prompt = tok.encode(unit * reps)[:a.prompt_tokens]
print(f"prompt: {len(prompt)} tokens", flush=True)
sampler = make_sampler(temp=0.0)


def run():
    t = time.perf_counter()
    for resp in stream_generate(model, tok, prompt, max_tokens=1, sampler=sampler):
        pass
    dt = time.perf_counter() - t
    return len(prompt) / dt, dt


rate, dt = run()
print(f"THROWAWAY (cold): {rate:.1f} tok/s prefill ({dt:.2f}s)", flush=True)

rates = []
for i in range(a.repeats):
    r, dt = run()
    rates.append(r)
    print(f"timed run {i + 1}: {r:.1f} tok/s prefill ({dt:.2f}s)", flush=True)

res = {"artifact": a.artifact, "prompt_tokens": len(prompt), "rates": rates,
       "median": statistics.median(rates), "mean": statistics.fmean(rates),
       "peak_gib": mx.get_peak_memory() / 2**30}
print(json.dumps(res, indent=2))
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=2))
