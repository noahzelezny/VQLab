#!/usr/bin/env python3
"""Warmed decode tok/s for ONE artifact (run once per side; never two loaded).

Methodology, matched to the flash-next ledger's speed numbers:
  * mlx_lm.utils.load(..., lazy=False) so weights are resident before timing
  * a full THROWAWAY generation first (the first run after a load pages
    weights in off SSD at 2-3 tok/s and is not a measurement)
  * greedy, fixed token budget, first-token-to-last timing so prefill and
    the first-token latency are excluded from the decode rate
  * three timed repeats; report each and the median

    python scripts/bench_decode_ab.py <artifact-dir> [--tokens 378]
"""
import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

PROMPT = ("Write a detailed technical explanation of how vector quantization "
          "reduces the memory footprint of a mixture-of-experts language "
          "model, and what it costs in quality.")

ap = argparse.ArgumentParser()
ap.add_argument("artifact")
ap.add_argument("--tokens", type=int, default=378)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out")
a = ap.parse_args()

t0 = time.time()
model, tok = load(a.artifact, lazy=False)
print(f"loaded in {time.time() - t0:.1f}s   resident {mx.get_active_memory() / 2**30:.2f} GiB",
      flush=True)

msgs = [{"role": "user", "content": PROMPT}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
sampler = make_sampler(temp=0.0)


def run(n):
    first = None
    count = 0
    text = []
    for resp in stream_generate(model, tok, prompt, max_tokens=n, sampler=sampler):
        if first is None:
            first = time.perf_counter()          # first token emitted
        else:
            count += 1                            # tokens AFTER the first
        text.append(resp.text)
    last = time.perf_counter()
    return count / (last - first), count, "".join(text)


rate, n, text = run(a.tokens)
print(f"THROWAWAY (cold, weights paging in): {rate:.2f} tok/s over {n} tokens", flush=True)
print("--- sample of generated text (smoke) ---")
print(text[:300].replace("\n", " "))
print("---", flush=True)

rates = []
for i in range(a.repeats):
    r, n, _ = run(a.tokens)
    rates.append(r)
    print(f"timed run {i + 1}: {r:.3f} tok/s over {n} tokens", flush=True)

res = {"artifact": a.artifact, "tokens": a.tokens, "rates": rates,
       "median": statistics.median(rates), "mean": statistics.fmean(rates),
       "peak_gib": mx.get_peak_memory() / 2**30}
print(json.dumps(res, indent=2))
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=2))
