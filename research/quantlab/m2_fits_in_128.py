#!/usr/bin/env python
"""Does the 110.8 GiB VQ 397B actually GENERATE on a 128 GB Mac?

The referee streams the model, so it proves quality, NOT residency. This
holds the whole thing in memory and generates at growing context lengths
until it swaps — turning "should fit" into a measured claim we can publish.

Escalates carefully and ABORTS on swap growth, because thrashing a 128 GB
box with a 110 GiB model makes it unusable rather than merely slow. Each
step reports peak memory and decode tok/s.

  ./m2_fits_in_128.py --model <artifact> [--steps 512,2048,8192,16384,32768]

Run it on a machine with nothing else resident. Reports the honest headline:
"runs on 128 GB up to N tokens of context".
"""
import argparse
import subprocess
import sys
import time

import mlx.core as mx


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    # total = 6144.00M  used = 4164.88M  free = 1979.12M
    for tok in out.split():
        if tok.endswith("M") and "used" in out[:out.index(tok)][-8:]:
            try:
                return float(tok[:-1])
            except ValueError:
                pass
    try:
        return float(out.split("used =")[1].split("M")[0])
    except Exception:
        return -1.0


def gib(b):
    return b / 2**30


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--steps", default="512,2048,8192,16384,32768")
ap.add_argument("--gen", type=int, default=32, help="tokens to generate per step")
ap.add_argument("--swap-abort-mb", type=float, default=2048.0,
                help="abort if swap grows this much over baseline")
args = ap.parse_args()

steps = [int(x) for x in args.steps.split(",")]
swap0 = swap_used_mb()
print(f"baseline swap: {swap0:.0f} MB", flush=True)

# let MLX wire the whole model; default limit is a fraction of RAM
try:
    total = mx.metal.device_info()["max_recommended_working_set_size"]
    print(f"max recommended working set: {gib(total):.1f} GiB", flush=True)
    mx.set_wired_limit(int(total))
except Exception as e:
    print(f"(could not raise wired limit: {e})", flush=True)

from mlx_lm import load, generate                       # noqa: E402
from mlx_lm.sample_utils import make_sampler            # noqa: E402

print(f"\nloading {args.model} …", flush=True)
t0 = time.time()
model, tokenizer = load(args.model)
mx.eval(model.parameters())
load_s = time.time() - t0
peak_load = mx.get_peak_memory()
print(f"  loaded in {load_s:.0f}s   resident/peak {gib(peak_load):.1f} GiB   "
      f"swap now {swap_used_mb():.0f} MB (+{swap_used_mb()-swap0:.0f})", flush=True)

if swap_used_mb() - swap0 > args.swap_abort_mb:
    print("\n!! model alone pushed the box into swap — DOES NOT FIT")
    sys.exit(1)

base = "The history of scientific instrumentation is a story about measurement. "
results = []
for n in steps:
    # build a prompt of roughly n tokens
    reps = max(1, n // max(1, len(tokenizer.encode(base))))
    prompt = base * reps
    ntok = len(tokenizer.encode(prompt))
    print(f"\n--- context {ntok:,} tokens ---", flush=True)
    mx.reset_peak_memory()
    t0 = time.time()
    try:
        out = generate(model, tokenizer, prompt=prompt,
                       max_tokens=args.gen, verbose=False,
                       sampler=make_sampler(temp=0.0))
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        results.append((ntok, None, None, "failed"))
        break
    dt = time.time() - t0
    peak = mx.get_peak_memory()
    sw = swap_used_mb() - swap0
    tps = args.gen / dt
    print(f"  peak {gib(peak):.1f} GiB   {tps:.1f} tok/s   swap +{sw:.0f} MB")
    print(f"  sample: {out[:80]!r}")
    results.append((ntok, gib(peak), tps, f"swap+{sw:.0f}MB"))
    if sw > args.swap_abort_mb:
        print(f"  !! swap grew {sw:.0f} MB — stopping here, this is the ceiling")
        break

print("\n================ VERDICT ================")
print(f"{'context':>10}  {'peak GiB':>9}  {'tok/s':>7}  note")
for n, pk, tps, note in results:
    print(f"{n:>10,}  {pk if pk is None else f'{pk:9.1f}':>9}  "
          f"{tps if tps is None else f'{tps:7.1f}':>7}  {note}")
good = [r for r in results if r[1] is not None and "failed" not in r[3]]
if good:
    print(f"\nRUNS ON THIS MACHINE up to {good[-1][0]:,} tokens of context "
          f"(peak {good[-1][1]:.1f} GiB)")
else:
    print("\nDOES NOT RUN on this machine")
