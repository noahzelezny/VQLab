#!/usr/bin/env python
"""Separate PREFILL from DECODE throughput — the two numbers people actually
quote, which m2_fits_in_128.py conflated.

That test divided 32 generated tokens by total wall time including prefill,
so at a 7.5k prompt it was mostly measuring prefill and badly understated
decode. Users feel decode (tok/s while it types); prefill is a one-time cost
per request. Publish both, never a blend.

Also uses REAL varied text (the referee corpus) rather than one sentence
repeated — a degenerate prompt makes the model echo the pattern, which
muddies whether output is sane.

  ./m2_speed_split.py --model <artifact> [--contexts 512,2048,8192]
"""
import argparse
import pathlib
import time

import mlx.core as mx


def gib(b):
    return b / 2**30


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--contexts", default="512,2048,8192")
ap.add_argument("--decode-tokens", type=int, default=64)
ap.add_argument("--corpus",
                default=str(pathlib.Path(__file__).parent / "referee/referee_corpus.txt"))
args = ap.parse_args()

try:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    print(f"device: {info.get('device_name')}  RAM {gib(info['memory_size']):.1f} GiB",
          flush=True)
    mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
except Exception as e:
    print(f"(wired limit: {e})", flush=True)

from mlx_lm import load                                    # noqa: E402
from mlx_lm.models.cache import make_prompt_cache          # noqa: E402

print(f"loading {args.model} …", flush=True)
t0 = time.time()
model, tokenizer = load(args.model)
mx.eval(model.parameters())
print(f"  loaded {time.time()-t0:.0f}s, resident {gib(mx.get_peak_memory()):.1f} GiB",
      flush=True)

text = open(args.corpus, encoding="utf-8", errors="ignore").read()
all_ids = tokenizer.encode(text)
print(f"corpus: {len(all_ids):,} real tokens\n", flush=True)

print(f"{'context':>9} {'prefill tok/s':>14} {'decode tok/s':>13} {'peak GiB':>9}")
for n in (int(x) for x in args.contexts.split(",")):
    if n > len(all_ids):
        print(f"{n:>9}  (corpus too short, skipped)")
        continue
    ids = mx.array(all_ids[:n])
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)

    # --- PREFILL: one forward pass over the whole prompt
    mx.eval(ids)
    t0 = time.time()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    t_pre = time.time() - t0

    # --- DECODE: token-by-token continuation reusing that cache
    tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(tok)
    t0 = time.time()
    for _ in range(args.decode_tokens):
        logits = model(tok[None] if tok.ndim == 1 else tok, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
    t_dec = time.time() - t0

    print(f"{n:>9} {n/t_pre:>14.1f} {args.decode_tokens/t_dec:>13.2f} "
          f"{gib(mx.get_peak_memory()):>9.1f}", flush=True)
    del cache
    mx.clear_cache()

print("\nprefill = one-time cost per request; decode = what the user feels.")
