#!/usr/bin/env python3
"""Referee perplexity through the runtime the artifact SHIPS, resident.

WHY NOT `vqlab score`. src/vqlab/referee/score_streaming.py streams one
decoder block at a time and calls `blk(h, mask=mask, cache=None)` against a
`blk.is_linear` flag. The qwen4_exp arch installed in the exo env (the
ml-explore PR #1788 graft) renamed that to `layer_type` and changed the
block signature to (h, rope, mask, conv_mask, cache, idx_cache, ids,
prev_ctx), with PLE layers needing `ids`/`prev_ctx`. The streaming referee
cannot be shimmed onto that without rewriting its loop, which would make
its number unvalidatable. These artifacts fit in RAM (45 GiB on a 96 GiB
box), so the streaming trick is not needed at all: run the whole model.

METRIC. Identical to the streaming referee's: prefix of the corpus, next-
token NLL over positions 1..n-1, exp(mean). Chunked prefill with a KV cache
is mathematically identical to one causal full-forward for a causal model.
Absolute values may differ in the last decimals from the published
streaming numbers (different code path, different arch revision) — the
DELTA between two artifacts scored here is on one instrument and is the
claim this experiment makes.

    python scripts/score_ppl_resident.py --model DIR [--corpus F] [--max-tokens 2048]
"""
import argparse
import json
import math
import pathlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

REF = pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab" / "referee"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", default=str(REF / "referee_corpus.txt"))
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--out")
a = ap.parse_args()

mx.set_cache_limit(8 << 30)
model, tok = load(a.model, lazy=False)

text = pathlib.Path(a.corpus).read_text(errors="replace")
ids = tok.encode(text)[: a.max_tokens + 1]
n = len(ids)
arr = mx.array(ids)
print(f"corpus {pathlib.Path(a.corpus).name}: {n} tokens", flush=True)

cache = make_prompt_cache(model)
total, scored = 0.0, 0
for s in range(0, n - 1, a.chunk):
    e = min(s + a.chunk, n - 1)
    logits = model(arr[s:e][None], cache=cache).astype(mx.float32)
    tgt = arr[s + 1:e + 1][None]
    nll = nn.losses.cross_entropy(logits, tgt, reduction="sum")
    mx.eval(nll)
    total += float(nll)
    scored += e - s
    del logits, nll
    mx.clear_cache()

ppl = math.exp(total / scored)
res = {"model": a.model, "corpus": pathlib.Path(a.corpus).name,
       "tokens": scored, "nll": total / scored, "ppl": ppl}
print(json.dumps(res, indent=2))
if a.out:
    open(a.out, "w").write(json.dumps(res, indent=2))
