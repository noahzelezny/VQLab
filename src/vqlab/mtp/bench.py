"""Measure what MTP drafting is worth on a given model.

Separate from the generation path on purpose: benchmarking decodes the same
prompt twice, holds both streams, and runs a numerics control that has no
business in a library decode loop.

Three numbers come out and each answers a different question:

  speedup       speculative tok/s over plain single-token tok/s. Both paths
                are warmed first — cold Metal kernels alone were measured to
                move this from 1.62x to 1.16x on identical weights.
  acceptance    fraction of steps whose drafted token survived verification.
                This is the speed knob; it cannot affect quality.
  chunk control does the trunk reproduce its OWN single-token greedy choices
                when the same tokens are fed as one chunk? This separates
                runtime numerics — which every correct speculative
                implementation on MLX inherits — from a rollback bug. The gate
                is that any divergence sits at near-ties: compare the reported
                top-2 logit gaps at the divergent positions against the median.
"""
from __future__ import annotations

import time
from typing import Optional

import mlx.core as mx

from .loop import _encode, mtp_stream_generate
from .sampling import make_distribution


def _plain_decode(model, ids, n_tokens, dist=None):
    """Stock single-token decoding, same model and sampler. The baseline."""
    cache = model.make_cache()
    logits = model(ids, cache=cache)
    row = logits[:, -1]
    t = mx.argmax(row, axis=-1) if dist is None else dist(row).sample()
    mx.eval(t)
    out = []
    start = time.perf_counter()
    for _ in range(n_tokens):
        out.append(int(t.item()))
        row = model(t[None], cache=cache)[:, -1]
        t = mx.argmax(row, axis=-1) if dist is None else dist(row).sample()
        mx.eval(t)
    return out, time.perf_counter() - start


def benchmark(model, tokenizer, prompt, head, *, tokens: int = 128,
              temp: float = 0.0, warmup: int = 8, family: Optional[str] = None,
              align: str = "committed", **kwargs) -> dict:
    """Run both decode paths on `prompt` and return the comparison record."""
    ids = _encode(tokenizer, prompt)
    dist = make_distribution(temp)

    def speculative(n):
        toks, last = [], None
        for r in mtp_stream_generate(model, tokenizer, ids, head,
                                     max_tokens=n, temp=temp, family=family,
                                     align=align, **kwargs):
            if not r.tail:
                toks.append(r.token)
            last = r
        return toks, last

    speculative(warmup)                       # warm the 2-token kernels
    t0 = time.perf_counter()
    spec_out, last = speculative(tokens)
    spec_s = time.perf_counter() - t0

    _plain_decode(model, ids, warmup, dist)   # warm the seq=1 kernels
    base_out, base_s = _plain_decode(model, ids, tokens, dist)

    rec = {
        "tokens": len(spec_out),
        "temp": temp,
        "speculative_tok_s": round(len(spec_out) / spec_s, 2),
        "baseline_tok_s": round(len(base_out) / base_s, 2),
        "speedup": round((base_s / len(base_out)) / (spec_s / len(spec_out)), 3),
        "acceptance": round(last.acceptance, 4) if last else None,
        "steps": last.steps if last else None,
        "text": tokenizer.decode(spec_out),
    }

    if temp == 0:
        rec.update(_chunk_control(model, ids, base_out))
        rec["outputs_identical"] = base_out == spec_out
    return rec


def _chunk_control(model, ids, base):
    """Feed the baseline's own tokens back as one chunk and see where the
    trunk's greedy choice differs from what it produced one token at a time."""
    cache = model.make_cache()
    n = ids.shape[1]
    full = mx.concatenate([ids, mx.array([base[:-1]])], axis=1)
    row = model(full, cache=cache)[0, n - 1:].astype(mx.float32)
    pred = mx.argmax(row, axis=-1)
    top2 = mx.sort(row, axis=-1)[:, -2:]
    gaps = top2[:, 1] - top2[:, 0]
    mx.eval(pred, gaps)
    bad = [i for i, (p, b) in enumerate(zip(pred.tolist(), base)) if p != b]
    return {
        "chunk_control_disagreements": len(bad),
        "chunk_control_positions": len(base),
        "chunk_control_gaps": [round(float(gaps[i].item()), 3) for i in bad],
        "chunk_control_median_gap": round(float(mx.median(gaps).item()), 3),
    }
