"""vqlab prefill-timeline — every stage of ONE prefill, in order, timed, summing to the whole.

The prefill counterpart of decode-timeline, and built on the same principle:
CUMULATIVE PREFIXES, never per-op evals (F133). Prefix k runs the model's own
forward with the decoder truncated after stage k; stage k costs
cum[k] - cum[k-1], measured as an interleaved pair so machine drift cancels.

Generic over architectures: it never re-implements a forward. It truncates
the trunk's `layers` list (every mlx-lm trunk iterates `zip(self.layers,
cache)`) and splits a layer into its two halves by replacing the last kept
layer's `mlp` with a stub. So a layer's stages are

    Lnn.attn   everything in the layer up to the MLP call (norms, attention,
               any residual / hyper-connection machinery around it)
    Lnn.mlp    the MLP call itself (dense, or MoE: router + experts + shared)

and `head` is what runs after the last layer (final norm + lm_head). Each
prefix still pays the head, which cancels in every delta except the first.

The stub returns `x * 0`: it must depend on its input or laziness turns the
upstream graph into dead code (F48).

Deltas are only a PARTITION if stages run serially; the tool compares their
sum against the independently timed whole and reports the drift.

    vqlab prefill-timeline --art <dir> [--tokens 1024] [--reps 1] [--stride 1]
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn


class _ZeroMLP(nn.Module):
    def __call__(self, x, *args, **kwargs):
        return x * 0


def _trunk(m, depth=0):
    if depth > 6:
        return None
    if hasattr(m, "layers") and hasattr(m, "embed_tokens"):
        return m
    for attr in ("language_model", "model"):
        sub = getattr(m, attr, None)
        if sub is not None and sub is not m:
            got = _trunk(sub, depth + 1)
            if got is not None:
                return got
    return None


def _kind(layer):
    attn = "lin" if getattr(layer, "is_linear", False) else "attn"
    mlp = getattr(layer, "mlp", None)
    moe = mlp is not None and any(hasattr(mlp, a) for a in ("switch_mlp", "experts", "gate"))
    return attn, ("moe" if moe else "dense")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=1,
                    help="timed pairs per stage after one warm pair")
    ap.add_argument("--stride", type=int, default=1,
                    help="time every Nth layer boundary; stages then cover N layers")
    ap.add_argument("--import", dest="imports", action="append", default=[],
                    help="MODULE[:FUNC[:ARG]] to import (and call) before loading, "
                         "e.g. to register an architecture the installed mlx-lm lacks")
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)

    for spec in a.imports:
        mod, _, rest = spec.partition(":")
        m = importlib.import_module(mod)
        if rest:
            fn, _, arg = rest.partition(":")
            getattr(m, fn)(*( [arg] if arg else []))

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(a.art)
    inner = _trunk(model)
    if inner is None:
        raise SystemExit("FAIL: no trunk carrying both `layers` and `embed_tokens`")
    layers = list(inner.layers)
    n = len(layers)

    text = ("the quick brown fox jumps over the lazy dog while considering "
            "distributed inference ")
    ids = tok.encode(text * (a.tokens // 13 + 2))[: a.tokens]
    x = mx.array([ids])

    # stage boundaries: (n_layers_kept, stub_last_mlp)
    bounds = []
    for i in range(0, n, a.stride):
        last = min(i + a.stride, n)
        if a.stride == 1:
            bounds.append((f"L{i:02d}.attn", last, True))
        bounds.append((f"L{i:02d}.mlp" if a.stride == 1 else f"L{i:02d}-{last - 1:02d}",
                       last, False))

    def run(kept, stub):
        # the cache is sized from the FULL layer list: forwards index it by
        # absolute layer position (fa_idx / ssm_idx), not by the kept count
        cache = make_prompt_cache(model)
        inner.layers = layers[:kept]
        saved = None
        if stub and kept:
            saved = layers[kept - 1].mlp
            layers[kept - 1].mlp = _ZeroMLP()
        try:
            mx.synchronize()
            t0 = time.perf_counter()
            out = model(x, cache=cache)
            out = getattr(out, "logits", out)
            mx.eval(out)
            mx.synchronize()
            return time.perf_counter() - t0
        finally:
            if saved is not None:
                layers[kept - 1].mlp = saved
            inner.layers = layers

    def pair(prev, cur):
        best_a = best_b = float("inf")
        for rep in range(a.reps + 1):
            ta, tb = run(*prev), run(*cur)
            if rep:
                best_a, best_b = min(best_a, ta), min(best_b, tb)
        return best_b - best_a, best_b

    print(f"\nprefill-timeline  {pathlib.Path(a.art).name}")
    print(f"{len(ids)} tokens, {n} layers, {len(bounds)} stages, best-of-{a.reps}\n", flush=True)

    head_only = min(run(0, False) for _ in range(a.reps + 1))
    whole = min(run(n, False) for _ in range(a.reps + 1))
    rows = [("head+embed", head_only)]
    prev = (0, False)
    for name, kept, stub in bounds:
        d, _ = pair(prev, (kept, stub))
        rows.append((name, d))
        print(f"  {name:12s} {d * 1e3:9.1f} ms", flush=True)
        prev = (kept, stub)

    total = sum(d for _, d in rows)
    drift = (total - whole) / whole * 100
    print(f"\n  whole prefill {whole:.3f} s ({len(ids) / whole:.0f} tok/s); "
          f"sum of stages {total:.3f} s, drift {drift:+.1f}%")

    groups = {}
    for (name, d), (_, kept, stub) in zip(rows[1:], bounds):
        if a.stride != 1:
            key = "layers"
        else:
            attn, mlp = _kind(layers[kept - 1])
            key = attn if stub else mlp
        groups[key] = groups.get(key, 0.0) + d
    groups["head+embed"] = head_only
    print("\n  BY STAGE TYPE")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v:8.3f} s  {v / total * 100:5.1f}%")
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps({
            "art": a.art, "tokens": len(ids), "whole_s": whole, "sum_s": total,
            "drift_pct": drift, "stages": rows, "groups": groups}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
