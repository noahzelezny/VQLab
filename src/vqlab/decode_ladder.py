#!/usr/bin/env python
"""decode-ladder — per-component decode deletion arms, Flash-aware.

F47 decomposed 35B decode this way. The same arm on Flash CRASHED (F48) and
the slice was never split; F130 then found GatedDeltaNet is 42.0% of Flash's
per-token bytes, making it the ranked suspect. This is the probe that arm
needed.

THREE LESSONS ARE BAKED IN, each paid for:

* **Do not touch the cache.** F47's stub seeded `cache.keys` with a
  (1,1,0,1) zero tensor to keep mlx_lm's KVCache.state alive. Flash's linear
  layers carry `ArraysCache` (no `.keys`) and its full-attention layers
  `QSAKVCache`, whose `state` already branches on `keys is None` and then
  slices the seeded tensor. Both caches handle None correctly on their own.
  The seeding IS the crash.

* **Patch INSTANCES, not classes** (F48). A class-level stub of
  QuantizedLinear once deleted every attention projection because lm_head
  shares the class. But instance scoping CANNOT be done by assigning
  `mod.__call__`: Python resolves `mod(x)` through `type(mod).__call__` and
  never consults the instance dict for implicit special-method lookup. The
  first cut of this probe did exactly that, reported `patched_instances=36`,
  and produced a checksum byte-identical to baseline -- 36 assignments, zero
  effect. The correct move is to give each targeted instance its OWN
  throwaway subclass: type-based lookup then finds the stub, and no other
  object of the original class is touched.

* **A terminal stub MUST depend on its input** (F48). A free-standing
  `zeros_like(x)` makes the upstream graph dead code under laziness -- F48
  got "lm_head 78%" that way. Every stub here returns `x * 0`: one
  elementwise op, output deleted, input still forced. It also keeps the
  producer of `x` (the hyper-connection) honestly billed to its own arm.

ALL ARMS PRODUCE WRONG OUTPUT. Timing only. Quote RATIOS from one session,
never absolutes -- the ~100 GiB decode instrument is bimodal (rule III).
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


def _stub(_self, x, *args, **kwargs):
    """Delete the module's work, keep its input alive. See module docstring."""
    return x * 0


def _hc_stub(self, hyper):
    """Delete the hyper-connection's LEARNED MACHINERY, keep its plumbing.

    GatedResidual returns a 3-tuple and the decoder layer does
        injection = branch[..., None, :] * inject[..., None]
        hidden    = hyper + injection.reshape(*hyper.shape)
    so the shapes are load-bearing and the generic `x * 0` stub cannot be
    used. This drops hc_norm, both hc_dim x hc_lowrank linears and
    block_inject (the 0.682 GB/token) while preserving the residual contract:
    the mix becomes a plain mean over the hc streams, the gate becomes ones
    (the real gate is 2*sigmoid(.), centred on 1).

    ATTRIBUTE NAMES DIFFER BY RUNTIME. The artifact's bundled model.py
    resolves its arch from mlx_lm FIRST and mlx_vlm only as a fallback, and
    the two spell this class differently: mlx_lm's `GatedResidual` carries
    .hc/.d and a `block_inject_weight is None` sentinel, mlx_vlm's
    `Qwen4ExpGatedResidual` carries .hc_count/.hidden_size and omits the
    attribute entirely. Reading the wrong one cost this arm a run.
    """
    hc = getattr(self, "hc", None)
    if hc is None:
        hc = self.hc_count
    d = getattr(self, "d", None)
    if d is None:
        d = self.hidden_size
    mixed = hyper.reshape(*hyper.shape[:-1], hc, d).mean(axis=-2)
    if getattr(self, "block_inject_weight", None) is None:
        return mixed
    inject = mx.ones((*hyper.shape[:-1], hc), dtype=hyper.dtype)
    return mixed, hyper, inject


def _compile_hc(model):
    """REPLACEMENT arm, not a deletion: fuse each GatedResidual with mx.compile.

    The hc deletion arm measured 44.4% of Flash decode for 12.9% of the bytes
    -- ~1,200 dispatches per token across 97 modules, each a fixed chain of
    small elementwise ops (norm, two low-rank linears, silu, sigmoid,
    reshape, multiply, mean, gate, sigmoid) on a residual stream hc_count=4
    makes 4x wider than hidden. That is exactly the shape mx.compile fuses.

    Weights are captured as trace constants, which is valid for inference
    (they do not change between steps). The shapes are static at decode.

    THE CHECKSUM CHANNEL INVERTS HERE. A deletion arm MUST change the
    checksum. A numerics-preserving optimization MUST NOT: if this arm's
    checksum differs from baseline, the fusion changed the arithmetic and the
    speedup is not free, exactly as the CPU-stream load turned out not to be
    (F120). Equality is the gate, not a nicety.
    """
    n = 0
    for _name, mod in model.named_modules():
        if type(mod).__name__.endswith("GatedResidual"):
            orig = type(mod).__call__
            mod._vq_compiled = mx.compile(
                lambda h, _m=mod, _f=orig: _f(_m, h))
            cls = type(mod)
            mod.__class__ = type(
                f"Compiled{cls.__name__}", (cls,),
                {"__call__": lambda self, h: self._vq_compiled(h)})
            n += 1
    if n == 0:
        raise SystemExit("hc-compile matched nothing -- refusing to run.")
    return n


def _patch(model, predicate, label):
    """Re-type matching INSTANCES onto a stubbed subclass; return the count.

    Per-instance subclassing, not `mod.__call__ = ...`: see the module
    docstring. Each hit is verified by resolving __call__ through the TYPE,
    which is the path the interpreter will actually take.
    """
    n = 0
    for name, mod in model.named_modules():
        if predicate(name, type(mod).__name__):
            cls = type(mod)
            fn = _hc_stub if cls.__name__.endswith("GatedResidual") else _stub
            mod.__class__ = type(f"Deleted{cls.__name__}", (cls,),
                                 {"__call__": fn})
            if type(mod).__call__ is not fn:
                raise SystemExit(
                    f"arm '{label}': re-typing {name} did not take. Refusing "
                    "to report timings from unpatched arms (F129).")
            n += 1
    if n == 0:
        raise SystemExit(f"arm '{label}' matched NOTHING -- refusing to run. "
                         "A probe whose arms cannot be proven to differ is "
                         "the F129 defect: it reports 'no difference' whether "
                         "or not the edit took.")
    return n


ARMS = {
    "baseline":   (lambda n, t: False, "untouched"),
    "gdn":        (lambda n, t: n.endswith("linear_attn"),
                   "GatedDeltaNet deleted (42.0% of Flash bytes/token, F130)"),
    "fullattn":   (lambda n, t: n.endswith("self_attn"),
                   "full attention deleted (12.4% of bytes/token)"),
    "vq":         (lambda n, t: t.startswith("VQSwitch"),
                   "VQ expert module deleted (12.1% of bytes/token; F48 ref)"),
    "sharedexp":  (lambda n, t: n.endswith("shared_expert"),
                   "dense shared expert deleted (4.7% of bytes/token)"),
    "hc-compile": (lambda n, t: False,
                   "hyper-connections FUSED with mx.compile "
                   "(replacement arm: checksum MUST match baseline)"),
    "hc":         (lambda n, t: t.endswith("GatedResidual"),
                   "hyper-connection machinery deleted (12.9% of bytes/token)"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), default="baseline")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--mode", choices=("decode", "prefill"), default="decode",
                    help="prefill times ONE forward over --prefill-tokens; the "
                         "same arms then answer whether a component's decode "
                         "share mirrors at batch, where GEMVs become GEMMs and "
                         "per-row cost amortizes")
    ap.add_argument("--prefill-tokens", type=int, default=4096)
    a = ap.parse_args()

    from mlx_lm import load
    model, tok = load(a.art)

    arch = type(model).__mro__[1].__module__
    import sys as _sys
    print(f"arch={arch}  file={getattr(_sys.modules.get(arch), '__file__', '?')}",
          flush=True)

    predicate, label = ARMS[a.arm]
    if a.arm == "baseline":
        hits = 0
    elif a.arm == "hc-compile":
        hits = _compile_hc(model)
    else:
        hits = _patch(model, predicate, a.arm)
    print(f"arm={a.arm}  patched_instances={hits}  {label}", flush=True)

    word = ("the quick brown fox jumps over the lazy dog while considering "
            "distributed inference ")
    n_prompt = a.prefill_tokens if a.mode == "prefill" else a.prompt_tokens
    ids = tok.encode(word * (n_prompt // 13 + 2))[:n_prompt]

    if a.mode == "prefill":
        from mlx_lm.models.cache import make_prompt_cache
        times, checksum = [], None
        for rep in range(a.reps + 1):
            cache = make_prompt_cache(model)
            mx.synchronize()
            t0 = time.time()
            logits = model(mx.array([ids]), cache=cache)
            mx.eval(logits)
            mx.synchronize()
            if rep:
                times.append(time.time() - t0)
            checksum = int(mx.sum(mx.argmax(logits[:, -1, :], axis=-1)).item())
        best = min(times)
        spread = (max(times) - min(times)) / min(times) * 100
        print(f"  prefill {best:7.3f} s   {len(ids)/best:8.1f} tok/s   "
              f"best-of-{a.reps} spread {spread:4.1f}%", flush=True)
        print(f"  output_checksum {checksum}  (arms MUST differ here; equal "
              f"checksums across arms means the edit did not take)", flush=True)
        return 0

    # Manual step loop: one forward per token, GPU drained each step. It
    # UNDERSTATES absolute throughput ~12% vs stream_generate's async_eval
    # pipelining (F62) -- which is fine and intended, because every arm pays
    # the same understatement and only the RATIO is quoted.
    from mlx_lm.models.cache import make_prompt_cache

    times, checksum = [], None
    for rep in range(a.reps + 1):
        cache = make_prompt_cache(model)
        y = mx.array([ids])
        logits = model(y, cache=cache)
        mx.eval(logits)
        tokidx = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tokidx)

        mx.synchronize()
        t0 = time.time()
        acc = None
        for _ in range(a.tokens):
            logits = model(tokidx[None], cache=cache)
            tokidx = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tokidx)
            acc = tokidx if acc is None else acc + tokidx
        mx.synchronize()
        dt = time.time() - t0
        if rep:                       # rep 0 is warm-up
            times.append(dt)
        checksum = int(mx.sum(acc).item())

    best = min(times)
    ms = best / a.tokens * 1e3
    spread = (max(times) - min(times)) / min(times) * 100
    print(f"  ms/tok {ms:8.3f}   tok/s {1e3/ms:7.2f}   "
          f"best-of-{a.reps} spread {spread:4.1f}%", flush=True)
    # The channel that PROVES the arms differ, independent of timing (F129).
    print(f"  output_checksum {checksum}  (arms MUST differ here; equal "
          f"checksums across arms means the edit did not take)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
