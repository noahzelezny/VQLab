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
import os
import time

import mlx.core as mx


def _stub(_self, x, *args, **kwargs):
    """Delete the module's work, keep its input alive. See module docstring."""
    return x * 0


def _switch_stub(_self, x, indices, *args, **kwargs):
    """SwitchGLU returns one row per (token, expert): (*indices.shape, D).
    The generic stub's x-shaped output would not combine with the scores."""
    return mx.broadcast_to(mx.expand_dims(x, -2) * 0,
                           (*indices.shape, x.shape[-1]))


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


def _compile_router(model):
    """REPLACEMENT arm (F200): fuse the MoE ROUTER and COMBINE with mx.compile.

    F200 measured the 35B's MoE block at 54% of decode against 28% for the
    routed SwitchGLU and 4% for the shared expert: ~3.8 ms/token sits in the
    ~11 small launches per layer around them (softmax, argpartition,
    take_along_axis, normalize, weight, reduce, sigmoid gate, add). The
    linears (gate, switch_mlp, shared_expert, shared_expert_gate) stay
    uncompiled -- the VQ kernels read host-side routing in prefill. Checksum
    MUST match baseline (see _compile_hc).
    """
    k_cache = {}

    def route(gates, k, norm):
        gates = mx.softmax(gates, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if norm:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def combine(y, scores, shared_y, sg):
        y = (y * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(sg) * shared_y

    comb = mx.compile(combine)

    def call(self, x):
        key = (self.top_k, bool(self.norm_topk_prob))
        if key not in k_cache:
            k_cache[key] = mx.compile(
                lambda g, _k=key[0], _n=key[1]: route(g, _k, _n))
        inds, scores = k_cache[key](self.gate(x))
        y = self.switch_mlp(x, inds)
        return comb(y, scores, self.shared_expert(x), self.shared_expert_gate(x))

    n = 0
    for _name, mod in model.named_modules():
        cls = type(mod)
        if (cls.__name__.endswith("SparseMoeBlock") and hasattr(mod, "shared_expert_gate")
                and getattr(mod, "sharding_group", None) is None):
            mod.__class__ = type(f"Compiled{cls.__name__}", (cls,), {"__call__": call})
            n += 1
    if n == 0:
        raise SystemExit("router-compile matched nothing -- refusing to run.")
    return n


def _delete_gdn_scan(model):
    """DELETION arm (F200, R3-3): delete ONLY the gated-delta recurrence.

    The `gdn` arm deletes the whole GatedDeltaNet (projections, conv, norm,
    recurrence). This one rebinds `gated_delta_update` in every loaded arch
    module that imported it, so the projections and conv still run and only
    the sequential scan is gone: gdn - gdn-scan = the projections' share.
    The stub keeps q, k, a, b alive (F48: a stub must depend on its inputs).
    """
    import sys as _sys

    def stub(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
             use_kernel=True):
        live = (q.sum() + k.sum() + a.sum() + b.sum()) * 0
        if state is None:
            B, _, _, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        return v * 0 + live.astype(v.dtype), state

    n = 0
    for name, mod in list(_sys.modules.items()):
        if mod is not None and name.startswith(("mlx_lm.", "vqlab", "knurlogic")) \
                and getattr(mod, "gated_delta_update", None) is not None \
                and name != "mlx_lm.models.gated_delta":
            mod.gated_delta_update = stub
            n += 1
    if n == 0:
        raise SystemExit("gdn-scan matched nothing -- refusing to run.")
    return n


def _chunk_gdn_scan(model):
    """REPLACEMENT arm (R3-3): rebind gated_delta_update to the chunked WY
    form (vqlab.runtime.gated_delta_chunked) for T >= VQ_GDN_CHUNK_MIN_T;
    decode stays on the stock kernel. CHANGES NUMERICS (fp32, different
    summation order): its checksum is NOT expected to match baseline.
    """
    import sys as _sys
    from vqlab.runtime.gated_delta_chunked import gated_delta_update as gdu
    n = 0
    for name, mod in list(_sys.modules.items()):
        if mod is not None and name.startswith(("mlx_lm.", "vqlab", "knurlogic")) \
                and getattr(mod, "gated_delta_update", None) is not None \
                and name != "mlx_lm.models.gated_delta" \
                and not name.startswith("vqlab.runtime.gated_delta_chunked"):
            mod.gated_delta_update = gdu
            n += 1
    if n == 0:
        raise SystemExit("gdn-chunked matched nothing -- refusing to run.")
    return n


def _compile_gdn(model, fuse_proj=False):
    """REPLACEMENT arm (F200): fuse the pure segments AROUND the gated-delta
    scan with mx.compile, for qwen3_5-style GatedDeltaNet (in_proj_qkv/z/b/a).

    F200's gdn-scan arm put ~0% of decode and ~10% of prefill in the
    recurrence and the rest of GDN's 27%/34% in the work around it: conv,
    silu, split, a hand-written q/k L2 norm (~8 elementwise ops each), and
    the gated RMSNorm. Segment 1 = conv_input -> (q, k, v); segment 2 =
    norm(out, z). Cache writes, projections and the scan stay outside the
    trace. Checksum (and logits) MUST match baseline.
    """
    import sys as _sys
    n = 0
    for _name, mod in model.named_modules():
        cls = type(mod)
        if not (cls.__name__.endswith("GatedDeltaNet")
                and hasattr(mod, "in_proj_qkv") and hasattr(mod, "in_proj_a")):
            continue
        gdu = getattr(_sys.modules[cls.__module__], "gated_delta_update")

        def seg1(conv_input, _m=mod):
            conv_out = mx.sigmoid(_c := _m.conv1d(conv_input)) * _c
            B, S = conv_out.shape[0], conv_out.shape[1]
            q, k, v = [
                t.reshape(B, S, h, d) for t, h, d in zip(
                    mx.split(conv_out, [_m.key_dim, 2 * _m.key_dim], -1),
                    [_m.num_k_heads, _m.num_k_heads, _m.num_v_heads],
                    [_m.head_k_dim, _m.head_k_dim, _m.head_v_dim])]
            inv_scale = k.shape[-1] ** -0.5
            q = inv_scale * q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6)
            k = k * mx.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)
            return q, k, v

        if fuse_proj:
            # gdn-fuseproj: qkv+z as ONE affine matmul (same bits/group/mode,
            # rows concatenated) and b+a as ONE bf16 matmul -- 4 launches -> 2.
            import mlx.nn as _nn
            p, zz = mod.in_proj_qkv, mod.in_proj_z
            if not (isinstance(p, _nn.QuantizedLinear) and isinstance(zz, _nn.QuantizedLinear)
                    and (p.bits, p.group_size, getattr(p, "mode", None))
                    == (zz.bits, zz.group_size, getattr(zz, "mode", None))):
                raise SystemExit("gdn-fuseproj: qkv/z are not matching QuantizedLinears")
            fq = _nn.QuantizedLinear(p.group_size, 32, bias=False, group_size=p.group_size, bits=p.bits)
            fq.weight = mx.concatenate([p.weight, zz.weight], 0)
            fq.scales = mx.concatenate([p.scales, zz.scales], 0)
            fq.biases = mx.concatenate([p.biases, zz.biases], 0)
            if hasattr(p, "mode"):
                fq.mode = p.mode
            mod._vq_qkvz, mod._vq_nqkv = fq, p.weight.shape[0]
            mod._vq_ba = mx.concatenate([mod.in_proj_b.weight, mod.in_proj_a.weight], 0)
            mod._vq_nb = mod.in_proj_b.weight.shape[0]
            mx.eval(fq.parameters(), mod._vq_ba)
        mod._vq_seg1 = mx.compile(seg1)
        mod._vq_seg2 = mx.compile(lambda out, z, _m=mod: _m.norm(out, z))

        def call(self, inputs, mask=None, cache=None, _gdu=gdu):
            B, S, _ = inputs.shape
            if hasattr(self, "_vq_qkvz"):
                qkvz = self._vq_qkvz(inputs)
                qkv, z = qkvz[..., :self._vq_nqkv], qkvz[..., self._vq_nqkv:]
                if os.environ.get("VQ_LADDER_FUSE_BA", "1") == "1":
                    ba = inputs @ self._vq_ba.T
                    b, a = ba[..., :self._vq_nb], ba[..., self._vq_nb:]
                else:
                    b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
            else:
                qkv = self.in_proj_qkv(inputs)
                z = self.in_proj_z(inputs)
                b = self.in_proj_b(inputs)
                a = self.in_proj_a(inputs)
            z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim),
                                      dtype=inputs.dtype)
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                n_keep = self.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
            q, k, v = self._vq_seg1(conv_input)
            state = cache[1] if cache else None
            out, state = _gdu(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                              use_kernel=not self.training)
            if cache is not None:
                cache[1] = state
                cache.advance(S)
            return self.out_proj(self._vq_seg2(out, z).reshape(B, S, -1))

        mod.__class__ = type(f"Compiled{cls.__name__}", (cls,), {"__call__": call})
        n += 1
    if n == 0:
        raise SystemExit("gdn-compile matched nothing -- refusing to run.")
    return n


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
            fn = (_hc_stub if cls.__name__.endswith("GatedResidual")
                  else _switch_stub if label == "switch" else _stub)
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


# Arms that preserve the model's arithmetic. Their checksum must MATCH the
# baseline; deletion arms' must differ. See the print at the end of main().
_REPLACEMENT_ARMS = {"hc-compile", "router-compile", "gdn-compile", "gdn-fuseproj"}

ARMS = {
    "baseline":   (lambda n, t: False, "untouched"),
    "gdn":        (lambda n, t: n.endswith("linear_attn"),
                   "GatedDeltaNet deleted (42.0% of Flash bytes/token, F130)"),
    "fullattn":   (lambda n, t: n.endswith("self_attn"),
                   "full attention deleted (12.4% of bytes/token)"),
    "vq":         (lambda n, t: t.startswith("VQSwitch"),
                   "VQ expert module deleted (12.1% of bytes/token; F48 ref)"),
    "sharedexp":  (lambda n, t: n.endswith(("shared_expert", "shared_experts")),
                   "dense shared expert deleted (4.7% of bytes/token)"),
    # The MoE partition (F200): moe - switch = router + shared expert +
    # combine; switch - vq = SwitchGLU glue (sort, activation, scatter).
    "moe":        (lambda n, t: t.endswith(("SparseMoeBlock", "MoEBlock", "MoE")),
                   "whole MoE block deleted (router, experts, shared, combine)"),
    "switch":     (lambda n, t: n.endswith("switch_mlp"),
                   "routed SwitchGLU deleted (VQ linears + sort/act/scatter glue)"),
    "gdn-scan":   (lambda n, t: False,
                   "gated-delta RECURRENCE deleted, projections/conv kept"),
    "gdn-chunked": (lambda n, t: False,
                   "gated-delta scan replaced by the chunked WY form at prefill "
                   "(numerics change: checksum NOT expected to match)"),
    "gdn-fuseproj": (lambda n, t: False,
                   "gdn-compile + qkv/z and b/a projections merged "
                   "(replacement arm: checksum MUST match baseline)"),
    "gdn-compile": (lambda n, t: False,
                   "GDN conv/norm segments FUSED with mx.compile "
                   "(replacement arm: checksum MUST match baseline)"),
    "router-compile": (lambda n, t: False,
                   "MoE router + combine FUSED with mx.compile "
                   "(replacement arm: checksum MUST match baseline)"),
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

    # First non-mlx.nn class in the MRO. __mro__[1] alone is right for a VQ
    # BUNDLE (whose Model subclasses the arch's Model) but reports
    # mlx.nn.layers.base for a stock affine model, where the parent is just
    # nn.Module -- i.e. it printed nothing useful on exactly the comparator
    # arms you most need to identify.
    arch = next((c.__module__ for c in type(model).__mro__
                 if not c.__module__.startswith(("mlx.nn", "builtins"))),
                type(model).__module__)
    import sys as _sys
    print(f"arch={arch}  file={getattr(_sys.modules.get(arch), '__file__', '?')}",
          flush=True)

    predicate, label = ARMS[a.arm]
    if a.arm == "baseline":
        hits = 0
    elif a.arm == "hc-compile":
        hits = _compile_hc(model)
    elif a.arm == "gdn-compile":
        hits = _compile_gdn(model)
    elif a.arm == "gdn-fuseproj":
        hits = _compile_gdn(model, fuse_proj=True)
    elif a.arm == "gdn-scan":
        hits = _delete_gdn_scan(model)
    elif a.arm == "gdn-chunked":
        hits = _chunk_gdn_scan(model)
    elif a.arm == "router-compile":
        hits = _compile_router(model)
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
            # ALL positions, not just the last. A single final-token argmax
            # is far too weak a channel: the vq arm changed prefill time by
            # 30% and still returned the baseline token, because one draw
            # from a 248320-vocab argmax collides easily. The decode side
            # sums 200 tokens; this must sum every position or it cannot do
            # the job F129 requires of it.
            checksum = int(mx.sum(mx.argmax(logits[0], axis=-1)).item())
        best = min(times)
        spread = (max(times) - min(times)) / min(times) * 100
        print(f"  prefill {best:7.3f} s   {len(ids)/best:8.1f} tok/s   "
              f"best-of-{a.reps} spread {spread:4.1f}%", flush=True)
        _kind = ("NUMERICS CHANGE: checksum may differ; gate on kernel-truth/KL"
                 if a.arm == "gdn-chunked" else
                 "REPLACEMENT: checksum must MATCH baseline"
                 if a.arm in _REPLACEMENT_ARMS else
                 "baseline" if a.arm == "baseline" else
                 "DELETION: checksum must DIFFER from baseline")
        print(f"  output_checksum {checksum}  ({_kind})", flush=True)
        _print_mem()
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
    # The EXPECTED DIRECTION depends on the arm type, and stating one rule
    # for both is how this line told six of eight F136 arms the wrong thing.
    # A DELETION arm produces wrong output, so its checksum must MOVE. A
    # REPLACEMENT arm (a bit-exact switch, a fusion) must leave it IDENTICAL;
    # a moved checksum there means the arithmetic changed and the timing is
    # an instrument change, not a free win (F120, and F136's VQ_DENSE_SS).
    _kind = ("REPLACEMENT: checksum must MATCH baseline"
             if a.arm in _REPLACEMENT_ARMS else
             "baseline" if a.arm == "baseline" else
             "DELETION: checksum must DIFFER from baseline")
    print(f"  output_checksum {checksum}  ({_kind})", flush=True)
    _print_mem()
    return 0


def _print_mem():
    # Resident cost of the arm: weights plus any load-time scratch (e.g.
    # VQ_PREFILL_EXPAND). Peak includes the transient activations.
    gib = 1024 ** 3
    print(f"  memory active {mx.get_active_memory() / gib:6.2f} GiB   "
          f"peak {mx.get_peak_memory() / gib:6.2f} GiB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
