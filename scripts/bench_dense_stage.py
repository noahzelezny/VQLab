#!/usr/bin/env python
"""Cost map of the DENSE packed-d4 tiled kernel, and the arc-5/arc-4 twins.

(2026-09-03, dense-kernel arc.  Sibling of scripts/bench_d8_stage.py, which
did this for the packed-d8 EXPERT kernel.)

WHY.  The 27B dense line (Qwen3.8-27B-VQ-3.9bpw and its 4.5/4.8 siblings) is
d=4 / K=4096 / pack_bits=12 / G=64, so every one of its 192 VQ linears
dispatches `_SRC_DENSE_PACKED_D4_TILED` at decode.  That kernel has BOTH
structures the expert arcs found money in:

  * it stages x into a threadgroup half4 tile in 32-group blocks, with two
    threadgroup_barriers per block  -> arc 5's `devx` shape;
  * it reduces with a 32-step serial simd_shuffle chain, character-identical
    to the packed-d8 reduction text -> arc 4's `simd_sum` shape.

Same discipline as the expert arcs: every `cost` arm is WRONG ON PURPOSE and
is therefore a LOWER BOUND on the cost of the thing it deletes; the
bit-identical arms have mx.array_equal asserted against base BEFORE any
timing is printed.

  cost arms
    empty       everything after the pointer setup deleted; one guarded
                store.  The launch + grid floor for this exact grid.
    nostage     the x staging loop deleted (barriers kept): prices the
                staging device LOADS + threadgroup STORES.
    nobar       both threadgroup_barriers deleted (staging kept): prices the
                BARRIERS.
    nostagebar  both deleted: the joint price.
    nored       the simd_shuffle reduction chain deleted: prices the
                REDUCTION.

  arms that could ship
    devx        no threadgroup x at all: each lane reads its own x slice from
                device as vec<T,4> and narrows with the SAME half4 cast the
                staging loop applied, so the values entering every dot -- and
                the dot/fma order -- are exactly base's.  BIT-IDENTICAL by
                construction.
    devx_ss     devx + arc 4's simd_sum reduction.  NOT bit-identical (the
                32 scaled group partials are summed in simd_sum's tree order
                instead of ascending fma order); reported as max-ULP, never
                as identical.

Real tensors, read-only, from the local 27B 3.9bpw artifact.  No model load.

Run:  PYTHONPATH=src:scripts python scripts/bench_dense_stage.py
"""
import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx

from vqlab import vq_switch as V

ART_DEFAULT = os.path.expanduser(
    "~/.exo/models/TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw")

BASE = V._SRC_DENSE_PACKED_D4_TILED
PACK_BITS = 12
G = 64

# --- the two texts every arm below rewrites ---------------------------------
_STAGE = """    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = half4((half)xrow[o], (half)xrow[o+1],
                          (half)xrow[o+2], (half)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""
assert _STAGE in BASE, "dense d4 staging text drifted from vq_switch.py"

_RED = """        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
"""
assert _RED in BASE, "dense d4 reduction text drifted from vq_switch.py"

_DOTS = """                gacc += dot(float4(cb[VQ_CODE(crow, j)]),   float4(xs[m]))
                      + dot(float4(cb[VQ_CODE(crow, j+1)]), float4(xs[m+1]))
                      + dot(float4(cb[VQ_CODE(crow, j+2)]), float4(xs[m+2]))
                      + dot(float4(cb[VQ_CODE(crow, j+3)]), float4(xs[m+3]));
"""
assert _DOTS in BASE, "dense d4 inner-dot text drifted from vq_switch.py"


def _stage(body, src=None):
    out = (src or BASE).replace(_STAGE, body)
    assert out != (src or BASE), "staging replace did not apply"
    return out


# ---- cost-only arms (WRONG on purpose; lower bounds) -----------------------
COST = {"base": BASE}
COST["nostage"] = _stage("""    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);
""")
COST["nobar"] = _stage("""    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NSUB - base);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = half4((half)xrow[o], (half)xrow[o+1],
                          (half)xrow[o+2], (half)xrow[o+3]);
        }
""")
COST["nostagebar"] = _stage("""    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
""")
# reduction deleted: gacc never leaves the lane, acc keeps the last gacc so
# the inner loop cannot be dead-code-eliminated.
COST["nored"] = BASE.replace(_RED, """        acc += gacc;
""")
assert COST["nored"] != BASE
_TAIL = BASE[BASE.index("    float acc = 0.0f;"):]
COST["empty"] = BASE.replace(
    _TAIL, "    if (active && lane == 0) y[(size_t)t * OUT + r] = T(0);\n")
assert COST["empty"] != BASE

# ---- arms that could ship --------------------------------------------------
# devx: read x from device.  `half4(xr4[m])` is the elementwise (half) cast
# the staging loop applied, so this is value-identical for EVERY T -- not
# only for T=half, which a bare float4(xr4[m]) would have assumed.
_DEVX = _stage("""    const device vec<T,4>* xr4 = (const device vec<T,4>*)xrow;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = 0;
""").replace(_DOTS,
             """                gacc += dot(float4(cb[VQ_CODE(crow, j)]),   float4(half4(xr4[m])))
                      + dot(float4(cb[VQ_CODE(crow, j+1)]), float4(half4(xr4[m+1])))
                      + dot(float4(cb[VQ_CODE(crow, j+2)]), float4(half4(xr4[m+2])))
                      + dot(float4(cb[VQ_CODE(crow, j+3)]), float4(half4(xr4[m+3])));
""").replace("    threadgroup half4 xs[MAX_TILE];\n", "")
assert "xs[" not in _DEVX and "barrier" not in _DEVX, (
    "dense devx twin did not fully apply")

_DEVX_SS = _DEVX.replace(_RED, """        {
            const int gg = b * 32 + (int)lane;
            const float sv = (gg < NGRP) ? (float)srow[gg] : 0.0f;
            acc += simd_sum(sv * gacc);
        }
""")
assert _DEVX_SS != _DEVX, "dense simd_sum replace did not apply on devx"

SHIP = {"base": BASE, "devx": _DEVX, "devx_ss": _DEVX_SS}
# devx is bit-identical by construction; devx_ss reorders the reduction.
EXACT = ("devx",)


# ---- real tensors ----------------------------------------------------------
def load_proj(art, layer, proj):
    idx = json.load(open(os.path.join(
        art, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for part in ("codes", "codebook", "vq_scales"):
        k = f"language_model.model.layers.{layer}.mlp.{proj}.{part}"
        out[part] = mx.contiguous(mx.load(os.path.join(art, idx[k]))[k])
    mx.eval(list(out.values()))
    return out


def make_x(t, N, seed=0):
    IN = t["vq_scales"].shape[1] * G
    mx.random.seed(seed)
    x = mx.random.normal((N, IN)).astype(mx.float16)
    mx.eval(x)
    return x


def dispatch(name, src, x, codes, codebook, scales, rows=None):
    """Replica of the packed-d4 branch of V._dense_fused with the SOURCE
    pinned by the caller.  Names must start with `vq_dense` -- that is how
    V._get_kernel picks the eidx-free dense input signature."""
    rows = V._DENSE_ROWS_TG if rows is None else rows
    N, IN = x.shape
    OUT = codes.shape[0]
    K, D = codebook.shape
    assert D == 4
    dims = mx.array([OUT, IN, D, G, N, K], dtype=mx.int32)
    (y,) = V._get_kernel(name, src)(
        inputs=[x, codes, codebook, scales, dims],
        template=[("T", x.dtype), ("MAX_TILE", 32 * (G // 4)),
                  ("BITS", PACK_BITS)],
        grid=(32, ((OUT + rows - 1) // rows) * rows, N),
        threadgroup=(32, rows, 1),
        output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
    return y


def interleaved(arms, reps=15, warm=2):
    for fn in arms.values():
        for _ in range(warm):
            mx.eval(fn())
    acc = {n: [] for n in arms}
    for _ in range(reps):
        for name, fn in arms.items():
            o = fn()
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(o)
            acc[name].append(time.perf_counter() - t0)
    return {n: (min(v), statistics.median(v)) for n, v in acc.items()}


def _ulps(a, b):
    """max |a-b| in units of b's fp16 ULP, for the non-bit-identical arm."""
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    ulp = mx.maximum(mx.abs(bf), 6.1e-5) * (2.0 ** -10)
    return float(mx.max(mx.abs(af - bf) / ulp))


SHAPES = ((20, "gate_proj"), (20, "down_proj"))


def run(art, arms, prefix, reps, M, ns, exact=(), rows=None):
    for layer, proj in SHAPES:
        t = load_proj(art, layer, proj)
        codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
        IN = sc.shape[1] * G
        print(f"\n=== L{layer}.{proj}  OUT={codes.shape[0]} IN={IN} "
              f"NGRP={sc.shape[1]}  rows={rows or V._DENSE_ROWS_TG}")
        for N in ns[:1] if not exact else ns:
            if not exact:
                break
            x = make_x(t, N, seed=N)
            ref = dispatch(prefix + "base", BASE, x, codes, cbk, sc, rows)
            mx.eval(ref)
            for n in arms:
                if n == "base":
                    continue
                y = dispatch(prefix + n, arms[n], x, codes, cbk, sc, rows)
                mx.eval(y)
                if n in exact:
                    if not bool(mx.array_equal(ref, y)):
                        raise SystemExit(
                            f"BIT-IDENTITY FAILED: {n} at N={N} on {proj} "
                            f"-- no timing reported")
                else:
                    print(f"    {n:>10s} N={N:<3d} NOT-bit-identical arm: "
                          f"max {_ulps(y, ref):.2f} ULP vs base")
        print("      N  " + "".join(f"{n:>17s}" for n in arms))
        for N in ns:
            x = make_x(t, N)

            def mk(name):
                def fn(name=name):
                    return [dispatch(prefix + name, arms[name], x, codes,
                                     cbk, sc, rows) for _ in range(M)]
                return fn

            r = interleaved({n: mk(n) for n in arms}, reps=reps)
            print(f"    {N:3d}  " + "".join(
                f"{r[n][0]/M*1e6:8.1f}/{r[n][1]/M*1e6:6.1f}" for n in arms))
            print("         vs base min/med: " + "  ".join(
                f"{n}={r[n][0]/r['base'][0]:.3f}/{r[n][1]/r['base'][1]:.3f}"
                for n in arms if n != "base"))
        del t
        mx.clear_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART_DEFAULT)
    ap.add_argument("--mode", default="all",
                    choices=["all", "cost", "ship", "rows"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--dispatches", type=int, default=50)
    a = ap.parse_args()
    if not os.path.isdir(a.art):
        sys.exit(f"artifact not readable: {a.art}")
    if a.mode in ("all", "cost"):
        print("\n####### COST-ONLY ARMS: each WRONG on purpose, LOWER BOUNDS")
        run(a.art, COST, "vq_dense_dsc_", a.reps, a.dispatches, (1, 5, 10, 20))
    if a.mode in ("all", "ship"):
        print("\n####### SHIPPABLE ARMS (bit-identity asserted before timing)")
        run(a.art, SHIP, "vq_dense_dsi_", a.reps, a.dispatches,
            (1, 5, 10, 20), exact=EXACT)
    if a.mode == "rows":
        # arc 5 lesson: a structural change re-weights the occupancy point,
        # so the rows sweep is re-run on the NEW kernel, not inherited.
        for rows in (4, 8, 16, 32):
            print(f"\n####### ROWS={rows}")
            run(a.art, {"base": BASE, "devx": _DEVX}, f"vq_dense_r{rows}_",
                a.reps, a.dispatches, (1, 10, 20), rows=rows)
