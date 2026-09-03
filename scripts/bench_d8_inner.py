#!/usr/bin/env python
"""Where the packed-d8 simd expert kernel's time actually goes.  (2026-09-02,
kernel arc 4.)

BACKGROUND.  `_SRC_FUSED_PACKED_D8_SIMD` is the hot path for 96% of the
Flash-Next 2.1bpw expert modules and for the 397B/GLM fleets.  Three prior arcs
refuted register buffering, codebook residency, wider code loads, dispatch
fusion, mx.compile and the rows/threadgroup shape (that last one shipped,
+3.6% end-to-end), leaving a standing diagnosis of "load-issue bound, ~4
dependent device loads per code, ~1 load issued per core per cycle".

THIS SCRIPT REFUTES THAT DIAGNOSIS AND REPLACES IT.  Deleting ANY ONE of the
inner loop's three loads buys only ~20%, so no single dependent chain owns the
cost; and a structure nobody had charged -- the 32-step sequentially dependent
`simd_shuffle` reduction that closes each block -- owns ~19% on its own.

ARMS.  `pipe` arms are BIT-IDENTICAL (asserted before any timing).  `cost`
arms are WRONG ON PURPOSE: each deletes work, so each is a LOWER BOUND on any
kernel that still has to do what it removed; they are never claimed as
speedups.  `ss` is the simd_sum reduction, which is real but not bit-identical.

  pipe  p2rot/p2pp/p3rot   depth-2 and depth-3 software pipelining of the
                           code fetch: issue code i+1's word load before
                           consuming code i's codebook loads.  NULL.
  pipe  sshuf/sshuf_h      the reduction's 32 scale loads replaced by
                           simd_shuffle of a per-lane scale.  Same value, same
                           fma order -> bit-identical.  NULL.
  cost  nocode/nogather/hotgather/warmgather/noxs/nored/noscale/redilp/noinner
  ss    simd_sum reduction: 1.16-1.20x through the real dispatcher, differs
        from base by one half-ULP on 0.05-0.16% of elements, ties against an
        fp32 reference.  Gated OFF by VQ_D8_SIMDSUM.

METHODOLOGY (the arc standard, and its three documented traps).
  * every variant JIT-warmed before timing;
  * arms timed INTERLEAVED (A,B,A,B) inside each rep so contention hits both;
  * headline is MIN over reps with the MEDIAN alongside; a claim holds on both;
  * `mx.eval` on the FULL output list -- evaluating only the last array leaves
    the rest dead and measures submit overhead;
  * `mx.eval` NEVER per dispatch -- that measures ~380 us of sync.

Run:  PYTHONPATH=src python scripts/bench_d8_inner.py [--mode all|pipe|cost|ss|num]
Reads three tensors per shape from the 2.1bpw artifact READ-ONLY (expert axis
sliced to E=64, ~1 GiB).  Never loads a model.
"""
import argparse
import json
import os
import statistics
import sys
import time

import mlx.core as mx

from vqlab import vq_switch as V

ART_DEFAULT = ("/Volumes/Thunderbay SSD/Exo Models/"
               "TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw")
E_SLICE = 64
BASE = V._SRC_FUSED_PACKED_D8_SIMD

# ---------------------------------------------------------------- loop bodies

_INNER = """            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
"""
_RED = """        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
"""
assert _INNER in BASE and _RED in BASE, (
    "kernel text drifted from vq_switch.py; every arm below would silently "
    "be a duplicate of the base kernel")


def _inner(body):
    return BASE.replace(_INNER, body)


def _red(body):
    return BASE.replace(_RED, body)


# --- BIT-IDENTICAL arms -----------------------------------------------------
PIPE = {"base": BASE}

# depth-2, rotation form: one code in flight, loop NOT unrolled (arc 3
# measured the unroll alone at 1.56x slower, so the unroll is avoided).
PIPE["p2rot"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            uint c0 = VQ_CODE(crow, j);
            for (int q = 0; q < SPG - 1; ++q, ++j, m += 2) {
                const uint c1 = VQ_CODE(crow, j + 1);
                gacc += dot(float4(cb4[2*c0]),   xs[m])
                      + dot(float4(cb4[2*c0+1]), xs[m+1]);
                c0 = c1;
            }
            gacc += dot(float4(cb4[2*c0]),   xs[m])
                  + dot(float4(cb4[2*c0+1]), xs[m+1]);
""")

# depth-2, explicit unroll-by-2 with compile-time ping/pong registers.
PIPE["p2pp"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            int q = 0;
            if (SPG >= 2) {
                uint ca = VQ_CODE(crow, j);
                for (; q + 2 <= SPG; q += 2, j += 2, m += 4) {
                    const uint cb1 = VQ_CODE(crow, j + 1);
                    gacc += dot(float4(cb4[2*ca]),   xs[m])
                          + dot(float4(cb4[2*ca+1]), xs[m+1]);
                    ca = (q + 2 < SPG) ? VQ_CODE(crow, j + 2) : 0u;
                    gacc += dot(float4(cb4[2*cb1]),   xs[m+2])
                          + dot(float4(cb4[2*cb1+1]), xs[m+3]);
                }
            }
            for (; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
""")

# depth-3: two codes in flight.
PIPE["p3rot"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            if (SPG >= 2) {
                uint c0 = VQ_CODE(crow, j);
                uint c1 = VQ_CODE(crow, j + 1);
                for (int q = 0; q < SPG - 2; ++q, ++j, m += 2) {
                    const uint c2 = VQ_CODE(crow, j + 2);
                    gacc += dot(float4(cb4[2*c0]),   xs[m])
                          + dot(float4(cb4[2*c0+1]), xs[m+1]);
                    c0 = c1; c1 = c2;
                }
                gacc += dot(float4(cb4[2*c0]),   xs[m])
                      + dot(float4(cb4[2*c0+1]), xs[m+1]);
                gacc += dot(float4(cb4[2*c1]),   xs[m+2])
                      + dot(float4(cb4[2*c1+1]), xs[m+3]);
            } else {
                const uint c = VQ_CODE(crow, j);
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
""")

# the reduction's 32 scale LOADS replaced by shuffles of a per-lane scale.
# simd_shuffle(sv, i) is exactly (float)srow[b*32+i], and the fma order is
# untouched, so this is bit-identical by construction.
PIPE["sshuf"] = _red("""        const int gmax = min(32, NGRP - b * 32);
        {
            const int gg = b * 32 + (int)lane;
            const float sv = (gg < NGRP) ? (float)srow[gg] : 0.0f;
            for (int i = 0; i < gmax; ++i)
                acc = fma(simd_shuffle(sv, (ushort)i),
                          simd_shuffle(gacc, (ushort)i), acc);
        }
""")
PIPE["sshuf_h"] = _red("""        const int gmax = min(32, NGRP - b * 32);
        {
            const int gg = b * 32 + (int)lane;
            const half sh = (gg < NGRP) ? srow[gg] : (half)0.0h;
            for (int i = 0; i < gmax; ++i)
                acc = fma((float)simd_shuffle(sh, (ushort)i),
                          simd_shuffle(gacc, (ushort)i), acc);
        }
""")

# --- COST-ONLY arms.  Every one computes a WRONG result on purpose. ---------
COST = {"base": BASE}
COST["nocode"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = (uint)(j & 1);
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
""")
COST["nogather"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += (float)c + xs[m].x;
            }
""")
COST["hotgather"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j) & 1u;
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
""")
# 128-entry (2 KiB) working set instead of 16384 (256 KiB): separates gather
# SCATTER from gather WORKING-SET SIZE, i.e. the codebook-residency question
# asked at the instruction level rather than the tensor level.
COST["warmgather"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j) & 127u;
                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
            }
""")
COST["noxs"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += dot(float4(cb4[2*c]),   float4(1.0f))
                      + dot(float4(cb4[2*c+1]), float4(1.0f));
            }
""")
COST["noinner"] = _inner("""            gacc = (float)g;
""")
COST["nored"] = _red("""        acc += gacc;
""")
COST["noscale"] = _red("""        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma(1.0f, simd_shuffle(gacc, (ushort)i), acc);
""")
COST["redilp"] = _red("""        const int gmax = min(32, NGRP - b * 32);
        float a0=0.0f, a1=0.0f, a2=0.0f, a3=0.0f;
        for (int i = 0; i < gmax; i += 4) {
            a0 = fma((float)srow[b*32+i+0], simd_shuffle(gacc,(ushort)(i+0)), a0);
            a1 = fma((float)srow[b*32+i+1], simd_shuffle(gacc,(ushort)(i+1)), a1);
            a2 = fma((float)srow[b*32+i+2], simd_shuffle(gacc,(ushort)(i+2)), a2);
            a3 = fma((float)srow[b*32+i+3], simd_shuffle(gacc,(ushort)(i+3)), a3);
        }
        acc += ((a0+a1)+(a2+a3));
""")

for _d in (PIPE, COST):
    for _n, _s in _d.items():
        assert (_s != BASE) == (_n != "base"), _n

SHAPES = ((20, "gate_proj"), (20, "up_proj"), (20, "down_proj"))


# ---------------------------------------------------------------- harness

def load_proj(art, layer, proj, e=E_SLICE):
    idx = json.load(open(os.path.join(art, "model.safetensors.index.json")))
    idx = idx["weight_map"]
    out = {}
    for part in ("codes", "codebook", "vq_scales"):
        k = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{part}"
        w = mx.load(os.path.join(art, idx[k]))[k]
        if part != "codebook" and e is not None:
            w = w[:e]
        out[part] = mx.contiguous(w)
    mx.eval(list(out.values()))
    return out


def dispatch(name, src, x, eidx, codes, codebook, scales, bits=14, rows=None):
    """Replica of the packed-d8-simd branch of V._fused with the SOURCE pinned
    by the caller, so a variant can be timed without touching the dispatcher.
    Kernel names must start with `vq_fused` -- that is how _get_kernel picks
    the input signature."""
    rows = V._EXPERT_ROWS_TG_D8_PACKED if rows is None else rows
    N, IN = x.shape
    E, OUT, _ = codes.shape
    K, D = codebook.shape
    assert D == 8
    G = IN // scales.shape[2]
    dims = mx.array([OUT, IN, D, G, N, K], dtype=mx.int32)
    (y,) = V._get_kernel(name, src)(
        inputs=[x, eidx, codes, codebook, scales, dims],
        template=[("T", x.dtype), ("MAX_TILE", 32 * (G // 8) * 2),
                  ("BITS", bits)],
        grid=(32, ((OUT + rows - 1) // rows) * rows, N),
        threadgroup=(32, rows, 1),
        output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
    return y


def make_inputs(t, N, seed=0):
    IN = t["vq_scales"].shape[2] * 64
    E = t["codes"].shape[0]
    mx.random.seed(seed)
    x = mx.random.normal((N, IN)).astype(mx.float16)
    eidx = (mx.arange(N, dtype=mx.uint32) % E).astype(mx.uint32)
    mx.eval(x, eidx)
    return x, eidx


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


def _timed(t, variants, prefix, reps, M, ns):
    codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
    for N in ns:
        x, eidx = make_inputs(t, N)

        def mk(name):
            def fn(name=name):
                return [dispatch(prefix + name, variants[name], x, eidx,
                                 codes, cbk, sc) for _ in range(M)]
            return fn

        r = interleaved({n: mk(n) for n in variants}, reps=reps)
        print(f"    {N:3d}  " + "".join(
            f"{r[n][0]/M*1e6:7.1f}/{r[n][1]/M*1e6:5.1f}" for n in variants))
        print("         vs base min/med: " + "  ".join(
            f"{n}={r[n][0]/r['base'][0]:.3f}/{r[n][1]/r['base'][1]:.3f}"
            for n in variants if n != "base"))


def run_pipe(art, reps, M, ns):
    print("\n########## BIT-IDENTICAL ARMS (identity asserted before timing)")
    for layer, proj in SHAPES[:2]:
        t = load_proj(art, layer, proj)
        codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
        print(f"\n=== L{layer}.{proj} OUT={codes.shape[1]} "
              f"IN={sc.shape[2]*64} NGRP={sc.shape[2]}")
        for N in ns:
            x, eidx = make_inputs(t, N, seed=N)
            ref = dispatch("vq_fused_a4p_base", BASE, x, eidx, codes, cbk, sc)
            mx.eval(ref)
            for n in PIPE:
                if n == "base":
                    continue
                y = dispatch("vq_fused_a4p_" + n, PIPE[n], x, eidx, codes,
                             cbk, sc)
                mx.eval(y)
                if not bool(mx.array_equal(ref, y)):
                    raise SystemExit(
                        f"BIT-IDENTITY FAILED: {n} at N={N} on {proj} "
                        f"-- no timing reported")
        print("      N  " + "".join(f"{n:>15s}" for n in PIPE))
        _timed(t, PIPE, "vq_fused_a4p_", reps, M, ns)
        del t
        mx.clear_cache()


def run_cost(art, reps, M, ns):
    print("\n########## COST-ONLY ARMS -- every one is WRONG on purpose and is"
          "\n########## a LOWER BOUND, never a speedup.")
    for layer, proj in SHAPES[:2]:
        t = load_proj(art, layer, proj)
        codes, sc = t["codes"], t["vq_scales"]
        print(f"\n=== L{layer}.{proj} OUT={codes.shape[1]} "
              f"IN={sc.shape[2]*64} NGRP={sc.shape[2]}")
        print("      N  " + "".join(f"{n:>15s}" for n in COST))
        _timed(t, COST, "vq_fused_a4c_", reps, M, ns)
        del t
        mx.clear_cache()


def run_ss(art, reps, M, ns):
    """The simd_sum reduction through the REAL dispatcher, VQ_D8_SIMDSUM off
    vs on.  down_proj is included precisely because it must NOT move: at
    NGRP=10 it is declined from the simd layout and has no reduction."""
    print("\n########## simd_sum reduction, REAL dispatcher (NOT bit-identical)")
    for layer, proj in SHAPES:
        t = load_proj(art, layer, proj)
        codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
        E, OUT, _ = codes.shape
        NGRP = sc.shape[2]
        per_e = codes.nbytes / E + sc.nbytes / E
        print(f"\n=== L{layer}.{proj} OUT={OUT} IN={NGRP*64} NGRP={NGRP} "
              f"(simd layout: {'YES' if NGRP >= 32 else 'NO -- thread-per-row'})")
        print("      N     off us(min/med)      on us(min/med)   "
              "GB/s off   GB/s on   speedup min/med")
        for N in ns:
            x, eidx = make_inputs(t, N)

            def mk(flag):
                def fn(flag=flag):
                    V._D8_SIMDSUM = flag
                    return [V._fused(x, eidx, codes, cbk, sc, pack_bits=14)
                            for _ in range(M)]
                return fn

            r = interleaved({"off": mk(False), "on": mk(True)}, reps=reps)
            V._D8_SIMDSUM = False
            by = per_e * N
            o, w = r["off"], r["on"]
            print(f"    {N:3d}   {o[0]/M*1e6:7.1f}/{o[1]/M*1e6:6.1f}     "
                  f"{w[0]/M*1e6:7.1f}/{w[1]/M*1e6:6.1f}   "
                  f"{by/(o[0]/M)/1e9:8.1f}  {by/(w[0]/M)/1e9:8.1f}   "
                  f"{o[0]/w[0]:.3f}/{o[1]/w[1]:.3f}")
        del t
        mx.clear_cache()


def run_num(art, ns):
    """How far from bit-identical is simd_sum, and in which direction?"""
    print("\n########## simd_sum numerical delta on real tensors")
    ss = V._SRC_FUSED_PACKED_D8_SIMD_SS
    for layer, proj in SHAPES[:2]:
        t = load_proj(art, layer, proj)
        codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
        print(f"\n=== L{layer}.{proj}")
        for N in ns:
            x, eidx = make_inputs(t, N, seed=N)
            a = dispatch("vq_fused_a4n_base", BASE, x, eidx, codes, cbk, sc)
            b = dispatch("vq_fused_a4n_ss", ss, x, eidx, codes, cbk, sc)
            mx.eval(a, b)
            af, bf = a.astype(mx.float32), b.astype(mx.float32)
            neq = int((a != b).sum())
            print(f"  N={N:3d}  differing {neq:6d}/{a.size:6d} "
                  f"({100*neq/a.size:5.2f}%)  "
                  f"max|delta|={float(mx.abs(af-bf).max()):.3e}")
        del t
        mx.clear_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART_DEFAULT)
    ap.add_argument("--mode", default="all",
                    choices=["all", "pipe", "cost", "ss", "num"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--dispatches", type=int, default=50)
    a = ap.parse_args()
    if not os.path.isdir(a.art):
        sys.exit(f"artifact not readable: {a.art}")
    ns = (1, 5, 10, 20)
    if a.mode in ("all", "pipe"):
        run_pipe(a.art, a.reps, a.dispatches, ns)
    if a.mode in ("all", "cost"):
        run_cost(a.art, a.reps, a.dispatches, (1, 10, 20))
    if a.mode in ("all", "ss"):
        run_ss(a.art, a.reps, a.dispatches, ns)
    if a.mode in ("all", "num"):
        run_num(a.art, (1, 10, 20))
