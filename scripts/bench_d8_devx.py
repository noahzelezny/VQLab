#!/usr/bin/env python
"""Where the packed-d8 simd expert kernel's time goes NOW that devx is the
shipped default.  (2026-09-02, kernel arc 7 -- devx cost-map refresh.)

BACKGROUND.  Arc 4 mapped the OLD (staged) kernel: inner loop 43-49%,
reduction ~19%, staging+barriers+launch ~36%.  Arc 5 shipped `devx`
(VQ_D8_DEVX=1, on by default in vq_switch.py): the threadgroup tile, the
staging loop and both barriers are DELETED -- each lane reads its x slice
straight from device memory instead.  Arc 4's staging/barrier/launch split is
therefore stale for the kernel actually running today.  THIS SCRIPT RE-RUNS
THE SAME COST-ONLY ABLATION STYLE AGAINST `_SRC_FUSED_PACKED_D8_SIMD_DEVX` AS
BASE, to see where the remaining time goes and whether any single term is
concentrated enough (>25%) to be a plausible arc-6/7 target.

Every `cost` arm below is WRONG ON PURPOSE: each deletes work, so each is a
LOWER BOUND on any kernel that still has to do what it removed. Never report
an ablated arm as a speedup.

METHODOLOGY (the arc standard):
  * every variant JIT-warmed before timing;
  * arms timed INTERLEAVED (A,B,A,B) inside each rep so contention hits both;
  * headline is MIN over reps with the MEDIAN alongside; a claim holds on both;
  * `mx.eval` on the FULL output list;
  * `mx.eval` NEVER per dispatch.

Run:  PYTHONPATH=src:scripts python scripts/bench_d8_devx.py [--mode all|cost|ss]
Reads three tensors per shape from the 2.1bpw artifact READ-ONLY (expert axis
sliced to E=64, ~1 GiB). Never loads a model.
"""
import argparse
import os
import sys

import mlx.core as mx

from vqlab import vq_switch as V
from bench_d8_inner import ART_DEFAULT, load_proj, make_inputs, interleaved, dispatch

BASE = V._SRC_FUSED_PACKED_D8_SIMD_DEVX

# The devx inner loop: xs[m] reads replaced by device xr4[m] reads relative
# to the staged-kernel text bench_d8_inner.py ablated.
_INNER = """            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += dot(float4(cb4[2*c]),   float4(xr4[m]))
                      + dot(float4(cb4[2*c+1]), float4(xr4[m+1]));
            }
"""
_RED = """        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
"""
assert _INNER in BASE and _RED in BASE, (
    "devx kernel text drifted from vq_switch.py; every arm below would "
    "silently be a duplicate of the base kernel")


def _inner(body):
    src = BASE.replace(_INNER, body)
    assert src != BASE
    return src


def _red(body):
    src = BASE.replace(_RED, body)
    assert src != BASE
    return src


# ---- cost-only arms (WRONG on purpose; lower bounds) -----------------------
COST = {"base": BASE}
COST["nocode"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = (uint)(j & 1);
                gacc += dot(float4(cb4[2*c]),   float4(xr4[m]))
                      + dot(float4(cb4[2*c+1]), float4(xr4[m+1]));
            }
""")
COST["nogather"] = _inner("""            int j = g * SPG;
            int m = 2 * j - base;
            for (int q = 0; q < SPG; ++q, ++j, m += 2) {
                const uint c = VQ_CODE(crow, j);
                gacc += (float)c + (float)xr4[m].x;
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
# everything after the pointer setup deleted: launch + grid floor.
_TAIL = BASE[BASE.index("    float acc = 0.0f;"):]
COST["empty"] = BASE.replace(
    _TAIL, "    if (active && lane == 0) y[(size_t)t * OUT + r] = T(0);\n")
assert COST["empty"] != BASE

SHAPES = ((20, "gate_proj"), (20, "up_proj"), (20, "down_proj"))


def run_cost(art, reps, M, ns):
    print("\n########## devx COST-ONLY ARMS -- WRONG on purpose, LOWER BOUNDS")
    for layer, proj in SHAPES:
        t = load_proj(art, layer, proj)
        codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
        E, OUT, _ = codes.shape
        NGRP = sc.shape[2]
        per_e = codes.nbytes / E + sc.nbytes / E
        print(f"\n=== L{layer}.{proj} OUT={OUT} IN={NGRP*64} NGRP={NGRP}")
        print("      N  " + "".join(f"{n:>12s}" for n in COST) + "   GB/s(base)")
        for N in ns:
            x, eidx = make_inputs(t, N)

            def mk(name):
                def fn(name=name):
                    return [dispatch("vq_fused_a7c_" + name, COST[name], x,
                                      eidx, codes, cbk, sc)
                            for _ in range(M)]
                return fn

            r = interleaved({n: mk(n) for n in COST}, reps=reps)
            by = per_e * N
            gbs = by / (r["base"][0] / M) / 1e9
            print(f"    {N:3d}  " + "".join(
                f"{r[n][0]/M*1e6:6.1f}/{r[n][1]/M*1e6:4.1f}" for n in COST)
                + f"   {gbs:6.1f}")
            print("         vs base min/med: " + "  ".join(
                f"{n}={r[n][0]/r['base'][0]:.3f}/{r[n][1]/r['base'][1]:.3f}"
                for n in COST if n != "base"))
        del t
        mx.clear_cache()


def run_ss(art, reps, M, ns):
    """devx + simd_sum composition, module-global patch (arc 6 style):
    _SRC_FUSED_PACKED_D8_SIMD_SS is swapped in-process for the devx-derived
    twin, and the elif chain's _D8_SIMDSUM branch is used to dispatch it.
    NOT bit-identical (same half-ULP story as arc 4/6's SS)."""
    print("\n########## devx+simd_sum composition (module-global patch, "
          "NOT bit-identical)")
    devx_ss = BASE.replace(_RED, """        {
            const int gg = b * 32 + (int)lane;
            const float sv = (gg < NGRP) ? (float)srow[gg] : 0.0f;
            acc += simd_sum(sv * gacc);
        }
""")
    assert devx_ss != BASE
    orig_ss_src = V._SRC_FUSED_PACKED_D8_SIMD_SS
    orig_simdsum, orig_devx = V._D8_SIMDSUM, V._D8_DEVX
    try:
        V._SRC_FUSED_PACKED_D8_SIMD_SS = devx_ss
        for layer, proj in SHAPES[:2]:
            t = load_proj(art, layer, proj)
            codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
            E, OUT, _ = codes.shape
            NGRP = sc.shape[2]
            per_e = codes.nbytes / E + sc.nbytes / E
            print(f"\n=== L{layer}.{proj} OUT={OUT} IN={NGRP*64} NGRP={NGRP}")
            print("      N     devx us(min/med)   devx+ss us(min/med)   "
                  "GB/s devx   GB/s +ss   speedup min/med")
            for N in ns:
                x, eidx = make_inputs(t, N)

                def mk(simdsum):
                    def fn(simdsum=simdsum):
                        V._D8_SIMDSUM = simdsum
                        V._D8_DEVX = True
                        return [V._fused(x, eidx, codes, cbk, sc, pack_bits=14)
                                for _ in range(M)]
                    return fn

                r = interleaved({"devx": mk(False), "devx+ss": mk(True)},
                                 reps=reps)
                by = per_e * N
                a, b = r["devx"], r["devx+ss"]
                print(f"    {N:3d}   {a[0]/M*1e6:7.1f}/{a[1]/M*1e6:6.1f}      "
                      f"{b[0]/M*1e6:7.1f}/{b[1]/M*1e6:6.1f}        "
                      f"{by/(a[0]/M)/1e9:7.1f}   {by/(b[0]/M)/1e9:7.1f}   "
                      f"{a[0]/b[0]:.3f}/{a[1]/b[1]:.3f}")
            del t
            mx.clear_cache()
    finally:
        V._SRC_FUSED_PACKED_D8_SIMD_SS = orig_ss_src
        V._D8_SIMDSUM, V._D8_DEVX = orig_simdsum, orig_devx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART_DEFAULT)
    ap.add_argument("--mode", default="all", choices=["all", "cost", "ss"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--dispatches", type=int, default=50)
    a = ap.parse_args()
    if not os.path.isdir(a.art):
        sys.exit(f"artifact not readable: {a.art}")
    ns = (1, 5, 10, 20)
    if a.mode in ("all", "cost"):
        run_cost(a.art, a.reps, a.dispatches, (1, 10, 20))
    if a.mode in ("all", "ss"):
        run_ss(a.art, a.reps, a.dispatches, ns)
