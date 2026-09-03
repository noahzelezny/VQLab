#!/usr/bin/env python
"""The packed-d8 simd kernel's LAST un-probed slice: tile staging, barriers,
launch (~36% of a dispatch per arc 4's ablation).  (2026-09-02, kernel arc 5.)

Arc 4 priced the inner loop (~45%) and the reduction (~19%) and left the rest
uncharacterised.  This script splits the remainder with the same cost-only
discipline (every `cost` arm is WRONG ON PURPOSE and is a LOWER BOUND), then
times the two candidate fixes, both of which are BIT-IDENTICAL by
construction and asserted so before any timing:

  cost arms
    empty     everything after the address setup deleted; one guarded store.
              The launch + grid floor for this exact grid.
    nostage   the x staging loop deleted (barriers kept): prices the staging
              LOADS + threadgroup STORES.
    nobar     both threadgroup_barriers deleted (staging kept): prices the
              BARRIERS.
    nostagebar  both deleted: staging + barrier joint price.

  bit-identical arms
    fullstage stage the WHOLE x row ONCE before the group-block loop (one
              barrier pair total instead of one per 32-group block).  The
              inner arithmetic and its order are untouched.  Needs
              MAX_TILE = NX4 (10 KiB at IN=2560) <= 32 KiB.
    devx      no threadgroup x at all: each lane reads its own x slice
              straight from device memory as half4 and converts, which is
              value-identical to reading the same half4 out of the staged
              tile.  Zero barriers, zero staging.

Methodology: arc standard (JIT-warm, interleaved A,B,A,B, min over reps with
median, mx.eval on the full output list).  Real L20 gate/up tensors, E=64,
READ-ONLY, no model.

Run:  PYTHONPATH=src:scripts python scripts/bench_d8_stage.py
"""
import argparse
import os
import sys

import mlx.core as mx

from vqlab import vq_switch as V
from bench_d8_inner import (ART_DEFAULT, BASE, load_proj, make_inputs,
                            interleaved, dispatch)

_STAGE = """    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
        const int span = min(TILE, NX4 - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = float4((float)xrow[o], (float)xrow[o+1],
                           (float)xrow[o+2], (float)xrow[o+3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""
assert _STAGE in BASE, "staging text drifted from vq_switch.py"


def _stage(body):
    src = BASE.replace(_STAGE, body)
    assert src != BASE
    return src


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
        const int span = min(TILE, NX4 - base);
        for (uint i = lid; i < (uint)span; i += tgsize) {
            const int o = (base + (int)i) * 4;
            xs[i] = float4((float)xrow[o], (float)xrow[o+1],
                           (float)xrow[o+2], (float)xrow[o+3]);
        }
""")
COST["nostagebar"] = _stage("""    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = b * TILE;
""")
# everything after the pointer setup deleted: launch + grid floor.
_TAIL = BASE[BASE.index("    float acc = 0.0f;"):]
COST["empty"] = BASE.replace(
    _TAIL, "    if (active && lane == 0) y[(size_t)t * OUT + r] = T(0);\n")
assert COST["empty"] != BASE

# ---- bit-identical arms ----------------------------------------------------
IDENT = {"base": BASE}
# one-shot staging: whole x row up front, one barrier pair, no per-block
# staging.  `base` stays defined (0) so the inner `m = 2*j - base` is
# untouched and the arithmetic order is exactly the base kernel's.
IDENT["fullstage"] = _stage("""    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = 0;
""")
# no threadgroup x at all: read the same half4, convert the same way.
IDENT["devx"] = _stage("""    const device half4* xr4 = (const device half4*)xrow;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int base = 0;
""").replace(
    """                gacc += dot(float4(cb4[2*c]),   xs[m])
                      + dot(float4(cb4[2*c+1]), xs[m+1]);
""",
    """                gacc += dot(float4(cb4[2*c]),   float4(xr4[m]))
                      + dot(float4(cb4[2*c+1]), float4(xr4[m+1]));
""")
# drop the (now unused) threadgroup tile so it cannot cost occupancy
IDENT["devx"] = IDENT["devx"].replace(
    "    threadgroup float4 xs[MAX_TILE];\n", "")
assert "xs[" not in IDENT["devx"]

SHAPES = ((20, "gate_proj"), (20, "up_proj"))


def _tile_for(t, name):
    # fullstage needs the whole row staged: MAX_TILE = NX4.
    IN = t["vq_scales"].shape[2] * 64
    return IN // 4 if name == "fullstage" else None


def run(art, arms, prefix, reps, M, ns, assert_identity):
    for layer, proj in SHAPES:
        t = load_proj(art, layer, proj)
        codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
        IN = sc.shape[2] * 64
        print(f"\n=== L{layer}.{proj} OUT={codes.shape[1]} IN={IN} "
              f"NGRP={sc.shape[2]}")
        if assert_identity:
            for N in ns:
                x, eidx = make_inputs(t, N, seed=N)
                ref = dispatch(prefix + "base", BASE, x, eidx, codes, cbk, sc)
                mx.eval(ref)
                for n, src in arms.items():
                    if n == "base":
                        continue
                    y = _disp(prefix + n, src, t, x, eidx)
                    mx.eval(y)
                    if not bool(mx.array_equal(ref, y)):
                        raise SystemExit(f"BIT-IDENTITY FAILED: {n} at N={N} "
                                         f"on {proj} -- no timing reported")
        print("      N  " + "".join(f"{n:>17s}" for n in arms))
        for N in ns:
            x, eidx = make_inputs(t, N)

            def mk(name):
                def fn(name=name):
                    return [_disp(prefix + name, arms[name], t, x, eidx)
                            for _ in range(M)]
                return fn

            r = interleaved({n: mk(n) for n in arms}, reps=reps)
            print(f"    {N:3d}  " + "".join(
                f"{r[n][0]/M*1e6:8.1f}/{r[n][1]/M*1e6:6.1f}" for n in arms))
            print("         vs base min/med: " + "  ".join(
                f"{n}={r[n][0]/r['base'][0]:.3f}/{r[n][1]/r['base'][1]:.3f}"
                for n in arms if n != "base"))
        del t
        mx.clear_cache()


def _disp(name, src, t, x, eidx):
    codes, cbk, sc = t["codes"], t["codebook"], t["vq_scales"]
    IN = sc.shape[2] * 64
    if "fullstage" in name:
        # MAX_TILE must hold the whole row
        N = x.shape[0]
        OUT = codes.shape[1]
        rows = V._EXPERT_ROWS_TG_D8_PACKED
        dims = mx.array([OUT, IN, 8, 64, N, cbk.shape[0]], dtype=mx.int32)
        (y,) = V._get_kernel(name, src)(
            inputs=[x, eidx, codes, cbk, sc, dims],
            template=[("T", x.dtype), ("MAX_TILE", IN // 4), ("BITS", 14)],
            grid=(32, ((OUT + rows - 1) // rows) * rows, N),
            threadgroup=(32, rows, 1),
            output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
        return y
    return dispatch(name, src, x, eidx, codes, cbk, sc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=ART_DEFAULT)
    ap.add_argument("--mode", default="all", choices=["all", "cost", "ident"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--dispatches", type=int, default=50)
    a = ap.parse_args()
    if not os.path.isdir(a.art):
        sys.exit(f"artifact not readable: {a.art}")
    if a.mode in ("all", "cost"):
        print("\n####### COST-ONLY ARMS: each WRONG on purpose, LOWER BOUNDS")
        run(a.art, COST, "vq_fused_a5c_", a.reps, a.dispatches, (1, 10, 20), False)
    if a.mode in ("all", "ident"):
        print("\n####### BIT-IDENTICAL ARMS (identity asserted before timing)")
        run(a.art, IDENT, "vq_fused_a5i_", a.reps, a.dispatches, (1, 5, 10, 20),
            True)
