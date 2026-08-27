#!/usr/bin/env python
"""WIP: PACKED d=8 fused Metal kernel for quantlab (E100 unblock).

Why: the d8-K16384 artifact packs to ~101 GiB and scores better than the
shipped 2.2bpw at the same size, but `vq_switch._fused` raises
NotImplementedError for packed d=8, so it cannot generate.

What this is: the DEVICE-memory d8 codebook access pattern of
`vq_switch._SRC_FUSED_D8`, with the code fetch swapped for the
`_PACK_FETCH` bit-field read used by `_SRC_FUSED_PACKED` (d4) and
`_SRC_FUSED_PACKED_D2`. Nothing else changes: same float4 x threadgroup
cache, same fma-per-scale-group accumulation order, so the output should be
BIT-IDENTICAL to the unpacked d8 device kernel on the same codes (this is
the test in test_d8packed_wip.py, not an assumption).

A threadgroup-codebook variant is also provided for K <= _D8_TG_MAX_K, the
mirror of `_SRC_FUSED_D8_TG`. It is NOT bit-identical to the device variant
(it caches the codebook as half4 -> float4, which is exact, but x as half4
rather than float4, matching _SRC_FUSED_D8_TG); it is compared against
_SRC_FUSED_D8_TG, its own unpacked twin.

This file does NOT edit vq_switch.py. Call `install()` to monkey-patch the
packed dispatch at runtime, or lift the two source strings + the dispatch
branch into vq_switch.py once benchmarked.
"""
import mlx.core as mx

import vq_switch
from vq_switch import _PACK_FETCH, _D8_TG_MAX_K, _get_kernel


# d=8 PACKED, codebook in DEVICE memory (K*16 B; K16384 = 256 KB, far over
# the 32 KB threadgroup ceiling — same reason _SRC_FUSED_D8 exists).
# Codes are BITS-wide fields in row-local uint32 blocks of 32 (vq_pack v1);
# the layout is dim-agnostic, so WPR = NSUB/32*BITS exactly as at d4/d2 —
# but NSUB = IN/8 here, a QUARTER of the d4 value for the same IN. That is
# the one asymmetry that has silently corrupted this family before, so NSUB
# is derived from IN inside the kernel and never passed in.
_SRC_FUSED_PACKED_D8 = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int WPR  = NSUB / 32 * BITS;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 xs[MAX_NX4];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb4 = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = VQ_CODE(crow, j);
            gacc += dot(float4(cb4[2*c]),   xs[2*j])
                  + dot(float4(cb4[2*c+1]), xs[2*j+1]);
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


# d=8 PACKED, codebook in THREADGROUP memory — K <= _D8_TG_MAX_K only
# (K*16 B + x as half4). Mirror of _SRC_FUSED_D8_TG. Not used by the
# K16384 artifact; included so the packed dispatch has the same two-variant
# shape as the unpacked one instead of a K-dependent hole.
_SRC_FUSED_PACKED_D8_TG = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int WPR  = NSUB / 32 * BITS;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[2 * MAX_K];
    threadgroup half4 xs[MAX_NX4];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)(2 * K); i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NX4; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = VQ_CODE(crow, j);
            gacc += dot(float4(cb[2*c]),   float4(xs[2*j]))
                  + dot(float4(cb[2*c+1]), float4(xs[2*j+1]));
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


def fused_packed_d8(x, eidx, codes, codebook, scales, pack_bits):
    """Standalone dispatch for the packed d=8 path (no vq_switch edits)."""
    N, IN = x.shape
    E, OUT, WPR = codes.shape
    K, D = codebook.shape
    if D != 8:
        raise NotImplementedError(f"this kernel is d=8 only, got d={D}")
    NSUB = IN // D
    if NSUB % 32 != 0:
        raise NotImplementedError(
            f"vq_pack blocks are 32 codes; NSUB={NSUB} (IN={IN}, d=8) "
            f"must be a multiple of 32, i.e. IN % 256 == 0")
    if WPR != NSUB // 32 * pack_bits:
        raise ValueError(f"codes WPR={WPR} != NSUB/32*BITS="
                         f"{NSUB // 32 * pack_bits} (NSUB={NSUB}, "
                         f"BITS={pack_bits})")
    G = IN // scales.shape[2]
    if G % 8 != 0:
        raise NotImplementedError(f"d8 kernel needs G % 8 == 0, got {G}")
    dims = mx.array([OUT, IN, D, G, N, K], dtype=mx.int32)
    tgx = 256 if OUT >= 256 else OUT
    if K <= _D8_TG_MAX_K:
        name = f"vq_fused_packed{pack_bits}_d8_tg"
        src = _SRC_FUSED_PACKED_D8_TG
        template = [("T", x.dtype), ("MAX_K", K), ("MAX_NX4", IN // 4),
                    ("BITS", pack_bits)]
    else:
        name = f"vq_fused_packed{pack_bits}_d8"
        src = _SRC_FUSED_PACKED_D8
        template = [("T", x.dtype), ("MAX_NX4", IN // 4),
                    ("BITS", pack_bits)]
    (y,) = _get_kernel(name, src)(
        inputs=[x, eidx, codes, codebook, scales, dims],
        template=template,
        grid=(((OUT + tgx - 1) // tgx) * tgx, N, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(N, OUT)],
        output_dtypes=[x.dtype],
    )
    return y


_ORIG_FUSED = vq_switch._fused


def install():
    """Monkey-patch vq_switch._fused to route packed d=8 here.

    Deliberately additive: every other (pack_bits, D) pair falls through to
    the original dispatch untouched, including its hard raises.
    """
    def _fused_patched(x, eidx, codes, codebook, scales, pack_bits=0):
        if pack_bits and codebook.shape[1] == 8:
            return fused_packed_d8(x, eidx, codes, codebook, scales,
                                   pack_bits)
        return _ORIG_FUSED(x, eidx, codes, codebook, scales, pack_bits)
    vq_switch._fused = _fused_patched
    return _fused_patched


def uninstall():
    vq_switch._fused = _ORIG_FUSED
