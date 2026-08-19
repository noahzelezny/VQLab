# quantlab M1d — VQ expert runtime for mlx_lm.
# Canonical copy lives in quantlab/vq_switch.py; patch_mlx_lm.py installs it
# as mlx_lm/models/vq_switch.py and hooks load_model. Keep both in sync by
# re-running the patcher, never by editing the installed copy.
#
# Format (per VQ'd expert tensor, produced by vq_*_codes fitters):
#   {p}.codes      uint8 (K<=256) / uint16   [E, out, in/d]
#   {p}.codebook   fp16                      [K, d]      (d=2, 4 or 8)
#   {p}.vq_scales  fp16                      [E, out, in/group]
#
# d=8 (E36 mixed geometry, 2026-08-15): each module dispatches on its own
# codebook shape, so a model can mix d4 and d8 tensors freely. d8 codebooks
# above 2K entries (K4096 = 64 KB fp16) cannot live in Apple's 32 KB
# threadgroup memory — the d8 kernels keep the codebook in DEVICE memory and
# rely on L2 residency (measured, not assumed: see m1b_bench d8 rows). A
# threadgroup d8 variant exists for K<=1024 (16 KB codebook).
#
# Two execution regimes (measured on M4, m1b/m1c benches, 2026-08-15):
#   decode (small N):  fused LUT-matmul kernel, threadgroup codebook + x,
#                      uchar4 code loads — 0.66-0.88x gather_qmm
#   prefill (large N): decode experts to dense fp16 chunks + ONE padded
#                      batched GEMM per chunk — 1.21-1.28x gather_qmm
#   The row-batched gather_mm path is a known trap (0.43x): do not "simplify"
#   the prefill path back to it.
#
# READ THIS BEFORE TRUSTING THE 1.21-1.28x ABOVE (VQ-PF1, 2026-08-16): that
# figure was measured with an `rng.integers` router, which pads only 1.20x.
# REAL MoE routing is skewed ~8.7x (max 1574 rows/expert vs mean 180), and
# the padded GEMM pads every expert in a chunk to that chunk's MAX row count
# — so expert-id-ordered chunking did 5.80x the necessary FLOPs and the
# prefill path measured ~9x SLOWER than gather_qmm end-to-end, not faster.
# The kernel was never at fault: at the real workload shape it is at parity
# (61.5 ms both). The fix is in `_prefill` below — chunk experts by SIMILAR
# ROW COUNT, not by expert id. Any future prefill benchmark MUST use a real
# router histogram or it will reproduce the same blind spot.

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# below this many (token, expert) pairs use the fused kernel; above, the
# decode+padded-GEMM path (decode cost amortizes). Tune in M1e if needed.
VQ_FUSED_MAX_N = int(os.environ.get("SCOUT_VQ_FUSED_MAX_N", 4096))

# Experts decoded to dense fp16 per prefill chunk. THIS IS THE MEMORY KNOB,
# not the KV cache: measured 2026-08-15 on a 128 GB M4 Max running the
# 110.8 GiB 397B, prefill grew 3.35 MB/token where the KV cache theory is
# only 0.059 MB/token — a 57x gap that is entirely these buffers. The
# transient is chunk * out * in * 2 bytes:
#     chunk=128 -> 1.0 GiB (down_proj) / 2.0 GiB (gate_up)
#     chunk= 16 -> 0.12 GiB            / 0.25 GiB
# On a box where the model nearly fills RAM, they are what caps your context
# length. Auto-sized from free memory at import; override with
# SCOUT_VQ_DECODE_CHUNK.
#
# DEFAULT IS 32, AND 32 IS NOT ARBITRARY (2026-08-17, resident probes on the
# M4 across all three artifacts). Steady-state ms/bucket, real block, weights
# RESIDENT (probe_block_prefill.py, [256,36] bucket):
#
#     chunk |  2.2bpw  2.4bpw  3.1bpw
#       16  |   677.2   651.9   663.6
#       32  |   716.1   683.0   691.9
#       64  |   791.4   756.3   775.7
#      128  |   984.1   943.2   947.3     <- the old default
#
# 128 -> 32 is 1.37x / 1.38x / 1.37x — the knee is the SAME for K128, K256
# and K2048, so codebook size does not move it.
#
# 32 rather than 16, which is marginally faster still, because 32 is the
# smallest chunk that reproduces the published perplexity EXACTLY
# (nll 9452.9414 on 2.2bpw; 16 gives 9465.2217 and 8 gives 9450.3555).
# Those shifts are float ORDERING, not quality: `ne` is the batched-GEMM
# batch dim, a different ne picks a different Metal tiling, and fp16 sums in
# a different order — chunk 8 lands BELOW the published number and chunk 16
# above it. A lower ppl from reordering cannot be banked; it is the same
# model measured differently. So the rule is: take the knee, keep exactness.
#
# CAUTION FOR ANYONE RE-MEASURING THIS. Do NOT time it with
# score_tasks_streaming.py --selftest: that re-reads the whole model from
# disk every pass (~63 s of a ~100 s run at 100 GiB), so it is DISK-bound and
# reports "no difference" for a change worth 1.37x. That mistake was made on
# 08-17 and reversed by measuring a resident block instead.
def _default_decode_chunk():
    env = os.environ.get("SCOUT_VQ_DECODE_CHUNK")
    if env:
        return max(1, int(env))
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        headroom = info["max_recommended_working_set_size"] - mx.get_active_memory()
        # keep the largest transient (gate_up, out*in ~ 8M elems fp16 = 16 MB
        # per expert) under ~1/8 of remaining headroom
        per_expert = 2048 * 4096 * 2
        return max(4, min(32, int(headroom / 8 / per_expert)))
    except Exception:
        return 32


_DECODE_CHUNK = None  # resolved lazily on first prefill (after weights load)

_SRC_FUSED = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 cb[MAX_K];
    threadgroup float4 xs[MAX_NSUB];
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = float4((float)codebook[i*4], (float)codebook[i*4+1],
                       (float)codebook[i*4+2], (float)codebook[i*4+3]);
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(cb[(uint)crow[j]],   xs[j])
                  + dot(cb[(uint)crow[j+1]], xs[j+1])
                  + dot(cb[(uint)crow[j+2]], xs[j+2])
                  + dot(cb[(uint)crow[j+3]], xs[j+3]);
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=8, codebook in DEVICE memory (L2-resident; K4096 = 64 KB > 32 KB
# threadgroup). x cached in threadgroup as float4; each code costs two half4
# codebook loads. Codes are uint16 (12-bit values for K4096).
_SRC_FUSED_D8 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
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
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb4 = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = (uint)crow[j];
            gacc += dot(float4(cb4[2*c]),   xs[2*j])
                  + dot(float4(cb4[2*c+1]), xs[2*j+1]);
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=8, codebook in THREADGROUP memory — K<=1024 only (half4 pairs: K*16 B
# = 16 KB at K1024, + x as half4 = 8 KB at IN=4096 -> 24 KB total).
_SRC_FUSED_D8_TG = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
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
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = (uint)crow[j];
            gacc += dot(float4(cb[2*c]),   float4(xs[2*j]))
                  + dot(float4(cb[2*c+1]), float4(xs[2*j+1]));
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=2 (gemma d2 rungs, 2026-08-18). The mirror case of d8: where d8 needs
# two half4 loads per code, d2 needs only a half2 — so both codebook and x
# cache as half2. Threadgroup footprint is tiny (K*4 B codebook: 1 KB at
# K256, 8 KB even at K2048) but NSUB doubles vs d4 (IN/2), so the x cache is
# the bigger term (NSUB*4 B: 5.5 KB at IN=2816) — still nowhere near the
# 32 KB ceiling. half2 caching is value-identical, same argument as the
# d4 bigK kernel: codebook is fp16 on disk and x is cast to fp16 before
# dispatch. Codes stay one-per-subvector (uint8/uint16), crow steps by NSUB
# exactly like d4 — only the load width changes.
_SRC_FUSED_D2 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float2(cb[(uint)crow[j]]),   float2(xs[j]))
                  + dot(float2(cb[(uint)crow[j+1]]), float2(xs[j+1]))
                  + dot(float2(cb[(uint)crow[j+2]]), float2(xs[j+2]))
                  + dot(float2(cb[(uint)crow[j+3]]), float2(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# --- packed-code variants (d=4) -------------------------------------------
# Codes live as uint32 words, BITS per code, blocks of 32 codes = BITS words
# (see vq_pack.py for the format and its bit-exactness tests). The fetch is
# the ONLY difference from the unpacked kernels above: same threadgroup
# caching, same accumulation, same output. Row math is unchanged because
# packing is row-local — crow just steps by WPR instead of NSUB.
# NOTE: mx.fast.metal_kernel splices this source INSIDE a function body, so a
# nested `inline uint vq_code(...) {}` is a compile error ("function definition
# is not allowed here"). Hence macros. The `(sh + BITS > 32)` guard is a
# ternary, so the `32 - sh` shift is never evaluated when sh == 0 (which would
# be a 32-bit shift on a 32-bit type, i.e. UB).
_PACK_FETCH = r"""
    #define VQ_MASK   ((1u << BITS) - 1u)
    #define VQ_OFF(j) (((j) & 31) * BITS)
    #define VQ_W(j)   ((((j) >> 5) * BITS) + (VQ_OFF(j) >> 5))
    #define VQ_SH(j)  (VQ_OFF(j) & 31)
    #define VQ_CODE(crow, j) ( ( ((crow)[VQ_W(j)] >> VQ_SH(j)) \
        | ((VQ_SH(j) + BITS > 32) ? ((crow)[VQ_W(j) + 1] << (32 - VQ_SH(j))) : 0u) \
        ) & VQ_MASK )
"""

# d=4, LARGE K (>1024) — and every packed artifact.
#
# The original _SRC_FUSED caches the codebook as float4 (16 B/entry). At
# K2048 that alone is 32 KB, so with x cached the threadgroup allocation is
# 36 KB and Metal refuses to load the kernel ("Threadgroup memory size 36864
# exceeds the maximum 32768"). That is a hard ceiling on E's geometry
# REGARDLESS of packing — found 2026-08-15 while gating the packed kernels.
#
# Fix: cache codebook and x as half4 (8 B/entry). This is VALUE-IDENTICAL,
# not an approximation: the codebook is fp16 on disk and x is cast to fp16
# before dispatch, so both round-trips are exact; only the threadgroup
# footprint changes (K2048 + NSUB1024 -> 16 KB + 8 KB = 24 KB).
#
# BITS is the packed code width; BITS == 0 selects plain uint8/uint16 codes,
# so one source serves both formats.
_SRC_FUSED_D4_BIGK = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[MAX_K];
    threadgroup half4 xs[MAX_NSUB];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    #define VQ_AT(j) ((uint)crow[j])
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[VQ_AT(j)]),   float4(xs[j]))
                  + dot(float4(cb[VQ_AT(j+1)]), float4(xs[j+1]))
                  + dot(float4(cb[VQ_AT(j+2)]), float4(xs[j+2]))
                  + dot(float4(cb[VQ_AT(j+3)]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

_SRC_FUSED_PACKED = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    const int WPR  = NSUB / 32 * BITS;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 cb[MAX_K];
    threadgroup half4 xs[MAX_NSUB];
    const device half4* cbg = (const device half4*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
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
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[VQ_CODE(crow, j)]),   float4(xs[j]))
                  + dot(float4(cb[VQ_CODE(crow, j+1)]), float4(xs[j+1]))
                  + dot(float4(cb[VQ_CODE(crow, j+2)]), float4(xs[j+2]))
                  + dot(float4(cb[VQ_CODE(crow, j+3)]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

_SRC_DECODE_PACKED = _PACK_FETCH + r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    const int WPR  = NSUB / 32 * BITS;
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half* wrow = w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G;
    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = VQ_CODE(crow, j0 + q);
        for (int u = 0; u < D; ++u)
            wrow[q * D + u] = (half)(s * (float)codebook[c * D + u]);
    }
"""

_SRC_DECODE = r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half* wrow = w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G;
    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = (uint)crow[j0 + q];
        for (int u = 0; u < D; ++u)
            wrow[q * D + u] = (half)(s * (float)codebook[c * D + u]);
    }
"""

_KERNELS = {}


def _get_kernel(name, src):
    if name not in _KERNELS:
        if name.startswith("vq_fused"):
            inp, out = ["x", "eidx", "codes", "codebook", "scales", "dims"], ["y"]
        else:
            inp, out = ["codes", "codebook", "scales", "eidx", "dims"], ["w"]
        _KERNELS[name] = mx.fast.metal_kernel(
            name=name, input_names=inp, output_names=out, source=src)
    return _KERNELS[name]


# largest d8 codebook the threadgroup variant may cache (K*16 B + x half4)
_D8_TG_MAX_K = 1024


def _fused(x, eidx, codes, codebook, scales, pack_bits=0):
    N, IN = x.shape
    E, OUT, _ = codes.shape
    K, D = codebook.shape
    NSUB = IN // D
    G = IN // scales.shape[2]
    dims = mx.array([OUT, IN, D, G, N, K], dtype=mx.int32)
    tgx = 256 if OUT >= 256 else OUT
    # threadgroup budget: float4 codebook cache is 16 B/entry, so K>1024 with
    # x cached overflows Apple's 32 KB. Large-K (and all packed) go through
    # the half4 variant — value-identical, half the footprint.
    # Dispatch is EXPLICIT on D with a hard raise for anything unhandled.
    # This used to fall through to the d8 kernels for any non-d4 codebook —
    # a d=2 artifact then read across codebook entries and generated pure
    # <pad> with NO error (2026-08-18, gemma26b vq-K256-d2). A wrong-memory
    # read must never be the default branch.
    if pack_bits:
        if D != 4:
            raise NotImplementedError(
                f"packed codes are d=4 only (got d={D}); d=2 artifacts must "
                "ship unpacked (uint8) until a packed d2 kernel exists")
        name = f"vq_fused_packed{pack_bits}"
        src = _SRC_FUSED_PACKED
        template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB),
                    ("BITS", pack_bits)]
    elif D == 2:
        name, src = "vq_fused_d2", _SRC_FUSED_D2
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NSUB", NSUB)]
    elif D == 4 and K > 1024:
        name = "vq_fused_d4_bigk"
        src = _SRC_FUSED_D4_BIGK
        template = [("T", x.dtype), ("CT", codes.dtype), ("MAX_K", K),
                    ("MAX_NSUB", NSUB)]
    elif D == 4:
        name, src = "vq_fused", _SRC_FUSED
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NSUB", NSUB)]
    elif D == 8 and K <= _D8_TG_MAX_K:
        name, src = "vq_fused_d8_tg", _SRC_FUSED_D8_TG
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NX4", IN // 4)]
    elif D == 8:
        name, src = "vq_fused_d8", _SRC_FUSED_D8
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_NX4", IN // 4)]
    else:
        raise NotImplementedError(f"no fused kernel for subvector dim d={D}")
    (y,) = _get_kernel(name, src)(
        inputs=[x, eidx, codes, codebook, scales, dims],
        template=template,
        grid=(((OUT + tgx - 1) // tgx) * tgx, N, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(N, OUT)],
        output_dtypes=[x.dtype],
    )
    return y


def _decode_chunk(codes, codebook, scales, eidx_chunk, pack_bits=0,
                  in_features=None):
    NE = eidx_chunk.shape[0]
    E, OUT, _ = codes.shape
    D = codebook.shape[1]
    NGRP = scales.shape[2]
    # packed rows are WPR words wide, so shape no longer implies IN — the
    # caller passes it (VQSwitchLinear.input_dims knows it from the format).
    IN = in_features if in_features is not None else codes.shape[2] * D
    G = IN // NGRP
    dims = mx.array([OUT, IN, D, G, NE], dtype=mx.int32)
    if pack_bits:
        name, src = f"vq_decode_packed{pack_bits}", _SRC_DECODE_PACKED
        template = [("BITS", pack_bits)]
    else:
        name, src = "vq_decode", _SRC_DECODE
        template = [("CT", codes.dtype)]
    (w,) = _get_kernel(name, src)(
        inputs=[codes, codebook, scales, eidx_chunk, dims],
        template=template,
        grid=(NGRP, OUT, NE),
        threadgroup=(min(32, NGRP), 8, 1),
        output_shapes=[(NE, OUT, IN)],
        output_dtypes=[mx.float16],
    )
    return w


def _prefill(xf, idx_sorted_np, codes, codebook, scales, pack_bits=0,
             in_features=None):
    """xf [N, IN] rows sorted by expert; idx_sorted_np = matching np expert
    ids. Decode touched experts in chunks; one padded batched GEMM each."""
    global _DECODE_CHUNK
    if _DECODE_CHUNK is None:
        _DECODE_CHUNK = _default_decode_chunk()
    E, OUT, _ = codes.shape
    counts = np.bincount(idx_sorted_np, minlength=E)
    touched = np.nonzero(counts)[0]
    # COUNT-SORTED CHUNKING (VQ-PF1, 2026-08-16). The GEMM below pads every
    # expert in a chunk up to that chunk's MAX row count, so a chunk that
    # mixes a 1574-row expert with a 20-row one pays 1574 rows for both.
    # Real routing is skewed ~8.7x, so chunking by expert ID (the obvious
    # order, and what shipped) did 5.80x the necessary FLOPs. Grouping
    # experts of SIMILAR size makes cap track the chunk mean: pad falls
    # 5.92x -> 1.19x and _prefill goes 263 -> 75.9 ms at chunk=16.
    # This is a pure host-side reordering of WHICH experts share a GEMM —
    # codes/codebook/scales are read identically, nothing is repacked, so
    # the on-disk artifact layout is untouched (the HARD CONSTRAINT: a fix
    # must ship as a bundled model.py update, never a re-upload).
    touched = touched[np.argsort(counts[touched], kind="stable")]
    starts = np.zeros(E + 1, np.int64)
    starts[1:] = np.cumsum(counts)
    ys = []
    # Reordering experts reorders OUTPUT ROWS, and this function's contract
    # is to return rows in its input order (__call__ applies its own `inv`
    # on top). Track the xf row each output row came from and undo it below.
    row_ids = []
    for c0 in range(0, len(touched), _DECODE_CHUNK):
        eids = touched[c0:c0 + _DECODE_CHUNK]
        ne = len(eids)
        cap = int(counts[eids].max())
        # gather map rows -> [ne, cap]; pads point at row 0 (discarded)
        gmap = np.zeros((ne, cap), np.uint32)
        vmask = np.zeros((ne, cap), bool)
        for i, e in enumerate(eids):
            c = counts[e]
            gmap[i, :c] = np.arange(starts[e], starts[e] + c, dtype=np.uint32)
            vmask[i, :c] = True
        w = _decode_chunk(codes, codebook, scales,
                          mx.array(eids.astype(np.uint32)),
                          pack_bits=pack_bits, in_features=in_features)
        xp = xf[mx.array(gmap.reshape(-1))].reshape(ne, cap, -1)
        yp = xp @ mx.swapaxes(w, 1, 2)                      # [ne, cap, OUT]
        flat_valid = np.nonzero(vmask.reshape(-1))[0].astype(np.uint32)
        ys.append(yp.reshape(ne * cap, OUT)[mx.array(flat_valid)])
        row_ids.append(gmap.reshape(-1)[flat_valid])
        # CRITICAL: MLX is lazy. Without this eval the whole loop builds one
        # graph and EVERY chunk's decoded weights stay live until the final
        # concatenate — 4 chunks x 2 GiB for gate_up, which is what actually
        # capped context length on a 128 GB box (measured 2026-08-15: prefill
        # grew 3.35 MB/token vs 0.059 MB/token of real KV cache). Evaluating
        # per chunk lets each `w` be freed before the next is decoded.
        mx.eval(ys[-1])
        del w, xp, yp
    y = mx.concatenate(ys, axis=0)
    # Undo the count-sort: output row j currently holds input row row_ids[j].
    # inv[row_ids[j]] = j, so y[inv] restores the caller's row order. One
    # [N, OUT] gather (~2 ms at N=92k) against ~190 ms of padding saved.
    rid = np.concatenate(row_ids)
    inv = np.empty(rid.shape[0], np.uint32)
    inv[rid] = np.arange(rid.shape[0], dtype=np.uint32)
    return y[mx.array(inv)]


class VQSwitchLinear(nn.Module):
    """Drop-in for QuantizedSwitchLinear over VQ codes. No bias support
    (Qwen3.5 experts are bias-free)."""

    def __init__(self, codes, codebook, vq_scales, group_size: int = 64,
                 pack_bits: int = 0, in_features: int | None = None):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        # pack_bits = 0 -> legacy unpacked codes (uint8/uint16), the format
        # every shipped artifact before 08-16 uses. Non-zero -> uint32 words
        # holding pack_bits-wide fields (vq_pack.py).
        self.pack_bits = pack_bits
        if pack_bits and in_features is None:
            raise ValueError("packed codes need explicit in_features: a "
                             "packed row is WPR words wide, so its shape no "
                             "longer implies the input dimension")
        self._in_features = in_features
        self.freeze()

    @classmethod
    def from_weights(cls, codes, codebook, vq_scales):
        if codes.dtype == mx.uint32:
            # packed codes (vq_pack.py). Geometry is fully derivable from the
            # tensors: the packer stores ceil(log2(K))-bit fields (K is a
            # power of two by construction), and rows are NSUB/32*BITS words.
            k, d = codebook.shape
            bits = int(k - 1).bit_length()
            nsub = codes.shape[2] * 32 // bits
            return cls(codes, codebook, vq_scales,
                       pack_bits=bits, in_features=nsub * d)
        return cls(codes, codebook, vq_scales)

    @property
    def input_dims(self):
        # MUST be derived from the CURRENT tensors, never cached: exo's
        # tensor-parallel path shards `codes` IN PLACE after the module is
        # built (auto_parallel._sharded_to_all -> last axis). A cached value
        # then describes the pre-shard tensor, the kernel computes word
        # offsets for twice the data it holds, and the ring desyncs mid-load
        # (observed 2026-08-16: M4 66 GiB loaded, M3 stalled at 16.5 GiB).
        # Single-box never shards, which is why this only bites the cluster.
        if self.pack_bits:
            nsub = self.codes.shape[2] * 32 // self.pack_bits
            return nsub * self.codebook.shape[1]
        return self.codes.shape[2] * self.codebook.shape[1]

    @property
    def output_dims(self):
        return self.codes.shape[1]

    @property
    def num_experts(self):
        return self.codes.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        IN = self.input_dims
        OUT = self.output_dims
        idx_flat = indices.flatten()
        N = idx_flat.size
        xf = mx.broadcast_to(x, (*indices.shape, 1, IN)).reshape(N, IN)
        in_dtype = xf.dtype
        if in_dtype not in (mx.float16,):
            xf = xf.astype(mx.float16)
        pb = self.pack_bits
        if N <= VQ_FUSED_MAX_N:
            y = _fused(xf, idx_flat.astype(mx.uint32),
                       self["codes"], self["codebook"], self["vq_scales"],
                       pack_bits=pb)
        else:
            idx_np = np.array(idx_flat, copy=False)
            if not sorted_indices:
                order = np.argsort(idx_np, kind="stable")
                inv = np.argsort(order, kind="stable")
                y = _prefill(xf[mx.array(order.astype(np.uint32))],
                             idx_np[order],
                             self["codes"], self["codebook"], self["vq_scales"],
                             pack_bits=pb, in_features=IN)
                y = y[mx.array(inv.astype(np.uint32))]
            else:
                y = _prefill(xf, idx_np,
                             self["codes"], self["codebook"], self["vq_scales"],
                             pack_bits=pb, in_features=IN)
        return y.astype(in_dtype).reshape(*indices.shape, 1, OUT)
