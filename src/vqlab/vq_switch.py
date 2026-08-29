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
VQ_FUSED_MAX_N = int(os.environ.get("VQ_FUSED_MAX_N",
                     4096))

# Experts decoded to dense fp16 per prefill chunk. THIS IS THE MEMORY KNOB,
# not the KV cache: measured 2026-08-15 on a 128 GB M4 Max running the
# 110.8 GiB 397B, prefill grew 3.35 MB/token where the KV cache theory is
# only 0.059 MB/token — a 57x gap that is entirely these buffers. The
# transient is chunk * out * in * 2 bytes:
#     chunk=128 -> 1.0 GiB (down_proj) / 2.0 GiB (gate_up)
#     chunk= 16 -> 0.12 GiB            / 0.25 GiB
# On a box where the model nearly fills RAM, they are what caps your context
# length. Auto-sized from free memory at import; override with
# VQ_DECODE_CHUNK.
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
    env = os.environ.get("VQ_DECODE_CHUNK", None)
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
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
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

# d=4, codebook in DEVICE memory (E134, 2026-08-22).
#
# WHY. _SRC_FUSED_D4_BIGK and _SRC_FUSED_PACKED both cache the codebook in
# THREADGROUP memory as half4 (8 B/entry) alongside the x cache, so the
# allocation is (K + NSUB) * 8 bytes against Apple's hard 32768 cap. Measured
# on the M4 across four sibling 35B artifacts (IN=2048 -> NSUB=512):
#
#     K256   ( 256+512)*8 =  6,144 B   loads, generates
#     K2048  (2048+512)*8 = 20,480 B   loads, generates
#     K4096  (4096+512)*8 = 36,864 B   FAILS to load
#     K8192  (8192+512)*8 = 69,632 B   FAILS to load
#
# Metal reports this as "Unable to load kernel ... Compilation failed due to
# an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED", NOT as a
# threadgroup-size error, which is why it reads like a broken compiler
# service. It is not: a trivial custom kernel compiles on the same box
# seconds later. Do not chase the XPC message.
#
# The consequence was that a K>=4096 d4 artifact SCORED normally (the
# streaming referee is prefill-shaped and never dispatches this kernel) while
# being unable to generate a single token. Both 35B offering candidates
# reached release consideration in that state.
#
# FIX. Mirror _SRC_FUSED_D8, which has kept its codebook in device memory
# since K4096 for exactly this reason: drop the threadgroup codebook cache
# and read cb straight from device memory (L2-resident in practice). x stays
# cached in threadgroup, so the allocation becomes NSUB * 8 bytes and is
# independent of K. Arithmetic is UNCHANGED -- same half4 loads, same
# float4 dot, same fma accumulation order -- so results are bit-identical to
# the threadgroup variants, not merely close.
_SRC_FUSED_D4_DEVCB = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 xs[MAX_NSUB];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb = (const device half4*)codebook;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float4(cb[(uint)crow[j]]),   float4(xs[j]))
                  + dot(float4(cb[(uint)crow[j+1]]), float4(xs[j+1]))
                  + dot(float4(cb[(uint)crow[j+2]]), float4(xs[j+2]))
                  + dot(float4(cb[(uint)crow[j+3]]), float4(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# d=4 PACKED, codebook in device memory. _SRC_FUSED_PACKED with the
# threadgroup codebook cache removed; code fetch and accumulation identical.
_SRC_FUSED_PACKED_D4_DEVCB = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup half4 xs[MAX_NSUB];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half4((half)xrow[i*4], (half)xrow[i*4+1],
                      (half)xrow[i*4+2], (half)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device half4* cb = (const device half4*)codebook;
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


# d=2 packed (gemma d2 K512/K1024 rungs, 2026-08-19). Same shape as
# _SRC_FUSED_D2 — half2 codebook + x caches, QPG*4 subvectors per group —
# with the code fetch swapped for the VQ_CODE bit-field read. The packing
# layout (vq_pack.py) is dim-agnostic: blocks of 32 codes = BITS uint32
# words, row-local, so crow steps by WPR just like the d4 packed kernel.
# What is NOT dim-agnostic is NSUB (= IN/2, double the d4 value for the same
# IN) — deriving it inside the kernel from IN keeps the d4/d2 asymmetry that
# caused the original silent d2 bug out of the picture; threadgroup sizing
# uses MAX_NSUB from the host, which also computes NSUB = IN/D explicitly.
_SRC_FUSED_PACKED_D2 = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
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
    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(float2(cb[VQ_CODE(crow, j)]),   float2(xs[j]))
                  + dot(float2(cb[VQ_CODE(crow, j+1)]), float2(xs[j+1]))
                  + dot(float2(cb[VQ_CODE(crow, j+2)]), float2(xs[j+2]))
                  + dot(float2(cb[VQ_CODE(crow, j+3)]), float2(xs[j+3]));
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""


_SRC_FUSED_PACKED_D8 = _PACK_FETCH + r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / 8;
    const int NX4  = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 8;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
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
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
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

# --- DENSE d=2 kernels (e4b VQLinear decode path, 2026-08-19) --------------
# The expert kernels above give a dense layer at E=1 correctness but not
# speed: ONE thread walks a whole code row (NSUB = IN/2 = up to 5120 codes)
# sequentially, so decode is LATENCY-bound, not bandwidth-bound (measured
# 43 tok/s on e4b-VQ-d2K2048-packed vs 84 for 8-bit, whose qmv splits a row
# across a simdgroup). These dense variants do the same: a 32-lane simdgroup
# owns each output row, lane l handles scale-groups l, l+32, ..., and the
# per-group partials are stitched back together with a simd_shuffle broadcast
# loop so the group-order fma chain is executed in EXACTLY the sequential
# order of the expert kernel. That makes the output BIT-IDENTICAL to the
# _SRC_FUSED_D2 / _SRC_FUSED_PACKED_D2 path — required, because the shipped
# KL numbers (E62) were scored through those kernels, and a reduction in a
# different float order would move the printed number. The per-group inner
# expression (QPG iterations of four dots summed a+b+c+d) is copied verbatim
# for the same reason: `gacc += a+b+c+d` and four `gacc += d_i` round
# differently.
#
# No expert axis, no eidx gather: codes/scales are [OUT, ...], crow/srow are
# plain row offsets. Threadgroup caching is unchanged (half2 codebook + x:
# K2048 -> 8 KB, IN=10240 -> NSUB 5120 -> 20 KB; 28 KB < 32 KB ceiling).
_SRC_DENSE_D2 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int QPG  = G / 8;
    const int SPG  = G / 2;
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT) return;
    const device CT* crow = codes + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)r * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (g < NGRP) {
            int j = g * SPG;
            for (int q = 0; q < QPG; ++q) {
                gacc += dot(float2(cb[(uint)crow[j]]),   float2(xs[j]))
                      + dot(float2(cb[(uint)crow[j+1]]), float2(xs[j+1]))
                      + dot(float2(cb[(uint)crow[j+2]]), float2(xs[j+2]))
                      + dot(float2(cb[(uint)crow[j+3]]), float2(xs[j+3]));
                j += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# Packed twin. G=64 ONLY, and that is load-bearing: at d=2, a scale-group of
# 64 weights is 32 codes, and vq_pack blocks are 32 codes = BITS uint32
# words — so each lane's group is EXACTLY one word-aligned block. The lane
# copies its block's BITS words into registers ONCE and extracts all 32
# codes from registers; going through the generic VQ_CODE device-memory
# macro instead re-reads words per code and measured 169 us/matmul on a
# dependent e4b-shaped chain vs 82 for this version (M3 Ultra, 2026-08-19).
# Bit extraction order and arithmetic are unchanged — output stays
# bit-identical to _SRC_FUSED_PACKED_D2 (verified).
# The LC(i) offsets are compile-time (i and BITS are constants after
# unrolling), so wbuf indexing does not spill. Code 31 ends at bit 32*BITS-1
# exactly, so no extraction ever reads past wbuf[BITS-1].
_SRC_DENSE_PACKED_D2 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int K    = dims[5];
    const int NSUB = IN / 2;
    const int NGRP = IN / G;
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
    uint r = thread_position_in_grid.y;
    uint t = thread_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;
    uint lid = thread_position_in_threadgroup.y * 32 + lane;
    uint tgsize = threads_per_threadgroup.x * threads_per_threadgroup.y;

    threadgroup half2 cb[MAX_K];
    threadgroup half2 xs[MAX_NSUB];
    const device half2* cbg = (const device half2*)codebook;
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = cbg[i];
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = half2((half)xrow[i*2], (half)xrow[i*2+1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT) return;
    const device uint* crow = codes + (size_t)r * WPR;
    const device half* srow = scales + (size_t)r * NGRP;
    float acc = 0.0f;
    const int NBLK = (NGRP + 31) / 32;
    for (int b = 0; b < NBLK; ++b) {
        const int g = b * 32 + (int)lane;
        float gacc = 0.0f;
        if (g < NGRP) {
            uint wbuf[BITS];
            const device uint* blk = crow + (size_t)g * BITS;
            for (int wi = 0; wi < BITS; ++wi) wbuf[wi] = blk[wi];
            int j = g * 32;
            for (int q = 0; q < 8; ++q) {
                const int i0 = q * 4;
                #define LC(i) (((wbuf[((i)*BITS)>>5] >> (((i)*BITS)&31)) \
                    | (((((i)*BITS)&31) + BITS > 32) \
                       ? (wbuf[(((i)*BITS)>>5)+1] << (32-(((i)*BITS)&31))) \
                       : 0u)) & ((1u<<BITS)-1u))
                gacc += dot(float2(cb[LC(i0)]),   float2(xs[j]))
                      + dot(float2(cb[LC(i0+1)]), float2(xs[j+1]))
                      + dot(float2(cb[LC(i0+2)]), float2(xs[j+2]))
                      + dot(float2(cb[LC(i0+3)]), float2(xs[j+3]));
                j += 4;
            }
        }
        const int gmax = min(32, NGRP - b * 32);
        for (int i = 0; i < gmax; ++i)
            acc = fma((float)srow[b * 32 + i],
                      simd_shuffle(gacc, (ushort)i), acc);
    }
    if (lane == 0) y[(size_t)t * OUT + r] = static_cast<T>(acc);
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
    const int WPR  = (NSUB + 31) / 32 * BITS;  // ceil: tail block padded, pad codes never read (n < NSUB)
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
        if name.startswith("vq_dense"):
            inp, out = ["x", "codes", "codebook", "scales", "dims"], ["y"]
        elif name.startswith("vq_fused"):
            inp, out = ["x", "eidx", "codes", "codebook", "scales", "dims"], ["y"]
        else:
            inp, out = ["codes", "codebook", "scales", "eidx", "dims"], ["w"]
        _KERNELS[name] = mx.fast.metal_kernel(
            name=name, input_names=inp, output_names=out, source=src)
    return _KERNELS[name]


# largest d8 codebook the threadgroup variant may cache (K*16 B + x half4)
_D8_TG_MAX_K = 1024


# Apple's hard threadgroup allocation cap. The d4 threadgroup kernels cache
# BOTH the codebook and x as half4, so they need (K + NSUB) * 8 bytes; past
# this the kernel fails to LOAD (E134). Checked before dispatch rather than
# discovered at kernel load, matching the guard vq_dense.py uses for the
# dense d2 path.
_TG_CAP_BYTES = 32768


def _d4_tg_fits(K, NSUB):
    """True if the d4 threadgroup codebook+x cache fits Metal's cap."""
    return (K + NSUB) * 8 <= _TG_CAP_BYTES


def _fused(x, eidx, codes, codebook, scales, pack_bits=0):
    # U8-VIEW DISPATCH (E77/E90, 2026-08-20). Unpacked uint8 d4 rows are
    # byte-for-byte the pack_bits=8 word layout (little-endian; verified
    # against vq_pack.pack), and the packed fused kernel's simdgroup layout
    # measured 1.38-1.45x the one-thread-per-row unpacked kernel at every N
    # on both 35B expert shapes (+33% end-to-end prefill at step 512,
    # 1009-1020 vs 732-769 tok/s, rotlab--35B-vqK256codes). Zero-copy
    # reinterpret; output is BIT-IDENTICAL (mx.array_equal at N=8/512/4096
    # on real tensors, greedy 200-token generation byte-identical, KL gate
    # reproduced to every printed digit).
    if (pack_bits == 0 and codes.dtype == mx.uint8
            and codebook.shape[1] == 4 and codes.shape[2] % 4 == 0):
        codes = mx.view(codes, dtype=mx.uint32)
        pack_bits = 8
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
        # Explicit per-D dispatch, same rule as unpacked: a packed kernel for
        # one D reads wrong memory at another D (the d4 kernel at d=2 returns
        # NaN), so anything unimplemented must raise, never fall through.
        if D == 4:
            # E134: fall back to the device-memory codebook when the
            # threadgroup cache would exceed the cap. Bit-identical, and the
            # only thing that changes is where cb is read from.
            if _d4_tg_fits(K, NSUB):
                name = f"vq_fused_packed{pack_bits}"
                src = _SRC_FUSED_PACKED
            else:
                name = f"vq_fused_packed{pack_bits}_d4_devcb"
                src = _SRC_FUSED_PACKED_D4_DEVCB
        elif D == 2:
            # NSUB (= IN/2) doubles vs d4 for the same IN; MAX_NSUB below is
            # computed from the actual IN, so the threadgroup x cache is
            # sized for the doubled subvector count automatically.
            name = f"vq_fused_packed{pack_bits}_d2"
            src = _SRC_FUSED_PACKED_D2
        elif D == 8:
            # d=8 packed (E100, 2026-08-21). NSUB = IN/8 -- a QUARTER of the
            # d4 value for the same IN -- so the codes row is WPR = NSUB/32
            # *BITS words; derived inside the kernel from IN, never passed.
            # K16384*d8*fp16 = 256 KB does not fit threadgroup memory, so the
            # large-K variant streams the codebook from DEVICE memory exactly
            # like _SRC_FUSED_D8. Verified bit-identical to the unpacked d8
            # kernels on synthetic codes at K=256/1024/4096/16384.
            # Unaligned NSUB is legal since the padded-tail format: pack()
            # zero-pads the last block and every packed kernel computes
            # ceil-WPR with inner loops bounded n < NSUB, so pad codes are
            # never read. The WPR shape assert below (already ceil) is the
            # remaining guard against a mis-packed tensor.
            if K <= _D8_TG_MAX_K:
                name = f"vq_fused_packed{pack_bits}_d8_tg"
                src = _SRC_FUSED_PACKED_D8_TG
            else:
                name = f"vq_fused_packed{pack_bits}_d8"
                src = _SRC_FUSED_PACKED_D8
        else:
            raise NotImplementedError(
                f"no FUSED packed kernel for d={D}; only d=4, d=2 and d=8 "
                f"are implemented and each is dispatched explicitly.")
        if codes.shape[2] != (NSUB + 31) // 32 * pack_bits:
            raise ValueError(
                f"packed codes are {codes.shape[2]} words/row, expected "
                f"{(NSUB + 31) // 32 * pack_bits} for IN={IN}, d={D}, "
                f"bits={pack_bits}")
        if D == 8:
            template = [("T", x.dtype), ("MAX_NX4", IN // 4),
                        ("BITS", pack_bits)]
            if K <= _D8_TG_MAX_K:
                template.insert(1, ("MAX_K", K))
        elif D == 4 and not _d4_tg_fits(K, NSUB):
            template = [("T", x.dtype), ("MAX_NSUB", NSUB),
                        ("BITS", pack_bits)]
        else:
            template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB),
                        ("BITS", pack_bits)]
    elif D == 2:
        name, src = "vq_fused_d2", _SRC_FUSED_D2
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NSUB", NSUB)]
    elif D == 4 and K > 1024 and not _d4_tg_fits(K, NSUB):
        # E134: (K + NSUB) * 8 over the cap -> device-memory codebook.
        name = "vq_fused_d4_devcb"
        src = _SRC_FUSED_D4_DEVCB
        template = [("T", x.dtype), ("CT", codes.dtype), ("MAX_NSUB", NSUB)]
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


# rows each threadgroup owns in the dense kernels (one 32-lane simdgroup per
# row). Swept 2/4/8/16/32 on the dependent e4b-shaped chain (M3 Ultra,
# 2026-08-19): 211.9 / 118.8 / 79.6 / 73.0 / 68.1 us per matmul — bigger
# threadgroups amortise the codebook+x threadgroup loads over more rows.
_DENSE_ROWS_TG = 32


def _dense_fused(x, codes, codebook, scales, pack_bits=0, in_features=None):
    """Dense d=2 fused VQ matmul: y[N, OUT] = x [N, IN] @ decode(codes).T.

    codes [OUT, NSUB] (uint8/uint16) or [OUT, WPR] (uint32, pack_bits-wide
    fields); scales [OUT, IN/G]. Output is BIT-IDENTICAL to _fused with E=1
    and eidx=0 (see the kernel-source comment) — verified before first ship.
    Dispatch is explicit with hard raises, never a silent fallthrough.
    """
    N, IN = x.shape
    OUT = codes.shape[0]
    K, D = codebook.shape
    if D != 2:
        raise NotImplementedError(
            f"no DENSE fused kernel for d={D}; only d=2 is implemented and "
            f"dispatched explicitly (a kernel for one D reads wrong memory "
            f"at another D — see the expert-kernel dispatch above).")
    NSUB = IN // D
    NGRP = scales.shape[1]
    G = IN // NGRP
    if G % 8 != 0:
        raise NotImplementedError(f"dense d2 kernel needs G % 8 == 0, got {G}")
    if pack_bits:
        if G != 64:
            # the packed kernel's register block-fetch assumes one scale
            # group == one 32-code pack block (see its source comment).
            raise NotImplementedError(
                f"packed dense d2 kernel requires group_size=64, got {G}")
        exp_in = in_features
        if exp_in is not None and exp_in != IN:
            raise ValueError(f"packed dense: x is IN={IN} but module says "
                             f"in_features={exp_in}")
        if codes.shape[1] != (NSUB + 31) // 32 * pack_bits:
            raise ValueError(
                f"packed dense: codes are {codes.shape[1]} words/row, "
                f"expected {(NSUB + 31) // 32 * pack_bits} for IN={IN}, "
                f"bits={pack_bits}")
        name = f"vq_dense_packed{pack_bits}_d2"
        src = _SRC_DENSE_PACKED_D2
        template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB),
                    ("BITS", pack_bits)]
    else:
        if codes.shape[1] != NSUB:
            raise ValueError(f"dense: codes are {codes.shape[1]} cols, "
                             f"expected NSUB={NSUB} for IN={IN}, d={D}")
        name = "vq_dense_d2"
        src = _SRC_DENSE_D2
        template = [("T", x.dtype), ("CT", codes.dtype),
                    ("MAX_K", K), ("MAX_NSUB", NSUB)]
    dims = mx.array([OUT, IN, D, G, N, K], dtype=mx.int32)
    rows = _DENSE_ROWS_TG
    (y,) = _get_kernel(name, src)(
        inputs=[x, codes, codebook, scales, dims],
        template=template,
        grid=(32, ((OUT + rows - 1) // rows) * rows, N),
        threadgroup=(32, rows, 1),
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
    # PREFILL GUARD (2026-08-18). _fused dispatches explicitly on D, but this
    # path only branched on pack_bits — so d=2 with uint16 codes (K>256) read
    # wrong memory and returned SILENT GARBAGE: gemma vq-K512-d2 verified at
    # relerr 0.0589 (verify_artifact PASS, weights provably fine) yet scored
    # 3154 mnats / 47.14%, worse than the d4 artifact it should beat. d=2 with
    # uint8 (K<=256) is correct and is what ships. Raise rather than decode
    # wrongly — a plausible bad number costs more than a crash.
    # (2026-08-18, later) the vq_decode kernel is D-generic: verified
    # numerically for d2-uint16 (K=512) against a numpy reference, max rel
    # diff 2.5e-4. So unpacked d2 needs no dtype restriction here. The
    # garbage that prompted the original guard came from a STALE COPY of
    # this file inside the venv (site-packages/mlx_lm/models/vq_switch.py)
    # that still had the fall-through dispatch — see E47. Packed d2 remains
    # unimplemented and must still raise.
    # packed d=2 through THIS path is verified D-generic: decoded against a
    # numpy vq_pack.unpack reference at max rel 2.6e-4 (K=512, pack_bits=9).
    # The fused path has its own dedicated packed-d2 kernel (2026-08-19).
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
        # SHARDING GUARD (2026-08-19). The codebook is a shared [K, d] lookup
        # table indexed by the codes; it must be REPLICATED across tensor-
        # parallel ranks, never sliced. exo's tensor_auto_parallel sliced it
        # by default until PR #2268, and a sliced codebook does not error —
        # every rank decodes against a fraction of the table and the model
        # emits fluent garbage, which reads as "this quant is broken" rather
        # than "my cluster mis-sharded a LUT". Remember the size we were
        # built with so __call__ can say so plainly.
        self._k_expect = int(codebook.shape[0])
        self.freeze()

    @classmethod
    def from_weights(cls, codes, codebook, vq_scales):
        if codes.dtype == mx.uint32:
            # packed codes (vq_pack.py). Geometry is fully derivable from the
            # tensors: the packer stores ceil(log2(K))-bit fields (K is a
            # power of two by construction), and rows are NSUB/32*BITS words.
            k, d = codebook.shape
            bits = int(k - 1).bit_length()
            # WPR -> NSUB is lossy for padded-tail packs; the scales axis
            # (IN/group, default group 64) carries the true input width.
            return cls(codes, codebook, vq_scales,
                       pack_bits=bits, in_features=vq_scales.shape[2] * 64)
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
            # Padded-tail packs (NSUB % 32 != 0) make WPR -> NSUB lossy:
            # ceil(80/32)*BITS words decode back to 96 subvectors, not 80.
            # The scales tensor's last axis is IN/group and shards on the
            # same axis as codes, so it carries the true IN through both
            # the unaligned format and exo's in-place sharding.
            return self.vq_scales.shape[2] * self.group_size
        return self.codes.shape[2] * self.codebook.shape[1]

    @property
    def output_dims(self):
        return self.codes.shape[1]

    @property
    def num_experts(self):
        return self.codes.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        k_now = self.codebook.shape[0]
        if k_now != self._k_expect:
            raise RuntimeError(
                f"VQ codebook was sharded: K={k_now}, expected "
                f"{self._k_expect}. The codebook is a SHARED lookup table and "
                f"must be replicated across tensor-parallel ranks, not sliced "
                f"(codes and scales shard fine on the default axes). On exo, "
                f"apply the codebook guard in "
                f"src/exo/worker/engines/mlx/auto_parallel.py — upstream PR "
                f"https://github.com/exo-explore/exo/pull/2268. "
                f"Pipeline sharding and single-box "
                f"mlx-lm are unaffected.")
        IN = self.input_dims
        OUT = self.output_dims
        idx_flat = indices.flatten()
        N = idx_flat.size
        xf = mx.broadcast_to(x, (*indices.shape, 1, IN)).reshape(N, IN)
        in_dtype = xf.dtype
        if in_dtype not in (mx.float16,):
            xf = xf.astype(mx.float16)
        pb = self.pack_bits
        # Packed d=2 now has its own fused kernel (vq_fused_packed{bits}_d2,
        # 2026-08-19, verified against vq_pack.unpack numpy reference); the
        # old force-to-_prefill workaround is gone. _fused still raises
        # explicitly for any (D, pack_bits) without a dedicated kernel.
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


class VQPLEEmbedding(nn.Module):
    """VQ'd embedding / PLE table shard: decode is a pure gather.

    codes      uint16 [rows, cols/dim]     (unpacked; NSUB=40 defeats the
                                            32-aligned block pack for now)
    codebook   fp16   [K, dim]
    vq_scales  fp16   [rows, cols/group]

    __call__(ids) -> [.., cols] rows, decoded on the fly:
    codebook[codes[ids]] reshaped, times the row's group scales. No matmul
    anywhere in the path — the original module was already a lookup, so VQ
    composes as a second lookup (LUT of a LUT).
    """

    def __init__(self, codes, codebook, vq_scales, group_size: int = 32,
                 packed_nsub: int = 0):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        # packed rows: BITS-wide codes (BITS from the codebook size — a
        # hardcoded 11 here read K256 rows on an 11-bit stride and scored
        # NaN, 2026-08-29), byte-aligned because nsub*BITS % 8 == 0.
        # Constant gather tables map code i -> its 3-byte window + shift.
        self._pn = packed_nsub
        self._bits = max(1, (codebook.shape[0] - 1).bit_length())
        self._mask = (1 << self._bits) - 1
        if packed_nsub:
            import numpy as _np
            bit0 = _np.arange(packed_nsub) * self._bits
            self._b0 = mx.array(bit0 // 8)
            self._sh = mx.array((bit0 % 8).astype(_np.uint32))

    def _unpack(self, rows_u8):
        # rows_u8 [.., row_bytes] uint8 -> [.., nsub] uint32 codes
        b = rows_u8.astype(mx.uint32)
        # pad 2 bytes so the 3-byte window never reads past the row
        pad = mx.zeros((*b.shape[:-1], 2), dtype=mx.uint32)
        b = mx.concatenate([b, pad], axis=-1)
        w = (mx.take(b, self._b0, axis=-1)
             | (mx.take(b, self._b0 + 1, axis=-1) << 8)
             | (mx.take(b, self._b0 + 2, axis=-1) << 16))
        return (w >> self._sh) & self._mask

    def __call__(self, ids):
        if self._pn:
            c = self._unpack(self.codes[ids])            # [.., nsub]
        else:
            c = self.codes[ids]                              # [.., nsub]
        v = self.codebook[c.astype(mx.uint32)]           # [.., nsub, d]
        flat = v.reshape(*ids.shape, -1)                 # [.., cols]
        sc = self.vq_scales[ids]                         # [.., cols/G]
        sc = mx.repeat(sc, self.group_size, axis=-1)
        return (flat * sc).astype(mx.bfloat16)
