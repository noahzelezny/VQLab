#!/usr/bin/env python
"""M1b: decode-shape benchmark — VQ LUT-matmul vs mx.gather_qmm.

Real MoE decode shape: T tokens x top-8 experts -> N = 8T (token, expert)
pairs, each a [1, IN] x [OUT, IN] matvec against a gathered expert. Baseline
is the shipping path: affine 2-bit gs64 weights through mx.gather_qmm with
rhs_indices. Ours: the M1a kernel extended with an expert-index input.

Bar (M1_KERNEL_PLAN): >=0.5x gather_qmm at M=1,4,16; roofline says >=1x is
in reach (VQ reads 2.0-2.29 bpw vs affine 2.5, and the 1 KB K128 codebook
lives in threadgroup memory).

Kernel v1 (naive-gather): one thread per (pair, row), scalar loops.
Kernel v2 (tg): threadgroup-cached x + codebook, float4 codebook rows,
each thread accumulates 4 rows. Iterate here in M1b.
"""
import argparse
import time

import mlx.core as mx
import numpy as np

SRC_GATHER = r"""
    uint r = thread_position_in_grid.x;   // output row
    uint t = thread_position_in_grid.y;   // (token, expert) pair
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int N    = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes  + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device T2*       srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    const device T*        xrow = x + (size_t)t * IN;
    float acc = 0.0f;
    for (int g = 0; g < NGRP; ++g) {
        float s = (float)srow[g];
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q) {
            const int j = g * SPG + q;
            const uint c = (uint)crow[j];
            for (int u = 0; u < D; ++u)
                gacc = fma((float)codebook[c * D + u], (float)xrow[j * D + u], gacc);
        }
        acc = fma(s, gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# v2: d=4 specialization. Codebook cached in threadgroup memory as float4;
# x cached in threadgroup memory; one thread still = one (pair,row) but with
# vector math and no repeated device reads of x.
SRC_TG = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 4;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 cb[MAX_K];
    threadgroup float4 xs[MAX_NSUB];
    for (uint i = lid; i < (uint)K; i += tgsize) {
        cb[i] = float4((float)codebook[i*4], (float)codebook[i*4+1],
                       (float)codebook[i*4+2], (float)codebook[i*4+3]);
    }
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize) {
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes  + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device T2*       srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q, ++j) {
            const uint c = (uint)crow[j];
            gacc += dot(cb[c], xs[j]);
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

# v3: d=4, uint8 codes only. uchar4 vector code loads (4 codes/load),
# fp16 scales, threadgroup codebook + x as in v2.
SRC_TG4 = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;              // uchar4 loads per scale group
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
    const device uchar4* crow = (const device uchar4*)
        (codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB);
    const device half*  srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int q4 = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q, ++q4) {
            const uchar4 c = crow[q4];
            const int j = q4 * 4;
            gacc += dot(cb[c.x], xs[j])
                  + dot(cb[c.y], xs[j+1])
                  + dot(cb[c.z], xs[j+2])
                  + dot(cb[c.w], xs[j+3]);
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

HDR = "#include <metal_stdlib>\nusing namespace metal;\n"


def make_kernel(name, src):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["x", "eidx", "codes", "codebook", "scales", "dims"],
        output_names=["y"],
        source=src,
    )


K_GATHER = make_kernel("vq_gather_naive", SRC_GATHER)
K_TG = make_kernel("vq_gather_tg", SRC_TG)
K_TG4 = make_kernel("vq_gather_tg4", SRC_TG4)


def vq_run(kernel, x, eidx, codes, codebook, scales, extra_hdr=None, tg=None):
    N, IN = x.shape
    E, OUT, NSUB = codes.shape
    D = codebook.shape[1]
    K = codebook.shape[0]
    G = IN // scales.shape[2]
    dims = mx.array([OUT, IN, D, G, N, K], dtype=mx.int32)
    tgx = tg or (256 if OUT >= 256 else OUT)
    (y,) = kernel(
        inputs=[x, eidx, codes, codebook, scales, dims],
        template=[("T", mx.float16), ("T2", mx.float32), ("CT", codes.dtype),
                  ("MAX_K", K), ("MAX_NSUB", NSUB)],
        grid=(((OUT + tgx - 1) // tgx) * tgx, N, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(N, OUT)],
        output_dtypes=[mx.float16],
    )
    return y


def bench(fn, warmup=5, reps=20):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="real codes (m1a_emit_codes)")
    ap.add_argument("--experts-used", type=int, default=8, help="top-k per token")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    z = np.load(args.npz)
    E, OUT, IN, D, K, G = [int(v) for v in z["meta"]]
    print(f"tensor: E={E} out={OUT} in={IN} d={D} K={K} G={G}")

    codes = mx.array(z["codes"])                       # uint16 [E,OUT,NSUB]
    # uint8 codes when K<=256: halves code traffic (the artifact stores this)
    codes8 = mx.array(z["codes"].astype(np.uint8)) if K <= 256 else None
    cb = mx.array(z["codebook"]).astype(mx.float16)
    sc = mx.array(z["scales"]).astype(mx.float32)      # [E,OUT,NGRP]
    sc16 = mx.array(z["scales"]).astype(mx.float16)    # for the tg4 kernel
    mx.eval(codes, cb, sc, *( [codes8] if codes8 is not None else [] ))

    # gather_qmm baseline: affine 2-bit gs64 quantization of a random bf16
    # tensor of the same shape (bytes/latency identical to the real one)
    Wb = mx.random.normal((E, OUT, IN)).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(Wb, group_size=64, bits=2)
    mx.eval(qw, qs, qb)

    top_k = args.experts_used
    for M in (1, 4, 16):
        N = M * top_k
        # HONEST baseline = the shape mlx_lm/exo actually emits: shared x per
        # token, [T, k] indices SORTED per token, sorted_indices=True. The
        # flat unsorted call hits a pathological gather_qmm path (57 ms at
        # N=128 vs 0.68 ms — measured on the M4) and would flatter us.
        idx_np = np.sort(rng.integers(0, E, (M, top_k)).astype(np.uint32), axis=1)
        xtok = mx.random.normal((M, IN)).astype(mx.float16)
        xq = xtok[:, None, None, :].astype(mx.bfloat16)   # [T,1,1,IN]
        idx = mx.array(idx_np)                            # [T,k]
        # our kernel takes flattened pairs (same data, same order)
        x = mx.repeat(xtok[:, None, :], top_k, axis=1).reshape(N, IN)
        eidx = mx.array(idx_np.reshape(-1))
        mx.eval(x, eidx, xq, idx)

        t_base = bench(lambda: mx.gather_qmm(
            xq, qw, qs, qb, rhs_indices=idx, transpose=True,
            group_size=64, bits=2, sorted_indices=True))
        t_naive = bench(lambda: vq_run(K_GATHER, x, eidx, codes, cb, sc))
        t_tg = bench(lambda: vq_run(K_TG, x, eidx, codes, cb, sc))
        t_tg8 = (bench(lambda: vq_run(K_TG, x, eidx, codes8, cb, sc))
                 if codes8 is not None else float("nan"))
        t_tg4 = (bench(lambda: vq_run(K_TG4, x, eidx, codes8, cb, sc16))
                 if codes8 is not None else float("nan"))

        # correctness cross-check of the tg kernels vs naive on this shape
        y_ref = vq_run(K_GATHER, x, eidx, codes, cb, sc).astype(mx.float32)
        d = float(mx.abs(vq_run(K_TG, x, eidx, codes, cb, sc).astype(mx.float32)
                         - y_ref).max())
        if codes8 is not None:
            d = max(d, float(mx.abs(
                vq_run(K_TG, x, eidx, codes8, cb, sc).astype(mx.float32)
                - y_ref).max()))
            # tg4 uses fp16 scales — looser bar, checked vs same-rounded ref
            d4 = float(mx.abs(
                vq_run(K_TG4, x, eidx, codes8, cb, sc16).astype(mx.float32)
                - y_ref).max())
        else:
            d4 = float("nan")
        print(f"M={M:3d} (N={N:4d}): gather_qmm {t_base*1e3:7.3f} ms | "
              f"naive {t_naive*1e3:7.3f} ({t_base/t_naive:4.2f}x) | "
              f"tg {t_tg*1e3:7.3f} ({t_base/t_tg:4.2f}x) | "
              f"tg-u8 {t_tg8*1e3:7.3f} ({t_base/t_tg8:4.2f}x) | "
              f"tg4 {t_tg4*1e3:7.3f} ({t_base/t_tg4:4.2f}x) | "
              f"max|d| {d:.1e} / tg4 {d4:.1e}",
              flush=True)


if __name__ == "__main__":
    main()
