#!/usr/bin/env python
"""M1c: prefill-shape benchmark — VQ vs gather_qmm at large M.

Bar (M1_KERNEL_PLAN): prefill of 8192 tokens within 2x of gather_qmm.

Strategy v1 = decode-to-dense + gather_mm: a decode kernel expands VQ
experts to dense fp16 [Echunk, OUT, IN] in device memory, then mx.gather_mm
runs the sorted-token GEMMs against the dense chunk. Decode cost is O(bytes)
and amortizes over all tokens routed to the chunk; the GEMM is the same
steel matmul gather_qmm's prefill path uses (which ALSO dequantizes into
tiles — we just do it one level up). Peak transient = chunk experts of
fp16, e.g. 64 experts of down_proj = 512 MB.

If this lands within 2x, the fused tile kernel (plan risk #1) is optional
for v1 and prefill ships as decode+gather_mm.
"""
import argparse
import time

import mlx.core as mx
import numpy as np

SRC_DECODE = r"""
    // one thread per (expert-in-chunk, row, scale-group)
    uint g = thread_position_in_grid.x;   // scale group
    uint r = thread_position_in_grid.y;   // row
    uint ec = thread_position_in_grid.z;  // expert index within chunk
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 4;
    const int NE   = dims[4];             // experts in chunk
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];              // absolute expert id
    const device uchar* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half* wrow = w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G;
    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = (uint)crow[j0 + q];
        for (int u = 0; u < 4; ++u)
            wrow[q * 4 + u] = (half)(s * (float)codebook[c * 4 + u]);
    }
"""

DECODE = mx.fast.metal_kernel(
    name="vq_decode_dense",
    input_names=["codes", "codebook", "scales", "eidx", "dims"],
    output_names=["w"],
    source=SRC_DECODE,
)


def decode_chunk(codes8, cb, sc16, eidx_chunk):
    NE = eidx_chunk.shape[0]
    E, OUT, NSUB = codes8.shape
    IN = NSUB * 4
    NGRP = sc16.shape[2]
    G = IN // NGRP
    dims = mx.array([OUT, IN, 4, G, NE], dtype=mx.int32)
    (w,) = DECODE(
        inputs=[codes8, cb, sc16, eidx_chunk, dims],
        template=[("X", mx.float16)],
        grid=(NGRP, OUT, NE),
        threadgroup=(min(32, NGRP), 8, 1),
        output_shapes=[(NE, OUT, IN)],
        output_dtypes=[mx.float16],
    )
    return w


def vq_prefill(x_sorted, seg_expert, seg_tok_idx, codes8, cb, sc16, chunk=64):
    """x_sorted [Ntok*k, IN] grouped by expert. Per chunk of experts: one
    decode kernel + ONE gather_mm over the chunk's contiguous row range
    (local rhs indices), not a python loop of per-expert matmuls."""
    ys = []
    for c0 in range(0, len(seg_expert), chunk):
        segs = seg_tok_idx[c0:c0 + chunk]
        eids = seg_expert[c0:c0 + chunk]
        w = decode_chunk(codes8, cb, sc16, mx.array(np.array(eids, np.uint32)))
        s0, s1 = segs[0][0], segs[-1][1]
        local = np.zeros(s1 - s0, dtype=np.uint32)
        for i, (s, e2, _) in enumerate(segs):
            local[s - s0:e2 - s0] = i
        rows = s1 - s0
        y = mx.gather_mm(x_sorted[s0:s1, None, :], mx.swapaxes(w, 1, 2),
                         lhs_indices=mx.arange(rows, dtype=mx.uint32),
                         rhs_indices=mx.array(local),
                         sorted_indices=True)
        ys.append(y.reshape(rows, -1))
    return mx.concatenate(ys, axis=0)


def vq_prefill_padded(x_pad, codes8, cb, sc16, chunk=64):
    """x_pad [E, cap, IN]: per-expert rows compacted to a uniform cap
    (zero-padded). One decode + one batched GEMM per chunk — trades ~pad%
    extra FLOPs for real matmul tiling instead of 65k M=1 matvecs."""
    E = codes8.shape[0]
    ys = []
    for c0 in range(0, E, chunk):
        eids = mx.arange(c0, min(c0 + chunk, E), dtype=mx.uint32)
        w = decode_chunk(codes8, cb, sc16, eids)
        ys.append(x_pad[c0:c0 + chunk] @ mx.swapaxes(w, 1, 2))
    return mx.concatenate(ys, axis=0)


def bench(fn, warmup=2, reps=5):
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
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    rng = np.random.default_rng(2)
    z = np.load(args.npz)
    E, OUT, IN, D, K, G = [int(v) for v in z["meta"]]
    assert D == 4 and K <= 256
    print(f"tensor: E={E} out={OUT} in={IN} d={D} K={K} G={G}  "
          f"tokens={args.tokens} top_k={args.top_k}")

    codes8 = mx.array(z["codes"].astype(np.uint8))
    cb = mx.array(z["codebook"]).astype(mx.float16)
    sc16 = mx.array(z["scales"]).astype(mx.float16)
    sc32 = mx.array(z["scales"]).astype(mx.float32)
    mx.eval(codes8, cb, sc16)

    # baseline weights (same shape, affine 2-bit)
    Wb = mx.random.normal((E, OUT, IN)).astype(mx.bfloat16)
    qw, qs, qb = mx.quantize(Wb, group_size=64, bits=2)
    mx.eval(qw, qs, qb)

    T = args.tokens
    idx_np = np.sort(rng.integers(0, E, (T, args.top_k)).astype(np.uint32), axis=1)
    xtok = mx.random.normal((T, IN)).astype(mx.float16)
    xq = xtok[:, None, None, :].astype(mx.bfloat16)
    idx = mx.array(idx_np)
    mx.eval(xtok, xq, idx)

    t_base = bench(lambda: mx.gather_qmm(
        xq, qw, qs, qb, rhs_indices=idx, transpose=True,
        group_size=64, bits=2, sorted_indices=True))
    print(f"gather_qmm prefill: {t_base*1e3:8.2f} ms")

    # sort (token, expert) pairs by expert — what exo's runner does anyway
    pairs_e = idx_np.reshape(-1)
    order = np.argsort(pairs_e, kind="stable")
    tok_of_pair = np.repeat(np.arange(T), args.top_k)[order]
    e_sorted = pairs_e[order]
    x_sorted = mx.array(np.array(xtok))[mx.array(tok_of_pair.astype(np.uint32))]
    mx.eval(x_sorted)
    seg_expert, seg_tok_idx = [], []
    s = 0
    for e in range(E):
        cnt = int((e_sorted == e).sum())
        if cnt:
            seg_expert.append(e)
            seg_tok_idx.append((s, s + cnt, e))
            s += cnt
    print(f"experts touched: {len(seg_expert)}/{E}")

    t_vq = bench(lambda: vq_prefill(x_sorted, seg_expert, seg_tok_idx,
                                    codes8, cb, sc16, args.chunk))
    print(f"vq decode+matmul:   {t_vq*1e3:8.2f} ms  ({t_base/t_vq:.2f}x; "
          f"bar >= 0.5x i.e. within 2x)")

    # padded batched variant: compact rows per expert to a uniform cap
    cnts = np.bincount(e_sorted, minlength=E)
    cap = int(cnts.max())
    x_pad_np = np.zeros((E, cap, IN), np.float16)
    s = 0
    for e in range(E):
        c = cnts[e]
        if c:
            x_pad_np[e, :c] = np.array(x_sorted[s:s + c])
            s += c
    x_pad = mx.array(x_pad_np)
    mx.eval(x_pad)
    pad_ratio = cap * E / max(len(e_sorted), 1)
    t_pad = bench(lambda: vq_prefill_padded(x_pad, codes8, cb, sc16, args.chunk))
    print(f"vq padded batched:  {t_pad*1e3:8.2f} ms  ({t_base/t_pad:.2f}x; "
          f"cap={cap}, pad overhead {pad_ratio:.2f}x rows)")

    # decode-only cost (all E experts once) for the record
    t_dec = bench(lambda: decode_chunk(
        codes8, cb, sc16, mx.array(np.arange(min(args.chunk, E), dtype=np.uint32))))
    print(f"decode {min(args.chunk, E)} experts: {t_dec*1e3:8.2f} ms "
          f"(x{(E + args.chunk - 1)//args.chunk} chunks for full tensor)")

    # correctness: one expert's decoded W vs numpy decode
    e0 = seg_expert[0]
    wref = (z["codebook"].astype(np.float32)[z["codes"][e0]].reshape(OUT, IN // G, G)
            * z["scales"][e0].astype(np.float32)[:, :, None]).reshape(OUT, IN)
    wdec = np.array(decode_chunk(codes8, cb, sc16,
                                 mx.array(np.array([e0], np.uint32)))[0]
                    .astype(mx.float32))
    rel = np.abs(wdec - wref).max() / max(np.abs(wref).max(), 1e-9)
    print(f"decode kernel vs numpy (expert {e0}): max rel {rel:.2e}")


if __name__ == "__main__":
    main()
