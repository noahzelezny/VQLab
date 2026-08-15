#!/usr/bin/env python
"""M1a: VQ fused LUT-matmul Metal kernel — CORRECTNESS ONLY.

One thread per (token, output-row); scalar loops inside. Deliberately naive:
M1a retires the "can a JIT'd metal_kernel reproduce the decode math exactly"
risk, M1b makes it fast.

y[t, r] = sum_g scale[e, r, g] * sum_{j in g} codebook[codes[e, r, j]] . x[t, jD:(j+1)D]

Checks, in order:
  1. synthetic codes, one expert  -> vs numpy fp64 decode-then-matmul
  2. real codes from --npz (m1a_emit_codes.py), several experts
  3. dtype ladder: fp32 in/out, then fp16 x / fp16 codebook with fp32 accum

Bar: max |dy| / ||y||_inf within fp16 accumulation noise (<1e-3 for fp16
inputs, <1e-5 for fp32 inputs).
"""
import argparse
import time

import mlx.core as mx
import numpy as np

HDR = ""

SRC = r"""
    uint r = thread_position_in_grid.x;   // output row
    uint t = thread_position_in_grid.y;   // token
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int D    = dims[2];
    const int G    = dims[3];
    const int M    = dims[4];
    const int NSUB = IN / D;
    const int NGRP = IN / G;
    const int SPG  = G / D;               // subvectors per scale group
    if (r >= (uint)OUT || t >= (uint)M) return;
    float acc = 0.0f;
    for (int g = 0; g < NGRP; ++g) {
        float s = (float)scales[r * NGRP + g];
        float gacc = 0.0f;
        for (int q = 0; q < SPG; ++q) {
            const int j = g * SPG + q;
            const uint c = (uint)codes[r * NSUB + j];
            for (int u = 0; u < D; ++u) {
                gacc = fma((float)codebook[c * D + u],
                           (float)x[t * IN + j * D + u], gacc);
            }
        }
        acc = fma(s, gacc, acc);
    }
    y[t * OUT + r] = static_cast<T>(acc);
"""

KERNEL = mx.fast.metal_kernel(
    name="vq_lut_matmul_m1a",
    input_names=["x", "codes", "codebook", "scales", "dims"],
    output_names=["y"],
    source=SRC,
    header=HDR,
)


def vq_matmul(x, codes, codebook, scales, out_dtype):
    """x [M, IN]; codes [OUT, IN/d] uint16/32; codebook [K, d]; scales [OUT, IN/G]."""
    M, IN = x.shape
    OUT = codes.shape[0]
    D = codebook.shape[1]
    G = IN // scales.shape[1]
    dims = mx.array([OUT, IN, D, G, M], dtype=mx.int32)
    tg = 256 if OUT >= 256 else OUT
    (y,) = KERNEL(
        inputs=[x, codes, codebook, scales, dims],
        template=[("T", out_dtype)],
        grid=(((OUT + tg - 1) // tg) * tg, M, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(M, OUT)],
        output_dtypes=[out_dtype],
    )
    return y


def numpy_ref(x, codes, codebook, scales, G):
    """fp64 decode-then-matmul ground truth."""
    OUT, NSUB = codes.shape
    D = codebook.shape[1]
    IN = NSUB * D
    W = codebook.astype(np.float64)[codes].reshape(OUT, IN // G, G)
    W = W * scales.astype(np.float64)[:, :, None]
    W = W.reshape(OUT, IN)
    return x.astype(np.float64) @ W.T


def check(tag, x_np, codes_np, cb_np, sc_np, G, x_dtype, cb_dtype, out_dtype, bar):
    x = mx.array(x_np).astype(x_dtype)
    codes = mx.array(codes_np.astype(np.uint32))
    cb = mx.array(cb_np).astype(cb_dtype)
    sc = mx.array(sc_np).astype(mx.float32)
    t0 = time.time()
    y = vq_matmul(x, codes, cb, sc, out_dtype)
    mx.eval(y)
    dt = time.time() - t0
    # reference sees the SAME rounded inputs the kernel saw
    ref = numpy_ref(np.array(x.astype(mx.float32)), codes_np,
                    np.array(cb.astype(mx.float32)), sc_np, G)
    y_np = np.array(y.astype(mx.float32)).astype(np.float64)
    denom = max(np.abs(ref).max(), 1e-9)
    rel = np.abs(y_np - ref).max() / denom
    ok = rel < bar
    print(f"  {tag:44s} max|dy|/||y|| = {rel:.2e}  "
          f"({'OK' if ok else 'FAIL'} bar {bar:.0e})  [{dt*1e3:.0f} ms]", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="real codes from m1a_emit_codes.py")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--m", type=int, default=4, help="tokens")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    all_ok = True

    print("== synthetic (OUT=1024, IN=4096, d=4, K=128, G=64) ==", flush=True)
    OUT, IN, D, K, G = 1024, 4096, 4, 128, 64
    codes = rng.integers(0, K, (OUT, IN // D)).astype(np.uint16)
    cb = (rng.standard_normal((K, D)) * 0.4).astype(np.float16)
    sc = (np.abs(rng.standard_normal((OUT, IN // G))) * 0.02 + 1e-3).astype(np.float32)
    x = rng.standard_normal((args.m, IN)).astype(np.float32)
    all_ok &= check("fp32 x, fp16 cb, fp32 out", x, codes, cb, sc, G,
                    mx.float32, mx.float16, mx.float32, 1e-5)
    all_ok &= check("fp16 x, fp16 cb, fp32 out", x, codes, cb, sc, G,
                    mx.float16, mx.float16, mx.float32, 1e-5)
    all_ok &= check("fp16 x, fp16 cb, fp16 out", x, codes, cb, sc, G,
                    mx.float16, mx.float16, mx.float16, 1e-3)

    print("== synthetic d=8 K=16384 (premium geometry) ==", flush=True)
    D2, K2 = 8, 16384
    codes2 = rng.integers(0, K2, (OUT, IN // D2)).astype(np.uint16)
    cb2 = (rng.standard_normal((K2, D2)) * 0.4).astype(np.float16)
    all_ok &= check("fp16 x, fp16 cb, fp32 out", x, codes2, cb2, sc, G,
                    mx.float16, mx.float16, mx.float32, 1e-5)

    if args.npz:
        z = np.load(args.npz)
        E, OUTd, INd, D, K, G = [int(v) for v in z["meta"]]
        print(f"== real codes {args.npz}  E={E} out={OUTd} in={INd} "
              f"d={D} K={K} G={G} ==", flush=True)
        x = rng.standard_normal((args.m, INd)).astype(np.float32)
        for e in range(min(args.experts, E)):
            all_ok &= check(f"expert {e}: fp16 x, fp16 cb, fp32 out",
                            x, z["codes"][e], z["codebook"], z["scales"][e], G,
                            mx.float16, mx.float16, mx.float32, 1e-5)
        # and the decode itself vs the ORIGINAL weights: sanity that the
        # emitted format reproduces the fit's reconstruction (not a kernel
        # property, but catches emit-side layout bugs)
        if "ref_w" in z:
            e = 0
            W = z["ref_w"][e]
            Wh = z["codebook"].astype(np.float32)[z["codes"][e]].reshape(
                OUTd, INd // G, G) * z["scales"][e][:, :, None]
            relerr = np.linalg.norm(Wh.reshape(OUTd, INd) - W) / np.linalg.norm(W)
            print(f"  emit-format decode vs original W (expert 0): "
                  f"relerr {relerr:.4f} (matches fit log => layout correct)",
                  flush=True)

    print("\nM1a:", "ALL OK" if all_ok else "FAILURES ABOVE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
