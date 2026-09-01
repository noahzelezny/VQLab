#!/usr/bin/env python
"""GPU equality gate for the packed VQ kernels.

Packing changes REPRESENTATION, not values: a packed artifact must produce
EXACTLY what its unpacked twin produces. So this asserts bit-identical
outputs, never a tolerance — any drift here would move a referee number for
reasons that have nothing to do with quantization quality.

Covers both dispatch paths (fused decode-shape and the padded-GEMM prefill
shape) at both widths that matter: 7-bit (F) and 11-bit (E).
"""
import numpy as np
import mlx.core as mx

import vq_pack
from vq_switch import VQSwitchLinear

FAIL = 0


def check(tag, a, b):
    global FAIL
    a = np.array(a, copy=False)
    b = np.array(b, copy=False)
    same = np.array_equal(a, b)
    if not same:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f"  FAIL {tag}: max|diff|={d.max():.3e} "
              f"({np.count_nonzero(d)}/{d.size} differ)")
        FAIL += 1
    else:
        print(f"  ok   {tag}: bit-identical ({a.size} values)")


def build(k, e, out, in_f, seed):
    rng = np.random.default_rng(seed)
    bits = vq_pack.bits_for_k(k)
    nsub = in_f // 4
    codes = rng.integers(0, k, size=(e, out, nsub), dtype=np.uint16)
    cb = rng.standard_normal((k, 4)).astype(np.float16)
    sc = (rng.random((e, out, in_f // 64)) + 0.5).astype(np.float16)

    unpacked = VQSwitchLinear(
        mx.array(codes.astype(np.uint8 if bits <= 8 else np.uint16)),
        mx.array(cb), mx.array(sc))
    packed = VQSwitchLinear(
        mx.array(vq_pack.pack(codes, bits)), mx.array(cb), mx.array(sc),
        pack_bits=bits, in_features=in_f)
    return unpacked, packed, bits


def main():
    # (K, experts, out, in) — real Qwen3.5 expert shapes, small expert count
    cases = [
        (128, 8, 512, 1024),    # F geometry, down_proj-like
        (128, 8, 256, 4096),    # F geometry, gate_up-like
        (2048, 8, 512, 1024),   # E geometry, 11-bit straddles words
        (2048, 8, 256, 4096),
        (256, 8, 512, 1024),    # C geometry: 8-bit, exercises the aligned case
    ]
    for k, e, out, in_f in cases:
        u, p, bits = build(k, e, out, in_f, seed=k + in_f)
        print(f"K={k} bits={bits} experts={e} out={out} in={in_f}")

        # sanity: the packed row really is narrower on disk
        ub = u["codes"].size * u["codes"].dtype.size
        pb = p["codes"].size * p["codes"].dtype.size
        print(f"  codes bytes {ub} -> {pb}  ({pb / ub:.3f}x)")

        rng = np.random.default_rng(0)
        # --- fused path (decode-shaped: few rows)
        n = 4
        x = mx.array(rng.standard_normal((n, 1, in_f)).astype(np.float16))
        idx = mx.array(rng.integers(0, e, (n,)).astype(np.uint32))
        check("fused", u(x, idx), p(x, idx))

        # --- prefill path (padded GEMM: many rows, forces the other branch)
        n = 512
        x = mx.array(rng.standard_normal((n, 1, in_f)).astype(np.float16))
        idx_np = np.sort(rng.integers(0, e, (n,)).astype(np.uint32))
        idx = mx.array(idx_np)
        check("prefill(sorted)", u(x, idx, sorted_indices=True),
              p(x, idx, sorted_indices=True))
        idx = mx.array(rng.integers(0, e, (n,)).astype(np.uint32))
        check("prefill(unsorted)", u(x, idx), p(x, idx))

    print("\nPACKED KERNEL GATE:", "FAILED" if FAIL else "PASS (bit-identical)")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
