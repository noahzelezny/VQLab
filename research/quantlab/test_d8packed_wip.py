#!/usr/bin/env python
"""Correctness test for the packed d=8 fused kernel (vq_switch_d8packed_wip).

SMALL synthetic tensors only (a few MB, ~1 s of GPU) — this machine is busy.

Three checks per config:
  1. packed-d8 kernel vs UNPACKED d8 kernel on the same codes  -> expect
     bit-identical for the device variant (same accumulation order, same
     float4 x cache; only the code fetch differs).
  2. packed-d8 kernel vs a numpy dequantize+matmul reference   -> fp16-ish
     tolerance (the reference accumulates in fp64).
  3. round-trip of vq_pack.pack/unpack on the codes            -> sanity.
"""
import numpy as np
import mlx.core as mx

import vq_pack
import vq_switch
import vq_switch_d8packed_wip as wip


def run(K, IN=512, OUT=96, N=7, E=3, G=64, seed=0):
    bits = vq_pack.bits_for_k(K)
    D = 8
    NSUB = IN // D
    NGRP = IN // G
    rng = np.random.default_rng(seed)

    cb = rng.standard_normal((K, D)).astype(np.float16)
    codes = rng.integers(0, K, size=(E, OUT, NSUB), dtype=np.uint64)
    scales = (rng.random((E, OUT, NGRP)) * 0.5 + 0.25).astype(np.float16)
    x = rng.standard_normal((N, IN)).astype(np.float16)
    eidx = rng.integers(0, E, size=(N,)).astype(np.uint32)

    packed = vq_pack.pack(codes, bits)
    assert np.array_equal(vq_pack.unpack(packed, NSUB, bits), codes), \
        "vq_pack round-trip failed"

    cdt = np.uint8 if K <= 256 else np.uint16
    mcodes_u = mx.array(codes.astype(cdt))
    mcodes_p = mx.array(packed)
    mcb, msc = mx.array(cb), mx.array(scales)
    mx_x, mx_e = mx.array(x), mx.array(eidx)

    y_pack = wip.fused_packed_d8(mx_x, mx_e, mcodes_p, mcb, msc, bits)
    y_unp = vq_switch._fused(mx_x, mx_e, mcodes_u, mcb, msc, 0)
    mx.eval(y_pack, y_unp)
    a = np.array(y_pack).astype(np.float64)
    b = np.array(y_unp).astype(np.float64)
    identical = bool(np.array_equal(np.array(y_pack), np.array(y_unp)))

    # numpy reference: decode weights, then matmul in float64
    cbf = cb.astype(np.float64)
    W = cbf[codes]                        # [E, OUT, NSUB, 8]
    W = W.reshape(E, OUT, NGRP, G) * scales.astype(np.float64)[..., None]
    W = W.reshape(E, OUT, IN)
    ref = np.einsum('ni,noi->no', x.astype(np.float64), W[eidx])

    d_ref = np.abs(a - ref)
    d_unp = np.abs(a - b)
    denom = np.abs(ref).mean()
    print(f"K={K:6d} bits={bits:2d} IN={IN} OUT={OUT} N={N} E={E} G={G} "
          f"variant={'TG' if K <= vq_switch._D8_TG_MAX_K else 'device'}")
    print(f"  packed vs unpacked-d8 : bit-identical={identical} "
          f"max={d_unp.max():.3e} mean={d_unp.mean():.3e}")
    print(f"  packed vs fp64 ref    : max={d_ref.max():.3e} "
          f"mean={d_ref.mean():.3e} rel_mean={d_ref.mean()/denom:.3e} "
          f"|ref|mean={denom:.3f}")
    return identical, d_unp.max(), d_ref.max(), denom


if __name__ == "__main__":
    for K in (16384, 4096, 1024, 256):
        run(K)
    print("\n-- shape sweep at K=16384 (the E100 artifact's K) --")
    run(16384, IN=2048, OUT=256, N=1, E=2, G=64, seed=3)
    run(16384, IN=1024, OUT=64, N=33, E=1, G=128, seed=4)


def module_test(K=16384, IN=512, OUT=96, E=4, G=64, seed=7):
    """End-to-end VQSwitchLinear with packed d=8 weights, fused branch and
    prefill (decode+GEMM) branch, both vs the fp64 reference."""
    import mlx.nn as nn
    bits = vq_pack.bits_for_k(K)
    D, NSUB, NGRP = 8, IN // 8, IN // G
    rng = np.random.default_rng(seed)
    cb = rng.standard_normal((K, D)).astype(np.float16)
    codes = rng.integers(0, K, size=(E, OUT, NSUB), dtype=np.uint64)
    scales = (rng.random((E, OUT, NGRP)) * 0.5 + 0.25).astype(np.float16)
    packed = vq_pack.pack(codes, bits)

    W = cb.astype(np.float64)[codes].reshape(E, OUT, NGRP, G)
    W = (W * scales.astype(np.float64)[..., None]).reshape(E, OUT, IN)

    mod = vq_switch.VQSwitchLinear.from_weights(
        mx.array(packed), mx.array(cb), mx.array(scales))
    assert mod.pack_bits == bits and mod.input_dims == IN, (
        mod.pack_bits, mod.input_dims)

    for N, label in ((3, "fused"), (vq_switch.VQ_FUSED_MAX_N + 1, "prefill")):
        x = rng.standard_normal((N, 1, 1, IN)).astype(np.float16)
        idx = rng.integers(0, E, size=(N, 1)).astype(np.uint32)
        y = mod(mx.array(x), mx.array(idx))
        mx.eval(y)
        got = np.array(y).reshape(N, OUT).astype(np.float64)
        ref = np.einsum('ni,noi->no', x.reshape(N, IN).astype(np.float64),
                        W[idx.reshape(N)])
        d = np.abs(got - ref)
        print(f"  module {label:8s} N={N:5d}: max={d.max():.3e} "
              f"mean={d.mean():.3e} rel_mean={d.mean()/np.abs(ref).mean():.3e}")
