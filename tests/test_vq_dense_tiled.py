"""The NSUB-tiled dense d2 kernels: bit-exact where both can run, correct
where only tiling can.

Why this file exists. The untiled dense kernels stage the whole x row in
threadgroup memory, so their allocation grows with LAYER WIDTH and blows
Metal's 32 KB cap at the 27B mlp shapes. VQLinear then silently falls back to
decoding the entire weight on every forward call, which measured 0.43 tok/s
against stock's 16.7 (docs/DENSE-VQ-DECODE.md). The tiled twins stage one
block span instead, so the allocation is width-independent.

The whole value of the change rests on it being BIT-EXACT: these artifacts'
published scores were produced by the existing paths, and a kernel that is
merely "close" silently invalidates every number we have shipped. So the
first test is equality, not tolerance.
"""
import sys, pathlib
import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"))
import vq_switch as V
import vq_pack


def _rand_layer(OUT, IN, K, G, seed=0, packed=0):
    rng = np.random.default_rng(seed)
    NSUB = IN // 2
    codes = rng.integers(0, K, size=(OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    codebook = rng.normal(0, 1, size=(K, 2)).astype(np.float16)
    scales = rng.normal(0, 0.05, size=(OUT, IN // G)).astype(np.float16)
    x = rng.normal(0, 1, size=(1, IN)).astype(np.float16)
    # vq_pack works on [E, OUT, NSUB]; a dense layer is E=1.
    c = (mx.array(vq_pack.pack(codes[None], packed)[0]) if packed
         else mx.array(codes))
    return (mx.array(x), c, mx.array(codebook), mx.array(scales))


def _run(tiled, *args, packed=0, in_features=None):
    import os
    prev = os.environ.get("VQ_DENSE_TILED")
    os.environ["VQ_DENSE_TILED"] = "1" if tiled else "0"
    try:
        y = V._dense_fused(*args, pack_bits=packed, in_features=in_features)
        mx.eval(y)
        return np.array(y, copy=True)
    finally:
        if prev is None:
            os.environ.pop("VQ_DENSE_TILED", None)
        else:
            os.environ["VQ_DENSE_TILED"] = prev


# Shapes where the UNTILED kernel still fits, so both are legal and must agree
# exactly. IN=5120 is the 27B attention width, the half of that model that
# does fuse today.
@pytest.mark.parametrize("OUT,IN,K,G", [
    (128, 5120, 256, 64),
    (64, 5120, 512, 64),
    (96, 2048, 256, 64),
])
def test_tiled_is_bit_identical_to_untiled(OUT, IN, K, G):
    x, codes, cb, sc = _rand_layer(OUT, IN, K, G)
    assert V._dense_tg_bytes(K, IN // 2) <= V._TG_CAP, "shape must fit untiled"
    a = _run(False, x, codes, cb, sc)
    b = _run(True, x, codes, cb, sc)
    assert a.shape == b.shape
    assert np.array_equal(a, b), (
        f"tiled kernel is not bit-identical: max |diff| "
        f"{np.abs(a.astype(np.float32) - b.astype(np.float32)).max()}")


def test_tiled_is_bit_identical_packed():
    OUT, IN, K, G, BITS = 64, 5120, 512, 64, 9
    x, codes, cb, sc = _rand_layer(OUT, IN, K, G, packed=BITS)
    a = _run(False, x, codes, cb, sc, packed=BITS, in_features=IN)
    b = _run(True, x, codes, cb, sc, packed=BITS, in_features=IN)
    assert np.array_equal(a, b)


# The shape this whole change exists for: the 27B mlp width, where the untiled
# kernel cannot be built at all.
@pytest.mark.parametrize("K,packed", [(256, 0), (512, 0), (512, 9), (4096, 0)])
def test_wide_layer_runs_tiled_and_matches_reference(K, packed):
    OUT, IN, G = 64, 17408, 64
    assert V._dense_tg_bytes(K, IN // 2) > V._TG_CAP, "shape must NOT fit untiled"
    assert V.dense_fits(K, IN, G), "tiling must make this shape fusable"
    x, codes, cb, sc = _rand_layer(OUT, IN, K, G, seed=3, packed=packed)
    y = _run(True, x, codes, cb, sc, packed=packed,
             in_features=IN if packed else None)

    # Reference: decode the weight and matmul, i.e. exactly the fallback path
    # VQLinear takes today. Different accumulation order, so compare with a
    # tolerance rather than for equality.
    raw = (vq_pack.unpack(np.array(codes)[None], IN // 2, packed)[0]
           if packed else np.array(codes))
    cbn = np.array(cb).astype(np.float32)
    w = cbn[raw.astype(np.int32)].reshape(OUT, IN)
    w = w * np.repeat(np.array(sc).astype(np.float32), G, axis=1)
    ref = np.array(x).astype(np.float32) @ w.T
    got = y.astype(np.float32)
    denom = max(1e-6, float(np.abs(ref).max()))
    assert np.abs(got - ref).max() / denom < 2e-2, (
        f"tiled wide-layer output diverges from the decode reference: "
        f"rel {np.abs(got - ref).max() / denom:.4f}")


def test_dense_fits_reports_the_shapes_that_motivated_this():
    # 27B attention width already fused; mlp width did not, and now does.
    assert V.dense_fits(256, 5120, 64)
    assert V.dense_fits(512, 17408, 64)
    # d != 2 still has no dense kernel of any kind -- 3.9bpw's problem.
    assert not V.dense_fits(4096, 17408, 64, d=4)


def test_fused_is_more_accurate_than_the_decode_fallback():
    """The wide layers had no kernel before, only the decode fallback, so
    "bit-exact against what shipped" is not the right question for them --
    the right question is which path is closer to the truth.

    It is the kernel. The fallback materialises an fp16 weight and runs an
    fp16 GEMM; the kernel keeps a float accumulator per scale group and
    applies scales in float. Greedy generations from the two therefore
    diverge after a few hundred tokens, and it is the FALLBACK that was
    drifting.
    """
    OUT, IN, K, G = 256, 17408, 512, 64
    rng = np.random.default_rng(7)
    NSUB = IN // 2
    codes = rng.integers(0, K, size=(OUT, NSUB), dtype=np.uint16)
    cb = rng.normal(0, 1, size=(K, 2)).astype(np.float16)
    sc = rng.normal(0, 0.05, size=(OUT, IN // G)).astype(np.float16)
    x = rng.normal(0, 1, size=(1, IN)).astype(np.float16)

    # Exact: reconstruct in float64 exactly as the format defines it.
    w64 = cb.astype(np.float64)[codes.astype(np.int64)].reshape(OUT, IN)
    w64 = w64 * np.repeat(sc.astype(np.float64), G, axis=1)
    exact = x.astype(np.float64) @ w64.T
    scale = np.abs(exact).max()

    y = V._dense_fused(mx.array(x), mx.array(codes), mx.array(cb), mx.array(sc))
    mx.eval(y)
    fused_err = np.abs(np.array(y).astype(np.float64) - exact).max() / scale

    w16 = (cb.astype(np.float32)[codes.astype(np.int32)].reshape(OUT, IN)
           * np.repeat(sc.astype(np.float32), G, axis=1)).astype(np.float16)
    d = mx.array(x) @ mx.array(w16).T
    mx.eval(d)
    decode_err = np.abs(np.array(d).astype(np.float64) - exact).max() / scale

    assert fused_err < decode_err, (
        f"fused {fused_err:.3e} should beat decode {decode_err:.3e}")
    assert fused_err < 1e-3
