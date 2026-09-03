"""The dense DEVICE-X kernels: bit-identical, on by default, all six twins.

Why this file exists. Every dense VQ kernel staged the activation row through
threadgroup memory with barriers, and the cost ablation
(scripts/bench_dense_stage.py) priced that staging at ~66% of a dispatch and
the barriers at ~32-39% -- a much bigger prize than the ~48%/~2% arc 5 found
in the packed-d8 EXPERT kernel, because a dense threadgroup is 32 simdgroups
against the expert kernel's 8. The devx twins delete the tile and read x from
device instead, for 1.6-1.7x per dispatch on the tiled kernels.

The whole value rests on BIT-IDENTITY: these artifacts' published scores were
produced by the staged kernels, so the twins must reproduce them exactly, not
approximately. Every test here is equality, never tolerance.

Two things this file pins that are easy to break by "tidying":

  1. THE NARROWING CAST. The staging loop wrote `half4((half)xrow[o], ...)`,
     narrowing T to half on the way into the tile. The twins keep that with
     an explicit `half4(...)` around the device read. A bare
     `float4(xr4[m])` would be identical at T=half and would silently skip
     the narrowing at bf16/fp32 -- so identity is tested at every dtype the
     dispatcher accepts, not just fp16.

  2. THE d2 BARRIER. The d2 kernels stage the CODEBOOK in threadgroup
     memory, published by what used to be the first in-loop barrier.
     Deleting every barrier there would leave the codebook fill racing --
     garbage results, not a slower kernel. The d2 twins keep exactly one
     hoisted barrier; the d4 twins (device codebook) keep none.
"""
import pathlib
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "src" / "vqlab"))
import vq_pack  # noqa: E402
import vq_switch as V  # noqa: E402


def _layer(OUT, IN, K, D, G, seed=0, packed=0, dtype=mx.float16):
    rng = np.random.default_rng(seed)
    NSUB = IN // D
    codes = rng.integers(0, K, size=(OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    codebook = rng.normal(0, 1, size=(K, D)).astype(np.float16)
    scales = rng.normal(0, 0.05, size=(OUT, IN // G)).astype(np.float16)
    c = (mx.array(vq_pack.pack(codes[None], packed)[0]) if packed
         else mx.array(codes))
    return c, mx.array(codebook), mx.array(scales)


def _run(devx, x, codes, cbk, sc, packed=0, in_features=None, tiled=None):
    import os
    prev_devx, prev_tiled = V._DENSE_DEVX, os.environ.get("VQ_DENSE_TILED")
    if tiled is not None:
        os.environ["VQ_DENSE_TILED"] = "1" if tiled else "0"
    V._DENSE_DEVX = devx
    try:
        y = V._dense_fused(x, codes, cbk, sc, pack_bits=packed,
                           in_features=in_features)
        mx.eval(y)
        return np.array(y, copy=True)
    finally:
        V._DENSE_DEVX = prev_devx
        if tiled is not None:
            if prev_tiled is None:
                os.environ.pop("VQ_DENSE_TILED", None)
            else:
                os.environ["VQ_DENSE_TILED"] = prev_tiled


# (OUT, IN, K, D, G, pack_bits) -- every geometry the dispatcher can reach.
# The d4 pair is the 27B 3.9bpw rung's geometry (K=4096, packed-12); the d2
# pairs are the 4.5 (K256 unpacked) and 4.8 (K512 packed-9) rungs.
SHAPES = [
    (256, 512, 4096, 4, 64, 12),    # packed d4 tiled  <- 3.9bpw
    (256, 512, 4096, 4, 64, 0),     # unpacked d4 tiled (no artifact uses it)
    (256, 512, 256, 2, 64, 0),      # unpacked d2      <- 4.5bpw gate/up
    (256, 512, 512, 2, 64, 9),      # packed d2        <- 4.8bpw gate/up
]


@pytest.mark.parametrize("OUT,IN,K,D,G,bits", SHAPES)
@pytest.mark.parametrize("N", [1, 3, 20])
def test_devx_bit_identical(OUT, IN, K, D, G, bits, N):
    codes, cbk, sc = _layer(OUT, IN, K, D, G, seed=N, packed=bits)
    x = mx.array(np.random.default_rng(N).normal(0, 1, (N, IN))
                 .astype(np.float16))
    inf = IN if bits else None
    off = _run(False, x, codes, cbk, sc, bits, inf)
    on = _run(True, x, codes, cbk, sc, bits, inf)
    assert np.array_equal(off, on), "dense devx is not bit-identical"


@pytest.mark.parametrize("tiled", [True, False])
def test_devx_bit_identical_both_d2_kernels(tiled):
    """d2 has a tiled AND an untiled kernel, with different barrier shapes.

    The untiled one stages the whole row before a single barrier; the tiled
    one staged per block. Both twins must hold, so both are dispatched here
    explicitly rather than leaving it to the width heuristic.
    """
    codes, cbk, sc = _layer(128, 512, 256, 2, 64, seed=7)
    x = mx.array(np.random.default_rng(7).normal(0, 1, (4, 512))
                 .astype(np.float16))
    off = _run(False, x, codes, cbk, sc, tiled=tiled)
    on = _run(True, x, codes, cbk, sc, tiled=tiled)
    assert np.array_equal(off, on)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_devx_narrowing_cast_holds_at_every_dtype(dtype):
    """The staged path narrowed T to half; the twin must narrow identically.

    This is the test that fails if someone replaces `float4(half4(xr4[m]))`
    with `float4(xr4[m])`: at fp16 both are the same, at bf16/fp32 the bare
    widen keeps mantissa bits the tile would have thrown away.
    """
    codes, cbk, sc = _layer(128, 512, 4096, 4, 64, seed=3, packed=12)
    x = mx.array(np.random.default_rng(3).normal(0, 1, (4, 512))
                 .astype(np.float32)).astype(dtype)
    # stay in mlx: numpy has no bfloat16, so the usual np.array_equal route
    # cannot even materialise the bf16 arm's output.
    prev = V._DENSE_DEVX
    try:
        V._DENSE_DEVX = False
        off = V._dense_fused(x, codes, cbk, sc, pack_bits=12, in_features=512)
        V._DENSE_DEVX = True
        on = V._dense_fused(x, codes, cbk, sc, pack_bits=12, in_features=512)
        mx.eval(off, on)
    finally:
        V._DENSE_DEVX = prev
    assert bool(mx.array_equal(off, on)), f"devx diverges at {dtype}"


def test_d2_twins_keep_exactly_one_barrier_and_d4_none():
    """Barrier discipline, asserted on the SOURCE.

    The d2 kernels publish a threadgroup codebook; dropping their last
    barrier is a data race that this suite could otherwise pass by luck.
    """
    for name in ("_SRC_DENSE_D2_TILED_DEVX", "_SRC_DENSE_PACKED_D2_TILED_DEVX",
                 "_SRC_DENSE_D2_DEVX", "_SRC_DENSE_PACKED_D2_DEVX"):
        src = getattr(V, name)
        assert src.count("threadgroup_barrier") == 1, name
        assert "threadgroup half2 cb[" in src, f"{name} lost its tg codebook"
    for name in ("_SRC_DENSE_D4_TILED_DEVX",
                 "_SRC_DENSE_PACKED_D4_TILED_DEVX"):
        src = getattr(V, name)
        assert src.count("threadgroup_barrier") == 0, name


def test_no_devx_kernel_reads_threadgroup_x():
    for name in dir(V):
        if name.startswith("_SRC_DENSE") and name.endswith(
                ("_DEVX", "_DEVX_SS")):
            src = getattr(V, name)
            assert "xs[" not in src, f"{name} still reads a threadgroup tile"


def test_devx_is_on_by_default_and_flag_reverts():
    import importlib
    import os
    prev = os.environ.get("VQ_DENSE_DEVX")
    try:
        os.environ.pop("VQ_DENSE_DEVX", None)
        assert importlib.reload(V)._DENSE_DEVX is True
        os.environ["VQ_DENSE_DEVX"] = "0"
        assert importlib.reload(V)._DENSE_DEVX is False
    finally:
        if prev is None:
            os.environ.pop("VQ_DENSE_DEVX", None)
        else:
            os.environ["VQ_DENSE_DEVX"] = prev
        importlib.reload(V)


def test_dense_simd_sum_is_off_by_default():
    """The dense simd_sum twin exists, is measured, and must NOT be default.

    The 2026-09-02 1-ULP gate relaxation (f6aa628) is scoped to the packed-d8
    EXPERT reduction. The dense reduction diverges by up to 8 ULP, not one,
    so it is not covered and stays opt-in until it has its own referee pass.
    """
    import importlib
    import os
    prev = os.environ.get("VQ_DENSE_SS")
    try:
        os.environ.pop("VQ_DENSE_SS", None)
        m = importlib.reload(V)
        assert m._DENSE_SS is False
        assert hasattr(m, "_SRC_DENSE_PACKED_D4_TILED_DEVX_SS")
        os.environ["VQ_DENSE_SS"] = "1"
        assert importlib.reload(V)._DENSE_SS is True
    finally:
        if prev is None:
            os.environ.pop("VQ_DENSE_SS", None)
        else:
            os.environ["VQ_DENSE_SS"] = prev
        importlib.reload(V)


def test_kernel_sig_gives_dense_kernels_no_eidx():
    """Regression pin for the bug that locked dense out of arc 5's win.

    _get_kernel_spec used to hardcode the expert signature, so the dense
    path could never take the specialized (template-free) route.
    """
    inp, out = V._kernel_sig("vq_dense_packed12_d4_tiled")
    assert "eidx" not in inp and out == ["y"]
    inp, _ = V._kernel_sig("vq_fused_packed14_d8_simd")
    assert "eidx" in inp


def test_dense_spec_kernel_path_is_bit_identical():
    """The specialized (no `template=`) dense kernel must match the generic
    one exactly -- same generated Metal, values baked in as #defines."""
    codes, cbk, sc = _layer(128, 512, 4096, 4, 64, seed=11, packed=12)
    x = mx.array(np.random.default_rng(11).normal(0, 1, (4, 512))
                 .astype(np.float16))
    prev = V._SPEC_KERNELS
    try:
        V._SPEC_KERNELS = False
        V._KERNELS.clear()
        a = _run(True, x, codes, cbk, sc, 12, 512)
        V._SPEC_KERNELS = True
        V._KERNELS.clear()
        b = _run(True, x, codes, cbk, sc, 12, 512)
    finally:
        V._SPEC_KERNELS = prev
        V._KERNELS.clear()
    assert np.array_equal(a, b)
