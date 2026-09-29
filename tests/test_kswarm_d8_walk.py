"""kswarm d8-walk: VQ_D8_WALK (bit-walker code fetch on the thread-per-row
packed d8 kernel) must be BIT-IDENTICAL to the kernel it replaces,
vq_fused_packed{bits}_d8 -- compared as uint16 bit patterns, never a
tolerance. That is the kernel that runs today for 397B-2.2 down_proj
(IN=1024, G=64 -> NGRP=16 < 32 declines the simd arm) at every N, and for
all d8 layers at N > _EXPERT_SIMD_MAX_N. The simd arm is not touched.
"""
import sys, pathlib
import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"))
import vq_switch as V
import vq_pack


def _args(E, OUT, IN, K, BITS, N, G=64, seed=0, skew=False):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, K, size=(E, OUT, IN // 8), dtype=np.uint16)
    if skew:
        eidx = np.where(rng.random(N) < 0.8, 0, rng.integers(0, E, size=N))
    else:
        eidx = rng.integers(0, E, size=(N,))
    return (mx.array(rng.normal(0, 1, size=(N, IN)).astype(np.float16)),
            mx.array(eidx.astype(np.uint32)),
            mx.array(vq_pack.pack(codes, BITS)),
            mx.array(rng.normal(0, 1, size=(K, 8)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // G)).astype(np.float16)))


def _run(monkeypatch, walk, args, bits, simd):
    monkeypatch.setattr(V, "_D8_WALK", walk)
    y = V._fused(*args, pack_bits=bits, simd=simd)
    mx.eval(y)
    return np.array(y).view(np.uint16)


# (E, OUT, IN, K, BITS, simd)  simd=False forces the thread-per-row arm
SHAPES = [
    (4, 4096, 1024, 16384, 14, None),   # 397B-2.2 down: NGRP=16, runs thread-per-row at every N
    (4, 1024, 4096, 16384, 14, False),  # 397B gate/up at N > simd max
    (3,  100, 1024, 16384, 14, None),   # ragged OUT
    (3,  256,  320, 16384, 14, None),   # NSUB=40: padded tail block
    (3,  256, 1024,  4096, 12, None),   # other pack width
    (3,  256, 1024,  8192, 13, None),
]


@pytest.mark.parametrize("N", [1, 8, 21, 64])
@pytest.mark.parametrize("shape", SHAPES)
def test_d8_walk_bit_exact(monkeypatch, shape, N):
    E, OUT, IN, K, BITS, simd = shape
    args = _args(E, OUT, IN, K, BITS, N)
    assert np.array_equal(_run(monkeypatch, False, args, BITS, simd),
                          _run(monkeypatch, True, args, BITS, simd))


def test_d8_walk_bit_exact_large_skewed(monkeypatch):
    args = _args(8, 1024, 4096, 16384, 14, 2048, skew=True, seed=3)
    assert np.array_equal(_run(monkeypatch, False, args, 14, False),
                          _run(monkeypatch, True, args, 14, False))


def test_d8_walk_matches_reference(monkeypatch):
    # independent anchor: unpacked codes decoded in numpy, fp32 matmul
    E, OUT, IN, K, BITS, N = 2, 64, 1024, 16384, 14, 3
    args = _args(E, OUT, IN, K, BITS, N, seed=7)
    x, eidx, pc, cb, sc = (np.array(a) for a in args)
    codes = vq_pack.unpack(pc, IN // 8, BITS)
    y = np.array(_run(monkeypatch, True, args, BITS, None).view(np.float16), dtype=np.float32)
    for t in range(N):
        e = eidx[t]
        W = cb.astype(np.float32)[codes[e].astype(np.int64)].reshape(OUT, IN)
        W *= np.repeat(sc[e].astype(np.float32), 64, axis=1)
        ref = W @ x[t].astype(np.float32)
        assert np.allclose(y[t], ref, rtol=2e-2, atol=2e-2)


def test_d8_walk_dispatches(monkeypatch):
    monkeypatch.setattr(V, "_D8_WALK", True)
    V._fused(*_args(2, 256, 1024, 16384, 14, 1), pack_bits=14)
    names = {p[2] for p in V._KERNELS.values() if isinstance(p, tuple) and len(p) == 10}
    assert "vq_fused_packed14_d8_walk" in names


def test_simd_arm_untouched(monkeypatch):
    # N<=simd max at NGRP>=32 must still take the devx_ss kernel with the flag on
    monkeypatch.setattr(V, "_D8_WALK", True)
    V._fused(*_args(2, 256, 4096, 16384, 14, 1), pack_bits=14)
    names = {p[2] for p in V._KERNELS.values() if isinstance(p, tuple) and len(p) == 10}
    assert "vq_fused_packed14_d8_simd_devx_ss" in names


def test_default_off():
    import os
    if os.environ.get("VQ_D8_WALK") is None:
        assert V._D8_WALK is True  # default ON since the 397B-2.2 pair (x1.020)
