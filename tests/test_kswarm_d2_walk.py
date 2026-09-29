"""kswarm d2-walk: VQ_D2_WALK (bit-walker code fetch on the packed d2
threadgroup-codebook kernel) must be BIT-IDENTICAL to the shipped default
vq_fused_packed{bits}_d2 kernel. uint16 bit patterns, never a tolerance.

Fleet: 35B-4.6 K512 p9, 35B-5.4 / Flash-4.4 / Flash-5.5 K1024 p10,
gemma-26b-6.2 K2048 p11; decode N = 1/8/32, ragged OUT, padded tail block,
and a large skewed-routing N.
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
    codes = rng.integers(0, K, size=(E, OUT, IN // 2), dtype=np.uint16)
    if skew:
        eidx = np.where(rng.random(N) < 0.8, 0, rng.integers(0, E, size=N))
    else:
        eidx = rng.integers(0, E, size=(N,))
    return (mx.array(rng.normal(0, 1, size=(N, IN)).astype(np.float16)),
            mx.array(eidx.astype(np.uint32)),
            mx.array(vq_pack.pack(codes, BITS)),
            mx.array(rng.normal(0, 1, size=(K, 2)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // G)).astype(np.float16)))


def _run(monkeypatch, walk, args, bits):
    monkeypatch.setattr(V, "_D2_WALK", walk)
    y = V._fused(*args, pack_bits=bits)
    mx.eval(y)
    return np.array(y).view(np.uint16)


# (E, OUT, IN, K, BITS)
SHAPES = [
    (4,  512, 2048,  512,  9),   # 35B-4.6 gate/up
    (4, 2048,  512,  512,  9),   # 35B-4.6 down
    (4,  512, 2048, 1024, 10),   # 35B-5.4 / Flash K1024 gate/up
    (4, 2048,  512, 1024, 10),   # K1024 down
    (4,  704, 2816, 2048, 11),   # gemma-26b-ish gate/up
    (4, 2816,  704, 2048, 11),   # gemma-26b-ish down
    (3,  100, 2048, 1024, 10),   # ragged OUT
    (3,  256,  640, 2048, 11),   # NSUB=320 fine; G=64 groups
    (3,  256,  576,  512,  9),   # NSUB=288: padded tail block
]


@pytest.mark.parametrize("N", [1, 8, 32])
@pytest.mark.parametrize("shape", SHAPES)
def test_d2_walk_bit_exact(monkeypatch, shape, N):
    E, OUT, IN, K, BITS = shape
    assert V._d2_tg_fits(K, IN // 2)
    args = _args(E, OUT, IN, K, BITS, N)
    assert np.array_equal(_run(monkeypatch, False, args, BITS),
                          _run(monkeypatch, True, args, BITS))


def test_d2_walk_bit_exact_large_skewed(monkeypatch):
    args = _args(8, 512, 2048, 1024, 10, 4096, skew=True, seed=3)
    assert np.array_equal(_run(monkeypatch, False, args, 10),
                          _run(monkeypatch, True, args, 10))


def test_d2_walk_dispatches(monkeypatch):
    monkeypatch.setattr(V, "_D2_WALK", True)
    V._fused(*_args(2, 256, 2048, 1024, 10, 1), pack_bits=10)
    names = {p[2] for p in V._KERNELS.values() if isinstance(p, tuple) and len(p) == 10}
    assert "vq_fused_packed10_d2_walk" in names


def test_default_on():
    import os
    if os.environ.get("VQ_D2_WALK") is None:
        assert V._D2_WALK is True  # default ON since F182
