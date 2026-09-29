"""kswarm devcb-walk: VQ_D4_DEVCB_WALK (bit-walker code fetch on the packed d4
DEVICE-codebook kernel) must be BIT-IDENTICAL to the shipped default
vq_fused_packed{bits}_d4_devcb kernel.

Compared as uint16 bit patterns, never a tolerance. Shapes are the devcb
fleet: K8192 p13 and K16384 p14 at the 35B (IN 2048 / 512) and GLM
(IN 4096 / 1536) expert shapes, at decode N = 1/8/32, plus a ragged OUT,
an IN whose NSUB is not a multiple of 32 (padded tail block), and a large
skewed-routing N.
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
    codes = rng.integers(0, K, size=(E, OUT, IN // 4), dtype=np.uint16)
    if skew:
        eidx = np.where(rng.random(N) < 0.8, 0, rng.integers(0, E, size=N))
    else:
        eidx = rng.integers(0, E, size=(N,))
    return (mx.array(rng.normal(0, 1, size=(N, IN)).astype(np.float16)),
            mx.array(eidx.astype(np.uint32)),
            mx.array(vq_pack.pack(codes, BITS)),
            mx.array(rng.normal(0, 1, size=(K, 4)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // G)).astype(np.float16)))


def _run(monkeypatch, walk, args, bits):
    monkeypatch.setattr(V, "_D4_DEVCB_WALK", walk)
    y = V._fused(*args, pack_bits=bits)
    mx.eval(y)
    return np.array(y).view(np.uint16)


# (E, OUT, IN, K, BITS)
SHAPES = [
    (4,  512, 2048,  8192, 13),   # 35B-3.8 gate/up
    (4, 2048,  512,  8192, 13),   # 35B-3.8 down
    (4,  512, 2048, 16384, 14),   # 35B K16384 gate/up
    (4, 1536, 4096, 16384, 14),   # GLM gate/up
    (4, 4096, 1536,  8192, 13),   # GLM down
    (3,  100, 2048, 16384, 14),   # ragged OUT
    (3,  256,  640,  8192, 13),   # NSUB=160: padded tail block
]


@pytest.mark.parametrize("N", [1, 8, 32])
@pytest.mark.parametrize("shape", SHAPES)
def test_devcb_walk_bit_exact(monkeypatch, shape, N):
    E, OUT, IN, K, BITS = shape
    assert not V._d4_tg_fits(K, IN // 4)
    args = _args(E, OUT, IN, K, BITS, N)
    assert np.array_equal(_run(monkeypatch, False, args, BITS),
                          _run(monkeypatch, True, args, BITS))


def test_devcb_walk_bit_exact_large_skewed(monkeypatch):
    args = _args(8, 512, 2048, 8192, 13, 4096, skew=True, seed=3)
    assert np.array_equal(_run(monkeypatch, False, args, 13),
                          _run(monkeypatch, True, args, 13))


def test_devcb_walk_dispatches(monkeypatch):
    monkeypatch.setattr(V, "_D4_DEVCB_WALK", True)
    V._fused(*_args(2, 256, 2048, 8192, 13, 1), pack_bits=13)
    names = {p[2] for p in V._KERNELS.values() if isinstance(p, tuple) and len(p) == 10}
    assert "vq_fused_packed13_d4_devcb_walk" in names


def test_default_off():
    import os
    if os.environ.get("VQ_D4_DEVCB_WALK") is None:
        assert V._D4_DEVCB_WALK is False
