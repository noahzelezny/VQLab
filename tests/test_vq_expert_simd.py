"""The simdgroup-per-row EXPERT kernels: bit-exact against the thread-per-row
kernels they replace.

Why this file exists. The device-codebook expert kernels (_SRC_FUSED_D8 and
its packed twin) gave one output row to ONE thread, so a row was a chain of
~2*NSUB dependent device loads -- read the code, then index a 256 KB codebook
with it -- with a single load in flight. That is latency-bound, not
bandwidth-bound: the 397B shapes moved ~50 GB/s on a machine that does 546.
The simdgroup variants give each row a 32-lane simdgroup, one scale group per
lane, so a row has up to 32 independent loads outstanding, and combine the
per-group partials with a simd_shuffle reduction.

The whole change rests on being BIT-EXACT. Every published perplexity, KL
gate and MTP acceptance number for the 397B and Flash-Next rungs was produced
by the old kernels; a variant that is merely "close" silently invalidates all
of them. So the assertion is np.array_equal, never a tolerance. The scales
must therefore be applied in ASCENDING group order in both layouts -- that is
what the shuffle loop's `for i = 0..gmax` preserves.

Shapes are the ones the SHIPPED artifacts dispatch, read off their
safetensors headers rather than assumed:
    397B  2.2bpw  d=8 K=16384 packed14, down [512,4096,1024] / gate_up [.,1024,4096]
    F-Next 2.1bpw d=8 K=16384 unpacked, down_proj-shaped
"""
import sys, pathlib
import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"))
import vq_switch as V
import vq_pack


def _rand_experts(E, OUT, IN, K, d, G, N, packed=0, seed=0):
    rng = np.random.default_rng(seed)
    NSUB = IN // d
    codes = rng.integers(0, K, size=(E, OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    c = mx.array(vq_pack.pack(codes, packed)) if packed else mx.array(codes)
    return (mx.array(rng.normal(0, 1, size=(N, IN)).astype(np.float16)),
            mx.array(rng.integers(0, E, size=(N,)).astype(np.uint32)),
            c,
            mx.array(rng.normal(0, 1, size=(K, d)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // G)).astype(np.float16)))


# (E, OUT, IN, K, d, G) -- device-codebook geometries only; the threadgroup
# variants (K <= 1024 at d8, all d2) are untouched by this change.
# Only NGRP >= 32 reaches the simd kernel (see the gate); the narrower shapes
# are kept so the parametrisation also proves the dispatcher still returns the
# right answer when it DECLINES the new layout.
D8_SHAPES = [
    (8, 1024, 4096, 16384, 8, 64),   # 397B gate/up_proj   (NGRP = 64)  -> simd
    (4,   96, 4096,  4096, 8, 64),   # OUT ragged vs the 32-row tile    -> simd
    (4,  100, 8192,  4096, 8, 64),   # NGRP = 128, two 32-lane blocks   -> simd
    (8, 4096, 1024, 16384, 8, 64),   # 397B down_proj      (NGRP = 16)  -> declined
    (8, 2560,  640, 16384, 8, 64),   # Flash-Next down     (NGRP = 10)  -> declined
]

D4_SHAPES = [
    (4,  512, 2048, 8192, 4, 64),    # 35B up/gate shape, unpacked (NGRP = 32)
    (4,  100, 4096, 8192, 4, 64),    # ragged OUT
]


@pytest.mark.parametrize("N", [1, 2, 5, 8])
@pytest.mark.parametrize("shape", D8_SHAPES)
def test_d8_simd_bit_exact(shape, N):
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, simd=False))
    simd = np.array(V._fused(*args, simd=True))
    assert np.array_equal(base, simd)


@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("bits", [12, 14])
@pytest.mark.parametrize("shape", D8_SHAPES)
def test_d8_packed_simd_bit_exact(shape, bits, N):
    E, OUT, IN, K, d, G = shape
    if K > (1 << bits):
        pytest.skip(f"K={K} does not fit {bits}-bit codes")
    args = _rand_experts(E, OUT, IN, K, d, G, N, packed=bits)
    base = np.array(V._fused(*args, pack_bits=bits, simd=False))
    simd = np.array(V._fused(*args, pack_bits=bits, simd=True))
    assert np.array_equal(base, simd)


@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("shape", D4_SHAPES)
def test_d4_devcb_simd_bit_exact(shape, N):
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, simd=False))
    simd = np.array(V._fused(*args, simd=True))
    assert np.array_equal(base, simd)


def _dispatched(pack_bits, simd, **kw):
    """Name of the kernel a shape actually dispatches."""
    before = set(V._KERNELS)
    V._fused(*_rand_experts(**kw, N=2, packed=pack_bits),
             pack_bits=pack_bits, simd=simd)
    new = set(V._KERNELS) - before
    return new  # empty if the kernel was already compiled by an earlier test


@pytest.mark.parametrize("bits", [0, 14])
def test_ngrp_gate_selects_the_layout(bits):
    """The NGRP >= 32 gate is the whole dispatch rule: at NGRP=64 the simd
    kernel must be chosen, at NGRP=10 it must NOT be (it measured 0.94x on the
    M4 Max, where lanes 10..31 sit idle)."""
    V._KERNELS.clear()
    wide = dict(E=4, OUT=256, IN=4096, K=16384, d=8, G=64)   # NGRP = 64
    narrow = dict(E=4, OUT=256, IN=640, K=16384, d=8, G=64)  # NGRP = 10
    assert any("simd" in n for n in _dispatched(bits, True, **wide))
    V._KERNELS.clear()
    assert not any("simd" in n for n in _dispatched(bits, True, **narrow))
    V._KERNELS.clear()
    assert not any("simd" in n for n in _dispatched(bits, False, **wide))
    V._KERNELS.clear()


def test_packed_d4_has_no_simd_path():
    """The 35B-A3B geometry (d=4, K=8192, pack_bits=13) intentionally has NO
    simd variant: the packed thread-per-row kernel measured 0.94-1.07x at
    NGRP=32 and 0.70-0.89x at NGRP=8, so there is nothing to bank (see the
    _SRC_FUSED_D4_DEVCB_SIMD comment). This pins the scope: a packed d=4
    shape must dispatch the OLD kernel even with simd on -- and still be
    bit-identical to simd=False, which is trivially true but cheap to hold."""
    kw = dict(E=4, OUT=256, IN=2048, K=8192, d=4, G=64)  # NGRP = 32
    V._KERNELS.clear()
    assert not any("simd" in n for n in _dispatched(13, True, **kw))
    args = _rand_experts(**kw, N=4, packed=13)
    assert np.array_equal(np.array(V._fused(*args, pack_bits=13, simd=False)),
                          np.array(V._fused(*args, pack_bits=13, simd=True)))


def test_d4_simd_declined_below_32_groups():
    """NGRP < 32 starves lanes and measured 0.70-0.89x, so the dispatcher must
    stay on the thread-per-row d4 kernel there -- and still be correct."""
    E, OUT, IN, K, d, G = 4, 2048, 512, 8192, 4, 64
    assert IN // G < 32
    args = _rand_experts(E, OUT, IN, K, d, G, 8)
    assert np.array_equal(np.array(V._fused(*args, simd=False)),
                          np.array(V._fused(*args, simd=True)))


def test_large_n_declines_simd():
    """End-to-end, seq>=3 on the 397B measured 13.6% SLOWER with the simd
    layout even though the isolated microbench says it wins at the same N --
    so the default dispatch (simd=None) must fall back to thread-per-row
    above _EXPERT_SIMD_MAX_N pairs. Explicit simd=True still forces it (that
    is what the A/B harness uses)."""
    kw = dict(E=4, OUT=256, IN=4096, K=16384, d=8, G=64)
    V._KERNELS.clear()
    a = _rand_experts(**kw, N=V._EXPERT_SIMD_MAX_N)
    V._fused(*a)
    assert any("simd" in n for n in V._KERNELS)
    V._KERNELS.clear()
    b = _rand_experts(**kw, N=V._EXPERT_SIMD_MAX_N + 1)
    V._fused(*b)
    assert not any("simd" in n for n in V._KERNELS)
    V._KERNELS.clear()


def test_simd_is_the_default():
    """The d8 device-codebook path ships with the simd layout on; VQ_EXPERT_SIMD
    is only an A/B escape hatch."""
    assert V._EXPERT_SIMD is True
