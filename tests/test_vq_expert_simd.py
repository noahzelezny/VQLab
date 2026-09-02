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


# --- d=2 code-load width (E14x, 2026-09-02) --------------------------------
# _SRC_FUSED_D2_U32 reads FOUR unpacked uint8 codes per uint32 load instead of
# four uchar loads, by reinterpreting the code row with mx.view. That is the
# d2 analogue of the d4 U8-VIEW dispatch and is worth 1.47-1.86x on the d2
# expert path (real 2.1bpw layer-1 codes, M3 Ultra) -- but only if it is
# EXACT, for the same reason as everything else in this file: the gemma d2
# rungs' published KL numbers were scored through the uchar kernel.
#
# The shapes are the ones shipped artifacts dispatch: the Flash-Next 2.1bpw
# uses d2/K256 for layers 0-1 (gate/up NGRP=40, down NGRP=10), and the gemma
# d2 rungs use d2 throughout.
D2_SHAPES = [
    (8,  640, 2560, 256, 2, 64),   # F-Next 2.1bpw gate/up  (NGRP = 40)
    (8, 2560,  640, 256, 2, 64),   # F-Next 2.1bpw down     (NGRP = 10)
    (4,  100, 2816, 256, 2, 64),   # ragged OUT, gemma-ish width
    (4,  128, 2048, 256, 2, 32),   # G = 32
]


@pytest.mark.parametrize("N", [1, 2, 5, 8, 10, 20])
@pytest.mark.parametrize("shape", D2_SHAPES)
def test_d2_u32_bit_exact(shape, N):
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, d2_u32=False))
    u32 = np.array(V._fused(*args, d2_u32=True))
    assert np.array_equal(base, u32)


@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("shape", D2_SHAPES)
def test_d2_simd_bit_exact(shape, N):
    """_SRC_FUSED_D2_SIMD is measured-and-not-dispatched (0.80-1.09x), but it
    is kept as the reference twin, so it must stay correct."""
    E, OUT, IN, K, d, G = shape
    if IN // G < 32:
        pytest.skip("simd layout needs NGRP >= 32")
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, d2_u32=False, simd=False))
    _, eidx, codes, cb, sc = args
    x = args[0]
    NSUB = IN // d
    y, = V._get_kernel("vq_fused_d2_simd_t", V._SRC_FUSED_D2_SIMD)(
        inputs=[x, eidx, codes, cb, sc,
                mx.array([OUT, IN, d, G, N, K], dtype=mx.int32)],
        template=[("T", x.dtype), ("CT", codes.dtype),
                  ("MAX_K", K), ("MAX_TILE", 32 * (G // 2))],
        grid=(32, ((OUT + 31) // 32) * 32, N),
        threadgroup=(32, 32, 1),
        output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
    assert np.array_equal(base, np.array(y))


def test_d2_u32_is_the_default_and_dispatches():
    """Unpacked uint8 d2 codes must take the uint32 kernel by default."""
    assert V._D2_U32 is True
    V._KERNELS.clear()
    V._fused(*_rand_experts(E=4, OUT=256, IN=2560, K=256, d=2, G=64, N=8))
    assert any("d2_u32" in n for n in V._KERNELS)
    V._KERNELS.clear()


def test_ple_byte_codes_skips_the_identity_unpack():
    """At BITS=8 a 'packed' PLE row is one byte per code, so _unpack is the
    identity -- the fast path must be selected AND agree with it exactly."""
    rng = np.random.default_rng(0)
    rows, nsub, d, G = 512, 20, 8, 32
    cols = nsub * d
    codes = mx.array(rng.integers(0, 256, size=(rows, nsub), dtype=np.uint8))
    cb = mx.array(rng.normal(0, 1, size=(256, d)).astype(np.float16))
    sc = mx.array(rng.normal(0, 0.05, size=(rows, cols // G)).astype(np.float16))
    m = V.VQPLEEmbedding(codes, cb, sc, group_size=G, packed_nsub=nsub)
    assert m._byte_codes is True
    # rebuild the ORIGINAL extraction tables and run the old path
    bit0 = np.arange(nsub) * m._bits
    m._b0 = mx.array(bit0 // 8)
    m._sh = mx.array((bit0 % 8).astype(np.uint32))
    ids = mx.array(rng.integers(0, rows, size=16).astype(np.uint32))
    c = m._unpack(m.codes[ids])
    v = m.codebook[c.astype(mx.uint32)]
    old = (v.reshape(*ids.shape, -1)
           * mx.repeat(m.vq_scales[ids], G, axis=-1)).astype(mx.bfloat16)
    # compared with mx.array_equal: numpy has no bfloat16, so np.array() on
    # the bf16 output raises rather than converting.
    assert bool(mx.array_equal(old, m(ids)))
