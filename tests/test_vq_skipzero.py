"""SKIPZERO runtime switch (config `vq_skipzero`, docs/SKIPZERO.md).

A VQSwitchLinear built with a row table over COMPACT live rows must be
BYTE-EQUAL to the same module over EXPANDED rows (dead rows = code 0,
scale +0) -- that is the stage-1 layout F175/F176 gated on KL, so byte
equality with it is the whole correctness claim. Checked for every
geometry the switch serves (packed d4 at 11 and 8 bits, unpacked uint8
d4 via U8-VIEW, packed d8 K16384 at 14 bits -- the 397B 2.2's geometry,
packed d2 at K256-K2048),
at decode N (fused d4 WALK; d8 SIMD_DEVX_SS at N <= 20 and IN/G >= 32, d8
WALK otherwise) and prefill N (gemmseg2), with skewed routing, whole dead
experts and odd OUT. Compared as uint16 bit patterns.

The second test pins that adding the switch left the NON-sz kernels
unchanged: the SZ code sits behind `#if SZ`, so a runtime without the
row table must dispatch the same kernels to the same bytes.
"""
import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS
from vqlab import vq_pack as VP


def _bits(a):
    return np.array(a.astype(mx.float16)).view(np.uint16)


def _mk(E, OUT, IN, K, packed, dead_frac, seed=0, D=4):
    """(expanded module, compact sz module) for the same weights."""
    r = np.random.default_rng(seed)
    G = 64
    NSUB = IN // D
    codes = r.integers(0, K, (E, OUT, NSUB)).astype(np.uint16 if K > 256 else np.uint8)
    cb = (r.standard_normal((K, D)) * 0.05).astype(np.float16)
    sc = (r.standard_normal((E, OUT, IN // G)) * 0.1 + 1).astype(np.float16)
    live = r.random((E, OUT)) >= dead_frac
    live[1, :] = False                       # one whole dead expert
    live[2, :] = True                        # one fully live expert
    codes[~live] = 0                         # the expanded (stage-1) form
    sc[~live] = 0
    tbl = np.full((E, OUT), -1, np.int32)
    tbl[live] = np.arange(int(live.sum()), dtype=np.int32)
    if packed:
        bits = VP.bits_for_k(K)
        full = VP.pack(codes.astype(np.uint32), bits)
        comp = full[live]
        mk = lambda c, s, **kw: VS.VQSwitchLinear(  # noqa: E731
            mx.array(c), mx.array(cb), mx.array(s), group_size=G,
            pack_bits=bits, in_features=IN, **kw)
    else:
        full, comp = codes, codes[live]
        mk = lambda c, s, **kw: VS.VQSwitchLinear(  # noqa: E731
            mx.array(c), mx.array(cb), mx.array(s), group_size=G, **kw)
    return mk(full, sc), mk(comp, sc[live], row_table=mx.array(tbl))


def _idx(T, top, E, seed=1):
    r = np.random.default_rng(seed)
    p = np.ones(E); p[0] = 10; p /= p.sum()
    idx = r.choice(E, size=(T, top), p=p).astype(np.uint32)
    idx[0, 0] = 1                            # the dead expert gets traffic
    # (one slot only: at T=1 a whole row of it would make every output +0)
    return mx.array(idx)


# (name, d, K, packed, IN, OUT); the d8 rows are the 2.2's gate/up shape
# class (IN/G = 32 -> simd at small N) and down shape class (IN/G = 16 ->
# walk at every decode N), at odd OUT.
GEOMS = [("packed11", 4, 2048, True, 512, 96), ("packed8", 4, 256, True, 512, 96),
         ("u8view", 4, 256, False, 512, 96),
         ("d8_gateup", 8, 16384, True, 2048, 97), ("d8_down", 8, 16384, True, 1024, 97)]


@pytest.mark.parametrize("name,D,K,packed,IN,OUT", GEOMS)
@pytest.mark.parametrize("T,top", [(1, 8), (1, 10), (2, 10), (3, 10), (700, 8)])
def test_sz_byte_equal_to_expanded(name, D, K, packed, IN, OUT, T, top):
    # N = 8 / 10 / 20 (d8 simd), 30 (d8 walk), 5600 (prefill, gemmseg2)
    E = 8
    full, sz = _mk(E, OUT, IN, K, packed, dead_frac=0.4, D=D)
    assert sz.skipzero and not full.skipzero
    assert (sz.num_experts, sz.output_dims, sz.input_dims) == \
        (full.num_experts, full.output_dims, full.input_dims)
    x = mx.array(np.random.default_rng(2).standard_normal((T, 1, 1, IN))
                 .astype(np.float16))
    idx = _idx(T, top, E)
    a, b = full(x, idx), sz(x, idx)
    mx.eval(a, b)
    assert a.shape == b.shape
    assert np.array_equal(_bits(a), _bits(b)), name


@pytest.mark.parametrize("dead_frac", [0.0, 0.95])
@pytest.mark.parametrize("T,top", [(1, 10), (3, 10), (450, 10)])
def test_sz_d8_extremes(dead_frac, T, top):
    """No dead rows, and nearly all dead, at decode and prefill N, bf16 in."""
    E, OUT, IN = 8, 63, 2048
    full, sz = _mk(E, OUT, IN, 16384, True, dead_frac=dead_frac, seed=5, D=8)
    x = mx.array(np.random.default_rng(6).standard_normal((T, 1, 1, IN))
                 .astype(np.float32)).astype(mx.bfloat16)
    idx = _idx(T, top, E, seed=7)
    a, b = full(x, idx), sz(x, idx)
    mx.eval(a, b)
    assert np.array_equal(_bits(a), _bits(b))


# packed d2 at the fleet's K's (35B-4.6 K512, 35B-5.4 K1024, ...): gate/up
# shape class (IN 2048) and down shape class (IN 512), odd OUT. Decode is the
# d2 WALK at every N; prefill is gemmseg2 D_BAKE=2 with the threadgroup
# codebook, and again with the device codebook forced.
D2_GEOMS = [(K, IN, OUT) for K in (256, 512, 1024, 2048)
            for IN, OUT in ((2048, 97), (512, 129))]


@pytest.mark.parametrize("K,IN,OUT", D2_GEOMS)
@pytest.mark.parametrize("dead_frac", [0.0, 0.4, 0.95])
@pytest.mark.parametrize("T,top", [(1, 1), (1, 8), (2, 10), (3, 10), (700, 8)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sz_d2_byte_equal(K, IN, OUT, dead_frac, T, top, dtype):
    # N = 1 / 8 / 20 / 30 (decode, d2 WALK), 5600 (prefill, gemmseg2)
    E = 8
    full, sz = _mk(E, OUT, IN, K, True, dead_frac=dead_frac, seed=K + IN, D=2)
    x = mx.array(np.random.default_rng(6).standard_normal((T, 1, 1, IN))
                 .astype(np.float32)).astype(getattr(mx, dtype))
    idx = _idx(T, top, E, seed=7)
    cbdev = ["0", "1"] if T > 100 else [VS._GEMMSEG_CBDEV]
    old = VS._GEMMSEG_CBDEV
    try:
        for arm in cbdev:
            VS._GEMMSEG_CBDEV = arm
            a, b = full(x, idx), sz(x, idx)
            mx.eval(a, b)
            assert np.array_equal(_bits(a), _bits(b)), arm
    finally:
        VS._GEMMSEG_CBDEV = old


def test_sz_d2_refuses_unswitched_kernel():
    E, OUT, IN = 4, 32, 512
    _, sz = _mk(E, OUT, IN, 1024, True, dead_frac=0.3, D=2)
    old = VS._D2_WALK
    VS._D2_WALK = False
    try:
        with pytest.raises(NotImplementedError):
            sz(mx.zeros((1, 1, 1, IN), mx.float16), mx.zeros((1, 8), mx.uint32))
    finally:
        VS._D2_WALK = old


@pytest.mark.parametrize("flag", ["_D8_WALK", "_D8_SS", "_D8_DEVX"])
def test_sz_d8_refuses_unswitched_kernel(flag):
    E, OUT, IN = 4, 32, 2048
    _, sz = _mk(E, OUT, IN, 16384, True, dead_frac=0.3, D=8)
    T = 3 if flag == "_D8_WALK" else 1          # N=30 walks, N=10 is simd
    old = getattr(VS, flag)
    setattr(VS, flag, False)
    try:
        with pytest.raises(NotImplementedError):
            sz(mx.zeros((T, 1, 1, IN), mx.float16),
               mx.zeros((T, 10), mx.uint32))
    finally:
        setattr(VS, flag, old)


def test_sz_refuses_unswitched_kernel():
    E, OUT, IN = 4, 32, 256
    _, sz = _mk(E, OUT, IN, 256, True, dead_frac=0.3)
    old = VS._D4_WALK
    VS._D4_WALK = False
    try:
        with pytest.raises(NotImplementedError):
            sz(mx.zeros((1, 1, 1, IN), mx.float16), mx.zeros((1, 2), mx.uint32))
    finally:
        VS._D4_WALK = old


# The non-sz equivalence to the pre-switch runtime lives in
# tests/test_runtime_equivalence.py (every fleet geometry, every certified revision).
