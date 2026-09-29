"""SKIPZERO runtime switch (config `vq_skipzero`, docs/SKIPZERO.md).

A VQSwitchLinear built with a row table over COMPACT live rows must be
BYTE-EQUAL to the same module over EXPANDED rows (dead rows = code 0,
scale +0) -- that is the stage-1 layout F175/F176 gated on KL, so byte
equality with it is the whole correctness claim. Checked for every
geometry the switch serves (packed d4 at 11 and 8 bits, unpacked uint8
d4 via U8-VIEW), at decode N (fused WALK) and prefill N (gemmseg2), with
skewed routing and whole dead experts. Compared as uint16 bit patterns.

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


def _mk(E, OUT, IN, K, packed, dead_frac, seed=0):
    """(expanded module, compact sz module) for the same weights."""
    r = np.random.default_rng(seed)
    D, G = 4, 64
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
    idx[0, :] = 1                            # the dead expert gets traffic
    return mx.array(idx)


GEOMS = [("packed11", 2048, True), ("packed8", 256, True), ("u8view", 256, False)]


@pytest.mark.parametrize("name,K,packed", GEOMS)
@pytest.mark.parametrize("T", [1, 3, 700])     # decode, decode, prefill (N=5600)
def test_sz_byte_equal_to_expanded(name, K, packed, T):
    E, OUT, IN, top = 8, 96, 512, 8
    full, sz = _mk(E, OUT, IN, K, packed, dead_frac=0.4)
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


def test_non_sz_unchanged_vs_head():
    """The switch must not move a single byte of the non-sz paths."""
    import subprocess, types, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        src = subprocess.run(["git", "show", "HEAD:src/vqlab/runtime/vq_switch.py"],
                             cwd=root, capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("no git HEAD to compare against")
    if "#if SZ" in src:
        pytest.skip("HEAD already carries the switch")
    head = types.ModuleType("vq_switch_head")
    exec(compile(src, "vq_switch_head", "exec"), head.__dict__)
    r = np.random.default_rng(5)
    for K, packed in ((2048, True), (256, True), (256, False)):
        E, OUT, IN = 8, 96, 512
        codes = r.integers(0, K, (E, OUT, IN // 4)).astype(np.uint16 if K > 256 else np.uint8)
        cb = (r.standard_normal((K, 4)) * 0.05).astype(np.float16)
        sc = (r.standard_normal((E, OUT, IN // 64)) * 0.1 + 1).astype(np.float16)
        kw = {}
        if packed:
            bits = VP.bits_for_k(K)
            codes = VP.pack(codes.astype(np.uint32), bits)
            kw = dict(pack_bits=bits, in_features=IN)
        for T in (1, 700):
            x = mx.array(r.standard_normal((T, 1, 1, IN)).astype(np.float16))
            idx = _idx(T, 8, E, seed=T)
            ys = [M.VQSwitchLinear(mx.array(codes), mx.array(cb), mx.array(sc),
                                   group_size=64, **kw)(x, idx)
                  for M in (VS, head)]
            mx.eval(ys)
            assert np.array_equal(_bits(ys[0]), _bits(ys[1])), (K, packed, T)
