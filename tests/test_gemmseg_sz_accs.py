"""F206: gemmseg2's SKIPZERO (row-table) arm with VQ_GEMMSEG_ACCS.

Per skipzero geometry (packed d2/d4/d8, unpacked-uint8 d4; dead rows incl. a
whole dead expert), at N=512 and 8192 pairs on normal and student-t3 bf16
inputs, against a float64 exact reference:
  (a) sz gemmseg2+ACCS is at least as accurate as the sz fused walk (rule: no
      kernel change may reduce accuracy; F205 kept skipzero modules on the walk
      below 4096 pairs because this arm did not exist);
  (b) sz gemmseg2+ACCS is BYTE-EQUAL to the expanded module's gemmseg2+ACCS.
Run `python tests/test_gemmseg_sz_accs.py` for the table.
"""
import os
import sys

import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS
from vqlab import vq_pack as VP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gemmseg_flags_geoms import E, SLACK, _exact, _inputs, _rms, _run  # noqa: E402

# (d, K, packed, IN, OUT): the skipzero-served classes (docs/SKIPZERO.md)
SZ_GEOMS = [(4, 256, False, 512, 96), (4, 256, True, 640, 80), (4, 2048, True, 512, 97),
            (8, 16384, True, 1024, 64), (8, 16384, True, 2048, 97),
            (2, 256, True, 512, 96), (2, 1024, True, 512, 129), (2, 2048, True, 704, 64)]


def _setup_sz(d, K, packed, IN, OUT, dead=0.4, seed=0):
    r = np.random.default_rng(seed)
    codes = r.integers(0, K, (E, OUT, IN // d)).astype(np.uint16 if K > 256 else np.uint8)
    cb = (r.standard_normal((K, d)) * 0.05).astype(np.float16)
    sc = (r.standard_normal((E, OUT, IN // 64)) * 0.1 + 1).astype(np.float16)
    live = r.random((E, OUT)) >= dead
    live[1, :] = False
    live[2, :] = True
    codes[~live] = 0
    sc[~live] = 0
    W = cb.astype(np.float64)[codes.astype(np.int64)].reshape(E, OUT, IN)
    W = W * np.repeat(sc.astype(np.float64), 64, axis=2)
    tbl = np.full((E, OUT), -1, np.int32)
    tbl[live] = np.arange(int(live.sum()), dtype=np.int32)
    kw, c = {}, codes
    if packed:
        bits = VP.bits_for_k(K)
        c = VP.pack(codes.astype(np.uint32), bits)
        kw = dict(pack_bits=bits, in_features=IN)
    full = VS.VQSwitchLinear(mx.array(c), mx.array(cb), mx.array(sc), group_size=64, **kw)
    sz = VS.VQSwitchLinear(mx.array(c[live]), mx.array(cb), mx.array(sc[live]),
                           group_size=64, row_table=mx.array(tbl), **kw)
    return full, sz, W


def measure(geom, pairs, dist):
    full, sz, W = _setup_sz(*geom)
    x, idx = _inputs(geom[3], pairs, dist)
    ref = _exact(W, x, idx)
    walk = _run(sz, x, idx, 10 ** 9)
    g2 = _run(sz, x, idx, 0)
    g2a = _run(sz, x, idx, 0, accs=True, cvec=True)
    fa = _run(full, x, idx, 0, accs=True, cvec=True)
    return dict(walk=_rms(walk, ref), g2=_rms(g2, ref), g2accs=_rms(g2a, ref),
                eq_expanded=bool(np.array_equal(g2a, fa)))


CASES = [(g, p, dd) for g in SZ_GEOMS for p in (512, 8192) for dd in ("normal", "t3")]


@pytest.mark.parametrize("geom,pairs,dist", CASES,
                         ids=[f"d{g[0]}K{g[1]}{'p' if g[2] else 'u'}I{g[3]}-N{p}-{dd}"
                              for g, p, dd in CASES])
def test_sz_accs(geom, pairs, dist):
    m = measure(geom, pairs, dist)
    assert m["eq_expanded"], (geom, pairs, dist, "sz+ACCS != expanded+ACCS")
    assert m["g2accs"] <= m["walk"] * SLACK, (geom, pairs, dist, m)


if __name__ == "__main__":
    print(f"{'geometry':24s} {'N':>5s} {'dist':6s} {'walk':>10s} {'g2/walk':>8s} "
          f"{'accs/walk':>9s} =expanded")
    for g, p, dd in CASES:
        m = measure(g, p, dd)
        print(f"d{g[0]} K{g[1]:<6d}{'pk' if g[2] else 'u8':3s} IN{g[3]:<5d} {p:5d} {dd:6s} "
              f"{m['walk']:10.3e} {m['g2'] / m['walk']:8.3f} {m['g2accs'] / m['walk']:9.3f} "
              f"{m['eq_expanded']}")
