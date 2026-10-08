"""F203: the F200-F202 MoE prefill flags, per fleet GEOMETRY (synthetic, no model).

The kernel change does not vary by family, only by geometry (d, K, packing).
For every geometry class in test_runtime_equivalence.GEOMS, at a gemmseg2-sized
N (8192 pairs) and at N=512 pairs:
  (a) VQ_GEMMSEG_CVEC on vs off is BIT-EXACT (it applies to packed codes only);
  (b) gemmseg2 + VQ_GEMMSEG_ACCS is at least as accurate as the fused walk
      against a float64 exact reference, on normal and heavy-tailed
      (student-t df=3) bf16 inputs -- Noah's rule: no kernel change may reduce
      accuracy. Plain gemmseg2 (shipped above VQ_FUSED_MAX_N) is reported too.
Run `python tests/test_gemmseg_flags_geoms.py` for the table.
"""
import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS
from vqlab import vq_pack as VP

# (d, K, packed, IN, OUT) -- same classes as test_runtime_equivalence.GEOMS;
# (4, 256, False) is the unpacked-uint8 (u8view) arm.
GEOMS = [(4, 256, False, 512, 96), (4, 256, True, 640, 80), (4, 512, True, 512, 96),
         (4, 2048, True, 512, 96), (4, 8192, True, 512, 96), (4, 16384, True, 512, 64),
         (8, 16384, True, 1024, 64), (2, 256, False, 512, 96), (2, 512, True, 512, 96),
         (2, 1024, True, 512, 96), (2, 2048, True, 704, 64)]
E, TOP = 8, 8
SLACK = 1.01   # ACCS rms err may exceed the walk's by at most 1%


def _setup(d, K, packed, IN, OUT, seed=0):
    r = np.random.default_rng(seed)
    codes = r.integers(0, K, (E, OUT, IN // d)).astype(np.uint16 if K > 256 else np.uint8)
    cb = (r.standard_normal((K, d)) * 0.05).astype(np.float16)
    sc = (r.standard_normal((E, OUT, IN // 64)) * 0.1 + 1).astype(np.float16)
    W = cb.astype(np.float64)[codes.astype(np.int64)].reshape(E, OUT, IN)
    W = W * np.repeat(sc.astype(np.float64), 64, axis=2)
    kw = {}
    c = codes
    if packed:
        bits = VP.bits_for_k(K)
        c = VP.pack(codes.astype(np.uint32), bits)
        kw = dict(pack_bits=bits, in_features=IN)
    lin = VS.VQSwitchLinear(mx.array(c), mx.array(cb), mx.array(sc), group_size=64, **kw)
    return lin, W


def _inputs(IN, pairs, dist, seed=1):
    r = np.random.default_rng(seed)
    T = pairs // TOP
    x = r.standard_t(3, (T, 1, 1, IN)) if dist == "t3" else r.standard_normal((T, 1, 1, IN))
    xb = mx.array(x.astype(np.float32)).astype(mx.bfloat16)
    idx = r.integers(0, E, (T, TOP)).astype(np.uint32)
    return xb, mx.array(idx)


def _run(lin, x, idx, maxn, accs=False, cvec=False):
    old = (VS.VQ_FUSED_MAX_N, VS._GEMMSEG_ACCS, VS._GEMMSEG_CVEC)
    VS.VQ_FUSED_MAX_N, VS._GEMMSEG_ACCS, VS._GEMMSEG_CVEC = maxn, accs, cvec
    try:
        y = lin(x, idx)
        mx.eval(y)
    finally:
        VS.VQ_FUSED_MAX_N, VS._GEMMSEG_ACCS, VS._GEMMSEG_CVEC = old
    return np.array(y.astype(mx.float32)).astype(np.float64)


def _exact(W, x, idx):
    xf = np.array(x.astype(mx.float32)).astype(np.float64)[:, 0, 0, :]   # [T, IN]
    ii = np.array(idx)
    return np.einsum("tkoi,ti->tko", W[ii], xf, optimize=True)          # [T, TOP, OUT]


def _rms(y, ref):
    return float(np.sqrt(np.mean((y.reshape(ref.shape) - ref) ** 2)))


def measure(geom, pairs, dist):
    d, K, packed, IN, OUT = geom
    lin, W = _setup(*geom)
    x, idx = _inputs(IN, pairs, dist)
    ref = _exact(W, x, idx)
    walk = _run(lin, x, idx, 10 ** 9)
    g2 = _run(lin, x, idx, 0)
    g2a = _run(lin, x, idx, 0, accs=True)
    g2acv = _run(lin, x, idx, 0, accs=True, cvec=True)
    g2cv = _run(lin, x, idx, 0, cvec=True)
    return dict(walk=_rms(walk, ref), g2=_rms(g2, ref), g2accs=_rms(g2a, ref),
                cvec_exact=bool(np.array_equal(g2, g2cv) and np.array_equal(g2a, g2acv)))


CASES = [(g, p, dist) for g in GEOMS for p in (512, 8192) for dist in ("normal", "t3")]


@pytest.mark.parametrize("geom,pairs,dist", CASES,
                         ids=[f"d{g[0]}K{g[1]}{'p' if g[2] else 'u'}-N{p}-{dd}" for g, p, dd in CASES])
def test_gemmseg_flags(geom, pairs, dist):
    m = measure(geom, pairs, dist)
    assert m["cvec_exact"], (geom, pairs, dist, "CVEC not bit-exact")
    assert m["g2accs"] <= m["walk"] * SLACK, (geom, pairs, dist, m)


if __name__ == "__main__":
    print(f"{'geometry':16s} {'N':>5s} {'dist':6s} {'walk':>10s} {'g2/walk':>8s} {'accs/walk':>9s} cvec")
    for g, p, dist in CASES:
        try:
            m = measure(g, p, dist)
            print(f"d{g[0]} K{g[1]:<6d}{'pk' if g[2] else 'u8':4s} {p:5d} {dist:6s} {m['walk']:10.3e} "
                  f"{m['g2'] / m['walk']:8.3f} {m['g2accs'] / m['walk']:9.3f} "
                  f"{'exact' if m['cvec_exact'] else 'DIFF'}")
        except Exception as e:  # report geometries where a flag does not apply / compile
            print(f"d{g[0]} K{g[1]} {'pk' if g[2] else 'u8'} N{p} {dist}: FAIL {type(e).__name__}: {str(e)[:120]}")
