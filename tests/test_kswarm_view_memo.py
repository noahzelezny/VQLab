"""VQ_VIEW_MEMO: memoize the zero-copy u8->u32 codes view on the decode
plan-hit path instead of issuing a fresh mx.view per call. Claim: output
BIT-EXACT (uint16 patterns) vs the default path over fleet decode geometries,
repeated calls (plan hit), both xkrep settings; and the memo actually fires
on the u8 geometries.
"""

import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as V
from vqlab import vq_pack

# (label, E, OUT, IN, K, d, pack_bits)  -- pack_bits 0 = unpacked u8/u16
GEOMS = [
    ("d4K2048p11", 8, 512, 2048, 2048, 4, 11),
    ("d4K512p9", 8, 256, 1024, 512, 4, 9),
    ("d4K256u8", 8, 256, 1024, 256, 4, 0),
    ("d4K256p8", 8, 2560, 640, 256, 4, 8),
    ("d4K8192p13", 8, 256, 2048, 8192, 4, 13),
    ("d4K16384p14", 4, 128, 2048, 16384, 4, 14),
    ("d8K16384p14in4096", 8, 256, 4096, 16384, 8, 14),
    ("d8K16384p14in2560", 8, 640, 2560, 16384, 8, 14),
    ("d8K16384p14in1024", 8, 512, 1024, 16384, 8, 14),
    ("d2K1024p10", 8, 256, 1024, 1024, 2, 10),
    ("d2K512p9", 8, 256, 1024, 512, 2, 9),
    ("d2K2048p11", 8, 704, 2816, 2048, 2, 11),
    ("d2K256u8", 8, 256, 1024, 256, 2, 0),
]
# (T tokens, top_k): N = T * top_k
ROUTES = [(1, 1), (1, 8), (2, 10), (4, 8)]


def _module(E, OUT, IN, K, d, pb, seed=0):
    rng = np.random.default_rng(seed)
    nsub = IN // d
    codes = rng.integers(0, K, size=(E, OUT, nsub),
                         dtype=np.uint16 if K > 256 else np.uint8)
    c = mx.array(vq_pack.pack(codes, pb)) if pb else mx.array(codes)
    cb = mx.array(rng.normal(0, 1, size=(K, d)).astype(np.float16))
    sc = mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // 64))
                  .astype(np.float16))
    return V.VQSwitchLinear(c, cb, sc, pack_bits=pb,
                            in_features=IN if pb else None)



def _run(mod, x, idx, on, xk, monkeypatch):
    monkeypatch.setattr(V, "_VIEW_MEMO", on)
    monkeypatch.setattr(V, "_DECODE_XKREP", xk)
    outs = []
    for _ in range(3):              # 1st resolves the plan, 2nd/3rd hit it
        y = mod(x, idx)
        mx.eval(y)
        outs.append(np.array(y.view(mx.uint16)))
    return outs


@pytest.mark.parametrize("xk", [False, True])
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("geom", GEOMS, ids=[g[0] for g in GEOMS])
def test_view_memo_bit_exact(geom, dtype, route, xk, monkeypatch):
    _, E, OUT, IN, K, d, pb = geom
    T, k = route
    mod = _module(E, OUT, IN, K, d, pb)
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(0, 1, size=(T, 1, 1, IN)).astype(np.float32)
                 ).astype(dtype)
    idx = mx.array(np.stack([rng.permutation(E)[:k] if k <= E else
                             rng.integers(0, E, size=k) for _ in range(T)])
                   .astype(np.uint32))
    V._VIEW_CACHE.clear()
    base = _run(mod, x, idx, False, xk, monkeypatch)
    assert not V._VIEW_CACHE
    got = _run(mod, x, idx, True, xk, monkeypatch)
    for b, g in zip(base, got):
        assert np.array_equal(b, g), f"{int((b != g).sum())} differ"
    if pb == 0 and d in (2, 4) and K <= 256 and not (d == 2 and not V._D2_U32):
        assert V._VIEW_CACHE, "memo never fired on a u8-view geometry"
