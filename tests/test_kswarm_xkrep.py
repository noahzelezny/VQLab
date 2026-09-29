"""VQ_DECODE_XKREP: the decode branch reads the [T, IN] token matrix and
divides the x-row address by top_k in-kernel, instead of materializing the
[N, IN] broadcast copy on the host. The only claim is BIT-EXACTNESS against
the default decode path, compared as uint16 bit patterns, over every fleet
decode geometry (see the kernel map) at N = 1 / 8 / 20 / 32 and bf16 + fp16
activations. Also checks the rewrite reached the kernel (an _xk plan exists),
so a pass cannot come from the flag silently not applying.
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


def _run(mod, x, idx, on, monkeypatch):
    monkeypatch.setattr(V, "_DECODE_XKREP", on)
    y = mod(x, idx)
    mx.eval(y)
    return np.array(y.view(mx.uint16))


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("geom", GEOMS, ids=[g[0] for g in GEOMS])
def test_xkrep_bit_exact(geom, dtype, route, monkeypatch):
    _, E, OUT, IN, K, d, pb = geom
    T, k = route
    mod = _module(E, OUT, IN, K, d, pb)
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(0, 1, size=(T, 1, 1, IN)).astype(np.float32)
                 ).astype(dtype)
    idx = mx.array(np.stack([rng.permutation(E)[:k] if k <= E else
                             rng.integers(0, E, size=k) for _ in range(T)])
                   .astype(np.uint32))
    base = _run(mod, x, idx, False, monkeypatch)
    got = _run(mod, x, idx, True, monkeypatch)
    assert base.shape == got.shape == (T, k, 1, OUT)
    assert np.array_equal(base, got), (
        f"{int((base != got).sum())} of {base.size} differ")
    # the arm must actually have dispatched the rewritten kernel
    assert any(isinstance(kk, tuple) and kk[0] == "plan" and kk[-1] == k
               for kk in V._KERNELS), "xkrep plan never resolved"


def test_rewrite_refuses_foreign_x_reads():
    with pytest.raises(NotImplementedError):
        V._xkrep_src("const device T* xrow = x + (size_t)t * IN;\n"
                     "float a = x[0];")
    with pytest.raises(NotImplementedError):
        V._xkrep_src("float a = x[t];")


def test_every_fused_source_rewrites():
    for name in dir(V):
        if name.startswith("_SRC_FUSED"):
            V._xkrep_src(getattr(V, name))
