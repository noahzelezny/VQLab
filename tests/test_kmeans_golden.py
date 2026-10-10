"""fit/kmeans.py reproduces, bit for bit, every k-means it replaced.

The goldens in tests/golden/kmeans/ were captured from the ORIGINAL private
copies (fit_additive._kmeanspp/_assign/_assign_mx/_lloyd/fit_additive,
geo_build._kmeanspp/_assign/_lloyd, fit_moe.kmeanspp_init/kmeans,
fit_dense_vq.assign, fit_ple.assign) on the inputs `cases()` builds, before
any caller was switched. Fits are compared byte-for-byte downstream, so any
drift here is a numerics change, not a refactor.

GPU argmin over near-tied distances can in principle round differently on
another GPU generation; the inputs are random floats, so ties are improbable,
and the test is marked `gpu`.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "kmeans" / "kmeans.npz"
pytestmark = pytest.mark.gpu


def cases():
    """{name: (callable taking the implementation namespace, ...)}: each case
    seeds its own RNGs so it is independent of order."""
    def data(seed, n, d):
        return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)

    def additive_small(ns):
        X = data(1, 3000, 4)
        rng = np.random.default_rng(5)
        C0 = ns["kmeanspp_np"](X, 16, rng)
        a = ns["assign_np"](X, C0)
        C1 = ns["lloyd_np"](X, C0.copy(), 3, rng)
        books, codes = ns["fit_additive"](data(2, 2000, 4), 8, M=2, iters=2,
                                          rng=np.random.default_rng(7))
        return {"C0": C0, "a": a, "C1": C1, "b0": books[0], "b1": books[1], "codes": codes}

    def additive_mx(ns):
        X = data(3, 40000, 4)           # > 1<<15 rows: the MLX assignment path
        rng = np.random.default_rng(9)
        C = ns["kmeanspp_np"](X[:4000], 32, rng)
        return {"a": ns["assign_np"](X, C), "amx": ns["assign_mx_np"](X, C, 1 << 12),
                "C1": ns["lloyd_np"](X, C.copy(), 2, rng)}

    def geo(ns):
        X = data(4, 6000, 4)
        rng = np.random.default_rng(11)
        C0 = ns["kmeanspp_pool"](X, 32, rng)
        w = np.abs(data(5, 6000, 1)[:, 0]).astype(np.float64) + 0.1
        C1 = ns["lloyd_w"](X, mx.array(C0), w, 3, 32, rng)
        a = ns["assign_chunked"](X, C1)
        return {"C0": C0, "C1": np.array(C1), "a": a}

    def moe(ns):
        # fit_moe's centroid update is a GPU scatter-add, whose float
        # accumulation order is not fixed run to run; on the CPU device the
        # same ops are deterministic, so the code equivalence is checked there.
        dev = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            return _moe(ns)
        finally:
            mx.set_default_device(dev)

    def _moe(ns):
        X = mx.array(data(6, 5000, 4))
        W = mx.array(np.abs(data(7, 5000, 1)[:, 0]) + 0.1)
        mx.random.seed(21)
        S0 = ns["kmeanspp_mx"](X, 16, cap=2000)
        mx.random.seed(22)
        S1 = ns["kmeanspp_mx"](X, 16, cap=2000, W=W)
        mx.random.seed(23)
        K0 = ns["kmeans_mx"](X, 16, 4)
        mx.random.seed(24)
        K1 = ns["kmeans_mx"](X, 16, 4, W=W)
        mx.random.seed(25)
        K2 = ns["kmeans_mx"](X, 16, 3, init="random")
        return {k: np.array(v) for k, v in
                dict(S0=S0, S1=S1, K0=K0, K1=K1, K2=K2).items()}

    def assign_mx_cases(ns):
        X = mx.array(data(8, 7000, 4))
        C = mx.array(data(9, 64, 4))
        return {"dense": np.array(ns["assign_mx"](X, C)),
                "dense_chunk": np.array(ns["assign_mx"](X, C, chunk=1000)),
                "ple": np.array(ns["assign_mx"](X, C, chunk=1_000_000))}

    return {"additive_small": additive_small, "additive_mx": additive_mx,
            "geo": geo, "moe": moe, "assign_mx": assign_mx_cases}


def new_namespace():
    from vqlab.fit import kmeans as K
    from vqlab.fit.fit_additive import fit_additive
    return {"kmeanspp_np": K.kmeanspp_init, "assign_np": K.assign,
            "assign_mx_np": K.assign_mx_np, "lloyd_np": K.lloyd,
            "fit_additive": fit_additive,
            "kmeanspp_pool": K.kmeanspp_init_pooled, "lloyd_w": K.lloyd_weighted,
            "assign_chunked": K.assign_chunked,
            "kmeanspp_mx": K.kmeanspp_init_mx, "kmeans_mx": K.kmeans,
            "assign_mx": K.assign_mx}


def run_all(ns):
    out = {}
    for name, fn in cases().items():
        for k, v in fn(ns).items():
            out[f"{name}.{k}"] = np.asarray(v)
    return out


@pytest.fixture(scope="module")
def golden():
    return dict(np.load(GOLDEN))


@pytest.fixture(scope="module")
def got():
    return run_all(new_namespace())


@pytest.mark.parametrize("key", sorted(np.load(GOLDEN).files) if GOLDEN.exists() else [])
def test_bit_identical(golden, got, key):
    g, n = golden[key], got[key]
    assert g.dtype == n.dtype and g.shape == n.shape, (key, g.dtype, n.dtype, g.shape, n.shape)
    assert g.tobytes() == n.tobytes(), key


def test_every_golden_is_checked(golden, got):
    assert sorted(golden) == sorted(got)


def test_geo_build_pack_is_the_runtime_pack():
    """geo_build used to carry its own pack/words_per_row; the shipped
    runtime/vq_pack.py versions give the same words for every code width,
    aligned or not."""
    from vqlab.runtime import vq_pack
    rng = np.random.default_rng(3)
    for bits in range(1, 17):
        for nsub in (32, 64, 96, 40, 7):
            codes = rng.integers(0, 1 << bits, (2, 3, nsub)).astype(np.uint32)
            words = -(-nsub // 32) * bits
            assert vq_pack.words_per_row(nsub, bits) == words
            want = np.zeros((2, 3, words), np.uint64)
            for j in range(nsub):           # the old geo_build.pack, verbatim
                off = (j & 31) * bits
                w = (j >> 5) * bits + (off >> 5)
                sh = off & 31
                v = codes[:, :, j].astype(np.uint64)
                want[:, :, w] |= (v << sh) & 0xFFFFFFFF
                if sh + bits > 32:
                    want[:, :, w + 1] |= v >> (32 - sh)
            got = vq_pack.pack(codes, bits)
            assert got.dtype == np.uint32
            assert (got == want.astype(np.uint32)).all(), (bits, nsub)
    from vqlab.fit import geo_build
    assert geo_build.pack is vq_pack.pack and geo_build.words_per_row is vq_pack.words_per_row
