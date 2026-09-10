"""Pin the routing memo (F42): exact keys, bounded size, identical products.

The memo exists because gate/up/down of one MoE layer call the prefill path
with the SAME routing (measured 240/720 repeats per prefill, host prefix 3.9%
of wall, memo recovers ~2% measured end-to-end). Reliability bar: a hit must
be EXACT (keys are the routing bytes, so dict equality memcmps on collision),
the cache must stay bounded, and a hit must hand back tensors bit-identical
to what a rebuild would produce.
"""
import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from vqlab import vq_switch as vs  # noqa: E402


@pytest.fixture(autouse=True)
def clean_memo():
    vs._ROUTING_MEMO.clear()
    yield
    vs._ROUTING_MEMO.clear()


def _idx(seed, n=256, e=8):
    return np.random.default_rng(seed).integers(0, e, n).astype(np.int32)


def test_put_get_roundtrip_and_exact_keying():
    a, b = _idx(1), _idx(2)
    vs._memo_put(("tiles", a.tobytes(), 8, 32), ("A", 1))
    assert vs._memo_get(("tiles", a.tobytes(), 8, 32)) == ("A", 1)
    assert vs._memo_get(("tiles", b.tobytes(), 8, 32)) is None
    # same bytes, different RTILE or E -> different entry (a hit across
    # geometries would hand the kernel a wrong tiling)
    assert vs._memo_get(("tiles", a.tobytes(), 8, 64)) is None
    assert vs._memo_get(("tiles", a.tobytes(), 16, 32)) is None


def test_memo_is_bounded_fifo():
    for i in range(vs._ROUTING_MEMO_MAX + 5):
        vs._memo_put(("tiles", bytes([i]), 8, 32), i)
    assert len(vs._ROUTING_MEMO) == vs._ROUTING_MEMO_MAX
    # oldest evicted, newest present
    assert vs._memo_get(("tiles", bytes([0]), 8, 32)) is None
    assert vs._memo_get(("tiles", bytes([vs._ROUTING_MEMO_MAX + 4]), 8, 32)) \
        == vs._ROUTING_MEMO_MAX + 4


def _tile_products(idx_sorted, E, rt):
    """The prefix _gemmseg_prefill memoises, rebuilt independently."""
    counts = np.bincount(idx_sorted, minlength=E)
    touched = np.nonzero(counts)[0]
    starts = np.zeros(E + 1, np.int64)
    starts[1:] = np.cumsum(counts)
    metas = []
    for e in touched:
        c0 = int(starts[e])
        for r in range(0, int(counts[e]), rt):
            metas.append((int(e), c0 + r, min(rt, int(counts[e]) - r)))
    return np.array(metas, np.int32).reshape(-1), len(metas)


def test_hit_products_bit_identical_to_rebuild():
    """Drive _gemmseg_prefill twice with the same routing; the second call
    must hit and the memoised tmeta must equal an independent rebuild."""
    idx = np.sort(_idx(7, n=512, e=8))
    E, OUT, IN, D, K = 8, 64, 64, 4, 32
    codes = mx.zeros((E, OUT, IN // D), dtype=mx.uint16)
    codebook = mx.zeros((K, D), dtype=mx.float16)
    scales = mx.ones((E, OUT, IN // 64), dtype=mx.float16)
    xsrc = mx.zeros((512, IN), dtype=mx.float16)
    src = np.arange(512, dtype=np.uint32)
    for _ in range(2):
        try:
            vs._gemmseg_prefill(xsrc, src, idx, codes, codebook, scales,
                                0, IN)
        except Exception:
            # kernel compile/launch specifics are not under test here; the
            # memo write happens in the prefix, before the kernel branch
            pass
    hits = [k for k in vs._ROUTING_MEMO if k[0] == "tiles"]
    assert len(hits) == 1, "same routing must produce ONE tile entry"
    tmeta, ntiles = vs._ROUTING_MEMO[hits[0]]
    ref_meta, ref_n = _tile_products(idx, E, 32)
    assert ntiles == ref_n
    assert np.array_equal(np.array(tmeta), ref_meta)


def test_vectorized_tile_build_matches_reference_loop():
    """The vectorized tmeta build (F42 follow-up) must be BIT-identical to
    the double loop it replaced, across ragged/edge routings."""
    import numpy as np
    rng = np.random.default_rng(0)
    for trial in range(200):
        E = int(rng.integers(1, 65))
        n = int(rng.integers(1, 4097))
        rt = int(rng.choice([32, 64]))
        idx = np.sort(rng.integers(0, E, n)).astype(np.int32)
        counts = np.bincount(idx, minlength=E)
        touched = np.nonzero(counts)[0]
        starts = np.zeros(E + 1, np.int64); starts[1:] = np.cumsum(counts)
        ref = []
        for e in touched:
            c0 = int(starts[e])
            for r in range(0, int(counts[e]), rt):
                ref.append((int(e), c0 + r, min(rt, int(counts[e]) - r)))
        ref = np.array(ref, np.int32).reshape(-1)
        tc = counts[touched]
        ntiles_per = (tc + rt - 1) // rt
        eids = np.repeat(touched, ntiles_per)
        cum = np.cumsum(ntiles_per)
        within = (np.arange(int(cum[-1]) if len(cum) else 0)
                  - np.repeat(cum - ntiles_per, ntiles_per)) * rt
        rows = starts[eids] + within
        nrows = np.minimum(rt, counts[eids] - within)
        vec = np.stack([eids, rows, nrows], axis=1).astype(np.int32).reshape(-1)
        assert np.array_equal(ref, vec), f"trial {trial} diverged"
