"""The peak-bounded dense prefill path.

Why this file exists. Above the fused-kernel N cutoff, VQLinear materialises
a decoded fp16 weight per call (178 MB at 27B mlp shapes). mlx is lazy, so
nothing forced layer L's weight to be freed before layer L+1's was
allocated, and the live set grew with graph depth: measured 2.758 GiB of
transient across 16 real VQLinears and 6.811 GiB across 48
(scripts/bench_dense_prefill_peak.py) -- the ~7 GiB that made an 11.6 GiB
model need 18.7 GiB of peak on a 2048-token prompt. That matters because
Noah's use case is several models resident at once, where peak is the budget.

The fix forces each linear's product to evaluate before the next linear's
weight is decoded, which caps the live set at one layer's transient.

WHAT THIS FILE LEARNED THE HARD WAY. The plan was to row-tile and call it
bit-identical, on the (mathematically correct) reasoning that splitting along
OUT reorders no reduction. test_row_tiling_is_not_bit_identical_in_general is
the counterexample that killed that claim: mlx's GEMM picks its own K-split
from the operand SHAPE, so narrowing the output width changes the
accumulation order INSIDE the matmul even though this file reorders nothing.

So the shipped default is the forced mx.eval with NO row tiling -- same
shapes, same GEMM, provably same bits, and 13.1x of the available peak
reduction. Row tiling is opt-in. Both properties are pinned below; if
someone makes row tiling the default, test_defaults fails.
"""
import pathlib
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "src" / "vqlab"))
import vq_dense as VD  # noqa: E402
import vq_pack  # noqa: E402


def _linear(OUT, IN, K, D, G, seed=0, packed=0):
    rng = np.random.default_rng(seed)
    NSUB = IN // D
    codes = rng.integers(0, K, size=(OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    cbk = mx.array(rng.normal(0, 1, size=(K, D)).astype(np.float16))
    sc = mx.array(rng.normal(0, 0.05, size=(OUT, IN // G)).astype(np.float16))
    c = (mx.array(vq_pack.pack(codes[None], packed)[0]) if packed
         else mx.array(codes))
    return VD.VQLinear(c, cbk, sc, group_size=G, pack_bits=packed,
                       in_features=IN if packed else None)


def _at(mb, m, x):
    prev = VD._DENSE_DECODE_TILE_MB
    try:
        VD._DENSE_DECODE_TILE_MB = mb
        y = m(x)
        mx.eval(y)
        return np.array(y, copy=True)
    finally:
        VD._DENSE_DECODE_TILE_MB = prev


# N is above every fused cutoff, so these all take the decode path.
BIG_N = 256


@pytest.mark.parametrize("bits", [0, 12])
def test_default_path_is_bit_identical(bits):
    """The SHIPPED arm: forced eval, no row split. Must be exact.

    This is the one that guards every published score for these artifacts --
    the decode path is what they were produced on.
    """
    m = _linear(320, 512, 4096, 4, 64, seed=1, packed=bits)
    x = mx.array(np.random.default_rng(1).normal(0, 1, (BIG_N, 512))
                 .astype(np.float16))
    prev = VD._DENSE_DECODE_EVAL
    try:
        VD._DENSE_DECODE_EVAL = False
        ref = _at(0, m, x)
        VD._DENSE_DECODE_EVAL = True
        got = _at(0, m, x)
    finally:
        VD._DENSE_DECODE_EVAL = prev
    assert np.array_equal(ref, got), "the forced eval changed the bits"


@pytest.mark.parametrize("bits", [0, 12])
@pytest.mark.parametrize("mb", [512, 1])
def test_single_tile_budgets_are_bit_identical(bits, mb):
    """A budget that does not actually split (>= the whole weight) must be
    the default arm exactly -- it is the same one GEMM."""
    m = _linear(320, 512, 4096, 4, 64, seed=1, packed=bits)
    x = mx.array(np.random.default_rng(1).normal(0, 1, (BIG_N, 512))
                 .astype(np.float16))
    assert np.array_equal(_at(0, m, x), _at(mb, m, x))


def test_row_tiling_is_not_bit_identical_in_general():
    """THE REFUTATION, pinned so nobody re-derives the wrong claim.

    Row tiling splits along OUT and reorders nothing here -- and still moves
    the bits, because mlx's GEMM chooses its K-split by shape. This is why
    row tiling is opt-in and the default does not split.

    If a future mlx makes this identical, this test fails LOUDLY and the
    finding (and the default) can be revisited on purpose.
    """
    m = _linear(320, 512, 4096, 4, 64, seed=1, packed=12)
    x = mx.array(np.random.default_rng(1).normal(0, 1, (BIG_N, 512))
                 .astype(np.float16))
    ref = _at(0, m, x)
    tiled = _at(0.05, m, x)          # ~51 rows per tile
    assert ref.shape == tiled.shape
    np.testing.assert_allclose(ref, tiled, rtol=2e-2, atol=2e-2)
    assert not np.array_equal(ref, tiled), (
        "row tiling is now bit-identical at this shape -- mlx's GEMM "
        "split may have changed; re-examine the default in vq_dense.py")


def test_defaults():
    """Eval on, row tiling off. The non-bit-identical option must not be
    the default -- house rule."""
    import importlib
    m = importlib.reload(VD)
    assert m._DENSE_DECODE_EVAL is True
    assert m._DENSE_DECODE_TILE_MB == 0


def test_tiling_actually_tiles_and_covers_every_row():
    """A budget of one row's worth must produce OUT tiles, all rows correct.

    Guards the off-by-one that a row-tiling loop invites: a dropped or
    duplicated tail tile would still return the right SHAPE.
    """
    OUT, IN = 130, 512          # OUT deliberately not a multiple of anything
    m = _linear(OUT, IN, 256, 2, 64, seed=2)
    x = mx.array(np.random.default_rng(2).normal(0, 1, (8, IN))
                 .astype(np.float16))
    ref = _at(0, m, x)
    got = _at(IN * 2 / (1 << 20), m, x)     # exactly one row per tile
    assert got.shape == (8, OUT)
    assert np.array_equal(ref, got)


def test_eval_flag_reverts_to_the_lazy_pre_arc_path():
    """VQ_DENSE_DECODE_EVAL=0 is the documented A/B escape hatch."""
    import importlib
    import os
    prev = os.environ.get("VQ_DENSE_DECODE_EVAL")
    try:
        os.environ["VQ_DENSE_DECODE_EVAL"] = "0"
        assert importlib.reload(VD)._DENSE_DECODE_EVAL is False
        os.environ.pop("VQ_DENSE_DECODE_EVAL")
        assert importlib.reload(VD)._DENSE_DECODE_EVAL is True
    finally:
        if prev is None:
            os.environ.pop("VQ_DENSE_DECODE_EVAL", None)
        else:
            os.environ["VQ_DENSE_DECODE_EVAL"] = prev
        importlib.reload(VD)


def test_peak_is_bounded_and_flat_in_depth():
    """The actual property Noah asked for: transient must not grow with how
    many VQ linears the forward chains together.

    Small shapes, so the numbers are small -- what is asserted is the
    SCALING, which is the whole diagnosis: the lazy arm grows with depth,
    the forced-eval arm does not.
    """
    IN = 512
    mods = [_linear(2048, IN, 256, 2, 64, seed=i) for i in range(2)]
    # square-ish so the chain composes: IN -> 2048 -> IN
    back = [_linear(IN, 2048, 256, 2, 64, seed=10 + i) for i in range(2)]
    x = mx.array(np.random.default_rng(0).normal(0, 1, (BIG_N, IN))
                 .astype(np.float16))

    def chain(depth, ev):
        prev = VD._DENSE_DECODE_EVAL
        VD._DENSE_DECODE_EVAL = ev
        try:
            mx.eval(x)
            mx.synchronize()
            mx.clear_cache()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            h = x
            for _ in range(depth):
                h = back[0](mods[0](h))
            mx.eval(h)
            return mx.get_peak_memory() - base
        finally:
            VD._DENSE_DECODE_EVAL = prev
            mx.clear_cache()

    un1, un4 = chain(1, False), chain(4, False)
    ti1, ti4 = chain(1, True), chain(4, True)
    # unbounded grows with depth; tiled stays put.
    assert un4 > un1 * 1.5, (un1, un4)
    assert ti4 < ti1 * 1.5, (ti1, ti4)
    assert ti4 < un4, (ti4, un4)


def test_packed_cutoff_is_keyed_on_d():
    """The 96 cutoff was a d=2 measurement; d=4's own crossover is ~35.

    Pins the geometry lookup, and pins that the env override still wins --
    every A/B in the bench scripts depends on that.
    """
    import os
    assert VD._fused_max_n(12, 4) == 32
    assert VD._fused_max_n(9, 2) == 72
    assert VD._fused_max_n(0, 4) == VD._DENSE_FUSED_MAX_N_PLAIN
    prev = os.environ.get("VQ_DENSE_FUSED_MAX_N")
    try:
        os.environ["VQ_DENSE_FUSED_MAX_N"] = "777"
        import importlib
        m = importlib.reload(VD)
        assert m._fused_max_n(12, 4) == 777
    finally:
        if prev is None:
            os.environ.pop("VQ_DENSE_FUSED_MAX_N", None)
        else:
            os.environ["VQ_DENSE_FUSED_MAX_N"] = prev
        import importlib
        importlib.reload(VD)
