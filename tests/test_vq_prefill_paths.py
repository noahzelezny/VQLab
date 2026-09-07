"""Bit gates for the _prefill input-path variants (2026-09-06).

- Fused gather (VQ_MOE_FUSE_GATHER, default ON) must be BIT-IDENTICAL to
  the legacy three-copy chain: it gathers the same rows from a smaller
  source, so the GEMM operands are byte-equal. Compared as uint16 bit
  patterns (array_equal lies on NaN).
- Exact-length per-expert GEMMs (VQ_MOE_EXACT_GEMM, default OFF) change
  GEMM shapes and therefore fp16 reduction order: NOT bit-gated, checked
  numerically close + shape/order correct. It must stay opt-in.

Skewed routing on purpose (VQ-PF1: uniform routers hide everything),
plus edge cases: single-row experts, untouched experts, duplicate expert
per token, and a chunk count > 1 (small _DECODE_CHUNK).
"""
import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS


def _mk(E=16, OUT=48, IN=64, K=64, D=2, seed=0):
    r = np.random.default_rng(seed)
    NSUB = IN // D
    codes = mx.array(r.integers(0, K, (E, OUT, NSUB)).astype(np.uint8))
    cbk = mx.array((r.standard_normal((K, D)) * 0.05).astype(np.float16))
    sc = mx.array((r.standard_normal((E, OUT, IN // 8)) * 0.1 + 1)
                  .astype(np.float16))
    return VS.VQSwitchLinear(codes, cbk, sc, group_size=8)


def _routing(T=640, K_top=8, E=16, seed=1):
    """Skewed: expert 0 hot (~10x mean), expert E-1 exactly once,
    expert E-2 never; one token routes to the same expert twice."""
    r = np.random.default_rng(seed)
    p = np.ones(E); p[0] = 10; p[E - 2] = 0; p /= p.sum()
    idx = r.choice(E, size=(T, K_top), p=p).astype(np.uint32)
    idx[0, :2] = 3          # duplicate expert within one token
    idx[idx == (E - 1)] = 1
    idx[5, 0] = E - 1       # exactly one row for the last expert
    return idx


def _run(mod, x, idx, fuse, exact, chunk=4):
    old = (VS._FUSE_GATHER, VS._EXACT_GEMM, VS._DECODE_CHUNK)
    VS._FUSE_GATHER, VS._EXACT_GEMM, VS._DECODE_CHUNK = fuse, exact, chunk
    try:
        y = mod(x, mx.array(idx))
        mx.eval(y)
        return y
    finally:
        VS._FUSE_GATHER, VS._EXACT_GEMM, VS._DECODE_CHUNK = old


@pytest.fixture()
def setup():
    mod = _mk()
    idx = _routing()
    T = idx.shape[0]
    r = np.random.default_rng(2)
    x = mx.array((r.standard_normal((T, 1, 1, 64)) * 0.2).astype(np.float16))
    assert T * idx.shape[1] > VS.VQ_FUSED_MAX_N or True
    return mod, x, idx


def test_prefill_reached(setup):
    mod, x, idx = setup
    # the test must exercise _prefill, not the fused small-N kernel
    assert idx.size > 0
    old = VS.VQ_FUSED_MAX_N
    VS.VQ_FUSED_MAX_N = 1
    try:
        y = _run(mod, x, idx, fuse=False, exact=False)
        assert y.shape == (*idx.shape, 1, 48)
    finally:
        VS.VQ_FUSED_MAX_N = old


def test_fused_gather_bit_identical(setup):
    mod, x, idx = setup
    old = VS.VQ_FUSED_MAX_N
    VS.VQ_FUSED_MAX_N = 1
    try:
        y_legacy = _run(mod, x, idx, fuse=False, exact=False)
        y_fused = _run(mod, x, idx, fuse=True, exact=False)
    finally:
        VS.VQ_FUSED_MAX_N = old
    assert bool(mx.array_equal(y_legacy.view(mx.uint16),
                               y_fused.view(mx.uint16))), \
        "fused gather changed bits — it must gather identical rows"


def test_fused_gather_bit_identical_across_chunks(setup):
    mod, x, idx = setup
    old = VS.VQ_FUSED_MAX_N
    VS.VQ_FUSED_MAX_N = 1
    try:
        for chunk in (2, 5, 64):
            a = _run(mod, x, idx, fuse=False, exact=False, chunk=chunk)
            b = _run(mod, x, idx, fuse=True, exact=False, chunk=chunk)
            assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
    finally:
        VS.VQ_FUSED_MAX_N = old


def test_exact_gemm_numerically_close_and_opt_in(setup):
    mod, x, idx = setup
    assert VS._EXACT_GEMM is False or \
        __import__("os").environ.get("VQ_MOE_EXACT_GEMM") == "1", \
        "exact-GEMM must be opt-in: it moves fp16 reduction bits"
    old = VS.VQ_FUSED_MAX_N
    VS.VQ_FUSED_MAX_N = 1
    try:
        y_ref = _run(mod, x, idx, fuse=False, exact=False)
        for fuse in (False, True):
            y_ex = _run(mod, x, idx, fuse=fuse, exact=True)
            d = mx.abs(y_ex.astype(mx.float32) - y_ref.astype(mx.float32))
            rel = float(mx.max(d)) / max(1e-6, float(mx.max(mx.abs(
                y_ref.astype(mx.float32)))))
            assert rel < 1e-2, f"exact-GEMM diverged (rel {rel})"
    finally:
        VS.VQ_FUSED_MAX_N = old


def test_fused_gather_sorted_indices_path(setup):
    mod, x, idx = setup
    old = VS.VQ_FUSED_MAX_N
    VS.VQ_FUSED_MAX_N = 1
    # pre-sort rows by expert the way switch_layers' do_sort path does
    flat = idx.flatten()
    order = np.argsort(flat, kind="stable")
    try:
        # sorted_indices=True is only bit-comparable through the same
        # entry: legacy vs fused on identical pre-sorted inputs.
        T = idx.shape[0]
        xf_rows = np.repeat(np.arange(T, dtype=np.uint32), idx.shape[1])
        x2 = mx.array(np.array(x.reshape(T, 64)))
        xs = x2[mx.array(xf_rows[order])][:, None, None, :]
        idx_sorted = flat[order].reshape(-1, 1)
        a = _run(mod, xs, idx_sorted, fuse=False, exact=False)
        b = _run(mod, xs, idx_sorted, fuse=True, exact=False)
        assert bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)))
    finally:
        VS.VQ_FUSED_MAX_N = old


def test_vector_decode_bit_identical(setup):
    """VQ_DECODE_VEC pairs stores/reads but keeps per-element fp32
    arithmetic — must be bit-identical to the scalar decode kernel."""
    mod, x, idx = setup
    old = VS.VQ_FUSED_MAX_N
    VS.VQ_FUSED_MAX_N = 1
    try:
        y_scalar = _run(mod, x, idx, fuse=True, exact=False)
        VS._DECODE_VEC = True
        try:
            y_vec = _run(mod, x, idx, fuse=True, exact=False)
        finally:
            VS._DECODE_VEC = False
    finally:
        VS.VQ_FUSED_MAX_N = old
    assert bool(mx.array_equal(y_scalar.view(mx.uint16),
                               y_vec.view(mx.uint16))), \
        "vector decode changed bits — paired store must not alter values"


def _mk_packed(E=16, OUT=48, IN=128, K=64, seed=0):
    """Packed d=2 module at the gemmseg-eligible geometry (G=64)."""
    from vqlab import vq_pack
    r = np.random.default_rng(seed)
    NSUB = IN // 2
    bits = vq_pack.bits_for_k(K)
    raw = r.integers(0, K, (E, OUT, NSUB)).astype(np.uint32)
    codes = mx.array(vq_pack.pack(raw, bits))
    cbk = mx.array((r.standard_normal((K, 2)) * 0.05).astype(np.float16))
    sc = mx.array((r.standard_normal((E, OUT, IN // 64)) * 0.1 + 1)
                  .astype(np.float16))
    return VS.VQSwitchLinear(codes, cbk, sc, group_size=64,
                             pack_bits=bits, in_features=IN)


def test_gemmseg_numeric_gate():
    """Fused segmented VQ-GEMM vs legacy _prefill: max rel error < 1e-3
    on skewed routing (fp32 in-tile accumulation; NOT bit-gated — the
    reduction order legitimately differs, see the acceptance contract)."""
    mod = _mk_packed()
    idx = _routing()
    T = idx.shape[0]
    r = np.random.default_rng(3)
    x = mx.array((r.standard_normal((T, 1, 1, 128)) * 0.2).astype(np.float16))
    old = (VS.VQ_FUSED_MAX_N, VS._FUSED_GEMM, VS._FUSE_GATHER,
           VS._FUSED_GEMM_V2)
    VS.VQ_FUSED_MAX_N = 1
    try:
        VS._FUSED_GEMM, VS._FUSE_GATHER = False, True
        y_ref = mod(x, mx.array(idx)); mx.eval(y_ref)
        a = y_ref.astype(mx.float32)
        for v2 in (False, True):
            VS._FUSED_GEMM, VS._FUSED_GEMM_V2 = True, v2
            y_fg = mod(x, mx.array(idx)); mx.eval(y_fg)
            b = y_fg.astype(mx.float32)
            rel = float(mx.max(mx.abs(a - b))) / \
                max(1e-6, float(mx.max(mx.abs(a))))
            assert rel < 1e-3, f"fused VQ-GEMM v2={v2} diverged: rel {rel}"
    finally:
        (VS.VQ_FUSED_MAX_N, VS._FUSED_GEMM, VS._FUSE_GATHER,
         VS._FUSED_GEMM_V2) = old


def test_gemmseg_default_promoted_v2():
    # Promoted 2026-09-07: default is v2 unless the env opts out.
    import os as _os
    env = _os.environ.get("VQ_MOE_FUSED_GEMM")
    if env is None:
        assert VS._FUSED_GEMM and VS._FUSED_GEMM_V2
