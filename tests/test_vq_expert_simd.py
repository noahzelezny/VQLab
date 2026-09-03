"""The simdgroup-per-row EXPERT kernels: bit-exact against the thread-per-row
kernels they replace.

Why this file exists. The device-codebook expert kernels (_SRC_FUSED_D8 and
its packed twin) gave one output row to ONE thread, so a row was a chain of
~2*NSUB dependent device loads -- read the code, then index a 256 KB codebook
with it -- with a single load in flight. That is latency-bound, not
bandwidth-bound: the 397B shapes moved ~50 GB/s on a machine that does 546.
The simdgroup variants give each row a 32-lane simdgroup, one scale group per
lane, so a row has up to 32 independent loads outstanding, and combine the
per-group partials with a simd_shuffle reduction.

The whole change rests on being BIT-EXACT. Every published perplexity, KL
gate and MTP acceptance number for the 397B and Flash-Next rungs was produced
by the old kernels; a variant that is merely "close" silently invalidates all
of them. So the assertion is np.array_equal, never a tolerance. The scales
must therefore be applied in ASCENDING group order in both layouts -- that is
what the shuffle loop's `for i = 0..gmax` preserves.

Shapes are the ones the SHIPPED artifacts dispatch, read off their
safetensors headers rather than assumed:
    397B  2.2bpw  d=8 K=16384 packed14, down [512,4096,1024] / gate_up [.,1024,4096]
    F-Next 2.1bpw d=8 K=16384 unpacked, down_proj-shaped
"""
import sys, pathlib
import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"))
import vq_switch as V
import vq_pack


def _rand_experts(E, OUT, IN, K, d, G, N, packed=0, seed=0):
    rng = np.random.default_rng(seed)
    NSUB = IN // d
    codes = rng.integers(0, K, size=(E, OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    c = mx.array(vq_pack.pack(codes, packed)) if packed else mx.array(codes)
    return (mx.array(rng.normal(0, 1, size=(N, IN)).astype(np.float16)),
            mx.array(rng.integers(0, E, size=(N,)).astype(np.uint32)),
            c,
            mx.array(rng.normal(0, 1, size=(K, d)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // G)).astype(np.float16)))


# (E, OUT, IN, K, d, G) -- device-codebook geometries only; the threadgroup
# variants (K <= 1024 at d8, all d2) are untouched by this change.
# Only NGRP >= 32 reaches the simd kernel (see the gate); the narrower shapes
# are kept so the parametrisation also proves the dispatcher still returns the
# right answer when it DECLINES the new layout.
D8_SHAPES = [
    (8, 1024, 4096, 16384, 8, 64),   # 397B gate/up_proj   (NGRP = 64)  -> simd
    (4,   96, 4096,  4096, 8, 64),   # OUT ragged vs the 32-row tile    -> simd
    (4,  100, 8192,  4096, 8, 64),   # NGRP = 128, two 32-lane blocks   -> simd
    (8,  640, 2560, 16384, 8, 64),   # Flash-Next gate/up (NGRP = 40,
                                     #  partial second block)           -> simd
    (8, 4096, 1024, 16384, 8, 64),   # 397B down_proj      (NGRP = 16)  -> declined
    (8, 2560,  640, 16384, 8, 64),   # Flash-Next down     (NGRP = 10)  -> declined
]

D4_SHAPES = [
    (4,  512, 2048, 8192, 4, 64),    # 35B up/gate shape, unpacked (NGRP = 32)
    (4,  100, 4096, 8192, 4, 64),    # ragged OUT
]


@pytest.mark.parametrize("N", [1, 2, 5, 8])
@pytest.mark.parametrize("shape", D8_SHAPES)
def test_d8_simd_bit_exact(shape, N):
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, simd=False))
    simd = np.array(V._fused(*args, simd=True))
    assert np.array_equal(base, simd)


@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("bits", [12, 14])
@pytest.mark.parametrize("shape", D8_SHAPES)
def test_d8_packed_simd_bit_exact(shape, bits, N):
    E, OUT, IN, K, d, G = shape
    if K > (1 << bits):
        pytest.skip(f"K={K} does not fit {bits}-bit codes")
    args = _rand_experts(E, OUT, IN, K, d, G, N, packed=bits)
    base = np.array(V._fused(*args, pack_bits=bits, simd=False))
    simd = np.array(V._fused(*args, pack_bits=bits, simd=True))
    assert np.array_equal(base, simd)


@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("shape", D4_SHAPES)
def test_d4_devcb_simd_bit_exact(shape, N):
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, simd=False))
    simd = np.array(V._fused(*args, simd=True))
    assert np.array_equal(base, simd)


def _dispatched(pack_bits, simd, **kw):
    """Name of the kernel a shape actually dispatches."""
    before = set(V._KERNELS)
    V._fused(*_rand_experts(**kw, N=2, packed=pack_bits),
             pack_bits=pack_bits, simd=simd)
    new = set(V._KERNELS) - before
    return new  # empty if the kernel was already compiled by an earlier test


@pytest.mark.parametrize("bits", [0, 14])
def test_ngrp_gate_selects_the_layout(bits):
    """The NGRP >= 32 gate is the whole dispatch rule: at NGRP=64 the simd
    kernel must be chosen, at NGRP=10 it must NOT be (it measured 0.94x on the
    M4 Max, where lanes 10..31 sit idle)."""
    V._KERNELS.clear()
    wide = dict(E=4, OUT=256, IN=4096, K=16384, d=8, G=64)   # NGRP = 64
    narrow = dict(E=4, OUT=256, IN=640, K=16384, d=8, G=64)  # NGRP = 10
    assert any("simd" in n for n in _dispatched(bits, True, **wide))
    V._KERNELS.clear()
    assert not any("simd" in n for n in _dispatched(bits, True, **narrow))
    V._KERNELS.clear()
    assert not any("simd" in n for n in _dispatched(bits, False, **wide))
    V._KERNELS.clear()


def test_packed_d4_has_no_simd_path():
    """The 35B-A3B geometry (d=4, K=8192, pack_bits=13) intentionally has NO
    simd variant: the packed thread-per-row kernel measured 0.94-1.07x at
    NGRP=32 and 0.70-0.89x at NGRP=8, so there is nothing to bank (see the
    _SRC_FUSED_D4_DEVCB_SIMD comment). This pins the scope: a packed d=4
    shape must dispatch the OLD kernel even with simd on -- and still be
    bit-identical to simd=False, which is trivially true but cheap to hold."""
    kw = dict(E=4, OUT=256, IN=2048, K=8192, d=4, G=64)  # NGRP = 32
    V._KERNELS.clear()
    assert not any("simd" in n for n in _dispatched(13, True, **kw))
    args = _rand_experts(**kw, N=4, packed=13)
    assert np.array_equal(np.array(V._fused(*args, pack_bits=13, simd=False)),
                          np.array(V._fused(*args, pack_bits=13, simd=True)))


def test_d4_simd_declined_below_32_groups():
    """NGRP < 32 starves lanes and measured 0.70-0.89x, so the dispatcher must
    stay on the thread-per-row d4 kernel there -- and still be correct."""
    E, OUT, IN, K, d, G = 4, 2048, 512, 8192, 4, 64
    assert IN // G < 32
    args = _rand_experts(E, OUT, IN, K, d, G, 8)
    assert np.array_equal(np.array(V._fused(*args, simd=False)),
                          np.array(V._fused(*args, simd=True)))


def test_large_n_declines_simd():
    """End-to-end, seq>=3 on the 397B measured 13.6% SLOWER with the simd
    layout even though the isolated microbench says it wins at the same N --
    so the default dispatch (simd=None) must fall back to thread-per-row
    above _EXPERT_SIMD_MAX_N pairs. Explicit simd=True still forces it (that
    is what the A/B harness uses)."""
    kw = dict(E=4, OUT=256, IN=4096, K=16384, d=8, G=64)
    V._KERNELS.clear()
    a = _rand_experts(**kw, N=V._EXPERT_SIMD_MAX_N)
    V._fused(*a)
    assert any("simd" in n for n in V._KERNELS)
    V._KERNELS.clear()
    b = _rand_experts(**kw, N=V._EXPERT_SIMD_MAX_N + 1)
    V._fused(*b)
    assert not any("simd" in n for n in V._KERNELS)
    V._KERNELS.clear()


def test_simd_is_the_default():
    """The d8 device-codebook path ships with the simd layout on; VQ_EXPERT_SIMD
    is only an A/B escape hatch."""
    assert V._EXPERT_SIMD is True


# --- d=2 code-load width (E14x, 2026-09-02) --------------------------------
# _SRC_FUSED_D2_U32 reads FOUR unpacked uint8 codes per uint32 load instead of
# four uchar loads, by reinterpreting the code row with mx.view. That is the
# d2 analogue of the d4 U8-VIEW dispatch and is worth 1.47-1.86x on the d2
# expert path (real 2.1bpw layer-1 codes, M3 Ultra) -- but only if it is
# EXACT, for the same reason as everything else in this file: the gemma d2
# rungs' published KL numbers were scored through the uchar kernel.
#
# The shapes are the ones shipped artifacts dispatch: the Flash-Next 2.1bpw
# uses d2/K256 for layers 0-1 (gate/up NGRP=40, down NGRP=10), and the gemma
# d2 rungs use d2 throughout.
D2_SHAPES = [
    (8,  640, 2560, 256, 2, 64),   # F-Next 2.1bpw gate/up  (NGRP = 40)
    (8, 2560,  640, 256, 2, 64),   # F-Next 2.1bpw down     (NGRP = 10)
    (4,  100, 2816, 256, 2, 64),   # ragged OUT, gemma-ish width
    (4,  128, 2048, 256, 2, 32),   # G = 32
]


@pytest.mark.parametrize("N", [1, 2, 5, 8, 10, 20])
@pytest.mark.parametrize("shape", D2_SHAPES)
def test_d2_u32_bit_exact(shape, N):
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, d2_u32=False))
    u32 = np.array(V._fused(*args, d2_u32=True))
    assert np.array_equal(base, u32)


@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("shape", D2_SHAPES)
def test_d2_simd_bit_exact(shape, N):
    """_SRC_FUSED_D2_SIMD is measured-and-not-dispatched (0.80-1.09x), but it
    is kept as the reference twin, so it must stay correct."""
    E, OUT, IN, K, d, G = shape
    if IN // G < 32:
        pytest.skip("simd layout needs NGRP >= 32")
    args = _rand_experts(E, OUT, IN, K, d, G, N)
    base = np.array(V._fused(*args, d2_u32=False, simd=False))
    _, eidx, codes, cb, sc = args
    x = args[0]
    NSUB = IN // d
    y, = V._get_kernel("vq_fused_d2_simd_t", V._SRC_FUSED_D2_SIMD)(
        inputs=[x, eidx, codes, cb, sc,
                mx.array([OUT, IN, d, G, N, K], dtype=mx.int32)],
        template=[("T", x.dtype), ("CT", codes.dtype),
                  ("MAX_K", K), ("MAX_TILE", 32 * (G // 2))],
        grid=(32, ((OUT + 31) // 32) * 32, N),
        threadgroup=(32, 32, 1),
        output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
    assert np.array_equal(base, np.array(y))


def test_d2_u32_is_the_default_and_dispatches():
    """Unpacked uint8 d2 codes must take the uint32 kernel by default."""
    assert V._D2_U32 is True
    V._KERNELS.clear()
    V._fused(*_rand_experts(E=4, OUT=256, IN=2560, K=256, d=2, G=64, N=8))
    assert any("d2_u32" in n for n in V._KERNELS)
    V._KERNELS.clear()


def test_ple_byte_codes_skips_the_identity_unpack():
    """At BITS=8 a 'packed' PLE row is one byte per code, so _unpack is the
    identity -- the fast path must be selected AND agree with it exactly."""
    rng = np.random.default_rng(0)
    rows, nsub, d, G = 512, 20, 8, 32
    cols = nsub * d
    codes = mx.array(rng.integers(0, 256, size=(rows, nsub), dtype=np.uint8))
    cb = mx.array(rng.normal(0, 1, size=(256, d)).astype(np.float16))
    sc = mx.array(rng.normal(0, 0.05, size=(rows, cols // G)).astype(np.float16))
    m = V.VQPLEEmbedding(codes, cb, sc, group_size=G, packed_nsub=nsub)
    assert m._byte_codes is True
    # rebuild the ORIGINAL extraction tables and run the old path
    bit0 = np.arange(nsub) * m._bits
    m._b0 = mx.array(bit0 // 8)
    m._sh = mx.array((bit0 % 8).astype(np.uint32))
    ids = mx.array(rng.integers(0, rows, size=16).astype(np.uint32))
    c = m._unpack(m.codes[ids])
    v = m.codebook[c.astype(mx.uint32)]
    old = (v.reshape(*ids.shape, -1)
           * mx.repeat(m.vq_scales[ids], G, axis=-1)).astype(mx.bfloat16)
    # compared with mx.array_equal: numpy has no bfloat16, so np.array() on
    # the bf16 output raises rather than converting.
    assert bool(mx.array_equal(old, m(ids)))

# --- register-buffering arc (2026-09-02) -----------------------------------

@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("shape", [s for s in D8_SHAPES if s[2] // s[5] >= 32])
def test_d8_regbuf_bit_exact(shape, N, monkeypatch):
    """The register-buffered packed-d8 kernel is bit-identical to the VQ_CODE
    one it was meant to replace. It is measured SLOWER and therefore OFF by
    default (see _D8_REGBUF), but it stays in-tree as the reproducible record
    of that negative -- so it must stay CORRECT, or the record is worthless.
    The funnel-shift window it reads must also never leave the pack block;
    that is what _d8_regbuf_ok guards and what a wrong answer here would
    expose."""
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N, packed=14)
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_D8_REGBUF", False)
    base = np.array(V._fused(*args, pack_bits=14, simd=True))
    assert any(isinstance(n, str) and "_d8_simd" in n
               and "_d8_simd_rb" not in n for n in V._KERNELS)
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_D8_REGBUF", True)
    rb = np.array(V._fused(*args, pack_bits=14, simd=True))
    assert any(isinstance(n, str) and "_d8_simd_rb" in n
               for n in V._KERNELS)
    V._KERNELS.clear()
    assert np.array_equal(base, rb)


def test_regbuf_is_off_by_default():
    """It measured 1.66x SLOWER than the kernel it replaces (real gate_proj,
    N=10: 106 vs 64 us). Nothing may dispatch it unless VQ_D8_REGBUF=1."""
    assert V._D8_REGBUF is False


@pytest.mark.parametrize("G", [64, 128, 256])
def test_regbuf_window_stays_inside_the_pack_block(G):
    """_d8_regbuf_ok's whole job: at every bit width the packer emits, the
    NW-word funnel window of the LAST phase must end at or before the block's
    last word. A window that ran past it would read the next row (or off the
    end of the tensor) on the last block of a row."""
    spg = G // 8
    nph = 32 // spg
    for bits in range(2, 17):
        sh0max = ((nph - 1) * spg * bits) & 31
        nw = (sh0max + spg * bits + 31) >> 5
        w0max = ((nph - 1) * spg * bits) >> 5
        assert w0max + nw - 1 <= bits - 1, (G, bits)


def test_regbuf_declines_geometries_it_cannot_serve():
    """Only SPG in {8,16,32} (G in {64,128,256}) is emitted; anything else
    must fall back to the VQ_CODE kernel rather than read wrong memory."""
    assert not V._d8_regbuf_ok(32)     # SPG = 4  -> 8 phases, not emitted
    assert not V._d8_regbuf_ok(16)     # SPG = 2
    assert not V._d8_regbuf_ok(512)    # SPG = 64 -> straddles two blocks


def test_packed_d8_simd_rows_per_threadgroup():
    """Swept on real Flash-Next expert tensors 2026-09-02: 8 rows/threadgroup
    beats the inherited 32 by 1.18-1.24x at every shape and N measured, and
    is bit-identical (same kernel source, different threadgroup shape). This
    pins the value so a future edit to the DENSE constant cannot silently
    drag the expert path back to 32."""
    assert V._EXPERT_ROWS_TG_D8_PACKED == 8
    assert V._EXPERT_ROWS_TG == 32


@pytest.mark.parametrize("rows", [4, 8, 32])
def test_rows_per_threadgroup_is_bit_neutral(rows, monkeypatch):
    """Changing rows/threadgroup must not move a single bit: the reduction is
    within one simdgroup and only the number of simdgroups sharing the x tile
    changes."""
    E, OUT, IN, K, d, G = 8, 1024, 4096, 16384, 8, 64
    args = _rand_experts(E, OUT, IN, K, d, G, 8, packed=14)
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_EXPERT_ROWS_TG_D8_PACKED", 32)
    ref = np.array(V._fused(*args, pack_bits=14, simd=True))
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_EXPERT_ROWS_TG_D8_PACKED", rows)
    got = np.array(V._fused(*args, pack_bits=14, simd=True))
    V._KERNELS.clear()
    assert np.array_equal(ref, got)


def test_simdsum_is_off_by_default():
    """Arc 4's simd_sum reduction is 1.16-1.20x on gate/up but re-associates
    the fp32 sum, so it fails the bit-identity gate the kernel arcs hold. It
    must stay opt-in until that gate is deliberately relaxed."""
    assert V._D8_SIMDSUM is False
    assert V._SRC_FUSED_PACKED_D8_SIMD_SS != V._SRC_FUSED_PACKED_D8_SIMD
    assert "simd_sum" in V._SRC_FUSED_PACKED_D8_SIMD_SS
    assert "simd_sum" not in V._SRC_FUSED_PACKED_D8_SIMD


@pytest.mark.parametrize("N", [1, 10, 20])
def test_simdsum_matches_within_one_ulp(N, monkeypatch):
    """It is NOT bit-identical -- that is the whole reason it is off -- but it
    must be a last-place-bit difference and nothing larger. Measured on real
    2.1bpw tensors: 0.05-0.16% of elements differ, every one by one half-ULP,
    and the two tie against an fp32 reference. This pins that bound so a
    future edit cannot quietly turn a rounding tie into a real error."""
    E, OUT, IN, K, d, G = 8, 512, 2560, 16384, 8, 64
    args = _rand_experts(E, OUT, IN, K, d, G, N, packed=14)
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_D8_SIMDSUM", False)
    ref = np.array(V._fused(*args, pack_bits=14, simd=True)).astype(np.float32)
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_D8_SIMDSUM", True)
    got = np.array(V._fused(*args, pack_bits=14, simd=True)).astype(np.float32)
    V._KERNELS.clear()
    # one half-ULP is 2**-10 of the magnitude; allow a hair over for values
    # sitting just below a binade boundary.
    tol = np.maximum(np.abs(ref), np.abs(got)) * (2.0 ** -9) + 1e-6
    assert np.all(np.abs(ref - got) <= tol)
    assert np.mean(ref != got) < 0.02


# --- specialized (template-free) kernels + plan memo, arc 5 (2026-09-02) ----

def test_spec_kernels_on_by_default():
    """Baking template values into the source and calling without `template=`
    saves ~7 us of HOST time per dispatch (~1 ms/token over 144 expert
    calls). It is bit-identical -- same generated Metal code -- so it is ON;
    VQ_SPEC_KERNELS=0 restores the template path for A/B."""
    assert V._SPEC_KERNELS is True


@pytest.mark.parametrize("shape", D8_SHAPES)
@pytest.mark.parametrize("N", [1, 8])
def test_spec_kernels_bit_identical(shape, N, monkeypatch):
    """The #define-specialized kernel must produce the same bits as the
    template-instantiated one on every dispatched geometry, simd and not."""
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N, packed=14)
    monkeypatch.setattr(V, "_SPEC_KERNELS", False)
    ref = np.array(V._fused(*args, pack_bits=14, simd=True))
    monkeypatch.setattr(V, "_SPEC_KERNELS", True)
    got = np.array(V._fused(*args, pack_bits=14, simd=True))
    assert np.array_equal(ref, got)


def test_spec_kernels_bit_identical_unpacked_d4_and_d2(monkeypatch):
    """The non-packed dispatch branches go through the same specialization."""
    for (E, OUT, IN, K, d, G) in ((4, 128, 512, 512, 4, 64),
                                  (4, 128, 512, 256, 2, 64)):
        args = _rand_experts(E, OUT, IN, K, d, G, 4)
        monkeypatch.setattr(V, "_SPEC_KERNELS", False)
        ref = np.array(V._fused(*args))
        monkeypatch.setattr(V, "_SPEC_KERNELS", True)
        got = np.array(V._fused(*args))
        assert np.array_equal(ref, got)


def test_plan_memo_respects_flag_flips(monkeypatch):
    """The plan memo keys on the module-level kernel flags, so a bench
    script's V._D8_SIMDSUM flip must reach the dispatcher on the NEXT call,
    not serve a stale plan."""
    E, OUT, IN, K, d, G = 4, 96, 4096, 4096, 8, 64
    args = _rand_experts(E, OUT, IN, K, d, G, 2, packed=14)
    V._KERNELS.clear()
    monkeypatch.setattr(V, "_D8_SIMDSUM", False)
    V._fused(*args, pack_bits=14, simd=True)
    assert not any(isinstance(n, str) and "_d8_simd_ss" in n
                   for n in V._KERNELS)
    monkeypatch.setattr(V, "_D8_SIMDSUM", True)
    V._fused(*args, pack_bits=14, simd=True)
    assert any(isinstance(n, str) and "_d8_simd_ss" in n for n in V._KERNELS)
    V._KERNELS.clear()


def test_plan_memo_is_invalidated_by_kernel_clear():
    """Plans live inside _KERNELS under ("plan", ...) keys precisely so the
    existing clear() used across this suite wipes them too."""
    E, OUT, IN, K, d, G = 4, 96, 4096, 4096, 8, 64
    args = _rand_experts(E, OUT, IN, K, d, G, 2, packed=14)
    V._fused(*args, pack_bits=14, simd=True)
    assert any(isinstance(n, tuple) and n and n[0] == "plan"
               for n in V._KERNELS)
    V._KERNELS.clear()
    assert not V._KERNELS


# --- device-x packed-d8 simd kernel, arc 5 (2026-09-02) ---------------------

def test_devx_on_by_default():
    """Reading x from device memory instead of the staged threadgroup tile is
    BIT-IDENTICAL (same half4 values, same float widening, same dot/fma
    order) and measured 1.19-1.23x at decode N on real L20 gate/up -- the
    staging (not the barriers) was ~half the dispatch. VQ_D8_DEVX=0 restores
    the staged kernel for A/B."""
    assert V._D8_DEVX is True


@pytest.mark.parametrize("shape", [s for s in D8_SHAPES if s[2] // s[5] >= 32])
@pytest.mark.parametrize("N", [1, 8])
def test_devx_bit_identical_to_staged(shape, N, monkeypatch):
    """devx vs the staged kernel, bit for bit, on every simd-eligible packed
    geometry -- including the ragged-OUT and NGRP-not-multiple-of-32 shapes
    where the tile bookkeeping differed most."""
    E, OUT, IN, K, d, G = shape
    args = _rand_experts(E, OUT, IN, K, d, G, N, packed=14)
    monkeypatch.setattr(V, "_D8_DEVX", False)
    staged = np.array(V._fused(*args, pack_bits=14, simd=True))
    monkeypatch.setattr(V, "_D8_DEVX", True)
    devx = np.array(V._fused(*args, pack_bits=14, simd=True))
    assert np.array_equal(staged, devx)
