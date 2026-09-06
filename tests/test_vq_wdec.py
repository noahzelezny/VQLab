"""vq_wdec — the packed-decode fusion kernel's ship gate.

The kernel replaces _unpack_rows + _decode's gather + broadcast-scale with
one dispatch writing the fp16 weight tile directly. Its entire value rests on
being BIT-EXACT against the graph arm (E62: fused work must reproduce the
scored numbers exactly), so the tests compare uint16 BIT PATTERNS — never
array_equal, which reports NaN != NaN.

Test plan mapping (logs/reviews/dense-fusion-design.md):
  T1 extraction equivalence (host-only, no GPU)
  T2 kernel w vs graph w on both shipped geometries
  T3 both _decode_matmul tile arms
  T4 adversarial synthetics (subnormals, inf/nan scales)
  T5 _kernel_sig binding (also import-time assert in vq_switch)
  T6 straddle sweep bits 2..16 on a tail-padded NSUB
  T10 gate fallthrough
"""
import os
import pathlib
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "src" / "vqlab"))
import vq_dense  # noqa: E402
import vq_pack  # noqa: E402
import vq_switch as V  # noqa: E402


def _layer(OUT, IN, K, D, G, bits, seed=0, codebook=None, scales=None):
    """A packed dense layer at the given geometry; returns mlx tensors."""
    rng = np.random.default_rng(seed)
    NSUB = IN // D
    codes = rng.integers(0, K, size=(OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    if codebook is None:
        codebook = rng.normal(0, 1, size=(K, D)).astype(np.float16)
    if scales is None:
        scales = rng.normal(0, 0.05, size=(OUT, IN // G)).astype(np.float16)
    packed = mx.array(vq_pack.pack(codes[None], bits)[0])
    return (packed, mx.array(codes), mx.array(codebook), mx.array(scales))


def _graph_w(codes_unpacked, codebook, scales, G, OUT, IN):
    cbk = codebook.astype(mx.float16)
    w = vq_dense._decode(codes_unpacked, cbk, scales, G, OUT, IN)
    mx.eval(w)
    return np.array(w, copy=True)


def _kernel_w(packed, codebook, scales, OUT, IN, G, bits, r0=0, rows=None):
    cbk = codebook.astype(mx.float16)
    w = V.wdec_decode(packed, cbk, scales, rows or OUT, IN, G, bits, r0=r0)
    mx.eval(w)
    return np.array(w, copy=True)


def _bits_equal(a, b):
    return np.array_equal(a.view(np.uint16), b.view(np.uint16))


# --------------------------------------------------------------------------- #
# T1 — extraction formulas agree for every (lane, bits), host-only
# --------------------------------------------------------------------------- #

def _py_extract(words, j, bits):
    """_unpack_rows' formula (vq_dense.py)."""
    off = (j % 32) * bits
    w_idx = (j // 32) * bits + off // 32
    sh = off % 32
    v = words[w_idx] >> sh
    if sh + bits > 32:
        v |= (words[w_idx + 1] << (32 - sh)) & 0xFFFFFFFF
    return v & ((1 << bits) - 1)


def _metal_extract(words, j, bits):
    """VQ_CODE's formula (vq_switch._PACK_FETCH), transliterated."""
    OFF = (j & 31) * bits
    W = ((j >> 5) * bits) + (OFF >> 5)
    SH = OFF & 31
    v = words[W] >> SH
    if SH + bits > 32:
        v |= (words[W + 1] << (32 - SH)) & 0xFFFFFFFF
    return v & ((1 << bits) - 1)


def test_t1_extraction_equivalence_all_lanes_all_bits():
    rng = np.random.default_rng(7)
    for bits in range(2, 17):
        words = [int(x) for x in
                 rng.integers(0, 2**32, size=2 * bits, dtype=np.uint64)]
        for j in range(64):  # two full pack blocks
            assert _py_extract(words, j, bits) == \
                _metal_extract(words, j, bits), (bits, j)


# --------------------------------------------------------------------------- #
# T5 — signature binding
# --------------------------------------------------------------------------- #

def test_t5_kernel_sig_binds_wdec_buffers():
    for name in ("vq_wdec_packed9_d2", "vq_wdec_packed12_d4"):
        assert V._kernel_sig(name) == (
            ["codes", "codebook", "scales", "dims"], ["w"])


# --------------------------------------------------------------------------- #
# T2 — kernel w vs graph w, both shipped geometries
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("OUT,IN,K,D,G,bits", [
    (256, 1024, 512, 2, 64, 9),     # 4.8bpw class: K512/bits9/d2
    (256, 1024, 4096, 4, 64, 12),   # 3.9bpw class: K4096/bits12/d4
    (96, 2560, 512, 2, 64, 9),      # non-power-of-two rows, 27B-ish IN/G
])
def test_t2_bit_identical_to_graph_arm(OUT, IN, K, D, G, bits):
    packed, codes, cbk, sc = _layer(OUT, IN, K, D, G, bits, seed=OUT)
    ref = _graph_w(codes, cbk, sc, G, OUT, IN)
    got = _kernel_w(packed, cbk, sc, OUT, IN, G, bits)
    assert _bits_equal(ref, got)


# --------------------------------------------------------------------------- #
# T3 — the row-sliced launch (the tiled arm's shape) matches its graph slice
# --------------------------------------------------------------------------- #

def test_t3_row_slices_match_graph_slices():
    OUT, IN, K, D, G, bits = 200, 1024, 512, 2, 64, 9
    packed, codes, cbk, sc = _layer(OUT, IN, K, D, G, bits, seed=3)
    full = _graph_w(codes, cbk, sc, G, OUT, IN)
    for r0, r1 in ((0, 64), (64, 128), (128, 200)):
        got = _kernel_w(packed, cbk, sc, OUT, IN, G, bits,
                        r0=r0, rows=r1 - r0)
        assert _bits_equal(full[r0:r1], got), (r0, r1)


# --------------------------------------------------------------------------- #
# T4 — adversarial values: subnormal codebook, inf/nan scales
# --------------------------------------------------------------------------- #

def test_t4_subnormal_codebook_and_nonfinite_scales():
    OUT, IN, K, D, G, bits = 64, 512, 512, 2, 64, 9
    rng = np.random.default_rng(11)
    codebook = (rng.normal(0, 1, size=(K, D)).astype(np.float16)
                * np.float16(6e-8))            # subnormal magnitudes
    scales = rng.normal(0, 0.05, size=(OUT, IN // G)).astype(np.float16)
    scales[0, 0] = np.float16("inf")
    scales[1, 1] = np.float16("nan")
    scales[2, 2] = np.float16("-inf")
    packed, codes, cbk, sc = _layer(OUT, IN, K, D, G, bits, seed=11,
                                    codebook=codebook, scales=scales)
    ref = _graph_w(codes, cbk, sc, G, OUT, IN)
    got = _kernel_w(packed, cbk, sc, OUT, IN, G, bits)
    assert _bits_equal(ref, got)    # bit compare is exactly why: NaN == NaN


# --------------------------------------------------------------------------- #
# T6 — straddle sweep: every bits width against the graph unpack
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bits", list(range(2, 17)))
def test_t6_all_bit_widths_match(bits):
    K = min(512, 1 << bits)
    OUT, IN, D, G = 32, 256, 2, 64
    packed, codes, cbk, sc = _layer(OUT, IN, K, D, G, bits, seed=bits)
    ref = _graph_w(codes, cbk, sc, G, OUT, IN)
    got = _kernel_w(packed, cbk, sc, OUT, IN, G, bits)
    assert _bits_equal(ref, got)


# --------------------------------------------------------------------------- #
# T10 — gate fallthrough
# --------------------------------------------------------------------------- #

def test_t10_gate_rejects_ineligible_shapes():
    assert not V.wdec_fits(8, 1024, 64, 9)      # D=8 unsupported
    assert not V.wdec_fits(2, 1000, 64, 9)      # NSUB % 32 != 0 (PLE class)
    assert not V.wdec_fits(2, 1024, 64, 0)      # unpacked stays bit-frozen
    assert not V.wdec_fits(2, 1024, 63, 9)      # G % D != 0
    assert V.wdec_fits(2, 1024, 64, 9)
    assert V.wdec_fits(4, 1024, 64, 12)


def test_t10_env_escape_hatch_disables_gate(monkeypatch):
    monkeypatch.setattr(V, "_WDEC_FUSE", False)
    assert not V.wdec_fits(2, 1024, 64, 9)


# --------------------------------------------------------------------------- #
# End-to-end: _decode_matmul via VQLinear agrees across FUSE arms
# --------------------------------------------------------------------------- #

def test_e2e_decode_matmul_fuse_arms_bit_identical(monkeypatch):
    OUT, IN, K, D, G, bits = 128, 1024, 512, 2, 64, 9
    packed, codes, cbk, sc = _layer(OUT, IN, K, D, G, bits, seed=42)
    rng = np.random.default_rng(42)
    xf = mx.array(rng.normal(0, 1, size=(160, IN)).astype(np.float16))

    def run():
        y = vq_dense._decode_matmul(xf, packed, cbk, sc, G, OUT, IN, bits)
        mx.eval(y)
        return np.array(y, copy=True)

    monkeypatch.setattr(V, "_WDEC_FUSE", False)
    ref = run()
    monkeypatch.setattr(V, "_WDEC_FUSE", True)
    got = run()
    assert _bits_equal(ref, got)
