"""Bit-exactness ratchet for the VQ code packer.

Packing changes REPRESENTATION, never values — so every test here is an
equality assertion, not a tolerance. If any of these go red, a packed
artifact would silently decode to different weights than the unpacked one it
was built from, and the referee number would move for a reason that has
nothing to do with quantization quality.
"""
import numpy as np
import pytest

from vq_pack import BLOCK, bits_for_k, pack, packed_bytes, unpack, words_per_row


@pytest.mark.parametrize("k", [128, 256, 2048, 4096])
def test_bits_for_k(k):
    assert 1 << bits_for_k(k) == k


def test_bits_for_k_rejects_non_power_of_two():
    with pytest.raises(AssertionError):
        bits_for_k(1000)


@pytest.mark.parametrize("bits", [7, 8, 11, 12])
@pytest.mark.parametrize("nsub", [32, 256, 1024])
def test_roundtrip_is_bit_exact(bits, nsub):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1 << bits, size=(3, 5, nsub), dtype=np.uint16)
    out = unpack(pack(codes, bits), nsub, bits)
    assert np.array_equal(out.astype(np.uint16), codes)


@pytest.mark.parametrize("bits", [7, 11])
def test_roundtrip_extremes(bits):
    """All-zero and all-max codes: catches sign/shift errors that random
    data can mask."""
    nsub = 256
    for fill in (0, (1 << bits) - 1):
        codes = np.full((2, 2, nsub), fill, dtype=np.uint16)
        out = unpack(pack(codes, bits), nsub, bits)
        assert np.array_equal(out.astype(np.uint16), codes)


@pytest.mark.parametrize("bits", [7, 11])
def test_rows_are_independent(bits):
    """Row-local packing is what lets kernels keep O(1) row math. Changing
    one row must not perturb any other row's bits."""
    nsub = 256
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 1 << bits, size=(2, 4, nsub), dtype=np.uint16)
    a = pack(codes, bits)
    codes2 = codes.copy()
    codes2[1, 2, :] = 0
    b = pack(codes2, bits)
    untouched = np.ones(a.shape[:2], bool)
    untouched[1, 2] = False
    assert np.array_equal(a[untouched], b[untouched])


@pytest.mark.parametrize("bits", [7, 11])
def test_no_bits_above_width_survive(bits):
    """A code at the max value must not bleed into its neighbour's field."""
    nsub = 64
    codes = np.zeros((1, 1, nsub), dtype=np.uint16)
    codes[0, 0, 5] = (1 << bits) - 1
    out = unpack(pack(codes, bits), nsub, bits)
    assert out[0, 0, 5] == (1 << bits) - 1
    assert out[0, 0, 4] == 0 and out[0, 0, 6] == 0


def test_pack_rejects_oversized_code():
    codes = np.array([[[0, 128]]], dtype=np.uint16)
    codes = np.repeat(codes, BLOCK // 2, axis=2)
    with pytest.raises(AssertionError):
        pack(codes, 7)


def test_pack_rejects_unaligned_nsub():
    with pytest.raises(AssertionError):
        words_per_row(100, 7)


@pytest.mark.parametrize("bits,nsub,expect_words", [(7, 32, 7), (11, 32, 11),
                                                    (8, 256, 64)])
def test_words_per_row(bits, nsub, expect_words):
    assert words_per_row(nsub, bits) == expect_words


def test_packed_size_matches_the_claim():
    """The whole point: 11-bit codes must actually cost 11/4 bpw, not 16/4."""
    e, out, nsub = 512, 2048, 1024
    got = packed_bytes(e, out, nsub, 11) * 8 / (e * out * nsub * 4)
    assert abs(got - 11 / 4) < 1e-9


def test_uint8_output_for_narrow_codes():
    codes = np.random.default_rng(2).integers(0, 128, (2, 2, 64), dtype=np.uint16)
    assert unpack(pack(codes, 7), 64, 7).dtype == np.uint8
