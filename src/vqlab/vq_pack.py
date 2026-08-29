#!/usr/bin/env python
"""Sub-byte bit-packing for VQ codes — the lever to sizes between the
byte-aligned points.

WHY THIS EXISTS (the M2 stored-vs-analytic trap): fitters print an analytic
bpw of log2(K)/d + 16/group, but codes are STORED in whole bytes — uint8 for
K<=256, uint16 above. So d4 K128 (7 bits) and d4 K256 (8 bits) both occupy
110.8 GiB, and d4 K2048 (11 bits) occupies 196.3 GiB rather than 142.8. Every
size between the byte points is unreachable until codes are packed.

FORMAT (v1, `vq_packed` in config):
  codes_packed  uint32  [E, OUT, WPR]      WPR = NSUB // 32 * BITS
  - BITS = ceil(log2(K)); the fitter's K is authoritative.
  - Packing is ROW-LOCAL: each [OUT] row packs its own NSUB codes, so the
    row base pointer stays (e*OUT + r) * WPR — kernels keep O(1) row math and
    no row ever depends on its neighbour's bits.
  - Codes are laid out in BLOCKS OF 32, each block occupying exactly BITS
    uint32 words. 32 codes * BITS bits = BITS * 32 bits exactly, so blocks
    are self-contained and the (word, shift) pattern repeats every block.
    Requires NSUB % 32 == 0 — true for every Qwen3.5 expert shape (NSUB is
    256 or 1024 at d=4).
  - Within a block, code i sits at bit offset i*BITS, little-endian, and may
    straddle two words; the straddle never escapes the block (code 31 ends at
    bit 32*BITS-1, i.e. word BITS-1).

The unpacked format stays valid forever — packing is an optional final pass,
and `vq_packed` absent means "read codes as before".
"""
import numpy as np

BLOCK = 32  # codes per packing block; chosen so blocks are word-aligned


def bits_for_k(k: int) -> int:
    """Code width the fitter's K implies. K must be a power of two."""
    b = int(k - 1).bit_length()
    assert 1 << b == k, f"K={k} is not a power of two"
    return b


def words_per_row(nsub: int, bits: int) -> int:
    # ceil: an unaligned tail block is stored zero-PADDED (pack pads, unpack
    # slices, kernels never read past n < NSUB). Identical to the old value
    # for every aligned tensor, so existing artifacts are byte-unchanged.
    return (nsub + BLOCK - 1) // BLOCK * bits


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """[E, OUT, NSUB] integer codes -> [E, OUT, WPR] uint32.

    Vectorised over everything except the 32 slot positions, so cost is 32
    OR-passes regardless of tensor size (a per-code Python loop would be
    ~10^9 iterations on a 397B gate_up tensor).
    """
    assert codes.ndim == 3, codes.shape
    e, out, nsub = codes.shape
    pad = (-nsub) % BLOCK
    if pad:
        import numpy as _np
        codes = _np.concatenate(
            [codes, _np.zeros((e, out, pad), dtype=codes.dtype)], axis=2)
        nsub += pad
    assert int(codes.max(initial=0)) < (1 << bits), "code exceeds bit width"
    nblk = nsub // BLOCK
    wpr = words_per_row(nsub, bits)

    src = codes.reshape(e, out, nblk, BLOCK).astype(np.uint64)
    packed = np.zeros((e, out, nblk, bits), dtype=np.uint64)
    for i in range(BLOCK):
        off = i * bits
        w, sh = divmod(off, 32)
        v = src[:, :, :, i] << np.uint64(sh)
        packed[:, :, :, w] |= v & np.uint64(0xFFFFFFFF)
        if sh + bits > 32:                      # straddles into the next word
            packed[:, :, :, w + 1] |= v >> np.uint64(32)
    return packed.astype(np.uint32).reshape(e, out, wpr)


def unpack(packed: np.ndarray, nsub: int, bits: int) -> np.ndarray:
    """Inverse of pack(); the reference the Metal reader must agree with."""
    e, out, wpr = packed.shape
    assert wpr == words_per_row(nsub, bits), (wpr, nsub, bits)
    true_nsub = nsub
    nsub = (nsub + BLOCK - 1) // BLOCK * BLOCK   # padded extent on disk
    nblk = nsub // BLOCK
    src = packed.reshape(e, out, nblk, bits).astype(np.uint64)
    mask = np.uint64((1 << bits) - 1)
    codes = np.zeros((e, out, nblk, BLOCK), dtype=np.uint64)
    for i in range(BLOCK):
        off = i * bits
        w, sh = divmod(off, 32)
        v = src[:, :, :, w] >> np.uint64(sh)
        if sh + bits > 32:
            v = v | (src[:, :, :, w + 1] << np.uint64(32 - sh))
        codes[:, :, :, i] = v & mask
    return codes.reshape(e, out, nsub)[:, :, :true_nsub].astype(
        np.uint16 if bits > 8 else np.uint8)


def packed_bytes(e: int, out: int, nsub: int, bits: int) -> int:
    return e * out * words_per_row(nsub, bits) * 4


if __name__ == "__main__":
    import sys
    # size table for the artifacts on the table
    # 57 layers x (gate_up [512,2048,4096], down [512,4096,1024])
    shapes = [(512, 2048, 4096)] * 57 + [(512, 4096, 1024)] * 57

    def expert_gib(bits):
        tot = sum(packed_bytes(e, o, i // 4, bits) for e, o, i in shapes)
        tot += sum(e * o * (i // 64) * 2 for e, o, i in shapes)   # fp16 scales
        return tot / 1024 ** 3

    # Everything outside the VQ'd expert region (attention, structure, the
    # promoted tail, routers, embeddings) is byte-identical across these
    # geometries. Calibrate it from the MEASURED C artifact so the totals
    # below are comparable to referee-era numbers rather than expert-only.
    C_MEASURED_GIB = 110.8
    rest = C_MEASURED_GIB - expert_gib(bits_for_k(256))
    print(f"non-expert region (from measured C): {rest:.1f} GiB\n")
    print(f"{'geometry':<18}{'bits':>5}{'bpw':>7}{'experts':>10}{'ARTIFACT':>10}")
    for name, k in (("d4 K128 (F)", 128), ("d4 K256 (C)", 256),
                    ("d4 K2048 (E)", 2048)):
        b = bits_for_k(k)
        ex = expert_gib(b)
        print(f"{name:<18}{b:>5}{b / 4 + 16 / 64:>7.2f}"
              f"{ex:>10.1f}{ex + rest:>10.1f}")
    sys.exit(0)
