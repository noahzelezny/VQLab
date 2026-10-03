"""The one packing decision for MoE expert codes, shared by `vqlab pack`
(pack.py, after the fact) and `vqlab fit-moe --pack` (as each shard
is written). Both call `pack_codes`, so a fit packed in flight is byte for
byte the fit packed afterwards (tests/test_fit_pack.py pins that).

Rules (each one paid for; see pack.py for the history):
  * bits % 8 == 0 (K256, K65536): packing saves zero bytes and costs the
    packed kernel's bit extraction (37% decode tax measured), so the codes
    stay byte-aligned;
  * NSUB % 32 != 0: copied through unpacked unless `pack_unaligned`
    (GPU acceptance of the padded tail block is still pending);
  * every packed tensor must round-trip through vq_pack.unpack before it
    leaves the process.
"""
from __future__ import annotations

import numpy as np

import vq_pack


def pack_bits_for(k: int, nsub: int, pack_unaligned: bool = False) -> int:
    """Bit width a [.., NSUB] codes tensor at codebook size K packs to, or 0
    when it stays unpacked (byte-aligned width, or unaligned NSUB)."""
    bits = vq_pack.bits_for_k(int(k))
    if nsub % vq_pack.BLOCK and not pack_unaligned:
        return 0
    if bits % 8 == 0:
        return 0
    return bits


def pack_codes(mod: str, codes, k: int, pack_unaligned: bool = False):
    """(codes array, pack_bits). `codes` is an [E, OUT, NSUB] integer array
    (mx or numpy); the result is uint32 words (mx) when it packs, else the
    input unchanged with pack_bits 0. Raises SystemExit if the packed
    tensor does not round-trip."""
    import mlx.core as mx
    nsub = codes.shape[2]
    bits = pack_bits_for(k, nsub, pack_unaligned)
    if not bits:
        return codes, 0
    c = np.array(codes, copy=False).astype(np.uint16)
    packed = vq_pack.pack(c, bits)
    # a silent packing error decodes to plausible garbage, not an error
    back = vq_pack.unpack(packed, nsub, bits)
    if not np.array_equal(back.astype(np.uint16), c):
        raise SystemExit(f"FATAL: {mod} did not round-trip through the packer")
    return mx.array(packed), bits
