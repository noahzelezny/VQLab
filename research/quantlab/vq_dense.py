"""Dense VQ modules — VQLinear and VQEmbedding.

WHY THESE EXIST. vq_switch.VQSwitchLinear is EXPERT-shaped: its __call__
takes routing indices and its kernels index a leading expert axis. A dense
model (gemma-4-e4b) has neither, so shipping a VQ e4b needs drop-ins for
plain nn.Linear and nn.Embedding.

THE EMBEDDING IS THE GOOD CASE. A [262144, 10752] table is 35.5% of e4b's
bytes, and an embedding lookup only ever needs the ROWS a batch touches —
so VQEmbedding gathers those code rows and decodes just those, never
materialising the table. Memory saving with no throughput cost.

THE LINEAR IS THE HONEST CASE. A matmul needs the whole weight, so
VQLinear decodes W per call: codebook[codes] -> [OUT, NSUB, D] ->
[OUT, IN], scaled group-wise. That trades compute for resident memory and
will be SLOWER than an 8-bit matmul; measure before believing any speed
claim. Correctness first — a fused dense kernel is a later optimisation,
and per E62 the fused work has to reproduce the scored numbers exactly
before it may replace this path.

Both mirror the fitter's contract exactly: scales are fp16 max-abs per
group of G along `in`, codes index a [K, D] fp16 codebook.
"""
import os

import mlx.core as mx
import mlx.nn as nn


def _decode(codes, codebook, scales, group_size, out_d, in_d):
    """codes [R, NSUB] -> dense [R, in_d] fp16, scaled group-wise."""
    w = codebook[codes.reshape(-1)].reshape(out_d, in_d)
    w = (w.reshape(out_d, in_d // group_size, group_size)
         * scales[..., None]).reshape(out_d, in_d)
    return w


def _unpack_rows(packed, nsub, bits):
    """mlx-native inverse of vq_pack.pack for a [R, WPR] uint32 slab.

    Runs on-GPU at gather time so packed EMBEDDING rows never round-trip
    through numpy. Same 32-codes-per-block layout as vq_pack.py; verified
    against vq_pack.unpack bit-exactly before first use.
    """
    R, wpr = packed.shape
    nblk = nsub // 32
    src = packed.reshape(R, nblk, bits).astype(mx.uint64)
    mask = mx.array(( 1 << bits) - 1, dtype=mx.uint64)
    outs = []
    for i in range(32):
        off = i * bits
        w, sh = divmod(off, 32)
        v = src[:, :, w] >> mx.array(sh, dtype=mx.uint64)
        if sh + bits > 32:
            v = v | (src[:, :, w + 1] << mx.array(32 - sh, dtype=mx.uint64))
        outs.append((v & mask).astype(mx.uint32))
    # outs[i] is code i of every block -> interleave back to row order
    return mx.stack(outs, axis=-1).reshape(R, nsub)


class VQLinear(nn.Module):
    """Drop-in for a bias-free nn.Linear whose weight is VQ-coded.

    codes may be unpacked ([OUT, NSUB] uint8/uint16) or packed
    ([OUT, WPR] uint32 of pack_bits-wide fields, vq_pack.py layout).
    """

    def __init__(self, codes, codebook, vq_scales, group_size=64,
                 pack_bits=0, in_features=None):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        self.pack_bits = pack_bits
        if pack_bits and in_features is None:
            raise ValueError("packed codes need explicit in_features")
        self._in_features = in_features
        self._k_expect = int(codebook.shape[0])
        self.freeze()

    @property
    def input_dims(self):
        if self.pack_bits:
            return self._in_features
        return self.codes.shape[-1] * self.codebook.shape[1]

    @property
    def output_dims(self):
        return self.codes.shape[0]

    def __call__(self, x):
        # Same sharding guard as VQSwitchLinear: a sliced codebook does not
        # error, it silently decodes garbage (exo PR #2268).
        if int(self.codebook.shape[0]) != self._k_expect:
            raise RuntimeError(
                f"VQ codebook was sharded: K={self.codebook.shape[0]}, "
                f"expected {self._k_expect}. The codebook is a SHARED lookup "
                f"table and must be REPLICATED across tensor-parallel ranks.")
        OUT = self.codes.shape[0]
        IN = (self._in_features if self.pack_bits
              else self.codes.shape[1] * self.codebook.shape[1])
        # A dense linear IS an expert layer with E=1 and every token routed
        # to expert 0 — so the small-N path rides the PRODUCTION-VALIDATED
        # fused kernels from vq_switch (E62: bit-identical, ~3x decode)
        # instead of decoding the whole weight per call, which measured
        # 11.5 tok/s vs the incumbent's 84 (2026-08-19). Prefill (large N)
        # amortises one full decode across the whole batch, which is faster
        # than the fused gather at that shape — same split vq_switch uses.
        if self.pack_bits:
            IN = self._in_features
        orig_shape = x.shape
        xf = x.reshape(-1, IN)
        N = xf.shape[0]
        if N <= 32:
            # DENSE fused kernel (2026-08-19): one simdgroup per output row
            # instead of one thread, no expert axis. Bit-identical to the
            # expert-kernel path below (the kernel replicates its float
            # ordering exactly — verified, and the E62 KL number reproduces).
            # SCOUT_VQ_DENSE_REF=1 keeps the old expert-shaped path callable
            # as the reference for A/B checks.
            if os.environ.get("SCOUT_VQ_DENSE_REF"):
                from mlx_lm.models.vq_switch import _fused
                eidx = mx.zeros((N,), dtype=mx.uint32)
                y = _fused(xf, eidx, self.codes[None], self.codebook,
                           self.vq_scales[None], pack_bits=self.pack_bits)
            else:
                from mlx_lm.models.vq_switch import _dense_fused
                y = _dense_fused(xf, self.codes, self.codebook,
                                 self.vq_scales, pack_bits=self.pack_bits,
                                 in_features=self._in_features)
            return y.astype(x.dtype).reshape(*orig_shape[:-1], OUT)
        codes = self.codes
        if self.pack_bits:
            codes = _unpack_rows(codes, IN // self.codebook.shape[1],
                                 self.pack_bits)
        w = _decode(codes, self.codebook.astype(mx.float16),
                    self.vq_scales, self.group_size, OUT, IN)
        return (xf @ w.T.astype(x.dtype)).reshape(*orig_shape[:-1], OUT)


class VQEmbedding(nn.Module):
    """Drop-in for nn.Embedding whose table is VQ-coded.

    Decodes ONLY the gathered rows — the whole point of putting VQ on a
    262k-row table. as_linear() is provided because gemma ties the output
    head to the input embedding in some configs; it decodes the full table
    and is deliberately NOT used by the PLE path.
    """

    def __init__(self, codes, codebook, vq_scales, group_size=64,
                 pack_bits=0, in_features=None):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        self.pack_bits = pack_bits
        if pack_bits and in_features is None:
            raise ValueError("packed codes need explicit in_features")
        self._in_features = in_features
        self._k_expect = int(codebook.shape[0])
        self.freeze()

    @property
    def num_embeddings(self):
        return self.codes.shape[0]

    @property
    def dims(self):
        return self.codes.shape[1] * self.codebook.shape[1]

    def __call__(self, ids):
        if int(self.codebook.shape[0]) != self._k_expect:
            raise RuntimeError("VQ codebook was sharded; must be replicated.")
        flat = ids.reshape(-1)
        rows = self.codes[flat]                       # [N, NSUB] or [N, WPR]
        sc = self.vq_scales[flat]                     # [N, in/G]
        D = self.codebook.shape[1]
        if self.pack_bits:
            NSUB = self._in_features // D
            rows = _unpack_rows(rows, NSUB, self.pack_bits)
        N, NSUB = rows.shape
        IN = NSUB * D
        w = self.codebook.astype(mx.float16)[rows.reshape(-1)].reshape(N, IN)
        w = (w.reshape(N, IN // self.group_size, self.group_size)
             * sc[..., None]).reshape(N, IN)
        return w.reshape(*ids.shape, IN)

    def as_linear(self, x):
        OUT = self.codes.shape[0]
        IN = self.dims
        w = _decode(self.codes, self.codebook.astype(mx.float16),
                    self.vq_scales, self.group_size, OUT, IN)
        return x @ w.T.astype(x.dtype)
