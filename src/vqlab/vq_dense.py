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


# Rows below which the fused kernel beats decode-the-weight-then-GEMM.
#
# This is TWO numbers, because the decode path's cost depends entirely on
# whether the codes are packed. Measured 2026-09-01 on the 27B mlp shape
# (OUT 17408, IN 5120, d=2), ms per call:
#
#   packed (K512/bits9)   N      1     8    32    68   128   256
#                         fused  0.6   1.4   4.2   8.3  14.9  29.2
#                         decode 9.6   9.5   9.7  10.2  10.2  11.3   -> cross ~100
#
#   unpacked (K256)       N      1     8    16    32    68    96
#                         fused  0.5   1.2   2.2   3.9   7.9  11.1
#                         decode 1.7   1.5   1.7   1.7   2.3   2.3   -> cross ~12
#
# Packed decode is ~9.5 ms flat because _unpack_rows dominates it; unpacked
# decode is 1.65 ms flat. So the fused path stays ahead six times longer when
# codes are packed.
#
# Memory is the other reason, and for packed artifacts it is the bigger one.
# The decode path builds a 170 MB fp16 weight per layer, plus the unpack
# temporaries, and mlx builds the whole forward before evaluating -- so a
# packed 27B rung peaked at 30 GB for a 15.5 GB model, i.e. it needed twice
# its own weight in RAM and could OOM a machine its model card says it fits.
# Routing short prompts through the fused path, which materialises nothing,
# took that peak to 16.1 GB.
#
# Prompts longer than these thresholds still take the decode path and still
# pay the transient. Bounding it there needs chunked decode with forced
# evaluation and is NOT done -- see docs/DENSE-VQ-DECODE.md.
_DENSE_FUSED_MAX_N_PACKED = int(os.environ.get("VQ_DENSE_FUSED_MAX_N", 96))
_DENSE_FUSED_MAX_N_PLAIN = int(os.environ.get("VQ_DENSE_FUSED_MAX_N_PLAIN", 12))


def _resolve_kernel(name):
    """Find the fused kernel, most-local copy first.

    Order: (1) this module's own globals — the BUNDLED case, where model.py
    concatenates vq_switch + vq_dense and the kernel is already here;
    (2) a sibling vq_switch module — the PACKAGE / same-directory case, which
    works in a completely stock venv; (3) mlx_lm.models.vq_switch — a
    VQ-PATCHED install only. The old order tried (3) directly after (1) and
    therefore worked on every lab machine and failed on a stock one — found
    the first time the selftest ran in a genuinely fresh venv.
    """
    f = globals().get(name)
    if f is not None:
        return f
    import importlib
    for mod in ("vq_switch", "vqlab.vq_switch", "mlx_lm.models.vq_switch"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(
        f"no source for {name}: not bundled alongside this module, no sibling "
        f"vq_switch importable, and mlx_lm is not VQ-patched")


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

    MEMORY (2026-09-01). This is called on the PREFILL path, once per VQ
    linear, and mlx builds the whole forward before evaluating it -- so every
    layer's temporaries are alive at once and this function sets the peak.
    The original cast the slab to uint64 (8 bytes per element) and returned
    uint32, which cost ~630 MB of transient per 27B mlp layer against the
    unpacked path's 178 MB. Across 64 layers that is the difference between a
    15.9 GB peak and a 30.0 GB one: the 4.8bpw artifact needed twice its own
    weight in RAM, which defeats the entire point of packing and can OOM a
    machine the model card says it fits on.

    Two changes, both bit-preserving:
      - stay in uint32. bits <= 16 always, so a field either lies inside one
        word or straddles two, and both cases are exact in 32-bit as long as
        the shift that would be >= 32 is never evaluated (see below).
      - return the narrowest dtype the codebook needs, not always uint32.

    The `sh + bits > 32` branch is why the uint64 was there: `32 - sh` is a
    32-bit shift by 32 when sh == 0, which is undefined. sh == 0 also means
    the field cannot straddle (bits <= 32), so that branch is unreachable
    then, and it is a Python-level `if` on compile-time constants here rather
    than a data-dependent select -- the bad shift is never emitted at all.
    """
    R, wpr = packed.shape
    nblk = nsub // 32
    src = packed.reshape(R, nblk, bits)
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    outs = []
    for i in range(32):
        off = i * bits
        w, sh = divmod(off, 32)
        v = src[:, :, w] >> mx.array(sh, dtype=mx.uint32) if sh else src[:, :, w]
        if sh + bits > 32:
            # sh > 0 here, so 32 - sh is in [1, 31] and the shift is defined.
            v = v | (src[:, :, w + 1] << mx.array(32 - sh, dtype=mx.uint32))
        outs.append(v & mask)
    ct = mx.uint8 if bits <= 8 else mx.uint16
    # outs[i] is code i of every block -> interleave back to row order
    return mx.stack(outs, axis=-1).reshape(R, nsub).astype(ct)


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
        # d=2 is the only geometry with a dense fused kernel. The expert
        # kernel (_fused, E=1) was tried as the d!=2 small-N path and DIED AT
        # KERNEL LOAD on the first real dense model: at 27B mlp shapes
        # (IN 17408) its threadgroup allocation is 36864 bytes vs Metal's
        # 32768 cap — shapes the 397B's experts never produce. Caught by
        # III.10 smoke-gen (E95). Until a dense kernel exists for d>2,
        # small-N takes the decode path: bit-exact, just slow at decode
        # (~11.5 tok/s measured 08-19) — fine for gates and prefill-shaped
        # scoring, NOT a shipping configuration.
        # THREADGROUP CEILING (2026-08-21, E124). The d=2 dense kernel ALSO
        # dies at kernel load on 27B mlp shapes: NSUB = IN/d = 8704 needs
        # 35840 B of threadgroup memory against Metal's 32768 cap. So neither
        # d=2 NOR d>2 has a usable fused dense path at this model's widths —
        # the 397B's expert shapes simply never produce them. Measured, not
        # inferred: the failure is a RuntimeError naming both numbers.
        # Falling back to decode keeps the artifact CORRECT (bit-exact, the
        # same path that produced its scores) at decode speed. This is not a
        # performance fix and must not be read as one; a dense kernel that
        # fits the cap is the real fix and does not exist yet.
        # THREADGROUP BUDGET — must count the CODEBOOK too (E134 addendum,
        # 2026-08-22). Both dense d2 kernels cache `threadgroup half2
        # cb[MAX_K]` AND `threadgroup half2 xs[MAX_NSUB]`, i.e. 4 bytes per
        # entry each, so the requirement is (K + NSUB) * 4 + overhead. The
        # original guard counted only NSUB and would have reported "fits" for
        # a large-K artifact that then died at KERNEL LOAD — the same failure
        # the guard exists to prevent, and the same defect E134 fixed on the
        # MoE side, where a K>=4096 d4 artifact scored normally and could not
        # generate a token. Does not bite at any rung on disk today
        # (d2/K4096 at NSUB 2560 needs 26,624 B) and would have bitten at
        # d2/K8192.
        # NSUB TILING (2026-09-01). The budget above is the UNTILED
        # kernel's, and it grows with layer width: at 27B mlp shapes it
        # reports "too big" and every forward then decoded the whole weight,
        # which measured 0.43 tok/s against stock's 16.7. The tiled kernels
        # stage one block span instead of the whole x row, so their budget is
        # (K + 32*(G/2)) * 4 and does not depend on width at all. Ask
        # vq_switch whether EITHER kernel can serve the shape; it picks.
        # Bit-exactness against the untiled path is asserted in
        # tests/test_vq_dense_tiled.py. Note this gate is DECODE ONLY
        # (N <= 32): scoring and prefill are large-N and take the
        # decode-to-dense path below, which is untouched, so no published
        # quality number depends on anything here.
        _kk = int(self.codebook.shape[0])
        _G = IN // int(self.vq_scales.shape[-1])
        _fits_tg = _resolve_kernel("dense_fits")(
            _kk, IN, _G, int(self.codebook.shape[1]))
        # d is NOT re-checked here: dense_fits owns which geometries
        # have a kernel (d=2 tiled/untiled, d=4 device-codebook), and
        # duplicating that as `== 2` here is what kept the d4 rung on
        # the decode path after its kernel existed.
        _maxn = (_DENSE_FUSED_MAX_N_PACKED if self.pack_bits
                 else _DENSE_FUSED_MAX_N_PLAIN)
        if N <= _maxn and _fits_tg:
            # DENSE fused kernel (2026-08-19): one simdgroup per output row
            # instead of one thread, no expert axis. Bit-identical to the
            # expert-kernel path below (the kernel replicates its float
            # ordering exactly — verified, and the E62 KL number reproduces).
            # VQ_DENSE_REF=1 keeps the old expert-shaped path callable
            # as the reference for A/B checks.
            if os.environ.get("VQ_DENSE_REF") and self.pack_bits == 0 \
                    and int(self.codebook.shape[1]) == 2:
                _fused = _resolve_kernel("_fused")
                eidx = mx.zeros((N,), dtype=mx.uint32)
                y = _fused(xf, eidx, self.codes[None], self.codebook,
                           self.vq_scales[None], pack_bits=self.pack_bits)
            else:
                _dense_fused = _resolve_kernel("_dense_fused")
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
