"""vq-skipzero STAGE 2 (EXPERIMENTAL): keep the compact form RESIDENT.

Copied VERBATIM into a resident artifact next to its model.py (by
`vqlab sz-resident`). Stage 1 (skipzero_load.py) re-expands dead rows at load;
stage 2 keeps, per packed VQ switch module:
    codes      [NLIVE, W]  uint32   live rows only (row-major over (e, out))
    vq_scales  [NLIVE, G]  fp16
    codebook   [K, 4]      fp16     unchanged
    row_table  [E, OUT]    int32    compact row of (e, out), -1 = dead
and dead rows produce EXACTLY 0 without reading a code byte.

It does NOT edit the shipped runtime. `install(ns)` takes the namespace of the
artifact's model.py (the repo-vintage vq_switch runtime + arch shim) and
builds NEW kernels by patching the source text of exactly two kernels:
    decode : _SRC_FUSED_PACKED_D4_WALK  -> sz_fused_packed{B}_d4_walk
    prefill: _SRC_GEMMSEG2 (d4, packed) -> sz_gemmseg2_packed{B}_d4...
Every patch is an exact-text replacement asserted to match once, so a runtime
vintage whose kernel text differs REFUSES instead of guessing. Every other
geometry / flag combination raises NotImplementedError.

Dead-row arithmetic is chosen to be byte-identical with the expanded
(stage-1) path, not just "zero":
  * decode : expanded acc = fma(+0 scale, gacc, acc) from acc = +0 stays +0
             for finite x, so the compact kernel writes (T)0.0f.
  * prefill: the dead row's wtT column is filled with (half)(0.0f * cb[0].c),
             i.e. the exact values the expanded path stages (code 0, scale
             0, signed zeros included); only the code/scale READS are gone.
"""
from __future__ import annotations

FORMAT = "vq-skipzero-resident"
VERSION = 1


# ------------------------------------------------------------------ numpy
def row_table_np(rowmask, E, OUT):
    """bit-packed live mask [E, ceil(OUT/8)] -> int32 [E, OUT], -1 = dead."""
    import numpy as np
    bits = np.unpackbits(np.asarray(rowmask, np.uint8), axis=-1, bitorder="little")
    live = bits[:, :OUT].astype(bool).reshape(-1)
    tbl = np.full(E * OUT, -1, np.int32)
    tbl[live] = np.arange(int(live.sum()), dtype=np.int32)
    return tbl.reshape(E, OUT)


def row_table_mx(shape, rowmask):
    import mlx.core as mx
    shape = [int(v) for v in (shape.tolist() if hasattr(shape, "tolist") else shape)]
    E, OUT = shape[0], shape[1]
    rm = rowmask.astype(mx.uint32)
    bits = (rm[..., None] >> mx.arange(8, dtype=mx.uint32)) & 1
    live = bits.reshape(E, -1)[:, :OUT].reshape(-1).astype(mx.int32)
    pos = mx.cumsum(live) - 1
    return mx.where(live > 0, pos, mx.array(-1, mx.int32)).astype(mx.int32).reshape(E, OUT)


def resident_weights(weights):
    """sz quadruples -> {p}.codes / {p}.vq_scales COMPACT + {p}.row_table.
    Idempotent (no sz_ keys -> unchanged). Dict or list of pairs."""
    as_list = not isinstance(weights, dict)
    d = dict(weights) if as_list else weights
    pre = sorted({k[: -len(".sz_rowmask")] for k in d if k.endswith(".sz_rowmask")})
    if not pre:
        return weights
    kill = {p + "." + s for p in pre for s in ("sz_shape", "sz_rowmask", "sz_codes", "sz_scales")}
    out = {k: v for k, v in d.items() if k not in kill}
    for p in pre:
        out[p + ".codes"] = d[p + ".sz_codes"]
        out[p + ".vq_scales"] = d[p + ".sz_scales"]
        out[p + ".row_table"] = row_table_mx(d[p + ".sz_shape"], d[p + ".sz_rowmask"])
    return list(out.items()) if as_list else out


# ------------------------------------------------------- kernel source forks
def _sub1(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"sz-resident: kernel patch '{what}' matched {n}x (need 1): "
                           "this runtime vintage's kernel text differs; refusing")
    return src.replace(old, new)


def patch_decode_src(src):
    """_SRC_FUSED_PACKED_D4_WALK -> compact-row variant (extra input rowtbl)."""
    return _sub1(src,
        "    const device uint* crow = codes + (size_t)e * OUT * WPR + (size_t)r * WPR;\n"
        "    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;\n",
        "    const int sz_lr = rowtbl[(size_t)e * OUT + r];\n"
        "    if (sz_lr < 0) { y[(size_t)t * OUT + r] = static_cast<T>(0.0f); return; }\n"
        "    const device uint* crow = codes + (size_t)sz_lr * WPR;\n"
        "    const device half* srow = scales + (size_t)sz_lr * NGRP;\n",
        "decode row pointers")


_GS_BASE_OLD = """#if BITS == 0
    const device CT* wrow_base = codes
        + (size_t)e * OUT * WPR + (size_t)(o0 + wr) * WPR;
    #define VQ_FETCH(j) ((uint)wrow_codes[j])
#else
    const device uint* wrow_base = codes
        + (size_t)e * OUT * WPR + (size_t)(o0 + wr) * WPR;
    #define VQ_FETCH(j) VQ_CODE(wrow_codes, j)
#endif
    const device half* srow_base = scales
        + (size_t)e * OUT * NGRP + (size_t)(o0 + wr) * NGRP;
"""
_GS_BASE_NEW = """#if BITS == 0 || D_BAKE != 4
#error "sz-resident gemmseg: packed d4 only"
#endif
    #define VQ_FETCH(j) VQ_CODE(wrow_codes, j)
    // compact row of each output row this thread decodes (-1 = dead / OOB)
    int sz_lr[1 + OT2];
    for (int ob = 0; ob < 1 + OT2; ++ob) {
        const int orow = o0 + ob * 32 + wr;
        sz_lr[ob] = (orow < OUT) ? rowtbl[(size_t)e * OUT + orow] : -1;
    }
"""
_GS_LOOP_OLD = """#if BITS == 0
        const device CT* wrow_codes = wrow_base + (size_t)ob * 32 * WPR;
#else
        const device uint* wrow_codes = wrow_base + (size_t)ob * 32 * WPR;
#endif
        const device half* srow_w = srow_base + (size_t)ob * 32 * NGRP;
        if (oo + wr < OUT) {
"""
_GS_LOOP_NEW = """        const int sz_l = sz_lr[ob];
        const device uint* wrow_codes = codes + (size_t)max(sz_l, 0) * WPR;
        const device half* srow_w = scales + (size_t)max(sz_l, 0) * NGRP;
        if (sz_l >= 0) {
"""
_GS_ELSE_OLD = """        } else {
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                for (int u = 0; u < D_BAKE; ++u)
                    wtT[q * D_BAKE + u][wr] = (half)0;
            }
        }
"""
_GS_ELSE_NEW = """        } else if (oo + wr < OUT) {
            // DEAD row: stage exactly what the expanded path stages for
            // code 0 / scale +0 (signed zeros included); no code/scale reads.
            const float s = 0.0f;
            for (int q = q0; q < q0 + SPG / 4; ++q) {
                const half4 v = cb[0];
                wtT[q * 4][wr]     = (half)(s * (float)v.x);
                wtT[q * 4 + 1][wr] = (half)(s * (float)v.y);
                wtT[q * 4 + 2][wr] = (half)(s * (float)v.z);
                wtT[q * 4 + 3][wr] = (half)(s * (float)v.w);
            }
""" + _GS_ELSE_OLD


def patch_gemmseg_src(src):
    s = _sub1(src, _GS_BASE_OLD, _GS_BASE_NEW, "gemmseg row bases")
    s = _sub1(s, _GS_LOOP_OLD, _GS_LOOP_NEW, "gemmseg per-ob row pointers")
    s = _sub1(s, _GS_ELSE_OLD, _GS_ELSE_NEW, "gemmseg zero-tile branch")
    for bad in ("wrow_base", "srow_base"):
        if bad in s:
            raise RuntimeError(f"sz-resident: '{bad}' survived the gemmseg patch")
    return s


DECODE_SIG = (["x", "eidx", "codes", "codebook", "scales", "dims", "rowtbl"], ["y"])
GEMMSEG_SIG = (["codes", "codebook", "scales", "xsrc", "srcrows", "tmeta", "dims",
                "rowtbl"], ["y"])


# ----------------------------------------------------------------- runtime
def install(ns):
    """Build SZSwitchLinear against a vq_switch runtime namespace `ns`
    (the artifact model.py's globals(), or vars() of runtime/vq_switch.py).
    Returns the class; also stores it as ns['SZSwitchLinear']."""
    if "SZSwitchLinear" in ns:
        return ns["SZSwitchLinear"]
    mx, np = ns["mx"], ns["np"]
    VQSwitchLinear = ns["VQSwitchLinear"]
    need = ("_SRC_FUSED_PACKED_D4_WALK", "_SRC_GEMMSEG2", "_PACK_FETCH", "_dims_array",
            "_d4_tg_fits", "_memo_get", "_memo_put", "_idx_np", "gemmseg_fits",
            "_metal_type_name", "VQ_FUSED_MAX_N")
    miss = [n for n in need if n not in ns]
    if miss:
        raise NotImplementedError(
            f"sz-resident: this runtime vintage lacks {miss} (older vintages, e.g. "
            "the 397B's vq_fused_packed8 / vq_gemmseg2_u8_d4, need their own fork)")
    dec_src = patch_decode_src(ns["_SRC_FUSED_PACKED_D4_WALK"])
    gs_src = patch_gemmseg_src(ns["_SRC_GEMMSEG2"])
    cache = {}

    def kernel(name, src, sig, template):
        parts = []
        for tname, tval in template:
            if isinstance(tval, bool):
                parts.append((tname, "true" if tval else "false"))
            elif isinstance(tval, int):
                parts.append((tname, str(tval)))
            else:
                mt = ns["_metal_type_name"](tval)
                if mt is None:
                    raise NotImplementedError(f"sz-resident: no Metal spelling for {tval}")
                parts.append((tname, mt))
        key = name + "|" + "|".join(f"{n}={v}" for n, v in parts)
        k = cache.get(key)
        if k is None:
            hdr = "".join(f"#define {n} {v}\n" for n, v in parts)
            k = mx.fast.metal_kernel(
                name=name + "_s" + "_".join(v.replace(" ", "") for _, v in parts),
                input_names=sig[0], output_names=sig[1], source=src, header=hdr)
            cache[key] = k
        return k

    def sz_fused(x, eidx, codes, codebook, scales, rowtbl, pack_bits, OUT, IN):
        # mirrors _fused_resolve's packed-d4 WALK branch only
        if not (pack_bits and codebook.shape[1] == 4 and codes.dtype == mx.uint32):
            raise NotImplementedError("sz-resident decode: packed d4 only")
        if not ns.get("_D4_WALK", False):
            raise NotImplementedError("sz-resident decode: needs VQ_D4_WALK=1")
        N = x.shape[0]
        K, D = codebook.shape
        NSUB = IN // D
        G = IN // scales.shape[1]
        if not ns["_d4_tg_fits"](K, NSUB):
            raise NotImplementedError("sz-resident decode: threadgroup codebook only (no devcb fork)")
        if codes.shape[1] != (NSUB + 31) // 32 * pack_bits:
            raise ValueError(f"sz-resident: {codes.shape[1]} words/row, expected "
                             f"{(NSUB + 31) // 32 * pack_bits}")
        dims = ns["_dims_array"](OUT, IN, D, G, N, K)
        tgx = 256 if OUT >= 256 else OUT
        template = [("T", x.dtype), ("MAX_K", K), ("MAX_NSUB", NSUB), ("BITS", pack_bits)]
        k = kernel(f"sz_fused_packed{pack_bits}_d4_walk", dec_src, DECODE_SIG, template)
        (y,) = k(inputs=[x, eidx, codes, codebook, scales, dims, rowtbl],
                 grid=(((OUT + tgx - 1) // tgx) * tgx, N, 1), threadgroup=(tgx, 1, 1),
                 output_shapes=[(N, OUT)], output_dtypes=[x.dtype])
        return y

    def sz_gemmseg(xsrc, src_rows, idx_sorted_np, codes, codebook, scales, rowtbl,
                   pack_bits, IN, E, OUT):
        # mirrors _gemmseg_prefill's v2 branch, packed d4 only
        if not ns.get("_FUSED_GEMM_V2", False):
            raise NotImplementedError("sz-resident prefill: needs VQ_MOE_FUSED_GEMM=2")
        if ns.get("_GEMMSEG_PIPE", False):
            raise NotImplementedError("sz-resident prefill: VQ_GEMMSEG_PIPE not forked")
        K = codebook.shape[0]
        D = int(codebook.shape[1])
        if D != 4 or not pack_bits:
            raise NotImplementedError("sz-resident prefill: packed d4 only")
        TGC, TR32, XPB = ns["_TG_CAP_BYTES"], ns["_TILES_R32"], ns["_XT_PAD_BYTES_R32"]
        RT = ns["_GEMMSEG_RTILE"]
        _rt = 64 if (RT == 64 and K * 2 * D + TR32 + XPB > TGC) else 32
        _mk = ("tiles", idx_sorted_np.tobytes(), E, _rt)
        _hit = ns["_memo_get"](_mk)
        if _hit is not None:
            tmeta, ntiles = _hit
        else:
            counts = np.bincount(idx_sorted_np, minlength=E)
            touched = np.nonzero(counts)[0]
            starts = np.zeros(E + 1, np.int64)
            starts[1:] = np.cumsum(counts)
            tc = counts[touched]
            ntiles_per = (tc + _rt - 1) // _rt
            eids = np.repeat(touched, ntiles_per)
            cum = np.cumsum(ntiles_per)
            within = (np.arange(int(cum[-1]) if len(cum) else 0)
                      - np.repeat(cum - ntiles_per, ntiles_per)) * _rt
            rows = starts[eids] + within
            nrows = np.minimum(_rt, counts[eids] - within)
            metas_np = np.stack([eids, rows, nrows], axis=1).astype(np.int32)
            tmeta = mx.array(metas_np.reshape(-1))
            ntiles = int(metas_np.shape[0])
            ns["_memo_put"](_mk, (tmeta, ntiles))
        cbk = codebook.astype(mx.float16) if codebook.dtype != mx.float16 else codebook
        dims = ns["_dims_array"](OUT, IN, IN // 64, K, ntiles)
        name = f"sz_gemmseg2_packed{pack_bits}_d{D}"
        cb_bytes = K * 2 * D
        cb_dev = cb_bytes + TR32 + XPB > TGC or cb_bytes >= 16384
        mode = ns["_GEMMSEG_CBDEV"]
        if mode == "1":
            cb_dev = True
        elif mode == "0" and cb_bytes + TR32 + XPB <= TGC:
            cb_dev = False
        if cb_dev:
            name += "_cbdev"
        rtile = 64 if (RT == 64 and cb_dev) else 32
        if rtile != 32:
            name += f"_r{rtile}"
        if ns["_GEMMSEG_XT_PAD"]:
            name += "_xp8"
        ot2 = 1 if (ns["_GEMMSEG_OT2"] and rtile == 32) else 0
        if ot2:
            name += "_ot2"
        if ns["_GEMMSEG_DSTORE"]:
            name += "_ds"
        if ns["_GEMMSEG_PH2V"]:
            name += "_p2v"
        if xsrc.dtype == mx.bfloat16:
            name += "_bf16io"
        template = [("BITS", pack_bits), ("GROUP", 64), ("MAX_K", 1 if cb_dev else K),
                    ("D_BAKE", D), ("CB_DEV", 1 if cb_dev else 0), ("RTILE", rtile),
                    ("XPAD", ns["_XT_PAD_HALVES"]), ("TIO", xsrc.dtype), ("OT2", ot2),
                    ("DSTORE", 1 if ns["_GEMMSEG_DSTORE"] else 0), ("PIPE", 0),
                    ("PH2V", 1 if ns["_GEMMSEG_PH2V"] else 0)]
        N = int(idx_sorted_np.shape[0])
        ospan = 64 if ot2 else 32
        k = kernel(name, gs_src, GEMMSEG_SIG, template)
        (y,) = k(inputs=[codes, cbk, scales, xsrc, mx.array(src_rows), tmeta, dims, rowtbl],
                 grid=(32 * ((OUT + ospan - 1) // ospan), 4 * ntiles, 1), threadgroup=(32, 4, 1),
                 output_shapes=[(N, OUT)], output_dtypes=[xsrc.dtype])
        return y

    class SZSwitchLinear(VQSwitchLinear):
        """VQSwitchLinear over COMPACT live rows + an [E, OUT] row table."""

        def __init__(self, codes, codebook, vq_scales, row_table, group_size=64,
                     pack_bits=0, in_features=None):
            if not pack_bits or codebook.shape[1] != 4:
                raise NotImplementedError("SZSwitchLinear: packed d4 only")
            super().__init__(codes, codebook, vq_scales, group_size=group_size,
                             pack_bits=pack_bits, in_features=in_features)
            self.unfreeze()
            self.row_table = row_table
            self.freeze()

        @property
        def input_dims(self):
            return self.vq_scales.shape[1] * self.group_size

        @property
        def output_dims(self):
            return self.row_table.shape[1]

        @property
        def num_experts(self):
            return self.row_table.shape[0]

        def __call__(self, x, indices, sorted_indices=False):
            if self.codebook.shape[0] != self._k_expect:
                raise RuntimeError("VQ codebook was sharded; sz-resident is single-box only")
            IN, OUT, E = self.input_dims, self.output_dims, self.num_experts
            idx_flat = indices.flatten()
            N = idx_flat.size
            xf = mx.broadcast_to(x, (*indices.shape, 1, IN)).reshape(N, IN)
            in_dtype = xf.dtype
            keep_bf16 = ns.get("_DECODE_BF16IO", False) and in_dtype == mx.bfloat16
            if in_dtype not in (mx.float16,) and not keep_bf16:
                xf = xf.astype(mx.float16)
            pb = self.pack_bits
            if N <= ns["VQ_FUSED_MAX_N"]:
                y = sz_fused(xf, idx_flat.astype(mx.uint32), self["codes"], self["codebook"],
                             self["vq_scales"], self["row_table"], pb, OUT, IN)
            else:
                idx_np = ns["_idx_np"](indices, idx_flat)
                T = x.size // IN
                if not (ns.get("_FUSE_GATHER", False) and N % max(T, 1) == 0):
                    raise NotImplementedError("sz-resident prefill: fused-gather form only")
                if not ns["gemmseg_fits"](4, int(self.codebook.shape[0]),
                                          IN // self.vq_scales.shape[1], pb, IN):
                    raise NotImplementedError("sz-resident prefill: gemmseg path only")
                x2 = x.reshape(T, IN)
                if x2.dtype not in (mx.float16,) and not (
                        ns.get("_GEMMSEG_BF16IO", False) and x2.dtype == mx.bfloat16):
                    x2 = x2.astype(mx.float16)
                k_rep = N // T
                if not sorted_indices:
                    _sk = ("sort", idx_np.tobytes(), int(k_rep))
                    _sh = ns["_memo_get"](_sk)
                    if _sh is not None:
                        idx_sorted, src, inv_mx = _sh
                    else:
                        order = np.argsort(idx_np, kind="stable")
                        inv = np.argsort(order, kind="stable")
                        src = (order // k_rep).astype(np.uint32)
                        idx_sorted = idx_np[order]
                        inv_mx = mx.array(inv.astype(np.uint32))
                        ns["_memo_put"](_sk, (idx_sorted, src, inv_mx))
                    y = sz_gemmseg(x2, src, idx_sorted, self["codes"], self["codebook"],
                                   self["vq_scales"], self["row_table"], pb, IN, E, OUT)
                    y = y[inv_mx]
                else:
                    src = (np.arange(N, dtype=np.uint32) // k_rep).astype(np.uint32)
                    y = sz_gemmseg(x2, src, idx_np, self["codes"], self["codebook"],
                                   self["vq_scales"], self["row_table"], pb, IN, E, OUT)
            return y.astype(in_dtype).reshape(*indices.shape, 1, OUT)

    SZSwitchLinear.sz_fused = staticmethod(sz_fused)
    SZSwitchLinear.sz_gemmseg = staticmethod(sz_gemmseg)
    ns["SZSwitchLinear"] = SZSwitchLinear
    return SZSwitchLinear


def swap_modules(model, cfg, reach, ns):
    """Replace every module named in cfg['vq_skipzero']['modules'] with an
    SZSwitchLinear shaped for its compact tensors (shapes from the config)."""
    mx = ns["mx"]
    SZ = install(ns)
    vqm, sz = cfg["vq_modules"], cfg["vq_skipzero"]
    for p, m in sz["modules"].items():
        g = vqm[p]
        pb = g.get("pack_bits", 0)
        if not pb or g["dim"] != 4:
            raise NotImplementedError(f"sz-resident: {p} is not packed d4")
        nsub = g["in"] // g["dim"]
        W = (nsub + 31) // 32 * pb
        if m.get("code_words", W) != W:
            raise ValueError(f"sz-resident: {p} code_words {m['code_words']} != {W}")
        obj, leaf = reach(model, p)
        setattr(obj, leaf, SZ(
            mx.zeros((m["live_rows"], W), dtype=mx.uint32),
            mx.zeros((g["k"], g["dim"]), dtype=mx.float16),
            mx.zeros((m["live_rows"], g["in"] // g["group"]), dtype=mx.float16),
            mx.zeros((m["experts"], m["out"]), dtype=mx.int32),
            group_size=g["group"], pack_bits=pb, in_features=g["in"]))


# appended to the source bundle's model.py by `vqlab sz-resident`
MODEL_HOOK = '''

# ===========================================================================
# vq-skipzero STAGE 2 / RESIDENT (EXPERIMENTAL, not a shipped format) --
# appended by `vqlab sz-resident`. Everything ABOVE this line is the source
# bundle's model.py, byte-for-byte. Packed modules listed in
# config.vq_skipzero.modules become SZSwitchLinear (sz_resident.py): compact
# live rows + an [E, OUT] row table stay resident; dead rows read no code
# bytes and produce exact zeros. Both loaders are hooked (sanitize and
# load_weights); resident_weights() is idempotent.
# ===========================================================================
import importlib.util as _szr_util
_szr_spec = _szr_util.spec_from_file_location(
    "sz_resident", _pathlib.Path(__file__).parent / "sz_resident.py")
_szr = _szr_util.module_from_spec(_szr_spec)
_szr_spec.loader.exec_module(_szr)
if not _cfg.get("vq_skipzero", {}).get("resident"):
    raise RuntimeError("sz-resident hook on an artifact without vq_skipzero.resident=true")

_SZRBaseModel = Model


class Model(_SZRBaseModel):
    def __init__(self, args):
        super().__init__(args)
        _szr.swap_modules(self, _cfg, _reach_vq, globals())

    def sanitize(self, weights):
        weights = _szr.resident_weights(weights)
        _base = getattr(super(), "sanitize", None)
        return _base(weights) if _base is not None else weights

    def load_weights(self, file_or_weights, strict=True):
        if not isinstance(file_or_weights, (str, _pathlib.Path)):
            file_or_weights = _szr.resident_weights(file_or_weights)
        return super().load_weights(file_or_weights, strict=strict)
'''
