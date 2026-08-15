# quantlab M1d — VQ expert runtime for mlx_lm.
# Canonical copy lives in quantlab/vq_switch.py; patch_mlx_lm.py installs it
# as mlx_lm/models/vq_switch.py and hooks load_model. Keep both in sync by
# re-running the patcher, never by editing the installed copy.
#
# Format (per VQ'd expert tensor, produced by vq_*_codes fitters):
#   {p}.codes      uint8 (K<=256) / uint16   [E, out, in/d]
#   {p}.codebook   fp16                      [K, d]      (d=4 only, v1)
#   {p}.vq_scales  fp16                      [E, out, in/group]
#
# Two execution regimes (measured on M4, m1b/m1c benches, 2026-08-15):
#   decode (small N):  fused LUT-matmul kernel, threadgroup codebook + x,
#                      uchar4 code loads — 0.66-0.88x gather_qmm
#   prefill (large N): decode experts to dense fp16 chunks + ONE padded
#                      batched GEMM per chunk — 1.21-1.28x gather_qmm
#   The row-batched gather_mm path is a known trap (0.43x): do not "simplify"
#   the prefill path back to it.

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# below this many (token, expert) pairs use the fused kernel; above, the
# decode+padded-GEMM path (decode cost amortizes). Tune in M1e if needed.
VQ_FUSED_MAX_N = int(os.environ.get("SCOUT_VQ_FUSED_MAX_N", 4096))

# Experts decoded to dense fp16 per prefill chunk. THIS IS THE MEMORY KNOB,
# not the KV cache: measured 2026-08-15 on a 128 GB M4 Max running the
# 110.8 GiB 397B, prefill grew 3.35 MB/token where the KV cache theory is
# only 0.059 MB/token — a 57x gap that is entirely these buffers. The
# transient is chunk * out * in * 2 bytes:
#     chunk=128 -> 1.0 GiB (down_proj) / 2.0 GiB (gate_up)
#     chunk= 16 -> 0.12 GiB            / 0.25 GiB
# On a box with headroom, big chunks give better GEMM efficiency. On a box
# where the model nearly fills RAM, they are what caps your context length.
# Auto-sized from free memory at import; override with SCOUT_VQ_DECODE_CHUNK.
def _default_decode_chunk():
    env = os.environ.get("SCOUT_VQ_DECODE_CHUNK")
    if env:
        return max(1, int(env))
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        headroom = info["max_recommended_working_set_size"] - mx.get_active_memory()
        # keep the largest transient (gate_up, out*in ~ 8M elems fp16 = 16 MB
        # per expert) under ~1/8 of remaining headroom
        per_expert = 2048 * 4096 * 2
        return max(4, min(128, int(headroom / 8 / per_expert)))
    except Exception:
        return 32


_DECODE_CHUNK = None  # resolved lazily on first prefill (after weights load)

_SRC_FUSED = r"""
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int N    = dims[4];
    const int K    = dims[5];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int QPG  = G / 16;
    uint r = thread_position_in_grid.x;
    uint t = thread_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    threadgroup float4 cb[MAX_K];
    threadgroup float4 xs[MAX_NSUB];
    for (uint i = lid; i < (uint)K; i += tgsize)
        cb[i] = float4((float)codebook[i*4], (float)codebook[i*4+1],
                       (float)codebook[i*4+2], (float)codebook[i*4+3]);
    const device T* xrow = x + (size_t)t * IN;
    for (uint i = lid; i < (uint)NSUB; i += tgsize)
        xs[i] = float4((float)xrow[i*4], (float)xrow[i*4+1],
                       (float)xrow[i*4+2], (float)xrow[i*4+3]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= (uint)OUT || t >= (uint)N) return;
    const uint e = eidx[t];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const device half* srow = scales + (size_t)e * OUT * NGRP + (size_t)r * NGRP;
    float acc = 0.0f;
    int j = 0;
    for (int g = 0; g < NGRP; ++g) {
        float gacc = 0.0f;
        for (int q = 0; q < QPG; ++q) {
            gacc += dot(cb[(uint)crow[j]],   xs[j])
                  + dot(cb[(uint)crow[j+1]], xs[j+1])
                  + dot(cb[(uint)crow[j+2]], xs[j+2])
                  + dot(cb[(uint)crow[j+3]], xs[j+3]);
            j += 4;
        }
        acc = fma((float)srow[g], gacc, acc);
    }
    y[(size_t)t * OUT + r] = static_cast<T>(acc);
"""

_SRC_DECODE = r"""
    uint g = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    uint ec = thread_position_in_grid.z;
    const int OUT  = dims[0];
    const int IN   = dims[1];
    const int G    = dims[3];
    const int NE   = dims[4];
    const int NSUB = IN / 4;
    const int NGRP = IN / G;
    const int SPG  = G / 4;
    if (g >= (uint)NGRP || r >= (uint)OUT || ec >= (uint)NE) return;
    const uint e = eidx[ec];
    const device CT* crow = codes + (size_t)e * OUT * NSUB + (size_t)r * NSUB;
    const float s = (float)scales[(size_t)e * OUT * NGRP + (size_t)r * NGRP + g];
    device half* wrow = w + (size_t)ec * OUT * IN + (size_t)r * IN + (size_t)g * G;
    const int j0 = g * SPG;
    for (int q = 0; q < SPG; ++q) {
        const uint c = (uint)crow[j0 + q];
        for (int u = 0; u < 4; ++u)
            wrow[q * 4 + u] = (half)(s * (float)codebook[c * 4 + u]);
    }
"""

_KERNELS = {}


def _get_kernel(name, src):
    if name not in _KERNELS:
        if name == "vq_fused":
            inp, out = ["x", "eidx", "codes", "codebook", "scales", "dims"], ["y"]
        else:
            inp, out = ["codes", "codebook", "scales", "eidx", "dims"], ["w"]
        _KERNELS[name] = mx.fast.metal_kernel(
            name=name, input_names=inp, output_names=out, source=src)
    return _KERNELS[name]


def _fused(x, eidx, codes, codebook, scales):
    N, IN = x.shape
    E, OUT, NSUB = codes.shape
    K = codebook.shape[0]
    G = IN // scales.shape[2]
    dims = mx.array([OUT, IN, 4, G, N, K], dtype=mx.int32)
    tgx = 256 if OUT >= 256 else OUT
    (y,) = _get_kernel("vq_fused", _SRC_FUSED)(
        inputs=[x, eidx, codes, codebook, scales, dims],
        template=[("T", x.dtype), ("CT", codes.dtype),
                  ("MAX_K", K), ("MAX_NSUB", NSUB)],
        grid=(((OUT + tgx - 1) // tgx) * tgx, N, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(N, OUT)],
        output_dtypes=[x.dtype],
    )
    return y


def _decode_chunk(codes, codebook, scales, eidx_chunk):
    NE = eidx_chunk.shape[0]
    E, OUT, NSUB = codes.shape
    IN = NSUB * 4
    NGRP = scales.shape[2]
    G = IN // NGRP
    dims = mx.array([OUT, IN, 4, G, NE], dtype=mx.int32)
    (w,) = _get_kernel("vq_decode", _SRC_DECODE)(
        inputs=[codes, codebook, scales, eidx_chunk, dims],
        template=[("CT", codes.dtype)],
        grid=(NGRP, OUT, NE),
        threadgroup=(min(32, NGRP), 8, 1),
        output_shapes=[(NE, OUT, IN)],
        output_dtypes=[mx.float16],
    )
    return w


def _prefill(xf, idx_sorted_np, codes, codebook, scales):
    """xf [N, IN] rows sorted by expert; idx_sorted_np = matching np expert
    ids. Decode touched experts in chunks; one padded batched GEMM each."""
    global _DECODE_CHUNK
    if _DECODE_CHUNK is None:
        _DECODE_CHUNK = _default_decode_chunk()
    E, OUT, _ = codes.shape
    counts = np.bincount(idx_sorted_np, minlength=E)
    touched = np.nonzero(counts)[0]
    starts = np.zeros(E + 1, np.int64)
    starts[1:] = np.cumsum(counts)
    ys = []
    for c0 in range(0, len(touched), _DECODE_CHUNK):
        eids = touched[c0:c0 + _DECODE_CHUNK]
        ne = len(eids)
        cap = int(counts[eids].max())
        # gather map rows -> [ne, cap]; pads point at row 0 (discarded)
        gmap = np.zeros((ne, cap), np.uint32)
        vmask = np.zeros((ne, cap), bool)
        for i, e in enumerate(eids):
            c = counts[e]
            gmap[i, :c] = np.arange(starts[e], starts[e] + c, dtype=np.uint32)
            vmask[i, :c] = True
        w = _decode_chunk(codes, codebook, scales,
                          mx.array(eids.astype(np.uint32)))
        xp = xf[mx.array(gmap.reshape(-1))].reshape(ne, cap, -1)
        yp = xp @ mx.swapaxes(w, 1, 2)                      # [ne, cap, OUT]
        flat_valid = np.nonzero(vmask.reshape(-1))[0].astype(np.uint32)
        ys.append(yp.reshape(ne * cap, OUT)[mx.array(flat_valid)])
        # CRITICAL: MLX is lazy. Without this eval the whole loop builds one
        # graph and EVERY chunk's decoded weights stay live until the final
        # concatenate — 4 chunks x 2 GiB for gate_up, which is what actually
        # capped context length on a 128 GB box (measured 2026-08-15: prefill
        # grew 3.35 MB/token vs 0.059 MB/token of real KV cache). Evaluating
        # per chunk lets each `w` be freed before the next is decoded.
        mx.eval(ys[-1])
        del w, xp, yp
    return mx.concatenate(ys, axis=0)


class VQSwitchLinear(nn.Module):
    """Drop-in for QuantizedSwitchLinear over VQ codes. No bias support
    (Qwen3.5 experts are bias-free)."""

    def __init__(self, codes, codebook, vq_scales, group_size: int = 64):
        super().__init__()
        self.codes = codes
        self.codebook = codebook
        self.vq_scales = vq_scales
        self.group_size = group_size
        self.freeze()

    @classmethod
    def from_weights(cls, codes, codebook, vq_scales):
        return cls(codes, codebook, vq_scales)

    @property
    def input_dims(self):
        return self.codes.shape[2] * self.codebook.shape[1]

    @property
    def output_dims(self):
        return self.codes.shape[1]

    @property
    def num_experts(self):
        return self.codes.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        IN = self.input_dims
        OUT = self.output_dims
        idx_flat = indices.flatten()
        N = idx_flat.size
        xf = mx.broadcast_to(x, (*indices.shape, 1, IN)).reshape(N, IN)
        in_dtype = xf.dtype
        if in_dtype not in (mx.float16,):
            xf = xf.astype(mx.float16)
        if N <= VQ_FUSED_MAX_N:
            y = _fused(xf, idx_flat.astype(mx.uint32),
                       self["codes"], self["codebook"], self["vq_scales"])
        else:
            idx_np = np.array(idx_flat, copy=False)
            if not sorted_indices:
                order = np.argsort(idx_np, kind="stable")
                inv = np.argsort(order, kind="stable")
                y = _prefill(xf[mx.array(order.astype(np.uint32))],
                             idx_np[order],
                             self["codes"], self["codebook"], self["vq_scales"])
                y = y[mx.array(inv.astype(np.uint32))]
            else:
                y = _prefill(xf, idx_np,
                             self["codes"], self["codebook"], self["vq_scales"])
        return y.astype(in_dtype).reshape(*indices.shape, 1, OUT)
