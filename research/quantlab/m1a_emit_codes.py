#!/usr/bin/env python
"""M1a fit-side: emit REAL codes/codebook/vq_scales for one expert tensor.

Standalone on purpose — vq_397b_fused.py is mid-run on the M3 queue (E/F/G)
and must not be edited under a live chain. Same math as vq_fit.vq_tensor,
but instead of decoding to bf16 it SAVES the M1 kernel format:

    codes     uint16   [E, out, in/d]      (uint8-safe when K<=256, kept
                                            uint16 in the npz for simplicity;
                                            the artifact writer narrows)
    codebook  fp16     [K, d]
    scales    fp32*    [E, out, in/64]     (*bf16-rounded values; npz has no
                                            bf16 dtype)

Also saves the original bf16 tensor (as fp32) sliced to --ref-experts so the
kernel correctness test can decode-and-matmul in numpy against ground truth.

Run on the M4 with the tensor's shard STAGED to local APFS first (mx.load
mmap + Metal is broken from T7/exFAT/SMB — E35 rig gotcha).
"""
import argparse
import time

import mlx.core as mx
import numpy as np

mx.set_cache_limit(8 << 30)

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True, help="staged .safetensors path (local APFS)")
ap.add_argument("--tensor", required=True, help="full tensor key inside the shard")
ap.add_argument("--out", required=True, help="output .npz")
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--k", type=int, default=128)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--sample", type=int, default=2_000_000)
ap.add_argument("--ref-experts", type=int, default=4,
                help="how many experts' ORIGINAL weights to save for reference")
args = ap.parse_args()
D, K, G = args.dim, args.k, args.group


def kmeans(X, k, iters):
    n = X.shape[0]
    idx = mx.random.randint(0, n, (k,))
    C = X[idx]
    xn = mx.sum(X * X, axis=1, keepdims=True)
    # K-scaled chunking (probe_dk lesson: 4M x K x 4B blows Metal buffers)
    step = max(50_000, int(5e8 / k))
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        oh_sum = mx.zeros((k, X.shape[1]))
        cnt = mx.zeros((k,))
        for s in range(0, n, step):
            xb, xnb = X[s:s + step], xn[s:s + step]
            a = mx.argmin(xnb - 2 * (xb @ C.T) + cn[None, :], axis=1)
            oh = (a[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
            oh_sum = oh_sum + oh.T @ xb
            cnt = cnt + mx.sum(oh, axis=0)
            mx.eval(oh_sum, cnt)
        newC = oh_sum / mx.maximum(cnt[:, None], 1.0)
        C = mx.where(cnt[:, None] > 0, newC, C)
        mx.eval(C)
    return C


def assign(X, C):
    cn = mx.sum(C * C, axis=1)
    step = max(50_000, int(5e8 / C.shape[0]))
    outs = []
    for s in range(0, X.shape[0], step):
        xb = X[s:s + step]
        d2 = mx.sum(xb * xb, axis=1, keepdims=True) - 2 * (xb @ C.T) + cn[None, :]
        a = mx.argmin(d2, axis=1)
        mx.eval(a)
        outs.append(a)
    return mx.concatenate(outs, axis=0)


t0 = time.time()
shard = mx.load(args.shard)
W = shard[args.tensor]
print(f"{args.tensor}  shape {W.shape}  dtype {W.dtype}", flush=True)
shape = W.shape
in_dim = shape[-1]
assert in_dim % G == 0 and G % D == 0

Wf = W.astype(mx.float32).reshape(-1, in_dim)
rows = Wf.shape[0]
Wg = Wf.reshape(rows, in_dim // G, G)
scale = mx.max(mx.abs(Wg), axis=2, keepdims=True)
scale = mx.maximum(scale, 1e-8).astype(mx.bfloat16).astype(mx.float32)
Wn = (Wg / scale).reshape(-1, D)
mx.eval(Wn)

n = Wn.shape[0]
take = min(args.sample, n)
sel = mx.random.randint(0, n, (take,))
C = kmeans(Wn[sel], K, args.iters)
print(f"kmeans done ({time.time()-t0:.0f}s)", flush=True)

codes = assign(Wn, C)                       # [rows * in/d]
# fp16-round the codebook NOW — that is what ships and what the kernel reads.
C16 = C.astype(mx.float16)

# reconstruction relerr with the SHIPPED (fp16 codebook) values
# (chunked: rows*in exceeds MLX's int32 shape limit on 397B tensors)
Cf = C16.astype(mx.float32)
nsub = in_dim // D
num = mx.zeros(())
den = mx.zeros(())
rstep = 200_000
for s in range(0, rows, rstep):
    e = min(s + rstep, rows)
    Rb = (Cf[codes[s * nsub:e * nsub]].reshape(e - s, in_dim // G, G)
          * scale[s:e]).reshape(e - s, in_dim)
    Wb = Wf[s:e]
    num = num + mx.sum((Rb - Wb) ** 2)
    den = den + mx.sum(Wb ** 2)
    mx.eval(num, den)
err = float(mx.sqrt(num / den))
print(f"assign done, relerr(fp16 codebook) {err:.4f} ({time.time()-t0:.0f}s)",
      flush=True)

E = shape[0] if len(shape) == 3 else 1
out_dim = shape[-2]
codes_np = np.array(codes.astype(mx.uint32), copy=False).astype(np.uint16)
codes_np = codes_np.reshape(E, out_dim, in_dim // D)
cb_np = np.array(C16, copy=False)           # fp16
sc_np = np.array(scale.reshape(E, out_dim, in_dim // G), copy=False)
ref = np.array(Wf.reshape(shape)[: args.ref_experts].astype(mx.float32), copy=False)

np.savez(args.out, codes=codes_np, codebook=cb_np, scales=sc_np,
         ref_w=ref, meta=np.array([E, out_dim, in_dim, D, K, G], dtype=np.int64))
print(f"saved {args.out}  codes {codes_np.shape} uint16, "
      f"codebook {cb_np.shape} fp16, scales {sc_np.shape}, "
      f"ref_w {ref.shape}  ({time.time()-t0:.0f}s total)", flush=True)
