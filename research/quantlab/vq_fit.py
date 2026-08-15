#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E35 M0: product-quantize the expert weights, emit a DECODED bf16 model.

Quality/kernel decoupling (VECTORQUANT_PLAN.md): this writes the VQ
RECONSTRUCTION as bf16 so the existing referee can score it with no Metal
kernel and no exo change. Bytes are computed analytically; what we're
buying here is the answer to "is VQ better than scalar 2-bit at all".

Format being emulated (d=4, K=256 default):
  codes   log2(K)/d = 2.00 bpw     <- SAME bit budget as scalar 2-bit
  scale   fp16 per (row, 64)       = 0.25 bpw
  total   2.25 bpw   vs RTN affine 2.50 bpw (scale AND bias per group)
So VQ is 10% SMALLER than the RTN baseline while getting 256 LEARNED joint
patterns per 4-weight group instead of 256 rigid grid combinations. VQ needs
no bias term: centroids are arbitrary 4-vectors, asymmetry is free.

Codebooks are fit in PURE WEIGHT SPACE (k-means, no Hessian, no activations)
— E34 put activation-fitted methods at 0-for-4 on this family.
"""
import argparse
import gc
import json
import pathlib
import shutil
import time

import mlx.core as mx

mx.set_cache_limit(8 << 30)

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-35B-A3B")
ap.add_argument("--out", required=True)
ap.add_argument("--dim", type=int, default=4, help="subvector dim d")
ap.add_argument("--k", type=int, default=256, help="codebook size K")
ap.add_argument("--group", type=int, default=64, help="scale granularity")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--sample", type=int, default=2_000_000, help="subvectors for the fit")
ap.add_argument("--chunk", type=int, default=4_000_000, help="assign chunk")
ap.add_argument("--bf16-experts", action="store_true",
                help="CONTROL: skip VQ, just copy experts (upper bound)")
args = ap.parse_args()

SRC = pathlib.Path(args.src)
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
D, K, G = args.dim, args.k, args.group

EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj",
               "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj",
               "mlp.switch_mlp.down_proj")


def kmeans(X, k, iters):
    """k-means++ light: seed from a random sample, then Lloyd on GPU."""
    n = X.shape[0]
    idx = mx.random.randint(0, n, (k,))
    C = X[idx]
    xn = mx.sum(X * X, axis=1, keepdims=True)
    for it in range(iters):
        cn = mx.sum(C * C, axis=1)
        # [n,k] distances via GEMM
        d2 = xn - 2 * (X @ C.T) + cn[None, :]
        a = mx.argmin(d2, axis=1)
        mx.eval(a)
        # sum per cluster via one-hot matmul (k is small)
        oh = (a[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
        cnt = mx.sum(oh, axis=0)
        newC = (oh.T @ X) / mx.maximum(cnt[:, None], 1.0)
        # keep empties where they were
        C = mx.where(cnt[:, None] > 0, newC, C)
        mx.eval(C)
    return C


def assign_and_decode(X, C, chunk):
    cn = mx.sum(C * C, axis=1)
    outs = []
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk]
        d2 = mx.sum(xb * xb, axis=1, keepdims=True) - 2 * (xb @ C.T) + cn[None, :]
        a = mx.argmin(d2, axis=1)
        outs.append(C[a])
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=0)


def vq_tensor(W):
    """W: [...,, out, in] bf16 -> VQ reconstruction, same shape/dtype."""
    shape = W.shape
    in_dim = shape[-1]
    assert in_dim % G == 0 and G % D == 0
    Wf = W.astype(mx.float32).reshape(-1, in_dim)          # [rows, in]
    rows = Wf.shape[0]
    # per (row, group) scale = max|w|
    Wg = Wf.reshape(rows, in_dim // G, G)
    scale = mx.max(mx.abs(Wg), axis=2, keepdims=True)
    scale = mx.maximum(scale, 1e-8).astype(mx.bfloat16).astype(mx.float32)
    Wn = (Wg / scale).reshape(-1, D)                        # subvectors
    mx.eval(Wn)
    n = Wn.shape[0]
    take = min(args.sample, n)
    sel = mx.random.randint(0, n, (take,))
    C = kmeans(Wn[sel], K, args.iters)
    R = assign_and_decode(Wn, C, args.chunk)
    R = (R.reshape(rows, in_dim // G, G) * scale).reshape(shape)
    err = float(mx.linalg.norm(R.astype(mx.float32) - Wf.reshape(shape).astype(mx.float32))
                / mx.linalg.norm(Wf.reshape(shape).astype(mx.float32)))
    return R.astype(mx.bfloat16), err


idx_path = SRC / "model.safetensors.index.json"
wmap = json.load(open(idx_path))["weight_map"]
files = sorted(set(wmap.values()))
out_map = {}
errs = []
t0 = time.time()
for fi, f in enumerate(files):
    shard = mx.load(str(SRC / f))
    new = {}
    for k, v in shard.items():
        if any(e in k for e in EXPERT_KEYS) and not args.bf16_experts:
            r, err = vq_tensor(v)
            errs.append(err)
            new[k] = r
            print(f"  {k.split('layers.')[-1][:40]:42s} relerr {err:.4f}", flush=True)
        else:
            new[k] = v
    mx.save_safetensors(str(OUT / f), new)
    for k in new:
        out_map[k] = f
    del shard, new
    gc.collect()
    mx.clear_cache()
    print(f"[{fi+1}/{len(files)}] {f}  ({time.time()-t0:.0f}s)", flush=True)

tsz = sum((OUT / f).stat().st_size for f in files)
json.dump({"metadata": {"total_size": tsz}, "weight_map": out_map},
          open(OUT / "model.safetensors.index.json", "w"))
for extra in SRC.iterdir():
    if extra.suffix != ".safetensors" and extra.is_file() and "index" not in extra.name:
        shutil.copy2(extra, OUT / extra.name)

bpw = (0 if args.bf16_experts else
       __import__("math").log2(K) / D + 16.0 / G)
print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")
if errs:
    print(f"mean expert reconstruction relerr {sum(errs)/len(errs):.4f} "
          f"over {len(errs)} tensors")
    print(f"emulated format: d={D} K={K} -> {bpw:.2f} bpw "
          f"(RTN affine 2-bit gs64 = 2.50 bpw)")
