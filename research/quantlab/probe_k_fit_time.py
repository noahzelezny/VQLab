#!/usr/bin/env python
"""Timing probe: how long does a CODES fit cost as a function of K?

Answers "what is the ETA on an E (d4 K2048) codes fit" without spending the
fit. Replicates vq_397b_codes.py's two hot loops EXACTLY (kmeans over a
sample, then assign-all per expert chunk) on ONE REAL tensor, at several K.

Per the E37 lesson (E36's d8 win was a layer-0 artifact) this defaults to a
DEEP layer — timing is less depth-sensitive than quality, but there is no
reason to reintroduce the habit.

Reports load / kmeans / assign separately, because only the K-dependent
parts scale: kmeans and assign both cost O(N*K*d), the source read does not.
"""
import argparse
import json
import math
import pathlib
import time

import mlx.core as mx
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/"
                                 "Qwen--Qwen3.5-397B-A17B-bf16")
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--proj", default="down_proj", choices=["down_proj", "gate_up_proj"])
ap.add_argument("--ks", default="256,2048")
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--sample", type=int, default=2_000_000)
ap.add_argument("--expert-chunk", type=int, default=32)
ap.add_argument("--experts", type=int, default=64,
                help="how many experts to actually assign (scaled up in the "
                     "report); the full tensor is 512")
args = ap.parse_args()

D, G = args.dim, args.group
SRC = pathlib.Path(args.src)


def kmeans(X, k, iters):
    """Verbatim shape-for-shape copy of vq_397b_codes.kmeans."""
    step = max(50_000, int(5e8 / k))
    n = X.shape[0]
    C = X[mx.random.randint(0, n, (k,))]
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        parts = []
        for s0 in range(0, n, step):
            xb = X[s0:s0 + step]
            parts.append(mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                                   - 2 * (xb @ C.T) + cn[None, :], axis=1))
            mx.eval(parts[-1])
        a = mx.concatenate(parts)
        oh_sum = mx.zeros((k, X.shape[1]))
        cnt = mx.zeros((k,))
        for s0 in range(0, n, 2_000_000):
            ab = a[s0:s0 + 2_000_000]
            oh = (ab[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
            oh_sum = oh_sum + oh.T @ X[s0:s0 + 2_000_000]
            cnt = cnt + mx.sum(oh, axis=0)
            mx.eval(oh_sum, cnt)
        C = mx.where(cnt[:, None] > 0,
                     oh_sum / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C


def normalize(blk):
    out_d, in_d = blk.shape[1], blk.shape[2]
    g = blk.reshape(-1, in_d // G, G)
    scale = mx.maximum(mx.max(mx.abs(g), axis=2, keepdims=True), 1e-8)
    return (g / scale).reshape(-1, D), scale


name = (f"model.language_model.layers.{args.layer}.mlp.experts.{args.proj}")
wm = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
assert name in wm, name

t0 = time.time()
# mlx's own safetensors reader — the `safetensors` package cannot express
# bf16 through numpy/mlx frameworks (TypeError: data type 'bfloat16').
T = mx.load(str(SRC / wm[name]))[name]
mx.eval(T)
t_load = time.time() - t0
n_exp_full = T.shape[0]
T = T[:args.experts]
print(f"{name}  shape={list(T.shape)} (of {n_exp_full} experts)  "
      f"load {t_load:.1f}s", flush=True)

# --- sample for kmeans (same as the fitter: a fixed-size subsample) --------
sub_all, _ = normalize(T[:args.expert_chunk].astype(mx.float32))
idx = mx.random.randint(0, sub_all.shape[0], (min(args.sample, sub_all.shape[0]),))
X = sub_all[idx]
mx.eval(X)
del sub_all

rows = []
for K in (int(x) for x in args.ks.split(",")):
    t0 = time.time()
    C16 = kmeans(X, K, args.iters).astype(mx.float16)
    mx.eval(C16)
    t_km = time.time() - t0

    Cf = C16.astype(mx.float32)
    cn = mx.sum(Cf * Cf, axis=1)
    step = max(50_000, int(5e8 / K))
    t0 = time.time()
    num = den = 0.0
    for s in range(0, T.shape[0], args.expert_chunk):
        blk = T[s:s + args.expert_chunk].astype(mx.float32)
        sub, scale = normalize(blk)
        aparts = []
        for c in range(0, sub.shape[0], step):
            xb = sub[c:c + step]
            aparts.append(mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                                    - 2 * (xb @ Cf.T) + cn[None, :], axis=1))
            mx.eval(aparts[-1])
        a = mx.concatenate(aparts)
        R = (Cf[a].reshape(-1, blk.shape[2] // G, G) * scale).reshape(blk.shape)
        mx.eval(R)
        num += float(mx.sum((R - blk) ** 2))
        den += float(mx.sum(blk ** 2))
        del blk, sub, scale, a, R, aparts
        mx.clear_cache()
    t_as = time.time() - t0
    relerr = math.sqrt(num / den)

    # scale the assign to the full tensor, then to the whole model
    full = t_as * (n_exp_full / args.experts)
    rows.append((K, t_km, t_as, full))
    print(f"K={K:<6} kmeans {t_km:7.1f}s   assign({args.experts} experts) "
          f"{t_as:7.1f}s   -> full tensor {full:7.1f}s   relerr {relerr:.4f}",
          flush=True)

print("\n--- extrapolation (57 layers x 3 projections = 171 tensors) ---")
print("assign scales with tensor size; gate_up is 2x down_proj, so a layer")
print("costs ~2x this tensor's gate_up + 1x its down_proj. Rough model:")
for K, t_km, t_as, full in rows:
    per_layer = full * (2.0 if args.proj == "down_proj" else 1.5)
    total = (per_layer + t_km) * 57
    print(f"K={K:<6} ~{per_layer/60:5.1f} min/layer  ->  "
          f"~{total/3600:4.1f} h for the full 397B fit")
print("\nCaveat: excludes source I/O (~390 GB read) and artifact write.")
