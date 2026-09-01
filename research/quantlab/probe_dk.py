#!/usr/bin/env python3
"""Accessibility probe: which (d,K) hits ~90-100 GiB with the least damage?

Target: a 397B that RUNS on a 128 GB machine. 128 GB marketed = 119.2 GiB
addressable, so the artifact must land <= ~100 GiB to leave room for macOS
+ KV cache. Sweeps d and K at several bit budgets, reporting relerr against
the RTN 2-bit control at the SAME layer. Ranking only (E34: relerr is not
the objective) — the winner gets built and scored end-to-end.
"""
import json, math, sys, time
import mlx.core as mx
mx.set_cache_limit(6 << 30)
SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vqprobe"
LAYERS = (10, 35)
NEXP, G = 32, 64

def kmeans(X, k, iters=15):
    C = X[mx.random.randint(0, X.shape[0], (k,))]
    xn = mx.sum(X * X, axis=1, keepdims=True)
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        a = mx.argmin(xn - 2 * (X @ C.T) + cn[None, :], axis=1); mx.eval(a)
        oh = (a[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
        cnt = mx.sum(oh, axis=0)
        C = mx.where(cnt[:, None] > 0, (oh.T @ X) / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C

def relerr(W, d, k):
    e, o, i = W.shape
    Wg = W.reshape(-1, i // G, G)
    sc = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-8)
    sc = sc.astype(mx.bfloat16).astype(mx.float32)
    sub = (Wg / sc).reshape(-1, d); mx.eval(sub)
    n = sub.shape[0]
    samp = sub[mx.random.randint(0, n, (min(max(200_000, int(2e8/k)), n),))]
    C = kmeans(samp, k)
    cn = mx.sum(C * C, axis=1); parts = []
    step = max(50_000, int(4e9 / k))
    for c in range(0, n, step):
        xb = sub[c:c + step]
        a = mx.argmin(mx.sum(xb*xb,axis=1,keepdims=True) - 2*(xb@C.T) + cn[None,:], axis=1)
        parts.append(C[a]); mx.eval(parts[-1])
    R = (mx.concatenate(parts, axis=0).reshape(-1, i // G, G) * sc).reshape(e, o, i)
    return float(mx.linalg.norm(R - W) / mx.linalg.norm(W))

import os
shards = [f for f in os.listdir(SRC) if f.endswith(".safetensors")]
print(f"probing {len(shards)} staged shard(s)\n")
print(f"{'setting':16s} {'bpw':>5s} {'~GiB':>6s} {'relerr':>8s}  {'vs RTN':>8s}")
for sh in shards[:1]:
    d0 = mx.load(os.path.join(SRC, sh))
    key = [k for k in d0 if "gate_up_proj" in k][0]
    W = d0[key][:NEXP].astype(mx.float32)
    wq, s2, b2 = mx.quantize(d0[key][:NEXP], group_size=64, bits=2)
    rtn = float(mx.linalg.norm(mx.dequantize(wq,s2,b2,group_size=64,bits=2).astype(mx.float32) - W) / mx.linalg.norm(W))
    print(f"{'RTN 2-bit':16s} {2.50:5.2f} {122.3:6.1f} {rtn:8.4f}  {'--':>8s}")
    for d, k in ((4,128),(4,256),(4,512),(8,1024),(8,4096),(8,16384),(8,65536)):
        bpw = math.log2(k)/d + 16.0/G
        size = 122.3 - 109*(1 - bpw/2.50)
        t0=time.time(); e = relerr(W, d, k)
        print(f"d={d} K={k:<7d}  {bpw:5.2f} {size:6.1f} {e:8.4f}  {(1-e/rtn)*100:+7.1f}%  ({time.time()-t0:.0f}s)", flush=True)
