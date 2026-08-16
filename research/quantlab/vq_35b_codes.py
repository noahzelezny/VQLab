#!/usr/bin/env python
"""M1d/M1e: fit 35B expert VQ and write TWO artifacts from the SAME fit:

  --out-codes  the real thing: codes uint8 + codebook fp16 + vq_scales fp16
               per expert tensor (loadable via patch_mlx_lm's VQSwitchLinear)
  --out-proxy  bf16 decode of the SAME codebooks/codes (referee-scorable twin)

M1e bar: referee(codes artifact) == referee(proxy twin) — same values, so
any gap is a runtime bug. The twin's score should also land near the E35
K256 record (7.1807 wiki / 3.0881 code); it won't be bit-identical to that
run because k-means reseeds.

NOTE scales are rounded to FP16 here (not bf16 as in the E35 proxies) so the
stored artifact and the fp16 kernels agree exactly.
"""
import argparse
import gc
import json
import pathlib
import shutil
import time

import mlx.core as mx
import numpy as np

mx.set_cache_limit(8 << 30)

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="rotlab-35B-bf16exp-struct6")
ap.add_argument("--out-codes", required=True)
ap.add_argument("--out-proxy", default=None,
                help="bf16 twin (M1e runtime check). OPTIONAL: skip it for a\n                      pure quality calibration — it costs ~6x the codes\n                      artifact in disk and the runtime-vs-twin gap is closed.")
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--sample", type=int, default=2_000_000)
args = ap.parse_args()

SRC = pathlib.Path(args.src)
OUTC = pathlib.Path(args.out_codes)
OUTP = pathlib.Path(args.out_proxy) if args.out_proxy else None
OUTC.mkdir(parents=True, exist_ok=True)
if OUTP is not None:
    OUTP.mkdir(parents=True, exist_ok=True)
D, K, G = args.dim, args.k, args.group
assert K <= 256, "uint8 codes only in this writer"

EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj",
               "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj",
               "mlp.switch_mlp.down_proj")


def kmeans(X, k, iters):
    n = X.shape[0]
    C = X[mx.random.randint(0, n, (k,))]
    xn = mx.sum(X * X, axis=1, keepdims=True)
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
        C = mx.where(cnt[:, None] > 0, oh_sum / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C


def assign(X, C):
    cn = mx.sum(C * C, axis=1)
    step = max(50_000, int(5e8 / C.shape[0]))
    outs = []
    for s in range(0, X.shape[0], step):
        xb = X[s:s + step]
        a = mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                      - 2 * (xb @ C.T) + cn[None, :], axis=1)
        mx.eval(a)
        outs.append(a)
    return mx.concatenate(outs, axis=0)


def vq_tensor(W):
    """-> (codes u8 [E,out,in/D], cb fp16 [K,D], scales fp16 [E,out,in/G],
           proxy bf16 same shape as W, relerr)"""
    shape = W.shape
    E, out_dim, in_dim = shape
    assert in_dim % G == 0 and G % D == 0
    Wf = W.astype(mx.float32).reshape(-1, in_dim)
    rows = Wf.shape[0]
    Wg = Wf.reshape(rows, in_dim // G, G)
    scale = mx.max(mx.abs(Wg), axis=2, keepdims=True)
    scale = mx.maximum(scale, 1e-6).astype(mx.float16).astype(mx.float32)
    Wn = (Wg / scale).reshape(-1, D)
    mx.eval(Wn)
    n = Wn.shape[0]
    sel = mx.random.randint(0, n, (min(args.sample, n),))
    C16 = kmeans(Wn[sel], K, args.iters).astype(mx.float16)
    codes = assign(Wn, C16.astype(mx.float32))
    # proxy decode + relerr, chunked (rows*in can exceed int32 limit)
    Cf = C16.astype(mx.float32)
    nsub = in_dim // D
    num = mx.zeros(())
    den = mx.zeros(())
    proxy_parts = []
    rstep = 200_000
    for s in range(0, rows, rstep):
        e2 = min(s + rstep, rows)
        Rb = (Cf[codes[s * nsub:e2 * nsub]].reshape(e2 - s, in_dim // G, G)
              * scale[s:e2]).reshape(e2 - s, in_dim)
        num = num + mx.sum((Rb - Wf[s:e2]) ** 2)
        den = den + mx.sum(Wf[s:e2] ** 2)
        proxy_parts.append(Rb.astype(mx.bfloat16))
        mx.eval(proxy_parts[-1], num, den)
    err = float(mx.sqrt(num / den))
    proxy = mx.concatenate(proxy_parts, axis=0).reshape(shape)
    codes_u8 = codes.astype(mx.uint8).reshape(E, out_dim, nsub)
    scales16 = scale.astype(mx.float16).reshape(E, out_dim, in_dim // G)
    mx.eval(codes_u8, scales16, proxy)
    return codes_u8, C16, scales16, proxy, err


idx_path = SRC / "model.safetensors.index.json"
wmap = json.load(open(idx_path))["weight_map"]
files = sorted(set(wmap.values()))
map_c, map_p = {}, {}
errs = []
t0 = time.time()
for fi, f in enumerate(files):
    shard = mx.load(str(SRC / f))
    new_c, new_p = {}, {}
    for k, v in shard.items():
        if any(e in k for e in EXPERT_KEYS):
            codes_u8, cb, sc, proxy, err = vq_tensor(v)
            errs.append(err)
            # module path, not tensor path: strip ".weight" so the loader
            # swaps gate_proj itself, not gate_proj.weight
            kb = k[:-7] if k.endswith(".weight") else k
            new_c[kb + ".codes"] = codes_u8
            new_c[kb + ".codebook"] = cb
            new_c[kb + ".vq_scales"] = sc
            new_p[k] = proxy
            print(f"  {k.split('layers.')[-1][:40]:42s} relerr {err:.4f}",
                  flush=True)
        else:
            new_c[k] = v
            new_p[k] = v
    mx.save_safetensors(str(OUTC / f), new_c)
    if OUTP is not None:
        mx.save_safetensors(str(OUTP / f), new_p)
    for k in new_c:
        map_c[k] = f
    for k in new_p:
        map_p[k] = f
    del shard, new_c, new_p
    gc.collect()
    mx.clear_cache()
    print(f"[{fi+1}/{len(files)}] {f}  ({time.time()-t0:.0f}s)", flush=True)

for out, wm in [(OUTC, map_c)] + ([(OUTP, map_p)] if OUTP is not None else []):
    tsz = sum((out / f).stat().st_size for f in files)
    json.dump({"metadata": {"total_size": tsz}, "weight_map": wm},
              open(out / "model.safetensors.index.json", "w"))
    for extra in SRC.iterdir():
        if extra.suffix != ".safetensors" and extra.is_file() \
                and "index" not in extra.name:
            shutil.copy2(extra, out / extra.name)

print(f"\ndone in {time.time()-t0:.0f}s")
print(f"mean relerr {sum(errs)/len(errs):.4f} over {len(errs)} tensors")
gib_c = sum((OUTC / f).stat().st_size for f in files) / 2**30
print(f"codes artifact {gib_c:.1f} GiB -> {OUTC}")
if OUTP is not None:
    gib_p = sum((OUTP / f).stat().st_size for f in files) / 2**30
    print(f"proxy twin     {gib_p:.1f} GiB -> {OUTP}")
