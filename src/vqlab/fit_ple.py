#!/usr/bin/env python3
"""VQ-fit per-layer-embedding (PLE) tables — sharded n-gram lookup tables.

Port of the gemma-e4b PLE fitter (one [262144, 10752] table) generalized to
qwen4_exp's layout: many shards of [~2.5M, 160] per PLE layer, matched by
--key-regex. Embeddings are the friendliest VQ case in the project — decode
is a row gather (codebook[codes[row]]), no matmul kernel — which is why the
e4b VQ-PLE artifact shipped.

Method per tensor (large-scale k-means practice, unchanged from the donor):
fit the codebook on a random SAMPLE of row-blocks, then ONE full assignment
pass in row chunks. The sample only has to represent the distribution; the
assignment is what has to be complete. Per-TENSOR codebooks, matching the
expert fitter's convention (codebook overhead at K2048/d4 fp16 is 16 KiB —
negligible x130).

qwen4_exp geometry note: rows are 160 wide, so --group must divide 160 —
the default here is 32 (NOT the body's 64). d4/K2048/g32 = 11/4 + 16/32
= 3.25 bpw; the 51.2 GiB of tables (bf16 102.4) land at ~20.8 GiB.

    python -m vqlab.fit_ple --src <bf16 dir> --out <dir> \
        [--key-regex ngram_embedding] [--k 2048 --dim 4 --group 32] \
        [--limit N]   # fit only the first N matching tensors (validation)

Scored like every artifact: relerr per tensor, reported worst-first; the
manifest records geometry + per-tensor relerr so a localized blowup cannot
hide in a mean.
"""
import argparse
import json
import pathlib
import re
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--key-regex", default="ngram_embedding")
ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=32)
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--sample", type=int, default=2_000_000,
                help="subvectors sampled for each tensor's codebook fit")
ap.add_argument("--rows-chunk", type=int, default=65536)
ap.add_argument("--limit", type=int, default=None,
                help="fit only the first N matching tensors (validation)")
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

SRC, OUT = pathlib.Path(a.src), pathlib.Path(a.out)
K, D, G = a.k, a.dim, a.group
OUT.mkdir(parents=True, exist_ok=True)
if a.seed >= 0:
    mx.random.seed(a.seed)

idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
keys = sorted(k for k in idx if re.search(a.key_regex, k))
if a.limit:
    keys = keys[: a.limit]
if not keys:
    raise SystemExit(f"FAIL: no keys match {a.key_regex!r}")
print(f"{len(keys)} tensors match {a.key_regex!r}", flush=True)


def norm_block(T):
    r, c = T.shape
    Wg = T.reshape(r, c // G, G)
    s = mx.maximum(mx.abs(Wg).max(axis=-1, keepdims=True), 1e-8)
    return (Wg / s).reshape(-1, D), s.astype(mx.float16)


def assign(X, C, chunk=1_000_000):
    outs = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = (x * x).sum(1, keepdims=True) - 2 * x @ C.T + (C * C).sum(1)
        outs.append(mx.argmin(d, axis=1).astype(mx.uint32))
        mx.eval(outs[-1])
    return mx.concatenate(outs)


manifest = {"geometry": {"k": K, "dim": D, "group": G, "iters": a.iters,
                         "sample": a.sample, "seed": a.seed},
            "tensors": {}}
t00 = time.time()
for ki, key in enumerate(keys):
    t0 = time.time()
    shard_path = OUT / f"ple-{ki:04d}.safetensors"
    if shard_path.exists():
        print(f"[{ki+1}/{len(keys)}] exists, skipping (resume)", flush=True)
        continue
    W = mx.load(str(SRC / idx[key]))[key]
    ROWS, COLS = W.shape
    if COLS % G or G % D:
        raise SystemExit(f"FAIL: {key} is [{ROWS},{COLS}]; group {G} must "
                         f"divide cols and dim {D} must divide group")

    per_block = COLS // D
    sel = mx.random.randint(0, ROWS, (max(1, a.sample // per_block),))
    # materialize the sample on the CPU stream BEFORE any GPU math: the
    # source is lazy (possibly over SMB), and a GPU kernel waiting on
    # storage inside a command buffer trips the Metal watchdog (M4,
    # 2026-08-28, 3 tensors in).
    with mx.stream(mx.cpu):
        Wsel = W[sel].astype(mx.float32)
        mx.eval(Wsel)
    Xs, _ = norm_block(Wsel)

    cap = min(200_000, Xs.shape[0])
    P = Xs[mx.random.randint(0, Xs.shape[0], (cap,))]
    C = P[mx.random.randint(0, P.shape[0], (1,))]
    d2 = mx.sum((P - C[0]) ** 2, axis=1)
    for _ in range(K - 1):
        r = float(mx.random.uniform().item())
        j = int(mx.argmax(mx.cumsum(d2 / mx.maximum(d2.sum(), 1e-12)) >= r).item())
        C = mx.concatenate([C, P[j:j + 1]], axis=0)
        d2 = mx.minimum(d2, mx.sum((P - P[j]) ** 2, axis=1))
        mx.eval(C, d2)

    for _ in range(a.iters):
        asn = assign(Xs, C)
        tot = mx.zeros((K, D)).at[asn].add(Xs)
        cnt = mx.zeros((K,)).at[asn].add(mx.ones((Xs.shape[0],)))
        C = mx.where((cnt > 0)[:, None], tot / mx.maximum(cnt, 1)[:, None], C)
        mx.eval(C)
    del Xs, P
    mx.clear_cache()

    codes_parts, scales_parts = [], []
    num = den = 0.0
    for s in range(0, ROWS, a.rows_chunk):
        with mx.stream(mx.cpu):
            blk = W[s:s + a.rows_chunk].astype(mx.float32)
            mx.eval(blk)
        Xn, sc = norm_block(blk)
        asn = assign(Xn, C)
        R = (C[asn].reshape(blk.shape[0], COLS // G, G)
             * sc.astype(mx.float32)).reshape(blk.shape)
        num += float(mx.sum((R - blk) ** 2).item())
        den += float(mx.sum(blk * blk).item())
        codes_parts.append(asn.reshape(blk.shape[0], per_block).astype(mx.uint16))
        scales_parts.append(sc.reshape(blk.shape[0], -1))
        del blk, Xn, R
        mx.clear_cache()

    relerr = (num / max(den, 1e-12)) ** 0.5
    mx.save_safetensors(str(OUT / f"ple-{ki:04d}.safetensors"),
                        {f"{key}.codes": mx.concatenate(codes_parts, axis=0),
                         f"{key}.codebook": C.astype(mx.float16),
                         f"{key}.vq_scales": mx.concatenate(scales_parts, axis=0)
                                               .astype(mx.float16)})
    manifest["tensors"][key] = {"shape": [ROWS, COLS], "relerr": round(relerr, 5),
                                "shard": f"ple-{ki:04d}.safetensors"}
    print(f"[{ki+1}/{len(keys)}] {key.split('.layers.')[-1]}  "
          f"relerr {relerr:.4f}  [{time.time()-t0:.0f}s]", flush=True)

mp = OUT / "ple_manifest.json"
if mp.exists():
    prev = json.loads(mp.read_text())
    prev["tensors"].update(manifest["tensors"])
    manifest = prev
mp.write_text(json.dumps(manifest, indent=1))
worst = sorted(manifest["tensors"].items(), key=lambda kv: -kv[1]["relerr"])[:5]
bits = (K - 1).bit_length()
print(f"\n{len(keys)} tensors  bpw {bits/D + 16/G:.2f}  worst:")
for k, v in worst:
    print(f"  {v['relerr']:.4f}  {k}")
