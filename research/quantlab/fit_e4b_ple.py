#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Fit the e4b PER-LAYER EMBEDDING table (5.25 GiB = 35.5% of the model).

WHY THIS ONE. E65 proved VQ transfers to e4b's dense mlp (mean relerr
0.0297, better than the 26b MoE). But the mlp trio only buys ~1.06 GiB
against the 8-bit incumbent. embed_tokens_per_layer is [262144, 10752] =
2.82B params, 35.5% of bf16 bytes, and it is the ONE tensor that could make
a VQ e4b decisively smaller than 8-bit. It is also the friendliest runtime
case in the whole project: an embedding is a row GATHER, so decode is
codes[row] -> codebook lookup, with no matmul kernel involved at all.

WHY A SEPARATE FITTER. 1.4B subvectors at d=2 will not take the mlp path's
full-data Lloyd iterations. Standard large-scale k-means practice: fit the
codebook on a large random SAMPLE, then do ONE full assignment pass. The
sample only has to represent the distribution; the assignment is what has
to be complete.

Scored the same way as every other artifact: relerr against bf16, reported
per row-block so a localized blowup cannot hide in the mean.
"""
import argparse
import json
import pathlib
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--key", default="language_model.model.embed_tokens_per_layer.weight")
ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--sample", type=int, default=20_000_000, help="rows for codebook fit")
ap.add_argument("--rows-chunk", type=int, default=8192, help="table rows per pass")
args = ap.parse_args()

SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)
K, D, G = args.k, args.dim, args.group
OUT.mkdir(parents=True, exist_ok=True)
idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
t0 = time.time()

W = mx.load(str(SRC / idx[args.key]))[args.key]
ROWS, COLS = W.shape
print(f"{args.key}  {W.shape}  {W.size*2/2**30:.2f} GiB bf16", flush=True)


def norm_block(T):
    r, c = T.shape
    Wg = T.reshape(r, c // G, G)
    s = mx.maximum(mx.abs(Wg).max(axis=-1, keepdims=True), 1e-8)
    return (Wg / s).reshape(-1, D), s.astype(mx.float16)


# ---- codebook on a random sample of ROW BLOCKS (rows are the natural unit)
per_block = (COLS // D)
n_blocks = max(1, args.sample // per_block)
sel = mx.random.randint(0, ROWS, (min(n_blocks, ROWS),))
Xs, _ = norm_block(W[sel].astype(mx.float32))
print(f"codebook sample: {Xs.shape[0]:,} subvectors from {sel.size:,} rows",
      flush=True)

# kmeans++ seed on a capped subsample, then Lloyd on the full sample
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
print(f"kmeans++ seeded [{time.time()-t0:.0f}s]", flush=True)


def assign(X, C, chunk=1_000_000):
    outs = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = (x * x).sum(1, keepdims=True) - 2 * x @ C.T + (C * C).sum(1)
        outs.append(mx.argmin(d, axis=1).astype(mx.uint32))
        mx.eval(outs[-1])
    return mx.concatenate(outs)


for it in range(args.iters):
    a = assign(Xs, C)
    tot = mx.zeros((K, D)).at[a].add(Xs)
    cnt = mx.zeros((K,)).at[a].add(mx.ones((Xs.shape[0],)))
    C = mx.where((cnt > 0)[:, None], tot / mx.maximum(cnt, 1)[:, None], C)
    mx.eval(C)
    print(f"  lloyd {it+1}/{args.iters} [{time.time()-t0:.0f}s]", flush=True)
del Xs, P
mx.clear_cache()

# ---- ONE full assignment pass over the whole table, in row chunks
codes_parts, scales_parts = [], []
num = den = 0.0
for s in range(0, ROWS, args.rows_chunk):
    blk = W[s:s + args.rows_chunk].astype(mx.float32)
    Xn, sc = norm_block(blk)
    a = assign(Xn, C)
    R = (C[a].reshape(blk.shape[0], COLS // G, G) * sc.astype(mx.float32)
         ).reshape(blk.shape)
    num += float(mx.sum((R - blk) ** 2).item())
    den += float(mx.sum(blk * blk).item())
    codes_parts.append(a.reshape(blk.shape[0], COLS // D).astype(mx.uint16))
    scales_parts.append(sc.reshape(blk.shape[0], -1))
    del blk, Xn, R
    mx.clear_cache()
    if (s // args.rows_chunk) % 4 == 0:
        print(f"  assigned {s+len(codes_parts[-1]):,}/{ROWS:,} rows  "
              f"running relerr {(num/max(den,1e-12))**0.5:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

relerr = (num / max(den, 1e-12)) ** 0.5
codes = mx.concatenate(codes_parts, axis=0)
scales = mx.concatenate(scales_parts, axis=0).astype(mx.float16)
mx.save_safetensors(str(OUT / "ple.safetensors"),
                    {"codes": codes, "codebook": C.astype(mx.float16),
                     "vq_scales": scales})
bits = (K - 1).bit_length()
bpw = bits / D + 16 / G
print(f"\nPLE relerr {relerr:.4f}   {bpw:.2f} bpw   "
      f"{W.size*bpw/8/2**30:.2f} GiB vs {W.size*2/2**30:.2f} GiB bf16 / "
      f"{W.size*8.5/8/2**30:.2f} GiB at 8-bit   [{time.time()-t0:.0f}s]")
