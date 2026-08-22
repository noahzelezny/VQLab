#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E120 — is the K256 fitter regression float SUMMATION ORDER?

Isolates ONE Lloyd centroid update, three ways, from a SHARED fixed
assignment so accumulation is the only variable:

    (a) scatter-add fp32       oh_sum.at[a].add(X)          [HEAD, a9f5c5c]
    (b) one-hot matmul fp32    oh.T @ X, chunked            [pre-a9f5c5c]
    (c) float64 reference      numpy, exact-enough truth

Hypothesis (pre-registered): serial fp32 accumulation loses precision with
the number of terms per bin, which is n/K — so error scales as 1/K, and the
damage is worst exactly where the regression is (K256), absent where it is
not (K2048).

CONFIRMED  if ||a-c|| > ||b-c|| AND the gap is larger at K256 than K2048.
FALSIFIED  if a and b agree to fp32 noise at both K.
"""
import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16")
ap.add_argument("--layer", type=int, default=30)
ap.add_argument("--proj", default="down_proj")
ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--ks", default="256,2048")
ap.add_argument("--out", default="results_e120_accum.json")
args = ap.parse_args()

SRC = pathlib.Path(args.src)
idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
# 397B source is HF layout: key is mlp.experts.<key> with NO .weight suffix,
# and gate/up live FUSED in one [E, 2I, H] stack taken as halves along OUT.
# (The first version of this probe guessed `switch_mlp.<proj>.weight` — the
# ARTIFACT convention, not the SOURCE one — and died on KeyError before doing
# any work. Read FAMILY["qwen3_5"] in vq_397b_codes.py, do not guess.)
_SUB = {"gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "down_proj": ("down_proj", None)}
_key, _half = _SUB[args.proj]
name = f"model.language_model.layers.{args.layer}.mlp.experts.{_key}"
T = mx.load(str(SRC / idx[name]))[name][:args.experts].astype(mx.float32)
if _half is not None:
    _h = T.shape[1] // 2
    T = T[:, _h * _half:_h * (_half + 1), :]
print(f"{name}  experts[:{args.experts}] {T.shape}", flush=True)

# normalize exactly as the fitter does: fp16 max-abs per group of G along `in`
E, OUT, IN = T.shape
G = args.group
Tg = T.reshape(E, OUT, IN // G, G)
sc = mx.max(mx.abs(Tg), axis=-1, keepdims=True)
sc = mx.maximum(sc.astype(mx.float16).astype(mx.float32), 1e-8)
X = (Tg / sc).reshape(-1, args.dim)
mx.eval(X)
n = X.shape[0]
Xnp64 = np.array(X, copy=True).astype(np.float64)
print(f"subvectors n={n:,}  d={args.dim}", flush=True)

rows = []
for K in [int(x) for x in args.ks.split(",")]:
    mx.random.seed(0)
    C = X[mx.random.randint(0, n, (K,))]
    # ONE shared assignment — identical input to all three accumulators
    cn = mx.sum(C * C, axis=1)
    step = max(50_000, int(4e8 // K))
    parts = []
    for s0 in range(0, n, step):
        xb = X[s0:s0 + step]
        parts.append(mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                               - 2 * (xb @ C.T) + cn[None, :], axis=1))
        mx.eval(parts[-1])
    a_idx = mx.concatenate(parts)
    mx.eval(a_idx)
    anp = np.array(a_idx)

    # (a) scatter-add fp32  [HEAD]
    sa = mx.zeros((K, args.dim))
    ca = mx.zeros((K,))
    sa = sa.at[a_idx].add(X)
    ca = ca.at[a_idx].add(mx.ones((n,), dtype=mx.float32))
    mx.eval(sa, ca)

    # (b) one-hot matmul fp32, chunked as pre-a9f5c5c  [old path]
    ob = mx.zeros((K, args.dim))
    cb = mx.zeros((K,))
    oh_chunk = max(50_000, int(5e8 / K))
    for s0 in range(0, n, oh_chunk):
        ab = a_idx[s0:s0 + oh_chunk]
        oh = (ab[:, None] == mx.arange(K)[None, :]).astype(mx.float32)
        ob = ob + oh.T @ X[s0:s0 + oh_chunk]
        cb = cb + mx.sum(oh, axis=0)
        mx.eval(ob, cb)
        del oh

    # (c) float64 reference
    oc = np.zeros((K, args.dim), dtype=np.float64)
    cc = np.zeros((K,), dtype=np.float64)
    np.add.at(oc, anp, Xnp64)
    np.add.at(cc, anp, 1.0)

    anp_sum = np.array(sa).astype(np.float64)
    bnp_sum = np.array(ob).astype(np.float64)
    live = cc > 0
    # centroid = sum/count; compare the CENTROIDS, which is what k-means uses
    ca_c = anp_sum[live] / np.maximum(np.array(ca).astype(np.float64)[live], 1)
    cb_c = bnp_sum[live] / np.maximum(np.array(cb).astype(np.float64)[live], 1)
    cc_c = oc[live] / cc[live]

    def relerr(P):
        return float(np.linalg.norm(P - cc_c) / np.linalg.norm(cc_c))

    r = {"K": K, "n": n, "live_bins": int(live.sum()),
         "terms_per_bin": float(n / K),
         "scatter_add_vs_f64": relerr(ca_c),
         "onehot_matmul_vs_f64": relerr(cb_c),
         "count_mismatch_a": int((np.array(ca)[live] != cc[live]).sum()),
         "count_mismatch_b": int((np.array(cb)[live] != cc[live]).sum())}
    r["ratio_a_over_b"] = (r["scatter_add_vs_f64"] /
                           max(r["onehot_matmul_vs_f64"], 1e-30))
    rows.append(r)
    print(f"K={K:5d}  terms/bin {r['terms_per_bin']:8.1f}  "
          f"scatter-add {r['scatter_add_vs_f64']:.3e}  "
          f"one-hot {r['onehot_matmul_vs_f64']:.3e}  "
          f"ratio {r['ratio_a_over_b']:.2f}x", flush=True)

json.dump({"tensor": name, "experts": args.experts, "dim": args.dim,
           "rows": rows}, open(args.out, "w"), indent=1)
print(f"\n-> {args.out}")
if len(rows) == 2:
    lo, hi = rows[0], rows[1]
    print(f"\nRATIO at K{lo['K']}: {lo['ratio_a_over_b']:.2f}x   "
          f"at K{hi['K']}: {hi['ratio_a_over_b']:.2f}x")
    print("CONFIRMED shape requires ratio > 1 at both AND larger at low K.")
