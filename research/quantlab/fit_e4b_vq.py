#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""VQ-fit the DENSE mlp trio of gemma-4-e4b-it from its bf16 source.

WHY A SEPARATE FITTER. vq_397b_codes.py targets modules that a prepared
struct BASE marked as 2-bit placeholders (is_vq_target checks bits==2);
dense e4b has no such base. This is the minimum honest experiment for
"does VQ transfer to a small dense model": fit gate/up/down_proj
(6.15 GiB = 42% of bf16 bytes) at d2-K2048 with the SAME contract as the
main fitter — group-64 max-abs fp16 scales along IN, kmeans++ init,
scatter-add Lloyd iterations — and write an artifact that
verify_artifact.py (--family gemma4_e4b) can decode independently.

Known-healthy reference for the geometry: 26b d2-K2048 mean relerr ~0.032.
A small dense model has less redundancy, so worse is EXPECTED; the
question is how much. This artifact is NOT runnable (no model.py splice) —
fit quality first, runtime integration only if the numbers earn it.

    ./fit_e4b_vq.py --src <bf16 snapshot> --out <dir> [--k 2048 --dim 2]
"""
import argparse
import json
import pathlib
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=2048)
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--layers", default="0-41")
args = ap.parse_args()

SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)
K, D, G = args.k, args.dim, args.group
LO, HI = (int(x) for x in args.layers.split("-"))
OUT.mkdir(parents=True, exist_ok=True)
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
PROJ = ["gate_proj", "up_proj", "down_proj"]
KEY = "language_model.model.layers.{li}.mlp.{key}.weight"


def normalize(T):
    """group-64 max-abs scales along IN, mirrored from the main fitter."""
    out_d, in_d = T.shape
    Wg = T.reshape(out_d, in_d // G, G)
    scale = mx.maximum(mx.abs(Wg).max(axis=-1, keepdims=True), 1e-8)
    Xn = (Wg / scale).reshape(-1, D)
    return Xn, scale.astype(mx.float16)


def kmeanspp(X, k, cap=200_000):
    n = X.shape[0]
    idx = mx.random.randint(0, n, (min(cap, n),))
    P = X[idx]
    C = P[mx.random.randint(0, P.shape[0], (1,))]
    d2 = mx.sum((P - C[0]) ** 2, axis=1)
    for _ in range(k - 1):
        probs = d2 / mx.maximum(d2.sum(), 1e-12)
        r = float(mx.random.uniform().item())
        cum = mx.cumsum(probs)
        j = int(mx.argmax(cum >= r).item())
        C = mx.concatenate([C, P[j:j + 1]], axis=0)
        d2 = mx.minimum(d2, mx.sum((P - P[j]) ** 2, axis=1))
        mx.eval(C, d2)
    return C


def assign(X, C, chunk=2_000_000):
    outs = []
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = (x * x).sum(1, keepdims=True) - 2 * x @ C.T + (C * C).sum(1)
        outs.append(mx.argmin(d, axis=1).astype(mx.uint32))
        mx.eval(outs[-1])
    return mx.concatenate(outs)


def fit_tensor(T):
    Xn, scale = normalize(T.astype(mx.float32))
    C = kmeanspp(Xn, K)
    for _ in range(args.iters):
        a = assign(Xn, C)
        oh = mx.zeros((K, D)).at[a].add(Xn)
        cnt = mx.zeros((K,)).at[a].add(mx.ones((Xn.shape[0],)))
        C = mx.where((cnt > 0)[:, None], oh / mx.maximum(cnt, 1)[:, None], C)
        mx.eval(C)
    a = assign(Xn, C)
    R = C[a].reshape(T.shape[0], T.shape[1] // G, G) * scale.astype(mx.float32)
    R = R.reshape(T.shape)
    num = float(mx.sum((R - T) ** 2).item())
    den = float(mx.sum(T.astype(mx.float32) ** 2).item())
    relerr = (num / max(den, 1e-12)) ** 0.5
    codes = a.reshape(1, T.shape[0], T.shape[1] // D).astype(mx.uint16)
    return codes, C.astype(mx.float16), scale.reshape(1, T.shape[0], -1), relerr


weights, vq_modules, report = {}, {}, []
t0 = time.time()
for li in range(LO, HI + 1):
    for proj in PROJ:
        sk = KEY.format(li=li, key=proj)
        sh = src_idx[sk]
        T = mx.load(str(SRC / sh))[sk]
        codes, cb, sc, rel = fit_tensor(T)
        mod = f"language_model.model.layers.{li}.mlp.{proj}"
        weights[mod + ".codes"] = codes
        weights[mod + ".codebook"] = cb
        weights[mod + ".vq_scales"] = sc
        vq_modules[mod] = {"experts": 1, "out": int(T.shape[0]),
                           "in": int(T.shape[1]), "k": K, "dim": D, "group": G}
        report.append(rel)
        print(f"L{li:02d} {proj:10s} relerr {rel:.4f}  "
              f"[{time.time()-t0:6.0f}s]", flush=True)
        del T
        mx.clear_cache()

mx.save_safetensors(str(OUT / "model-00001-of-00001.safetensors"), weights)
idx = {"metadata": {}, "weight_map":
       {k: "model-00001-of-00001.safetensors" for k in weights}}
json.dump(idx, open(OUT / "model.safetensors.index.json", "w"), indent=1)
cfg = json.load(open(SRC / "config.json"))
cfg["vq_modules"] = vq_modules
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)
print(f"\nfit {len(report)} tensors, mean relerr "
      f"{sum(report)/len(report):.4f}, worst {max(report):.4f}, "
      f"{time.time()-t0:.0f}s total -> {OUT}")
