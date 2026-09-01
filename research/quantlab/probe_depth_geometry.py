#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""WHY does the ++ seeding penalty flip sign with depth? (E109 open question)

HYPOTHESIS. k-means++ seeds proportional to squared distance, so it places
centroids where points are FAR from what is already covered. What that buys
depends on the geometry k-means actually sees — the group-64-NORMALIZED
subvectors:

  - If the normalized cloud is HEAVY-TAILED (a few subvectors far from the
    bulk), distance-proportional seeding lands centroids ON that tail. Tail
    served, and ++ helps everywhere.
  - If the cloud is HOMOGENEOUS (no distinguished far points), ++ degenerates
    to a better-spread cover of the bulk — a better local optimum of the
    AVERAGE objective, which at scarce K is bought BY the tail.

PREDICTION, pre-registered: shallow layers (L00-L10, where ++ helps) show a
HEAVIER-TAILED normalized subvector-norm distribution than body layers
(L20-L55, where ++ sells the tail). Metrics on ||x|| of normalized subvectors:
excess kurtosis, top-1% energy share, and CV. FALSIFIED if shallow and body
are indistinguishable, or if body is the heavier-tailed one.
"""
import argparse, json, pathlib
import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/"
                                 "Qwen--Qwen3.5-397B-A17B-bf16")
ap.add_argument("--layers", default="0,5,10,15,20,25,30,35,40,45,50,55")
ap.add_argument("--projs", default="down_proj,gate_proj,up_proj")
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--experts", type=int, default=8)
args = ap.parse_args()

SRC = pathlib.Path(args.src)
D, G = args.dim, args.group
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
PROJ_KEY = {"down_proj": ("down_proj", None),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1)}
_cache = {}

print(f"normalized-subvector geometry (what k-means actually sees), d={D}\n")
print(f"{'layer':>5} {'proj':>10} {'exc.kurt':>10} {'top1%%energy':>12} "
      f"{'CV':>8} {'p99/p50':>9}")
print("-" * 60)
rows = []
for layer in [int(x) for x in args.layers.split(",")]:
    for proj in args.projs.split(","):
        key, half = PROJ_KEY[proj]
        sk = f"model.language_model.layers.{layer}.mlp.experts.{key}"
        sh = src_idx[sk]
        if sh not in _cache:
            _cache.clear()
            with mx.stream(mx.cpu):
                _cache[sh] = mx.load(str(SRC / sh))
        with mx.stream(mx.cpu):
            T = _cache[sh][sk]
            if half is not None:
                mid = T.shape[1] // 2
                T = T[:, :mid, :] if half == 0 else T[:, mid:, :]
            T = T[:args.experts].astype(mx.float32)
            mx.eval(T)
        in_d = T.shape[2]
        Wg = T.reshape(-1, in_d // G, G)
        scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
        scale = scale.astype(mx.float16).astype(mx.float32)
        Xn = (Wg / scale).reshape(-1, D)
        nrm = np.array(mx.sqrt(mx.sum(Xn * Xn, axis=1)))
        del T, Wg, Xn
        mx.clear_cache()
        mu, sd = nrm.mean(), nrm.std()
        kurt = float(((nrm - mu) ** 4).mean() / max(sd ** 4, 1e-20) - 3.0)
        e = nrm ** 2
        thr = np.percentile(nrm, 99)
        top1 = float(e[nrm >= thr].sum() / max(e.sum(), 1e-20))
        cv = float(sd / max(mu, 1e-20))
        ratio = float(thr / max(np.percentile(nrm, 50), 1e-20))
        rows.append((layer, proj, kurt, top1, cv, ratio))
        print(f"{layer:5d} {proj:>10} {kurt:>10.3f} {top1:>12.4f} "
              f"{cv:>8.4f} {ratio:>9.3f}", flush=True)

print("-" * 60)
sh_rows = [r for r in rows if r[0] <= 10]
bd_rows = [r for r in rows if r[0] >= 20]
print("\n                 exc.kurt   top1%energy      CV     p99/p50")
for name, s in (("SHALLOW L00-L10 (++ helps)", sh_rows),
                ("BODY    L20-L55 (++ hurts)", bd_rows)):
    a = np.array([[r[2], r[3], r[4], r[5]] for r in s])
    print(f"{name}  {a[:,0].mean():8.3f} {a[:,1].mean():12.4f} "
          f"{a[:,2].mean():8.4f} {a[:,3].mean():9.3f}")
print("\nPREDICTION: shallow HEAVIER-TAILED (higher kurtosis / top1% energy /")
print("CV / p99-p50 ratio) than body. Falsified if indistinguishable or reversed.")
