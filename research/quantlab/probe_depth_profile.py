#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Do down_proj and gate/up have OPPOSING depth profiles? (397B session, E53)

WHY. A depth-tilted schedule only makes sense if BOTH expert projections get
harder in the SAME direction with depth. On the 397B they do not — down_proj
degrades with depth while gate_up improves, so any end-tilted schedule robs
one to pay the other, and the tail lever is dead there (E53).

That result is relerr-only, d=4, K128-vs-K2048 — it does not transfer by
assumption. This probes the same MECHANISM on another family/geometry for a
tenth of the cost of building the artifact: fit ONE tensor per (layer, proj)
and compare shallow vs deep.

    ./probe_depth_profile.py --family qwen3_5_mlx --src <bf16> --layers 3 38 \
        --dim 2 --k 512
"""
import argparse, json, math, pathlib, time
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--layers", nargs="+", type=int, required=True)
ap.add_argument("--projs", nargs="+", default=["down_proj", "gate_proj", "up_proj"])
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--k", type=int, default=512)
ap.add_argument("--experts", type=int, default=16, help="expert slice for speed")
ap.add_argument("--sample", type=int, default=1_000_000)
ap.add_argument("--group", type=int, default=64)
args = ap.parse_args()

src = (pathlib.Path(__file__).parent / "vq_397b_codes.py").read_text()
i = src.index("FAMILY = {"); depth = 0
for j in range(i + len("FAMILY = "), len(src)):
    if src[j] == "{": depth += 1
    elif src[j] == "}":
        depth -= 1
        if depth == 0: break
ns = {}; exec("FAMILY = " + src[i + len("FAMILY = "): j + 1], ns)
FAM = ns["FAMILY"][args.family]
i2 = src.index("def kmeanspp_init"); j2 = src.index("_staged = {}")
kns = {"mx": mx}; exec(src[i2:j2], kns)
kmeans = kns["kmeans"]

SRC = pathlib.Path(args.src)
smap = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
cache = {}

def load(li, proj):
    key_t, sub = FAM["proj"][proj]
    name = FAM["src_key"].format(li=li, key=key_t)
    sh = smap[name]
    if sh not in cache:
        cache.clear(); cache[sh] = mx.load(str(SRC / sh))
    T = cache[sh][name]
    if sub is not None:
        half = T.shape[1] // 2
        T = T[:, half * sub:half * (sub + 1), :]
    return T[:args.experts].astype(mx.float32)

G, D, K = args.group, args.dim, args.k
print(f"family={args.family} d={D} K={K} experts={args.experts}\n")
print(f"{'proj':10s} " + " ".join(f"L{l:<8d}" for l in args.layers) + "  direction")
rows = {}
for proj in args.projs:
    errs = []
    for li in args.layers:
        T = load(li, proj)
        in_d = T.shape[2]
        Wg = T.reshape(-1, in_d // G, G)
        sc = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
        sc = sc.astype(mx.float16).astype(mx.float32)
        X = (Wg / sc).reshape(-1, D)
        idx = mx.random.randint(0, X.shape[0], (min(args.sample, X.shape[0]),))
        mx.random.seed(0)
        C = kmeans(X[idx], K, 20).astype(mx.float16).astype(mx.float32)
        num = den = 0.0
        for s in range(0, X.shape[0], 400_000):
            xb = X[s:s + 400_000]
            d2 = mx.sum(xb * xb, axis=1, keepdims=True) - 2 * (xb @ C.T) + mx.sum(C * C, axis=1)
            rec = C[mx.argmin(d2, axis=1)]
            num += float(mx.sum((rec - xb) ** 2).item()); den += float(mx.sum(xb * xb).item())
        errs.append(math.sqrt(num / den))
        mx.clear_cache()
    rows[proj] = errs
    trend = "WORSE with depth" if errs[-1] > errs[0] else "BETTER with depth"
    print(f"{proj:10s} " + " ".join(f"{e:.4f}   " for e in errs) + f"  {trend}")

print()
# MAGNITUDE FIRST, then direction. An earlier version reported only the sign
# and called a 0.5% drift "same direction -> a depth effect is available",
# which is nonsense: if every layer fits equally well there is no gradient to
# exploit in either direction.
spans = {p: (max(r) - min(r)) / min(r) for p, r in rows.items()}
worst = max(spans.values())
print(f"largest spread across layers: {worst * 100:.1f}% of the smallest value")
if worst < 0.05:
    print("FLAT -> no depth gradient in FIT DIFFICULTY. No fit-side basis for any")
    print("depth-tilted schedule on this family/geometry.")
else:
    dirs = {p: (r[-1] > r[0]) for p, r in rows.items()}
    if len(set(dirs.values())) > 1:
        print("OPPOSING profiles -> an end-tilted schedule robs one projection to")
        print("pay the other. Tail lever dead on this family (matches E53).")
    else:
        print("SAME direction and material -> a depth effect is physically")
        print("available; an end-tilted schedule can win.")
print()
print("SCOPE: this measures how hard each tensor is to RECONSTRUCT, not how much")
print("its error COSTS at the output. A flat fit profile does not prove flat")
print("output sensitivity — only an output-scored control can settle that.")
