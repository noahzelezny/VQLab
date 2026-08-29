#!/usr/bin/env python3
"""COUNTING EXPERIMENT: on what fraction of tensors does ++ seeding sell the tail?

E107 found ++ seeding trades tail for bulk at K=256 (L30 down_proj). E108
reproduced it on L45 gate_proj and FAILED to reproduce on L15 up_proj — ++ was
uniformly better there. The artifact-level effect is a sum over 171 tensors, so
the mechanism survives if the penalty holds on a MAJORITY. That is a counting
question, and this answers it by walking layers x all three projections.

Controls, identical to probe_init_tail.py: init is the ONLY variable within a
pair (same tensor, same sampled subvectors, same seed), evaluation on HELD-OUT
experts, --reps seeds so a delta smaller than the spread is called noise.

K=256 only: that is where the penalty lives. The K=2048 arm is 5x the cost and
E107/E108 already showed the penalty shrinks or reverses there.
"""
import argparse, json, pathlib, sys
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from families import FAMILY
import expert_src

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="bf16 source model dir")
ap.add_argument("--layers", default="0,5,10,15,20,25,30,35,40,45,50,55")
ap.add_argument("--projs", default="down_proj,gate_proj,up_proj")
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--sample", type=int, default=300_000)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--family", default="qwen3_5", choices=sorted(FAMILY),
                help="source-key family (families.py). Default preserves the "
                     "original hardcoded qwen3_5 behaviour. For unfused "
                     "layouts (glm5_next) only the first --experts experts' "
                     "tensors are read at all.")
args = ap.parse_args()

SRC = pathlib.Path(args.src)
K, D, G = args.k, args.dim, args.group
QS = [0, 50, 90, 99, 99.9, 100]
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
FAM = FAMILY[args.family]
_shard_cache = {}


def load_tensor(layer, proj):
    T = expert_src.load_expert_stack(SRC, src_idx, FAM, layer, proj,
                                     experts=args.experts,
                                     shard_cache=_shard_cache)
    with mx.stream(mx.cpu):
        T = T.astype(mx.float32)
        mx.eval(T)
    return T


def lloyd(X, C, iters):
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        a = mx.argmin(mx.sum(X * X, axis=1, keepdims=True) - 2 * (X @ C.T) + cn[None, :], axis=1)
        num = mx.zeros(C.shape).at[a].add(X)
        den = mx.zeros((C.shape[0],)).at[a].add(mx.ones((X.shape[0],), dtype=mx.float32))
        C = mx.where(den[:, None] > 0, num / mx.maximum(den[:, None], 1e-20), C)
        mx.eval(C)
    return C


def seed_pp(X, k):
    cap = min(X.shape[0], 200_000)
    Xc = X[mx.random.randint(0, X.shape[0], (cap,))]
    seeds = [Xc[mx.random.randint(0, cap, (1,))[0]]]
    d2 = mx.sum((Xc - seeds[0]) ** 2, axis=1)
    for _ in range(k - 1):
        tot = float(mx.sum(d2).item())
        if tot <= 0:
            seeds.append(Xc[mx.random.randint(0, cap, (1,))[0]])
        else:
            cdf = mx.cumsum(d2 / tot)
            j = int(mx.sum(cdf < mx.random.uniform()).item())
            seeds.append(Xc[min(j, cap - 1)])
        d2 = mx.minimum(d2, mx.sum((Xc - seeds[-1]) ** 2, axis=1))
        mx.eval(d2)
    return mx.stack(seeds)


print(f"K={K} d{D}  experts={args.experts} (half train / half held-out)  "
      f"reps={args.reps}\n")
print(f"{'layer':>5} {'proj':>10} {'d(99-99.9)':>12} {'d(99.9-100)':>13} "
      f"{'d(MEAN)':>10}  verdict")
print("-" * 66)

results = []
for layer in [int(x) for x in args.layers.split(",")]:
    for proj in args.projs.split(","):
        T = load_tensor(layer, proj)
        n_exp, out_d, in_d = T.shape
        n_train = max(1, n_exp // 2)
        Wg = T.reshape(-1, in_d // G, G)
        scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
        scale = scale.astype(mx.float16).astype(mx.float32)
        Xn = (Wg / scale).reshape(-1, D)
        per = Xn.shape[0] // n_exp
        Xtr = Xn[:n_train * per]
        a_np = np.array(T[n_train:]).ravel()
        edges = np.percentile(np.abs(a_np), QS)
        mask = [(np.abs(a_np) >= edges[i]) &
                (np.abs(a_np) <= edges[i+1] if i == len(QS)-2 else np.abs(a_np) < edges[i+1])
                for i in range(len(QS)-1)]
        denom = [max(float(np.sqrt((a_np[m]**2).mean())), 1e-12) for m in mask]

        def ev(C):
            cn = mx.sum(C*C, axis=1)
            step = max(1, int(4e8 // C.shape[0]))
            parts = []
            for s0 in range(0, Xn.shape[0], step):
                x = Xn[s0:s0+step]
                parts.append(mx.argmin(mx.sum(x*x, axis=1, keepdims=True)
                                       - 2*(x @ C.T) + cn[None, :], axis=1))
                mx.eval(parts[-1])
            a = mx.concatenate(parts)
            R = (C[a].reshape(-1, in_d//G, G) * scale).reshape(T.shape)
            mx.eval(R)
            err = np.array(R)[n_train:].ravel() - a_np
            out = [float(np.sqrt((err[m]**2).mean()))/d for m, d in zip(mask, denom)]
            out.append(float(np.sqrt((err**2).sum()/(a_np**2).sum())))
            return out

        ds = []
        for r in range(args.reps):
            mx.random.seed(2000 + r)
            idx = mx.random.randint(0, Xtr.shape[0], (min(args.sample, Xtr.shape[0]),))
            Xs = Xtr[idx]; mx.eval(Xs)
            mx.random.seed(2000 + r)
            rnd = ev(lloyd(Xs, Xs[mx.random.randint(0, Xs.shape[0], (K,))], args.iters))
            mx.random.seed(2000 + r)
            pp = ev(lloyd(Xs, seed_pp(Xs, K), args.iters))
            ds.append([p - q for p, q in zip(pp, rnd)])
            mx.clear_cache()
        A = np.array(ds)
        m, sd = A.mean(0), A.std(0)
        # PENALTY only if BOTH tail buckets are worse by more than the spread.
        pen = (m[3] > max(sd[3], 1e-9)) and (m[4] > max(sd[4], 1e-9))
        uni = (m[3] < -max(sd[3], 1e-9)) and (m[4] < -max(sd[4], 1e-9))
        v = "PENALTY" if pen else ("++ better" if uni else "mixed/noise")
        results.append((layer, proj, m[3], m[4], m[5], v))
        print(f"{layer:5d} {proj:>10} {m[3]:>+12.5f} {m[4]:>+13.5f} {m[5]:>+10.5f}  {v}",
              flush=True)
        del T, Xn, Xtr, Wg, scale
        mx.clear_cache()

n = len(results)
npen = sum(1 for r in results if r[5] == "PENALTY")
nuni = sum(1 for r in results if r[5] == "++ better")
print("-" * 66)
print(f"\nTENSORS: {n}   PENALTY: {npen} ({100*npen/n:.0f}%)   "
      f"++ better: {nuni} ({100*nuni/n:.0f}%)   other: {n-npen-nuni}")
print("\nby projection:")
for p in args.projs.split(","):
    sub = [r for r in results if r[1] == p]
    if sub:
        k = sum(1 for r in sub if r[5] == "PENALTY")
        print(f"  {p:>10}: {k}/{len(sub)} penalty")
print("\nby depth:")
for lo, hi in ((0, 14), (15, 34), (35, 56)):
    sub = [r for r in results if lo <= r[0] <= hi]
    if sub:
        k = sum(1 for r in sub if r[5] == "PENALTY")
        print(f"  L{lo:02d}-L{hi:02d}: {k}/{len(sub)} penalty")
print("\nA majority PENALTY supports 'ple++ seeding causes the K256 regression'.")
print("A minority means the artifact-level effect needs another explanation.")
