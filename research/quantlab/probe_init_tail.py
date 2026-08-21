#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Does k-means++ SEEDING cause the bulk-for-tail trade, and only at low K?

WHY THIS EXISTS. Commit 689e03c (2026-08-18) added ++ seeding, described as
"robustness, NOT a quality lever". It is one of the two changes that made the
K2048 (144 GiB) and K8192 refits BETTER and the K256 (111 GiB) refit WORSE
(E92/E101). E102 showed the K256 loss is a bulk-for-tail trade. A first, noisy
look suggested ++ reproduces that trade.

PRE-REGISTERED PREDICTION (written before running):
  At K=256   ++ is WORSE than random on the 99-99.9 and 99.9-100 buckets.
  At K=2048  that penalty is ABSENT or much smaller.
Both halves are required. If ++ hurts the tail at BOTH K, the effect is real
but does NOT explain the K-dependence, and the mechanism stays open.

CONTROLS. Same tensor, same sampled subvectors, same Lloyd iterations, same
seed per pair -- init is the ONLY difference within a pair. Repeated over
--reps seeds so the spread is visible; a delta smaller than the spread is
noise, not a finding. Evaluation is on HELD-OUT experts (E106).
"""
import argparse, json, pathlib
import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/"
                                 "Qwen--Qwen3.5-397B-A17B-bf16")
ap.add_argument("--layer", type=int, default=30)
ap.add_argument("--proj", default="down_proj")
ap.add_argument("--ks", default="256,2048")
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--experts", type=int, default=16)
ap.add_argument("--sample", type=int, default=400_000)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

SRC = pathlib.Path(args.src)
D, G = args.dim, args.group
QS = [0, 50, 90, 99, 99.9, 100]
KS = [int(x) for x in args.ks.split(",")]

src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
KEY, HALF = {"down_proj": ("down_proj", None),
             "gate_proj": ("gate_up_proj", 0),
             "up_proj": ("gate_up_proj", 1)}[args.proj]
sk = f"model.language_model.layers.{args.layer}.mlp.experts.{KEY}"
with mx.stream(mx.cpu):
    T = mx.load(str(SRC / src_idx[sk]))[sk]
    if HALF is not None:
        mid = T.shape[1] // 2
        T = T[:, :mid, :] if HALF == 0 else T[:, mid:, :]
    T = T[:args.experts].astype(mx.float32)
    mx.eval(T)
n_exp, out_d, in_d = T.shape
N_TRAIN = max(1, n_exp // 2)
print(f"{sk}  [{n_exp} experts, {out_d}, {in_d}]  d{D}  "
      f"train 0-{N_TRAIN-1} / held-out {N_TRAIN}-{n_exp-1}  reps={args.reps}",
      flush=True)

Wg = T.reshape(-1, in_d // G, G)
scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
scale = scale.astype(mx.float16).astype(mx.float32)
Xn = (Wg / scale).reshape(-1, D)
PER = Xn.shape[0] // n_exp
Xtr = Xn[:N_TRAIN * PER]

a_np = np.array(T[N_TRAIN:]).ravel()
edges = np.percentile(np.abs(a_np), QS)
mask = [(np.abs(a_np) >= edges[i]) &
        (np.abs(a_np) <= edges[i+1] if i == len(QS)-2 else np.abs(a_np) < edges[i+1])
        for i in range(len(QS)-1)]
denom = [max(float(np.sqrt((a_np[m]**2).mean())), 1e-12) for m in mask]


def lloyd(X, C, iters):
    for _ in range(iters):
        cn = mx.sum(C*C, axis=1)
        a = mx.argmin(mx.sum(X*X, axis=1, keepdims=True) - 2*(X @ C.T) + cn[None, :], axis=1)
        num = mx.zeros(C.shape).at[a].add(X)
        den = mx.zeros((C.shape[0],)).at[a].add(mx.ones((X.shape[0],), dtype=mx.float32))
        C = mx.where(den[:, None] > 0, num/mx.maximum(den[:, None], 1e-20), C)
        mx.eval(C)
    return C


def seed_pp(X, k):
    cap = min(X.shape[0], 200_000)
    Xc = X[mx.random.randint(0, X.shape[0], (cap,))]
    seeds = [Xc[mx.random.randint(0, cap, (1,))[0]]]
    d2 = mx.sum((Xc - seeds[0])**2, axis=1)
    for _ in range(k-1):
        tot = float(mx.sum(d2).item())
        if tot <= 0:
            seeds.append(Xc[mx.random.randint(0, cap, (1,))[0]])
        else:
            cdf = mx.cumsum(d2/tot)
            j = int(mx.sum(cdf < mx.random.uniform()).item())
            seeds.append(Xc[min(j, cap-1)])
        d2 = mx.minimum(d2, mx.sum((Xc - seeds[-1])**2, axis=1))
        mx.eval(d2)
    return mx.stack(seeds)


def evaluate(C):
    # CHUNKED: Xn @ C.T at K=2048 over ~17M subvectors is a 137 GB buffer and
    # Metal caps at 62 GB. Assignment is row-independent, so chunk it.
    cn = mx.sum(C*C, axis=1)
    STEP = max(1, int(4e8 // C.shape[0]))
    parts = []
    for s0 in range(0, Xn.shape[0], STEP):
        x = Xn[s0:s0+STEP]
        parts.append(mx.argmin(mx.sum(x*x, axis=1, keepdims=True)
                               - 2*(x @ C.T) + cn[None, :], axis=1))
        mx.eval(parts[-1])
    a = mx.concatenate(parts)
    R = (C[a].reshape(-1, in_d//G, G) * scale).reshape(T.shape)
    mx.eval(R)
    del parts
    err = np.array(R)[N_TRAIN:].ravel() - a_np
    out = [float(np.sqrt((err[m]**2).mean()))/d for m, d in zip(mask, denom)]
    out.append(float(np.sqrt((err**2).sum()/(a_np**2).sum())))
    return out


hdr = [f"{QS[i]:g}-{QS[i+1]:g}" for i in range(len(QS)-1)] + ["MEAN"]
for K in KS:
    deltas = []
    for r in range(args.reps):
        mx.random.seed(1000 + r)
        idx = mx.random.randint(0, Xtr.shape[0], (min(args.sample, Xtr.shape[0]),))
        Xs = Xtr[idx]
        mx.eval(Xs)
        mx.random.seed(1000 + r)                      # same seed for both arms
        rnd = evaluate(lloyd(Xs, Xs[mx.random.randint(0, Xs.shape[0], (K,))], args.iters))
        mx.random.seed(1000 + r)
        pp = evaluate(lloyd(Xs, seed_pp(Xs, K), args.iters))
        deltas.append([p - q for p, q in zip(pp, rnd)])
        print(f"  K{K} rep{r}: random MEAN {rnd[-1]:.5f}   ++ MEAN {pp[-1]:.5f}", flush=True)
        mx.clear_cache()
    A = np.array(deltas)
    print(f"\n  K={K}  delta (++ minus random); POSITIVE = ++ WORSE")
    print("    " + "".join(f"{h:>13}" for h in hdr))
    print("    " + "".join(f"{v:>+13.5f}" for v in A.mean(0)))
    print("    " + "".join(f"{'±'+format(v,'.5f'):>13}" for v in A.std(0)) + "   (spread over reps)\n")

print("PREDICTION: ++ WORSE on the two tail buckets at K=256, and that penalty")
print("absent or much smaller at K=2048. A delta inside the spread is noise.")
