#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""STAGE-0 SCREEN for --tail-weight-pow (patches/tail-weighted-kmeans.patch).

E102 measured WHY the K256 refit loses: k-means minimizes AVERAGE distortion,
so with scarce centroids it wins the bulk and abandons the |w| 99-100th
percentile, which is what the output notices. The patch adds a weighted
objective, w = (scale_g * ||x||_2)^p per training subvector, p=0 being today's
fit exactly.

This probe answers ONE cheap question before anyone spends a 397B fit:
**does p>0 actually move the tail bucket, and how much bulk does it cost?**
It fits two codebooks on the SAME sampled subvectors, differing only in p, and
reports the E102 bucket table for each against the bf16 source.

WHAT IT IS NOT. It is not an answer about the model. Law 6 / law 11: weight
space does not rank output damage, and this probe reads weight space only. Its
job is to pick p and to catch a no-op — a p that does not move the 99-100
bucket cannot possibly fix E92, so the fit is not worth launching. A p that
DOES move it still has to be scored end-to-end.

HELD-OUT BY DEFAULT (fixed 2026-08-21). The first version fit the codebook on
subvectors from --experts experts and measured reconstruction on THOSE SAME
experts. That is an in-sample number, and magnitude weighting games it: the
weighting concentrates centroids on the tail OF THE SAMPLE, which then
reconstructs well and generalizes worse. Measured consequence — L00 down_proj,
p=4: the in-sample screen said MEAN relerr 0.1586 (BETTER than unweighted
0.1925), the real 512-expert fit came out at 0.7111 and aborted the run. A
3.7x miss in the flattering direction, and it cost a launched fit.

The screen now splits experts into TRAIN and HELD-OUT halves: the codebook is
fit only on train experts and scored only on held-out ones, which is what the
real fitter does across all 512. Both columns are printed so the in-sample gap
stays visible rather than being something you have to remember.

    ./probe_tailweight.py --layer 30 --proj down_proj --k 256 --dim 4 \
        --pows 0,2,4,8
"""
import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/"
                                 "Qwen--Qwen3.5-397B-A17B-bf16")
ap.add_argument("--layer", type=int, default=30)
ap.add_argument("--proj", default="down_proj",
                choices=["down_proj", "gate_proj", "up_proj"])
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--experts", type=int, default=16,
                help="experts to LOAD; split half train / half held-out")
ap.add_argument("--sample", type=int, default=2_000_000)
ap.add_argument("--pows", default="0,2,4,8")
ap.add_argument("--init", choices=("kmeans++", "random"), default="kmeans++",
                help="MIRROR THE FITTER (kmeans++, default) or isolate the "
                     "centroid-update change (random). The patch weights BOTH "
                     "the update and the ++ seeding; a screen that only "
                     "randomises init tests half of it. Measured 2026-08-21: "
                     "random-init screen said p=4 IMPROVES L00 down_proj "
                     "(0.169 vs 0.204) while the real ++-seeded fit came out "
                     "at 0.7111 and aborted.")
args = ap.parse_args()

SRC = pathlib.Path(args.src)
K, D, G = args.k, args.dim, args.group
QS = [0, 50, 90, 99, 99.9, 100]          # the E102 buckets, unchanged
POWS = [float(x) for x in args.pows.split(",")]
assert G % D == 0, "a subvector must sit inside one scale group"

# --- source read, mirroring vq_397b_codes.load_src_expert (qwen3_5 family:
# gate/up live fused in a [E, 2I, H] stack and are taken as OUT-axis halves).
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
KEY, HALF = {"down_proj": ("down_proj", None),
             "gate_proj": ("gate_up_proj", 0),
             "up_proj": ("gate_up_proj", 1)}[args.proj]
sk = f"model.language_model.layers.{args.layer}.mlp.experts.{KEY}"
with mx.stream(mx.cpu):                  # SMB read inside a GPU command
    T = mx.load(str(SRC / src_idx[sk]))[sk]   # buffer = watchdog kill
    if HALF is not None:
        mid = T.shape[1] // 2
        T = T[:, :mid, :] if HALF == 0 else T[:, mid:, :]
    T = T[:args.experts].astype(mx.float32)
    mx.eval(T)
n_exp, out_d, in_d = T.shape
N_TRAIN = max(1, n_exp // 2)
print(f"{sk}  [{n_exp} of full E, {out_d}, {in_d}]  K{K} d{D}", flush=True)
print(f"  train experts 0-{N_TRAIN-1}   HELD-OUT experts {N_TRAIN}-{n_exp-1}",
      flush=True)


def normalize(blk):
    """group-64 max-abs along IN, with the SHIPPED fp16 rounding."""
    Wg = blk.reshape(-1, in_d // G, G)
    scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
    scale = scale.astype(mx.float16).astype(mx.float32)
    return (Wg / scale).reshape(-1, D), scale


def subvector_scales(scale):
    """per-subvector scale. Flatten order is (rows, groups, G//D, D), so each
    group's scale repeats G//D times CONSECUTIVELY. Verified numerically:
    Xn * ssub[:, None] reconstructs the block."""
    return mx.repeat(scale.reshape(-1, in_d // G), G // D, axis=1).reshape(-1)


def kmeans(X, W, iters):
    """weighted Lloyd, random init. Assignment does not see W (a positive
    scalar cannot change an argmin); only the centroid update does. Init is
    random rather than ++ so that the ONLY difference between arms is p."""
    n = X.shape[0]
    if args.init == "random":
        C = X[mx.random.randint(0, n, (K,))]
    else:
        # weighted k-means++, mirroring kmeanspp_init in vq_397b_codes.py:
        # each new centre is drawn with probability proportional to w * d^2.
        cap = min(n, 200_000)
        Xc = X[mx.random.randint(0, n, (cap,))] if n > cap else X
        # Unweighted, mirroring the fitter after E106: weighting the seeding
        # as well as the update compounds and starves the bulk.
        Wc = None
        seeds = [Xc[mx.random.randint(0, Xc.shape[0], (1,))[0]]]
        d2 = mx.sum((Xc - seeds[0]) ** 2, axis=1)
        for _ in range(K - 1):
            pr = d2 if Wc is None else d2 * Wc
            pr = mx.maximum(pr, 0)
            tot = float(mx.sum(pr).item())
            if tot <= 0:
                seeds.append(Xc[mx.random.randint(0, Xc.shape[0], (1,))[0]])
            else:
                cdf = mx.cumsum(pr / tot)
                j = int(mx.sum(cdf < mx.random.uniform()).item())
                seeds.append(Xc[min(j, Xc.shape[0] - 1)])
            d2 = mx.minimum(d2, mx.sum((Xc - seeds[-1]) ** 2, axis=1))
            mx.eval(d2)
        C = mx.stack(seeds)
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        a = mx.argmin(mx.sum(X * X, axis=1, keepdims=True)
                      - 2 * (X @ C.T) + cn[None, :], axis=1)
        num = mx.zeros((K, D)).at[a].add(X if W is None else X * W[:, None])
        den = mx.zeros((K,)).at[a].add(
            mx.ones((n,), dtype=mx.float32) if W is None else W)
        C = mx.where(den[:, None] > 0, num / mx.maximum(den[:, None], 1e-20), C)
        mx.eval(C)
    return C


Xn, scale = normalize(T)
ssub = subvector_scales(scale)

# TRAIN/HELD-OUT split along the EXPERT axis. Xn is row-major over experts, so
# the first N_TRAIN experts own the first N_TRAIN*PER subvector rows. Sampling
# the fit ONLY from those, and scoring ONLY on the rest, is what makes this
# screen predictive of a 512-expert fit; sampling across all of them is what
# made the first version flatter magnitude weighting by 3.7x.
PER = Xn.shape[0] // n_exp
CUT = N_TRAIN * PER
Xtr, Str = Xn[:CUT], ssub[:CUT]
idx = mx.random.randint(0, Xtr.shape[0], (min(args.sample, Xtr.shape[0]),))
Xs, Ss = Xtr[idx], Str[idx]
mag = Ss * mx.sqrt(mx.sum(Xs ** 2, axis=1))       # weight-space subvector norm
mx.eval(Xs, mag)

# held-out weights, in the same row-major order, for the bucket masks
a_np = np.array(T[N_TRAIN:]).ravel()
a_in = np.array(T[:N_TRAIN]).ravel()
edges = np.percentile(np.abs(a_np), QS)
mask = [(np.abs(a_np) >= edges[i]) &
        (np.abs(a_np) <= edges[i + 1] if i == len(QS) - 2
         else np.abs(a_np) < edges[i + 1]) for i in range(len(QS) - 1)]
denom = [max(float(np.sqrt((a_np[m] ** 2).mean())), 1e-12) for m in mask]

rows = {}
for p in POWS:
    W = None
    if p:
        W = mag ** p
        W = W / mx.maximum(mx.mean(W), 1e-20)
        mx.eval(W)
    C = kmeans(Xs, W, args.iters)
    cn = mx.sum(C * C, axis=1)
    a = mx.argmin(mx.sum(Xn * Xn, axis=1, keepdims=True)
                  - 2 * (Xn @ C.T) + cn[None, :], axis=1)
    R = (C[a].reshape(-1, in_d // G, G) * scale).reshape(T.shape)
    mx.eval(R)
    R_np = np.array(R)
    err = R_np[N_TRAIN:].ravel() - a_np          # HELD-OUT experts only
    err_in = R_np[:N_TRAIN].ravel() - a_in       # train experts, for the gap
    rows[p] = [float(np.sqrt((err[m] ** 2).mean())) / d
               for m, d in zip(mask, denom)]
    rows[p].append(float(np.sqrt((err ** 2).sum() / (a_np ** 2).sum())))
    rows[p].append(float(np.sqrt((err_in ** 2).sum() / (a_in ** 2).sum())))
    del R_np, err_in
    print(f"  p={p:<4g} fit done", flush=True)
    del C, a, R, err
    mx.clear_cache()

hdr = ([f"{QS[i]:g}-{QS[i+1]:g}" for i in range(len(QS) - 1)]
       + ["MEAN held-out", "MEAN in-sample"])
print("\nnormalized RMS error by |w| percentile bucket "
      "(lower is better; MEAN relerr is the gated bulk statistic)")
print(f"{'p':>5} " + " ".join(f"{h:>12}" for h in hdr))
for p in POWS:
    print(f"{p:>5g} " + " ".join(f"{v:12.5f}" for v in rows[p]))
if 0.0 in rows:
    print("\ndelta vs p=0 (negative = tail-weighted BETTER in that bucket)")
    for p in POWS:
        if p == 0.0:
            continue
        d = [rows[p][i] - rows[0.0][i] for i in range(len(hdr))]
        print(f"{p:>5g} " + " ".join(f"{v:+12.5f}" for v in d))
    print("\nDECISION RULE (pre-registered): a p is worth a 397B fit only if it"
          "\nturns the 99-99.9 and 99.9-100 deltas NEGATIVE. Expect the MEAN"
          "\nrelerr column to get WORSE — that is the trade being bought, and"
          "\nit is why --relerr-abort must be raised for the real fit.")
