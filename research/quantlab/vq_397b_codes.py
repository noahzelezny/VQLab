#!/usr/bin/env python
"""M2: fused VQ-fit + assemble the REAL 397B codes artifact (no proxy).

vq_397b_fused.py's twin, emitting the M1 runtime format instead of a bf16
reconstruction: each 2-bit expert tensor in --vq-layers becomes

    {mod}.codes      uint8 (K<=256) / uint16   [E, out, in/d]
    {mod}.codebook   fp16                      [K, d]
    {mod}.vq_scales  fp16                      [E, out, in/group]

and the artifact is born SELF-CONTAINED: config.json gains
  "model_file": "model.py"   +   "vq_modules": {path: geometry}
and model.py (generated here from quantlab/vq_switch.py + a shim) gives any
stock `pip install mlx-lm` user a working model with zero patching —
mlx_lm.utils.load_model imports the model class from inside the checkpoint.

Scales are FP16-rounded (matches the M1 kernels exactly; the E35 proxies
used bf16 — sub-0.1% referee difference, see M1_KERNEL_PLAN M1e).

Designed to run on the M4 while the M3 grinds its own queue:
  --base     staged LOCAL copy of the shipped artifact (mmap-safe APFS)
  --src      bf16 source; over SMB use --stage-dir (cp -> mx.load -> evict)
  --ship-to  after each output shard is written, rsync it to this dir
             (plain file IO, SMB-safe) and delete the local copy so
             base(122G) + output(~40G VQ region) never overflows the disk.

  ./vq_397b_codes.py --base /tmp/m1d/...tail3x3 --vq-layers 0-56 --k 256 \
      --out /tmp/m1d/out-C --stage-dir /tmp/m1d/srcstage \
      --src "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16" \
      --ship-to "/Volumes/Thunderbay SSD/Exo Models/rotlab--397B-vqK256codes"
"""
import argparse
import gc
import json
import math
import pathlib
import shutil
import subprocess
import sys
import re
import time

import mlx.core as mx

mx.set_cache_limit(8 << 30)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True,
                help="output dir. EXISTING SHARDS ARE SKIPPED (the fit RESUMES). "
                     "Do NOT delete this dir to 'start clean' on a retry -- that "
                     "throws away hours of completed, valid work. Two relaunches "
                     "cost ~107 GiB of finished shards to exactly that reflex.")
ap.add_argument("--vq-layers", required=True)
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--init", choices=("kmeans++", "random"),
                default="kmeans++",
                help="codebook seeding. Artifacts built before 08-18 used 'random'.")
ap.add_argument("--expert-chunk", type=int, default=32)
ap.add_argument("--family", default="qwen3_5", choices=["qwen3_5", "gemma4", "qwen3_5_mlx"],
                help="module naming / source-key family. Default reproduces "
                     "the shipped 397B behaviour exactly.")
ap.add_argument("--sample", type=int, default=2_000_000)
ap.add_argument("--stage-dir", default=None)
ap.add_argument("--ship-to", default=None,
                help="rsync each finished shard here, then delete local")
ap.add_argument("--geom", default=None,
                help="per-projection geometry override, e.g. "
                     "'gate_proj=d4k256,up_proj=d4k256,down_proj=d8k4096' "
                     "(E36: down_proj prefers d8; gate/up never recovers)")
ap.add_argument("--relerr-abort", type=float, default=0.35,
                help="refit, then abort, if any tensor exceeds this relerr. "
                     "Healthy d4k2048 is ~0.19, d2k2048 ~0.032, worst "
                     "legitimate (L00) ~0.215 — so 0.35 is well clear of "
                     "normal and well under a collapse (1.0).")
ap.add_argument("--max-refit", type=int, default=2,
                help="refit attempts for a tensor over --relerr-abort")
ap.add_argument("--tail-weight-from", type=int, default=0,
                help="apply --tail-weight-pow only from this layer onward. "
                     "Shallow layers are HEAVY-TAILED (E110: L00-L10 excess "
                     "kurtosis +1.25 vs body -0.38) and magnitude weighting "
                     "destroys them — L00 down_proj went 0.118 -> 0.487 mean "
                     "relerr at p=4 and aborted a 397B fit (E106). The body "
                     "is sub-Gaussian and is where the tail trade is live. "
                     "0 = every layer.")
ap.add_argument("--tail-weight-pow", type=float, default=0.0,
                help="MAGNITUDE-WEIGHTED k-means for the codebook fit. 0.0 "
                     "(default) is the unweighted objective every existing "
                     "artifact was fit with — bit-identical code path. p>0 "
                     "weights each training subvector by its WEIGHT-SPACE L2 "
                     "norm to the p: w = (scale_g * ||x||_2)^p, where scale_g "
                     "is the group-64 max-abs the subvector was normalized by "
                     "(64 %% d == 0, so a subvector never straddles two "
                     "groups). p=2 makes Lloyd minimize true weight-space MSE "
                     "instead of normalized-space MSE; p>2 buys the tail at "
                     "the bulk's expense. WHY (E102): k-means minimizes "
                     "AVERAGE distortion, so at low K it packs centroids into "
                     "the dense middle and abandons the rare large weights "
                     "that dominate the output — measured as a monotonic "
                     "crossover, better on |w| 0-90 pct, worse on 99-100. "
                     "Only expected to matter where centroids are scarce; at "
                     "K2048/K8192 the codebook already serves both.")
ap.add_argument("--tail-from", type=int, default=None,
                help="first layer index of the TAIL (see --tail-geom)")
ap.add_argument("--tail-geom", default=None,
                help="geometry for layers >= --tail-from, e.g. 'd2k2048', "
                     "applied to all three projections. Spends bytes where "
                     "they help: E25/E29 put the 397B knee at ~tail30 of 60. "
                     "This is a QWEN-FAMILY law and does NOT transfer to "
                     "gemma (LADDER_GEMMA.md:180; gemma vq-tail10 scored "
                     "BELOW flat K256, 76.92 vs 79.81).")
args = ap.parse_args()

BASE, SRC, OUT = pathlib.Path(args.base), pathlib.Path(args.src), pathlib.Path(args.out)
SHIP = pathlib.Path(args.ship_to) if args.ship_to else None
LO, HI = (int(x) for x in args.vq_layers.split("-"))
G = args.group

# per-projection (D, K); default = the uniform --dim/--k
GEOM = {p: (args.dim, args.k) for p in ("gate_proj", "up_proj", "down_proj")}
if args.geom:
    for part in args.geom.split(","):
        proj, dk = part.split("=")
        d_s, k_s = dk.lstrip("d").split("k")
        assert proj in GEOM, proj
        GEOM[proj] = (int(d_s), int(k_s))

# tail override: layers >= TAIL_FROM use TAIL_GEOM for every projection
TAIL_GEOM, TAIL_FROM = None, args.tail_from
if args.tail_geom:
    assert TAIL_FROM is not None, "--tail-geom requires --tail-from"
    _d, _k = args.tail_geom.lstrip("d").split("k")
    TAIL_GEOM = (int(_d), int(_k))


def geom_for(li, proj):
    """(D, K) for this layer+projection — the tail may differ from the body."""
    if TAIL_GEOM is not None and li >= TAIL_FROM:
        return TAIL_GEOM
    return GEOM[proj]


def code_dtype(k):
    return mx.uint8 if k <= 256 else mx.uint16

base_idx = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
base_cfg = json.load(open(BASE / "config.json"))

# FAMILY TABLE. Default is qwen3_5 and is BIT-IDENTICAL to the behaviour that
# produced the shipped 397B artifacts — the constants below are exactly what
# was hardcoded before. gemma4 differs only in the module substring and the
# source-key template; its fused gate_up stack and split axis are the same
# (mlx_lm gemma4_text.py:627 splits axis=-2 of a rank-3 [E,2I,H], which IS
# axis 1, the same OUT-dim half-slice used here).
FAMILY = {
    "qwen3_5": {
        "target_substr": "switch_mlp",
        # HF-layout source: gate and up live FUSED in one [E, 2I, H] stack,
        # taken as halves along the OUT axis.
        "src_key": "model.language_model.layers.{li}.mlp.experts.{key}",
        "proj": {"gate_proj": ("gate_up_proj", 0),
                 "up_proj": ("gate_up_proj", 1),
                 "down_proj": ("down_proj", None)},
    },
    "gemma4": {
        "target_substr": "switch_glu",
        # mlx-community's gemma bf16 is an MLX-FORMAT conversion, so
        # mlx_lm's sanitize has ALREADY split experts.gate_up_proj into
        # switch_glu.{gate,up}_proj (gemma4_text.py:625-634). There is no
        # fused stack in the checkpoint — verified: 'gate_up_proj' appears
        # in zero keys. So every projection is direct, no half-slicing, and
        # the prefix is language_model.model.* not model.language_model.*.
        "src_key": "language_model.model.layers.{li}.experts.switch_glu.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "gemma4_e4b": {
        # gemma-4-e4b-it: DENSE mlp (no experts), weights are 2D [OUT, IN].
        # Only used for VERIFICATION of e4b VQ artifacts (fit_e4b_vq.py) —
        # the main fitter's is_vq_target() wants a 2-bit-marked struct BASE,
        # which dense e4b builds do not have. Consumers must treat a 2D
        # source tensor as [1, OUT, IN] (E=1).
        "target_substr": ".mlp.",
        "src_key": "language_model.model.layers.{li}.mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "qwen3_8_dense": {
        # DENSE Qwen3.8-27B. Not an MoE: the mlp trio lives directly on the
        # layer with no .experts./.switch_mlp. segment, and each tensor is 2D
        # ([OUT, IN]) rather than [E, OUT, IN] — verify_artifact adds the E=1
        # axis for dense families. Source is the HF-format bf16 checkpoint,
        # whose prefix is model.language_model.* (the ARTIFACT uses
        # language_model.model.*; build_dense_vq.py owns that remap).
        "target_substr": "mlp",
        "src_key": "model.language_model.layers.{li}.mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "qwen3_5_mlx": {
        # SAME architecture as qwen3_5 (qwen3_5_moe, switch_mlp, shared
        # expert) but sourced from an mlx-community MLX-FORMAT bf16
        # conversion rather than the original HF-format one. Verified on
        # mlx-community/Qwen3.6-35B-A3B-bf16: sanitize already split
        # gate_up_proj (zero fused keys), prefix is language_model.model.*
        # (no leading "model."), and there is no .experts. path segment —
        # the module IS switch_mlp.{gate,up,down}_proj directly. Do not
        # point this at an HF-format source (use "qwen3_5" for that).
        "target_substr": "switch_mlp",
        "src_key": "language_model.model.layers.{li}.mlp.switch_mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
}
FAM = FAMILY[args.family]
PROJ = FAM["proj"]


def layer_of(name):
    try:
        return int(name.split("layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def is_vq_target(name):
    if FAM["target_substr"] not in name:
        return False
    li = layer_of(name)
    if not (LO <= li <= HI):
        return False
    mod = name.rsplit(".", 1)[0]
    q = base_cfg["quantization"].get(mod)
    return isinstance(q, dict) and q.get("bits") == 2


def kmeanspp_init(X, k, cap=200_000, W=None):
    """k-means++ seeding: each new centre is drawn with probability
    proportional to its squared distance from the nearest chosen centre.

    WHY. Uniform-random seeding (the original) draws k points from the raw
    density, so dense regions get many near-duplicate centres and sparse
    regions get none — wasted codebook and, at worst, a collapsed fit (E44:
    tail30 L26 down_proj hit relerr 1.0 from a bad draw). ++ spreads the
    seeds and is the standard fix.

    Textbook ++ is k sequential passes over X; at k=2048 and |X|=2M that is
    ~4e9 distance evals. So seed on a SUBSAMPLE (cap) and hand the result to
    full k-means over all of X — the usual scalable compromise, and the
    refinement iterations see every point regardless.
    """
    n = X.shape[0]
    if n > cap:
        sel = mx.random.randint(0, n, (cap,))
        X = X[sel]
        if W is not None:
            W = W[sel]
        n = cap
    first = int(mx.random.randint(0, n, (1,)).item())
    picks = [X[first]]
    d2 = mx.sum((X - picks[0]) ** 2, axis=1)
    for _ in range(k - 1):
        # With W, seeds are drawn proportional to w*d2 rather than d2, so the
        # SEEDING obeys the same objective as the refinement below. Seeding
        # unweighted and refining weighted starts every tail centre in the
        # bulk and asks 20 Lloyd steps to walk it out.
        pd2 = d2 if W is None else d2 * W
        tot = mx.sum(pd2)
        if float(tot.item()) <= 1e-12:          # X has <k distinct points
            picks.append(X[int(mx.random.randint(0, n, (1,)).item())])
            continue
        cdf = mx.cumsum(pd2 / tot)
        r = mx.random.uniform(shape=(1,))
        j = min(int(mx.sum(cdf < r).item()), n - 1)
        c = X[j]
        picks.append(c)
        d2 = mx.minimum(d2, mx.sum((X - c) ** 2, axis=1))
    return mx.stack(picks)


def kmeans(X, k, iters, init="kmeans++", W=None):
    """W=None reproduces the unweighted fit exactly. W is a per-row weight;
    ASSIGNMENT is unaffected (argmin of a distance does not see a positive
    scalar), only the centroid update and the ++ seeding are."""
    step = max(50_000, int(5e8 / k))
    n = X.shape[0]
    # SEEDING IS DELIBERATELY UNWEIGHTED. Weighting BOTH the ++ seeding and the
    # centroid update compounds instead of composing: ++ already spreads seeds
    # by squared distance, so multiplying by mag^p puts nearly every seed in
    # the extreme tail and starves the bulk of centroids. Measured 2026-08-21
    # (E106), L00 down_proj K256 d4, mean held-out relerr:
    #     random init  p=0 0.2035  p=4 0.1685   (weighting helps)
    #     ++ seeding   p=0 0.1177  p=4 0.4869   (weighting destroys it)
    # and the real 397B fit aborted at 0.7111. The tail weighting belongs on
    # the UPDATE only; ++ keeps its own unweighted objective.
    C = (kmeanspp_init(X, k) if init == "kmeans++"
         else X[mx.random.randint(0, n, (k,))])
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        parts = []
        for s0 in range(0, n, step):
            xb = X[s0:s0 + step]
            parts.append(mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                                   - 2 * (xb @ C.T) + cn[None, :], axis=1))
            mx.eval(parts[-1])
        a = mx.concatenate(parts)
        oh_sum = mx.zeros((k, X.shape[1]))
        cnt = mx.zeros((k,))
        # ONE-HOT CHUNK MUST SCALE WITH K. This was a fixed 2,000,000 rows,
        # so the [rows, k] fp32 one-hot below is 2e6*k*4 bytes — 65.5 GB at
        # k=8192, over Metal's 62.6 GB single-buffer cap. That is what killed
        # every K8192 attempt (the failure reads as
        # "[metal::malloc] Attempting to allocate 65536000000 bytes", NOT as
        # the command-buffer timeout it was previously blamed on). Budget the
        # chunk by k instead, matching how `step` above is already computed.
        # SCATTER-ADD, not one-hot. The one-hot form built an [rows, k] fp32
        # matrix: 2e6*k*4 bytes = 65.5 GB at k=8192 (over Metal's 62.6 GB cap,
        # which is what killed every K8192 run). Chunking it small enough to
        # fit made it 33x more iterations and K8192 did not finish a single
        # tensor in 58 minutes. scatter-add is O(n) in the assignment instead
        # of O(n*k), so cost stops depending on k entirely.
        # Weighted centroid = sum(w*x)/sum(w). Same two scatter-adds, so the
        # O(n) cost and the k-independence of the update are unchanged.
        oh_sum = oh_sum.at[a].add(X if W is None else X * W[:, None])
        cnt = cnt.at[a].add(mx.ones((n,), dtype=mx.float32) if W is None else W)
        mx.eval(oh_sum, cnt)
        C = mx.where(cnt[:, None] > 0,
                     oh_sum / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C


_staged = {}


def _shard_path(fname):
    if not args.stage_dir:
        return str(SRC / fname)
    st = pathlib.Path(args.stage_dir)
    st.mkdir(parents=True, exist_ok=True)
    local = st / fname
    if fname not in _staged:
        for old in _staged.values():
            old.unlink(missing_ok=True)
        _staged.clear()
        shutil.copy2(SRC / fname, local)
        _staged[fname] = local
    return str(local)


def load_src_expert(li, proj):
    key, half = PROJ[proj]
    sk = FAM["src_key"].format(li=li, key=key)
    # Materialize the source read ON THE CPU STREAM. The lazy shard read of a
    # remote 751G bf16 source stalls on disk INSIDE a GPU command buffer and
    # the Metal watchdog kills the wait (kIOGPUCommandBufferCallbackErrorTimeout)
    # — seen at the fit's sampling eval when SRC is on SMB rather than local
    # disk. Same fault and same cure as verify_artifact.py: the stream binds at
    # OP-CREATION time, not eval time, so the load AND the slice must be created
    # under mx.cpu or the read chain stays on the watchdog'd stream.
    with mx.stream(mx.cpu):
        T = mx.load(_shard_path(src_idx[sk]))[sk]
        if half is not None:
            mid = T.shape[1] // 2
            T = T[:, :mid, :] if half == 0 else T[:, mid:, :]
        mx.eval(T)
    return T


def vq_tensor_codes(li, proj, want_shape):
    """Fit one expert tensor -> (codes, codebook fp16, scales fp16, relerr)."""
    D, K = geom_for(li, proj)
    T = load_src_expert(li, proj)
    assert tuple(T.shape) == tuple(want_shape), (li, proj, T.shape, want_shape)
    n_exp, out_d, in_d = T.shape
    EC = args.expert_chunk

    def normalize(blk):
        Wg = blk.reshape(-1, in_d // G, G)
        scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
        scale = scale.astype(mx.float16).astype(mx.float32)   # SHIPPED rounding
        return (Wg / scale).reshape(-1, D), scale

    # --- codebook from sampled subvectors
    P = args.tail_weight_pow if li >= args.tail_weight_from else 0.0
    samples, sample_w = [], []
    per = max(1, args.sample // max(1, (n_exp // EC)))
    for s in range(0, n_exp, EC):
        sub, sc = normalize(T[s:s + EC].astype(mx.float32))
        idx = mx.random.randint(0, sub.shape[0], (min(per, sub.shape[0]),))
        samples.append(sub[idx])
        if P:
            # scale is per (row, group-of-G); a subvector lives wholly inside
            # one group (G % D == 0) and the flatten order is
            # (rows, groups, G//D, D), so repeating each group's scale G//D
            # times CONSECUTIVELY lines the two up. Getting this wrong is
            # silent: it mislabels which subvectors are the tail.
            ssub = mx.repeat(sc.reshape(-1, in_d // G), G // D,
                             axis=1).reshape(-1)
            # magnitude in ORIGINAL weight units — the same quantity E102
            # bucketed on, lifted from per-weight to per-subvector.
            mag = ssub[idx] * mx.sqrt(mx.sum(sub[idx] ** 2, axis=1))
            sample_w.append(mag ** P)
            mx.eval(sample_w[-1])
        mx.eval(samples[-1])
        del sub, sc
    Xs = mx.concatenate(samples, axis=0)
    Ws = None
    if P:
        Ws = mx.concatenate(sample_w, axis=0)
        Ws = Ws / mx.maximum(mx.mean(Ws), 1e-20)   # mean 1: keeps the fp32
        mx.eval(Ws)                                # scatter sums well-scaled
    C16 = kmeans(Xs, K, args.iters,
                 init=args.init, W=Ws).astype(mx.float16)
    mx.eval(C16)
    del samples, sample_w, Xs, Ws
    Cf = C16.astype(mx.float32)               # assign against SHIPPED values
    cn = mx.sum(Cf * Cf, axis=1)

    # --- assign per chunk; relerr against the bf16 source with shipped values
    codes_parts, scales_parts = [], []
    num = den = 0.0
    step = max(50_000, int(5e8 / K))
    for s in range(0, n_exp, EC):
        blk = T[s:s + EC].astype(mx.float32)
        sub, scale = normalize(blk)
        aparts = []
        for c in range(0, sub.shape[0], step):
            xb = sub[c:c + step]
            aparts.append(mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                                    - 2 * (xb @ Cf.T) + cn[None, :], axis=1))
            mx.eval(aparts[-1])
        a = mx.concatenate(aparts)
        R = (Cf[a].reshape(-1, in_d // G, G) * scale).reshape(blk.shape)
        mx.eval(R)
        num += float(mx.sum((R - blk) ** 2))
        den += float(mx.sum(blk ** 2))
        e = blk.shape[0]
        codes_parts.append(a.astype(code_dtype(K)).reshape(e, out_d, in_d // D))
        scales_parts.append(scale.astype(mx.float16).reshape(e, out_d, in_d // G))
        mx.eval(codes_parts[-1], scales_parts[-1])
        del blk, sub, scale, a, R, aparts
        gc.collect()
        mx.clear_cache()
    del T, Cf
    codes = mx.concatenate(codes_parts, axis=0)
    scales = mx.concatenate(scales_parts, axis=0)
    del codes_parts, scales_parts
    return codes, C16, scales, math.sqrt(num / den)


def ship(shard_name):
    if SHIP is None:
        return
    SHIP.mkdir(parents=True, exist_ok=True)
    src = OUT / shard_name
    subprocess.run(["rsync", "-a", str(src), str(SHIP) + "/"], check=True)
    src.unlink()


# ---- plan -----------------------------------------------------------------
targets = sorted({n.rsplit(".", 1)[0] for n in base_idx if is_vq_target(n)})


def stored_bpw(d, k):
    # codes are whole bytes (the M2 trap): uint8 for K<=256 else uint16
    return (8 if k <= 256 else 16) / d + 16.0 / G


geom_str = ", ".join(f"{p}=d{d}k{k} ({stored_bpw(d, k):.2f} bpw stored)"
                     for p, (d, k) in GEOM.items())
if TAIL_GEOM is not None:
    _d, _k = TAIL_GEOM
    geom_str += (f"  ||  TAIL layers {TAIL_FROM}-{HI}: d{_d}k{_k} "
                 f"({stored_bpw(_d, _k):.2f} bpw stored)")
print(f"base {BASE.name}: {len(targets)} expert tensors -> CODES "
      f"(layers {LO}-{HI})  {geom_str}", flush=True)
if not targets:
    raise SystemExit("no 2-bit expert tensors in range")

OUT.mkdir(parents=True, exist_ok=True)
new_cfg = json.loads(json.dumps(base_cfg))
vq_modules = {}
for m in targets:
    new_cfg["quantization"].pop(m, None)
    new_cfg.get("quantization_config", {}).pop(m, None)

t0 = time.time()
out_map, errs = {}, []
shard_sizes = {}
shards = sorted(set(base_idx.values()))
for si, sh in enumerate(shards):
    dst = OUT / sh
    def _complete(f):
        # A shard is resumable only if its safetensors header parses AND the
        # header-declared payload matches the bytes on disk. Existence alone
        # is NOT enough: a file caught mid-transfer is indistinguishable from
        # a finished one by name, and the skip would then bake a truncated
        # shard into the artifact silently. (Caught live: model-00005 at
        # 3.6/8.2 GiB during a resume-copy. Same failure family as the
        # tokenizer that loaded and encoded nothing.)
        try:
            import struct as _st
            with open(f, "rb") as fh:
                n = _st.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            want = 8 + n + max(v["data_offsets"][1] for k, v in hdr.items()
                               if k != "__metadata__")
            return f.stat().st_size == want
        except Exception:
            return False

    shipped = SHIP / sh if SHIP else None
    if dst.exists() and not _complete(dst):
        print(f"[{si+1}/{len(shards)}] {sh} PARTIAL on disk — refitting it",
              flush=True)
        dst.unlink()
    if dst.exists() or (shipped is not None and shipped.exists()):
        for k in (k for k, v in base_idx.items() if v == sh):
            mod = k.rsplit(".", 1)[0]
            if is_vq_target(k):
                for suf in (".codes", ".codebook", ".vq_scales"):
                    out_map[mod + suf] = sh
            else:
                out_map[k] = sh
        have = dst if dst.exists() else shipped
        shard_sizes[sh] = have.stat().st_size
        print(f"[{si+1}/{len(shards)}] {sh} exists, skip", flush=True)
        continue
    # Materialize the BASE shard ON THE CPU STREAM. Tensors we pass through
    # untouched go into `new` still LAZY, so the only pending work when
    # mx.save_safetensors fires below is this read — and it would then be paid
    # INSIDE a GPU command buffer, where a slow (e.g. network-mounted) read
    # trips the Metal watchdog. That failure presents at the SAVE, which is
    # misleading: the write is fine, the read never happened yet. General rule:
    # any lazy read still pending when a save forces evaluation is paid inside
    # a GPU command buffer. Stream binds at op creation, so the load must be
    # created here, not merely evaluated here.
    with mx.stream(mx.cpu):
        data = mx.load(str(BASE / sh))
        # This eval is LOAD-BEARING, not belt-and-braces. Empirical, two runs
        # on the same box differing only in this line (08-20): with it, 65 min
        # clean; without it (cpu-stream creation only), watchdog kill at the
        # save, original traceback. So creation-under-cpu is necessary but not
        # sufficient -- the deferred read still lands on a GPU command buffer
        # when save forces it. It also costs ~2x per tensor by materialising
        # the full base shard eagerly. Both facts are measured. Do not
        # "optimise" this line out.
        mx.eval(list(data.values()))
    new = {}
    done = set()
    for name, val in data.items():
        if is_vq_target(name):
            mod = name.rsplit(".", 1)[0]
            if mod in done:
                continue
            proj = mod.rsplit(".", 1)[1]
            li = layer_of(mod)
            sc = data[mod + ".scales"]
            want = (sc.shape[0], sc.shape[1], sc.shape[2] * G)
            codes, cb, vsc, err = vq_tensor_codes(li, proj, want)
            # SANITY GATE. kmeans init is random and unseeded, and a fit can
            # collapse: tail30 (08-18) produced L26 down_proj at relerr
            # 1.0000 (reconstruction ~= zero, i.e. that weight destroyed)
            # plus four more >0.12, while the SAME layers fitted cleanly at
            # 0.032 in tail20. Non-deterministic, so retry with a fresh init;
            # a silently broken tensor must never reach an artifact.
            for attempt in range(args.max_refit):
                if err <= args.relerr_abort:
                    break
                print(f"    !! L{li:02d} {proj} relerr {err:.4f} > "
                      f"{args.relerr_abort} — refit {attempt + 1}/"
                      f"{args.max_refit}", flush=True)
                codes, cb, vsc, err = vq_tensor_codes(li, proj, want)
            if err > args.relerr_abort:
                sys.exit(f"FATAL: L{li} {proj} relerr {err:.4f} still above "
                         f"{args.relerr_abort} after {args.max_refit} refits. "
                         f"The fit is unstable for this tensor — do not ship "
                         f"this artifact.")
            new[mod + ".codes"] = codes
            new[mod + ".codebook"] = cb
            new[mod + ".vq_scales"] = vsc
            pd, pk = geom_for(li, proj)
            vq_modules[mod] = {"experts": want[0], "out": want[1],
                              "in": want[2], "k": pk, "dim": pd, "group": G}
            errs.append(err)
            done.add(mod)
            print(f"    L{li:02d} {proj:10s} relerr {err:.4f}", flush=True)
            del codes, cb, vsc
            gc.collect()
            mx.clear_cache()
        elif name.rsplit(".", 1)[0] in done:
            continue
        else:
            new[name] = val
    tmp = OUT / sh.replace(".safetensors", ".tmp.safetensors")
    mx.save_safetensors(str(tmp), new)
    tmp.rename(dst)
    shard_sizes[sh] = dst.stat().st_size
    for k in new:
        out_map[k] = sh
    del data, new
    gc.collect()
    mx.clear_cache()
    ship(sh)
    print(f"[{si+1}/{len(shards)}] {sh}  ({time.time()-t0:.0f}s)", flush=True)

# vq_modules may be partial on resume — rebuild for every target from config
for m in targets:
    if m not in vq_modules:
        q = base_cfg["quantization"][m]
        # shapes recoverable only from the base scales tensor; load lazily
        sc = mx.load(str(BASE / base_idx[m + ".scales"]))[m + ".scales"]
        # geom_for, NOT GEOM: this rebuild runs for shards SKIPPED on resume,
        # and GEOM alone ignores --tail-geom — a resumed tail shard would get
        # stamped with the SHALLOW K and the runtime would decode against the
        # wrong codebook size, silently. (Latent until a tail run resumes;
        # audited tonight's rung's config — it did not fire.)
        li = int(re.search(r"layers\.(\d+)\.", m).group(1))
        pd, pk = geom_for(li, m.rsplit(".", 1)[1])
        vq_modules[m] = {"experts": sc.shape[0], "out": sc.shape[1],
                         "in": sc.shape[2] * G, "k": pk, "dim": pd, "group": G}

new_cfg["model_file"] = "model.py"
new_cfg["vq_modules"] = vq_modules

tsz = sum(shard_sizes.values())
json.dump({"metadata": {"total_size": tsz}, "weight_map": out_map},
          open(OUT / "model.safetensors.index.json", "w"))
json.dump(new_cfg, open(OUT / "config.json", "w"), indent=1)

# ---- model.py: vq_switch runtime + loader shim, one self-contained file ---
runtime = (pathlib.Path(__file__).parent / "vq_switch.py").read_text()
shim = '''

# ---------------------------------------------------------------------------
# mlx_lm `model_file` shim: stock mlx_lm imports Model/ModelArgs from THIS
# file (it lives inside the checkpoint). We reuse the registry architecture
# and swap each VQ'd expert module for VQSwitchLinear before weights load.
# ---------------------------------------------------------------------------
import importlib as _importlib
import json as _json
import pathlib as _pathlib

_cfg = _json.load(open(_pathlib.Path(__file__).parent / "config.json"))
_arch = _importlib.import_module(f"mlx_lm.models.{_cfg['model_type']}")
ModelArgs = _arch.ModelArgs


class Model(_arch.Model):
    def __init__(self, args):
        super().__init__(args)
        for _path, _m in _cfg.get("vq_modules", {}).items():
            _obj = self
            _parts = _path.split(".")
            for _c in _parts[:-1]:
                _obj = _obj[int(_c)] if _c.isdigit() else getattr(_obj, _c)
            _ct = mx.uint8 if _m["k"] <= 256 else mx.uint16
            setattr(_obj, _parts[-1], VQSwitchLinear(
                mx.zeros((_m["experts"], _m["out"], _m["in"] // _m["dim"]),
                         dtype=_ct),
                mx.zeros((_m["k"], _m["dim"]), dtype=mx.float16),
                mx.zeros((_m["experts"], _m["out"], _m["in"] // _m["group"]),
                         dtype=mx.float16),
                group_size=_m["group"],
            ))
'''
(OUT / "model.py").write_text(runtime + shim)

for extra in BASE.iterdir():
    if extra.is_file() and extra.suffix != ".safetensors" \
            and extra.name not in ("config.json", "model.safetensors.index.json"):
        shutil.copy2(extra, OUT / extra.name)

if SHIP is not None:
    for f in OUT.iterdir():
        if f.is_file():
            subprocess.run(["rsync", "-a", str(f), str(SHIP) + "/"], check=True)

print(f"\ndone in {time.time()-t0:.0f}s  {tsz/2**30:.1f} GiB total "
      f"-> {SHIP or OUT}")
print(f"VQ'd {len(errs)} tensors, mean relerr {sum(errs)/max(len(errs),1):.4f}")
