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
import time

import mlx.core as mx

mx.set_cache_limit(8 << 30)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
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


def kmeanspp_init(X, k, cap=200_000):
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
        X = X[mx.random.randint(0, n, (cap,))]
        n = cap
    first = int(mx.random.randint(0, n, (1,)).item())
    picks = [X[first]]
    d2 = mx.sum((X - picks[0]) ** 2, axis=1)
    for _ in range(k - 1):
        tot = mx.sum(d2)
        if float(tot.item()) <= 1e-12:          # X has <k distinct points
            picks.append(X[int(mx.random.randint(0, n, (1,)).item())])
            continue
        cdf = mx.cumsum(d2 / tot)
        r = mx.random.uniform(shape=(1,))
        j = min(int(mx.sum(cdf < r).item()), n - 1)
        c = X[j]
        picks.append(c)
        d2 = mx.minimum(d2, mx.sum((X - c) ** 2, axis=1))
    return mx.stack(picks)


def kmeans(X, k, iters, init="kmeans++"):
    step = max(50_000, int(5e8 / k))
    n = X.shape[0]
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
        oh_sum = oh_sum.at[a].add(X)
        cnt = cnt.at[a].add(mx.ones((n,), dtype=mx.float32))
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
    T = mx.load(_shard_path(src_idx[sk]))[sk]
    if half is not None:
        mid = T.shape[1] // 2
        T = T[:, :mid, :] if half == 0 else T[:, mid:, :]
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
    samples = []
    per = max(1, args.sample // max(1, (n_exp // EC)))
    for s in range(0, n_exp, EC):
        sub, _ = normalize(T[s:s + EC].astype(mx.float32))
        idx = mx.random.randint(0, sub.shape[0], (min(per, sub.shape[0]),))
        samples.append(sub[idx])
        mx.eval(samples[-1])
        del sub
    C16 = kmeans(mx.concatenate(samples, axis=0), K, args.iters,
                 init=args.init).astype(mx.float16)
    mx.eval(C16)
    del samples
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
    shipped = SHIP / sh if SHIP else None
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
    data = mx.load(str(BASE / sh))
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
        pd, pk = GEOM[m.rsplit(".", 1)[1]]
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
