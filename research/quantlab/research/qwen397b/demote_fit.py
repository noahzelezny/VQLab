#!/usr/bin/env python3
"""Fit a DEMOTED (or refit) d8 expert layer for the 397B 2.2 base and build
the spliced candidate directly — stage 0+2 of V2-SWEEP-PLAN-2.2.md in one
tool.

WHY THIS EXISTS. fit-moe (vq_397b_codes.py) needs the affine 2-bit skeleton
as --base, and that artifact was deleted in the disk purge
(deletion_plan.json). But its fit math is SELF-CONTAINED given the bf16:
normalize() derives the per-group scales from the source weights (max-abs,
fp16-rounded), not from the base. So this tool copies that math VERBATIM
(same seed, init, iters, sample count, fp16 rounding, chunking) and only
needs: the bf16 source + the shipped 2.2 artifact as the splice base.

VINTAGE GATE. Run --k 16384 --gate first: refitting the shipped geometry
through this path and scoring the spliced result must land at ~base ppl
(3.0568). That validates fit math + packing + splice end-to-end before any
demotion row is trusted.

PACKING. Codes are block-packed via vq_pack.pack (12 bits for K4096, 13
for K8192, 14 reproduces the shipped K16384), so demoted layers actually
pay bytes: -0.1875 / -0.09375 GiB per layer vs the 1.75-bpw base.

USAGE (on the M4; bf16 over the SMB mount, staged one shard at a time):

  python3 demote_fit.py --layer 9 --k 4096 \
      --base ~/.exo/models/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw \
      --src "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16" \
      --stage-dir ~/v2sweep22/stage --out ~/v2sweep22/work/397b-v2-demoteL9-k4096
"""
import argparse
import gc
import json
import math
import os
import pathlib
import shutil
import sys

import mlx.core as mx
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# vqlab package modules shipped alongside on a box without the repo
for cand in (HERE, HERE.parent.parent.parent.parent / "src" / "vqlab"):
    if (cand / "vq_pack.py").exists():
        sys.path.insert(0, str(cand))
        break
import vq_pack
import expert_src
from families import FAMILY

GiB = 2 ** 30
G = 64                       # scale group, matches the shipped artifact
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
VQ_SUFFIXES = ("codebook", "codes", "vq_scales")

mx.set_cache_limit(8 << 30)


def module_name(layer, proj):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}"


# ---- fit math: copied VERBATIM from vq_397b_codes.py (E-series vintage) ---

def kmeanspp_init(X, k, cap=200_000):
    """fit-moe's seeding, verbatim (unweighted arm): ++ on a subsample."""
    n = X.shape[0]
    if n > cap:
        sel = mx.random.randint(0, n, (cap,))
        X = X[sel]
        n = cap
    first = int(mx.random.randint(0, n, (1,)).item())
    picks = [X[first]]
    d2 = mx.sum((X - picks[0]) ** 2, axis=1)
    for _ in range(k - 1):
        tot = mx.sum(d2)
        if float(tot.item()) <= 1e-12:
            picks.append(X[int(mx.random.randint(0, n, (1,)).item())])
            continue
        cdf = mx.cumsum(d2 / tot)
        r = mx.random.uniform(shape=(1,))
        j = min(int(mx.sum(cdf < r).item()), n - 1)
        c = X[j]
        picks.append(c)
        d2 = mx.minimum(d2, mx.sum((X - c) ** 2, axis=1))
    return mx.stack(picks)


def kmeans(X, k, iters, seed):
    mx.random.seed(seed)
    n = X.shape[0]
    C = kmeanspp_init(X, k)
    step = max(50_000, int(5e8 / k))
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
        oh_sum = oh_sum.at[a].add(X)
        cnt = cnt.at[a].add(mx.ones((n,), dtype=mx.float32))
        mx.eval(oh_sum, cnt)
        C = mx.where(cnt[:, None] > 0,
                     oh_sum / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C


def fit_tensor(T, D, K, iters, sample, seed, expert_chunk):
    """bf16 stack [E, out, in] -> (codes u16 [E,out,in/D], codebook fp16,
    scales fp16 [E,out,in/G], relerr). Verbatim fit-moe math, unweighted."""
    n_exp, out_d, in_d = (int(s) for s in T.shape)

    def normalize(blk):
        Wg = blk.reshape(-1, in_d // G, G)
        scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-6)
        scale = scale.astype(mx.float16).astype(mx.float32)  # SHIPPED rounding
        return (Wg / scale).reshape(-1, D), scale

    EC = expert_chunk
    samples = []
    per = max(1, sample // max(1, (n_exp // EC)))
    mx.random.seed(seed)
    for s in range(0, n_exp, EC):
        sub, _ = normalize(T[s:s + EC].astype(mx.float32))
        idx = mx.random.randint(0, sub.shape[0], (min(per, sub.shape[0]),))
        samples.append(sub[idx])
        mx.eval(samples[-1])
        del sub
    Xs = mx.concatenate(samples, axis=0)
    C16 = kmeans(Xs, K, iters, seed).astype(mx.float16)
    mx.eval(C16)
    del samples, Xs
    Cf = C16.astype(mx.float32)
    cn = mx.sum(Cf * Cf, axis=1)

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
        codes_parts.append(a.astype(mx.uint16).reshape(e, out_d, in_d // D))
        scales_parts.append(scale.astype(mx.float16).reshape(e, out_d,
                                                             in_d // G))
        mx.eval(codes_parts[-1], scales_parts[-1])
        del blk, sub, scale, a, R, aparts
        gc.collect()
        mx.clear_cache()
    codes = mx.concatenate(codes_parts, axis=0)
    scales = mx.concatenate(scales_parts, axis=0)
    mx.eval(codes, scales)
    return codes, C16, scales, math.sqrt(num / den)


# ---- staging (one bf16 shard resident at a time; M4 has ~17 GiB free) ----

_staged = {}


def make_shard_path(src, stage_dir):
    if not stage_dir:
        return lambda f: str(pathlib.Path(src) / f)
    st = pathlib.Path(stage_dir)
    # SWEEP THE DIR ON STARTUP, not just between shards. `_staged` is
    # per-process, so the last shard of every run survived its exit: a
    # 10-layer arc leaked ~8 GiB x 10 and filled the M4 to 112 MiB free,
    # killing its own last three layers (2026-09-06). This dir is scratch
    # by contract -- the caller passes --stage-dir precisely to say
    # "expendable copies live here".
    if st.is_dir():
        for stale in st.glob("*.safetensors"):
            stale.unlink(missing_ok=True)

    def _p(fname):
        st.mkdir(parents=True, exist_ok=True)
        local = st / fname
        if fname not in _staged:
            for old in list(_staged.values()):
                old.unlink(missing_ok=True)
            _staged.clear()
            shutil.copy2(pathlib.Path(src) / fname, local)
            _staged[fname] = local
        return str(local)
    return _p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--k", type=int, required=True,
                    help="4096 (pack 12), 8192 (pack 13); 16384 = vintage gate")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--base", required=True,
                    help="shipped 2.2 artifact dir (splice base, local APFS)")
    ap.add_argument("--src", required=True, help="bf16 source dir")
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--out", required=True, help="candidate output dir")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sample", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--expert-chunk", type=int, default=32)
    ap.add_argument("--family", default="qwen3_5")
    ap.add_argument("--save-fit", default=None,
                    help="dir for the raw fitted tensors (codes/codebook/"
                         "vq_scales, one safetensors per layer+K). Refits "
                         "have burned hours twice; ALWAYS pass the archive "
                         "(HDD: .../vqlab-fits/<model>/<geom>/) so a fit is "
                         "paid for once. Existing file = fit is SKIPPED and "
                         "loaded from the archive.")
    a = ap.parse_args()

    D, K = a.dim, a.k
    pack_bits = int(math.ceil(math.log2(K)))
    bp = pathlib.Path(a.base)
    out = pathlib.Path(a.out)
    idx = json.load(open(bp / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    cfg = json.load(open(bp / "config.json"))
    fam = FAMILY[a.family]
    src_idx = json.load(open(pathlib.Path(a.src) /
                             "model.safetensors.index.json"))["weight_map"]
    shard_path = make_shard_path(a.src, a.stage_dir)

    # fit all three projections on the CPU-stream-load / GPU-fit pattern
    fit_file = None
    if a.save_fit:
        fdir = pathlib.Path(a.save_fit)
        fdir.mkdir(parents=True, exist_ok=True)
        fit_file = fdir / f"layer{a.layer}-d{D}k{K}.safetensors"
    # CONVERT vs DEMOTE. A layer already carrying VQ codes is re-fitted at a
    # new K (demote). A layer still in `quantization` is AFFINE and has never
    # been VQ'd at all -- layers 57-59 of every shipped 397B rung, which the
    # original fit missed by running --vq-layers 0-56 on a 60-layer model
    # (found 2026-09-06). Converting one rewrites its tensor NAMES
    # (weight/scales/biases -> codes/codebook/vq_scales), so the weight_map
    # and both quantization dicts have to move with it.
    def _is_affine(m):
        return m not in cfg["vq_modules"] and m in cfg.get("quantization", {})

    convert = {module_name(a.layer, p) for p in PROJECTIONS
               if _is_affine(module_name(a.layer, p))}
    if convert:
        bits = cfg["quantization"][sorted(convert)[0]].get("bits")
        print(f"== CONVERT layer {a.layer}: {len(convert)} AFFINE modules "
              f"({bits}-bit) -> VQ d{D}/K{K}. Tensor names change.", flush=True)

    def _vq_entry(m):
        """vq_modules entry the bundled runtime needs (model.py reads
        experts/out/in/k/dim/group/pack_bits). For a demote, inherit the
        existing entry; for a convert, derive the shape from the affine
        weight, which is packed uint32: [experts, out, in*bits/32]."""
        if m in cfg["vq_modules"]:
            e = dict(cfg["vq_modules"][m])
        else:
            h = mx.load(str(bp / wm[m + ".weight"]))
            sc = h[m + ".scales"]          # [experts, out, in/group] -- exact
            e = {"experts": int(sc.shape[0]), "out": int(sc.shape[1]),
                 "in": int(sc.shape[2]) * G}
            del h, sc
            mx.clear_cache()
        e.update({"k": K, "dim": D, "group": G, "pack_bits": pack_bits})
        return e

    new_tensors = {}
    vq_entries = {}
    if fit_file is not None and fit_file.exists():
        print(f"== fit ARCHIVE HIT {fit_file.name} — skipping k-means",
              flush=True)
        new_tensors = dict(mx.load(str(fit_file)))
        for proj in PROJECTIONS:
            vq_entries[module_name(a.layer, proj)] = _vq_entry(
                module_name(a.layer, proj))
    for proj in (() if new_tensors else PROJECTIONS):
        m = module_name(a.layer, proj)
        assert m in cfg["vq_modules"] or m in cfg["quantization"], m
        print(f"== fit {m}  d{D}/K{K} (pack {pack_bits})", flush=True)
        T = expert_src.load_expert_stack(pathlib.Path(a.src), src_idx, fam,
                                         a.layer, proj, shard_path=shard_path)
        codes, cb, scales, relerr = fit_tensor(
            T, D, K, a.iters, a.sample, a.seed, a.expert_chunk)
        del T
        gc.collect(); mx.clear_cache()
        print(f"   relerr {relerr:.4f}", flush=True)
        packed = vq_pack.pack(np.array(codes, copy=False), pack_bits)
        new_tensors[m + ".codes"] = mx.array(packed)
        new_tensors[m + ".codebook"] = cb
        new_tensors[m + ".vq_scales"] = scales
        vq_entries[m] = _vq_entry(m)
        del codes, cb, scales, packed
        gc.collect(); mx.clear_cache()
    if fit_file is not None and not fit_file.exists() and new_tensors:
        tmp = fit_file.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(tmp), new_tensors)
        os.replace(tmp, fit_file)
        print(f"   archived fit -> {fit_file}", flush=True)

    # splice into a candidate: hardlink untouched shards, rewrite touched
    AFFINE_SUFFIXES = ("weight", "scales", "biases")
    old_keys = {}                       # module -> [key, ...] to DROP
    dest = {}                           # NEW tensor name -> shard it lands in
    touched = set()
    for proj in PROJECTIONS:
        m = module_name(a.layer, proj)
        sufs = AFFINE_SUFFIXES if m in convert else VQ_SUFFIXES
        keys = [m + "." + s for s in sufs if m + "." + s in wm]
        old_keys[m] = keys
        # a converted module's VQ tensors land in the shard its affine
        # weight occupied; a demoted module's names are unchanged.
        home = wm[keys[0]]
        touched.update(wm[k] for k in keys)
        for suf in VQ_SUFFIXES:
            dest[m + "." + suf] = wm.get(m + "." + suf, home)
    touched = sorted(touched)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # non-shard files come across as-is. A .safetensors file that the INDEX
    # does not reference is a SIDECAR (mtp-head-q6, vision grafts), not a
    # shard -- skipping every .safetensors dropped the 397B's published
    # 5.4 GiB MTP head from a rebuild (caught 2026-09-07, before it could
    # be uploaded over the live repo).
    _idx_shards = set(wm.values())
    for f in bp.iterdir():
        if f.is_dir():
            continue
        if f.suffix == ".safetensors" and f.name in _idx_shards:
            continue
        shutil.copy2(f, out / f.name)
    mx.set_default_device(mx.cpu)
    for sh in sorted(set(wm.values())):
        if sh not in touched:
            os.link(bp / sh, out / sh)
            continue
        t = mx.load(str(bp / sh))
        n = dropped = 0
        # drop the affine tensors of any module being converted
        for m in convert:
            for k in old_keys[m]:
                if k in t:
                    del t[k]
                    dropped += 1
        for name, arr in new_tensors.items():
            if dest.get(name) == sh:
                t[name] = arr
                n += 1
        tmp = out / (sh[:-len('.safetensors')] + '.tmp.safetensors')
        mx.save_safetensors(str(tmp), t)
        os.replace(tmp, out / sh)
        del t
        print(f"   rewrote {sh}: {n} tensors in, {dropped} affine dropped",
              flush=True)

    cfg["vq_modules"].update(vq_entries)
    # a converted module leaves BOTH quantization dicts (mlx reads
    # `quantization`, HF tooling reads `quantization_config`); leaving it in
    # either makes the loader build a QuantizedSwitchLinear for weights that
    # are no longer there.
    for m in convert:
        for d in ("quantization", "quantization_config"):
            cfg.get(d, {}).pop(m, None)
        # weight_map follows the rename, or the loader looks for .weight
        for k in old_keys[m]:
            wm.pop(k, None)
        for suf in VQ_SUFFIXES:
            wm[m + "." + suf] = dest[m + "." + suf]
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    idx["weight_map"] = wm
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(out / s) for s in sorted(set(wm.values())))
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=1)
    print(f"DONE {out.name}: {idx['metadata']['total_size'] / GiB:.3f} GiB",
          flush=True)


if __name__ == "__main__":
    main()
