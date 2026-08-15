#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E35 M0b: fused VQ-fit + assemble for the 397B quality proxy.

ONE pass over a shipped artifact's shards (no 390 GB intermediate — P1 showed
IO is the pinch): copy every tensor through, except the 2-bit expert tensors
in --vq-layers, which are replaced by a bf16 VQ reconstruction fitted from the
bf16 source. Structure / tail / routers / vision stay BYTE-IDENTICAL.

Quality proxy only (experts stored bf16): the emulated format is d/K codes +
per-(row,64) fp16 scale; real bytes are analytic. Same recipe that won at 35B
(E35 M0), codebooks fit in PURE WEIGHT SPACE.

Memory: 397B expert tensors are 4.3B params (17 GB fp32), so every stage
works in EXPERT CHUNKS; only the bf16 output tensor (8.6 GB) is resident.

  ./vq_397b_fused.py --base <artifact> --vq-layers 0-29 --out <dir> [--k 1024]
"""
import argparse
import gc
import json
import math
import pathlib
import shutil
import time

import mlx.core as mx

mx.set_cache_limit(8 << 30)

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="shipped artifact to start from")
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16")
ap.add_argument("--out", required=True)
ap.add_argument("--vq-layers", required=True, help="inclusive range, e.g. 0-29")
ap.add_argument("--k", type=int, default=1024)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--expert-chunk", type=int, default=32)
ap.add_argument("--sample", type=int, default=2_000_000)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

BASE, SRC, OUT = pathlib.Path(args.base), pathlib.Path(args.src), pathlib.Path(args.out)
LO, HI = (int(x) for x in args.vq_layers.split("-"))
D, K, G = args.dim, args.k, args.group
BPW = math.log2(K) / D + 16.0 / G

base_idx = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
src_idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
base_cfg = json.load(open(BASE / "config.json"))

PROJ = {"gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
        "down_proj": ("down_proj", None)}


def layer_of(name):
    try:
        return int(name.split("layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def is_vq_target(name):
    """a quantized switch_mlp expert tensor inside the VQ layer range"""
    if "switch_mlp" not in name:
        return False
    li = layer_of(name)
    if not (LO <= li <= HI):
        return False
    mod = name.rsplit(".", 1)[0]
    q = base_cfg["quantization"].get(mod)
    return isinstance(q, dict) and q.get("bits") == 2


def kmeans(X, k, iters):
    n = X.shape[0]
    C = X[mx.random.randint(0, n, (k,))]
    xn = mx.sum(X * X, axis=1, keepdims=True)
    for _ in range(iters):
        cn = mx.sum(C * C, axis=1)
        a = mx.argmin(xn - 2 * (X @ C.T) + cn[None, :], axis=1)
        mx.eval(a)
        oh = (a[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
        cnt = mx.sum(oh, axis=0)
        C = mx.where(cnt[:, None] > 0, (oh.T @ X) / mx.maximum(cnt[:, None], 1.0), C)
        mx.eval(C)
    return C


def vq_expert_block(W):
    """W: [e, out, in] fp32 -> (reconstruction bf16, subvectors for sampling)"""
    e, out_d, in_d = W.shape
    Wg = W.reshape(-1, in_d // G, G)
    scale = mx.maximum(mx.max(mx.abs(Wg), axis=2, keepdims=True), 1e-8)
    scale = scale.astype(mx.bfloat16).astype(mx.float32)
    return (Wg / scale).reshape(-1, D), scale, (e, out_d, in_d)


def load_src_expert(li, proj):
    """bf16 source tensor for (layer, mlx proj name), sliced from gate_up."""
    key, half = PROJ[proj]
    sk = f"model.language_model.layers.{li}.mlp.experts.{key}"
    T = mx.load(str(SRC / src_idx[sk]))[sk]
    if half is not None:
        mid = T.shape[1] // 2
        T = T[:, :mid, :] if half == 0 else T[:, mid:, :]
    return T


def vq_tensor(li, proj, want_shape):
    """Fit + reconstruct one expert tensor, in expert chunks."""
    T = load_src_expert(li, proj)
    assert tuple(T.shape) == tuple(want_shape), (li, proj, T.shape, want_shape)
    n_exp = T.shape[0]
    EC = args.expert_chunk
    # --- codebook: sample subvectors from a few chunks spread over experts
    samples = []
    per = max(1, args.sample // max(1, (n_exp // EC)))
    for s in range(0, n_exp, EC):
        sub, _, _ = vq_expert_block(T[s:s + EC].astype(mx.float32))
        idx = mx.random.randint(0, sub.shape[0], (min(per, sub.shape[0]),))
        samples.append(sub[idx])
        mx.eval(samples[-1])
        del sub
    S = mx.concatenate(samples, axis=0)
    del samples
    C = kmeans(S, K, args.iters)
    mx.eval(C)
    del S
    cn = mx.sum(C * C, axis=1)
    # --- assign + reconstruct per chunk
    outs, num, den = [], 0.0, 0.0
    for s in range(0, n_exp, EC):
        blk = T[s:s + EC].astype(mx.float32)
        sub, scale, (e, out_d, in_d) = vq_expert_block(blk)
        parts = []
        for c in range(0, sub.shape[0], 4_000_000):
            xb = sub[c:c + 4_000_000]
            a = mx.argmin(mx.sum(xb * xb, axis=1, keepdims=True)
                          - 2 * (xb @ C.T) + cn[None, :], axis=1)
            parts.append(C[a])
            mx.eval(parts[-1])
        R = mx.concatenate(parts, axis=0)
        R = (R.reshape(-1, in_d // G, G) * scale).reshape(e, out_d, in_d)
        mx.eval(R)
        num += float(mx.sum((R - blk) ** 2))
        den += float(mx.sum(blk ** 2))
        outs.append(R.astype(mx.bfloat16))
        del blk, sub, scale, R, parts
        gc.collect()
        mx.clear_cache()
    del T, C
    out = mx.concatenate(outs, axis=0)
    del outs
    return out, math.sqrt(num / den)


# ---- plan / dry run -------------------------------------------------------
targets = sorted({n.rsplit(".", 1)[0] for n in base_idx if is_vq_target(n)})
print(f"base {BASE.name}: {len(targets)} expert tensors to VQ "
      f"(layers {LO}-{HI}), d={D} K={K} -> {BPW:.2f} bpw emulated "
      f"(RTN 2-bit gs64 = 2.50)")
if not targets:
    raise SystemExit("no 2-bit expert tensors in range — check --vq-layers/base")
if args.dry_run:
    # predicted output size: base + (bf16 expert bytes - quantized bytes)
    add = 0
    for m in targets:
        q = mx.load(str(BASE / base_idx[m + ".weight"]))[m + ".weight"]
        sc = mx.load(str(BASE / base_idx[m + ".scales"]))[m + ".scales"]
        break
    print("dry run: use --no-dry-run to build")
    raise SystemExit(0)

OUT.mkdir(parents=True, exist_ok=True)
new_cfg = json.loads(json.dumps(base_cfg))
for m in targets:
    new_cfg["quantization"].pop(m, None)     # mark unquantized for the loader
    new_cfg.get("quantization_config", {}).pop(m, None)

t0 = time.time()
out_map, errs = {}, []
shards = sorted(set(base_idx.values()))
for si, sh in enumerate(shards):
    dst = OUT / sh
    if dst.exists():
        for k in (k for k, v in base_idx.items() if v == sh):
            out_map[k] = sh
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
            want = (val.shape[0],
                    base_cfg["quantization"][mod].get("out", None) or 0, 0)
            # derive true shape from scales: [e, out, in/64]
            sc = data[mod + ".scales"]
            want = (sc.shape[0], sc.shape[1], sc.shape[2] * G)
            R, err = vq_tensor(li, proj, want)
            new[mod + ".weight"] = R
            errs.append(err)
            done.add(mod)
            print(f"    L{li:02d} {proj:10s} relerr {err:.4f}", flush=True)
            del R
            gc.collect()
            mx.clear_cache()
        elif name.rsplit(".", 1)[0] in done:
            continue          # drop .scales/.biases of a VQ'd module
        else:
            new[name] = val
    tmp = OUT / sh.replace(".safetensors", ".tmp.safetensors")
    mx.save_safetensors(str(tmp), new)
    tmp.rename(dst)
    for k in new:
        out_map[k] = sh
    del data, new
    gc.collect()
    mx.clear_cache()
    print(f"[{si+1}/{len(shards)}] {sh}  ({time.time()-t0:.0f}s)", flush=True)

tsz = sum((OUT / f).stat().st_size for f in shards)
json.dump({"metadata": {"total_size": tsz}, "weight_map": out_map},
          open(OUT / "model.safetensors.index.json", "w"))
json.dump(new_cfg, open(OUT / "config.json", "w"), indent=1)
for extra in BASE.iterdir():
    if extra.is_file() and extra.suffix != ".safetensors" \
            and extra.name not in ("config.json", "model.safetensors.index.json"):
        shutil.copy2(extra, OUT / extra.name)
    elif extra.is_dir() and extra.name == "optiq":
        shutil.copytree(extra, OUT / "optiq", dirs_exist_ok=True)
print(f"\ndone in {time.time()-t0:.0f}s  {tsz/2**30:.1f} GiB -> {OUT}")
print(f"VQ'd {len(errs)} tensors, mean relerr "
      f"{sum(errs)/max(len(errs),1):.4f}  (35B reference: 0.222 @ K1024)")
