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
ap.add_argument("--expert-chunk", type=int, default=32)
ap.add_argument("--sample", type=int, default=2_000_000)
ap.add_argument("--stage-dir", default=None)
ap.add_argument("--ship-to", default=None,
                help="rsync each finished shard here, then delete local")
ap.add_argument("--geom", default=None,
                help="per-projection geometry override, e.g. "
                     "'gate_proj=d4k256,up_proj=d4k256,down_proj=d8k4096' "
                     "(E36: down_proj prefers d8; gate/up never recovers)")
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


def code_dtype(k):
    return mx.uint8 if k <= 256 else mx.uint16

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
    if "switch_mlp" not in name:
        return False
    li = layer_of(name)
    if not (LO <= li <= HI):
        return False
    mod = name.rsplit(".", 1)[0]
    q = base_cfg["quantization"].get(mod)
    return isinstance(q, dict) and q.get("bits") == 2


def kmeans(X, k, iters):
    step = max(50_000, int(5e8 / k))
    n = X.shape[0]
    C = X[mx.random.randint(0, n, (k,))]
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
        for s0 in range(0, n, 2_000_000):
            ab = a[s0:s0 + 2_000_000]
            oh = (ab[:, None] == mx.arange(k)[None, :]).astype(mx.float32)
            oh_sum = oh_sum + oh.T @ X[s0:s0 + 2_000_000]
            cnt = cnt + mx.sum(oh, axis=0)
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
    sk = f"model.language_model.layers.{li}.mlp.experts.{key}"
    T = mx.load(_shard_path(src_idx[sk]))[sk]
    if half is not None:
        mid = T.shape[1] // 2
        T = T[:, :mid, :] if half == 0 else T[:, mid:, :]
    return T


def vq_tensor_codes(li, proj, want_shape):
    """Fit one expert tensor -> (codes, codebook fp16, scales fp16, relerr)."""
    D, K = GEOM[proj]
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
    C16 = kmeans(mx.concatenate(samples, axis=0), K, args.iters).astype(mx.float16)
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
            new[mod + ".codes"] = codes
            new[mod + ".codebook"] = cb
            new[mod + ".vq_scales"] = vsc
            pd, pk = GEOM[proj]
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
