#!/usr/bin/env python3
"""sanitize-stream: convert a checkpoint to MLX layout ONE LAYER AT A TIME
through the runtime's own Model.sanitize, for families whose sanitize
materializes everything on load.

Why it exists (DeepSeek-V4-Flash, 2026-10-02): mlx-lm's deepseek_v4
sanitize mx.eval's every FP8 block dequant and every chunked expert stack,
so load_model(lazy=True) on the official release resident-materializes the
whole model and is OOM-killed on a 96 GiB and a 128 GiB box alike. The
sanitize itself is per-key (dequant, rename) plus per-layer (fuse attention
pairs, stack one layer's 256 experts), so feeding it one layer's tensors at
a time gives byte-identical output at one layer's memory.

The output is an EXACT teacher in the runtime's layout: whatever the
sanitize emits is saved as is (DeepSeek: routed experts as mxfp4 uint32 +
uint8 scales, FP8 attention dequantized to bf16), one shard per layer.
config.json gets `quantization` entries for exactly the modules that
came out quantized (a `.scales` sibling), so stock load_model builds the
same model the sanitize targeted.

Source tensors must be mx.load-able (F8_E8M0 relabelled U8 first).

    vqlab sanitize-stream --src <dir> --out <dir> [--layers 0-3]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import shutil
import sys
import time

import mlx.core as mx


def _group(key: str):
    m = re.match(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else -1          # -1 = top-level tensors


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab sanitize-stream")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", help="LO-HI subset (a preflight); default all")
    ap.add_argument("--expert-group", type=int, default=32,
                    help="group size recorded for quantized output modules "
                         "(mxfp4 = 32)")
    ap.add_argument("--mode", default="mxfp4")
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args(argv)

    from mlx_lm.utils import _get_classes
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    model_cls, args_cls = _get_classes(cfg)
    model = model_cls(args_cls.from_dict(cfg))   # never evaluated: params lazy
    # Keys the sanitize passes through but the runtime model has no slot for
    # (DeepSeek-V4-Flash-Vision-Exp: vision.*, aligner.*, image_*, the
    # image-token gate bias_vl) would make a strict load fail. Keep exactly
    # the model's own parameters and REPORT what was dropped, never silently.
    from mlx.utils import tree_flatten
    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    # quantized outputs carry .scales/.biases the unquantized model lacks
    model_mods = {k.rsplit(".", 1)[0] for k in model_keys}
    dropped = collections.Counter()

    wmap = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    groups = collections.defaultdict(list)
    for k in wmap:
        groups[_group(k)].append(k)
    order = sorted(groups)
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        order = [g for g in order if lo <= g <= hi]

    index, quant, t0 = {}, {}, time.time()
    for gi, g in enumerate(order):
        fn = f"model-{'top' if g < 0 else f'L{g:03d}'}.safetensors"
        dst = out / fn
        if dst.exists():                          # resumable, one file per group
            hdr = _header(dst)
            index.update({k: fn for k in hdr})
            quant.update(_quant_from_keys(hdr, a))
            print(f"[{gi + 1}/{len(order)}] {fn} exists, skip", flush=True)
            continue
        keys = groups[g]
        with mx.stream(mx.cpu):
            per_shard = collections.defaultdict(list)
            for k in keys:
                per_shard[wmap[k]].append(k)
            sub = {}
            for sh, ks in per_shard.items():
                data = mx.load(str(src / sh))
                sub.update({k: data[k] for k in ks})
                del data
            res = model.sanitize(sub)
            for k in [k for k in res if not _wanted(k, model_keys, model_mods)]:
                dropped[re.sub(r"\.\d+\.", ".N.", k)] += 1
                del res[k]
            mx.eval(list(res.values()))
        tmp = out / fn.replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(str(tmp), res, metadata={"format": "mlx"})
        tmp.rename(dst)
        index.update({k: fn for k in res})
        quant.update(_quant_from_keys(res, a))
        gib = dst.stat().st_size / 2**30
        print(f"[{gi + 1}/{len(order)}] {fn} {len(keys)} -> {len(res)} tensors, "
              f"{gib:.2f} GiB ({time.time() - t0:.0f}s)", flush=True)
        del sub, res
        mx.clear_cache()

    if dropped:
        print(f"dropped {sum(dropped.values())} tensors the runtime model has "
              f"no parameter for:", flush=True)
        for k, n in sorted(dropped.items()):
            print(f"  {n:5d} x {k}", flush=True)
    tot = sum((out / f).stat().st_size for f in set(index.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": tot}, "weight_map": index}, indent=1))
    new_cfg = dict(cfg)
    new_cfg.pop("quantization_config", None)     # the FP8 source recipe; consumed
    new_cfg["quantization"] = {"group_size": a.expert_group, "bits": a.bits,
                               "mode": a.mode, **quant}
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2))
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in (
                "config.json", "model.safetensors.index.json"):
            shutil.copy(f, out)
    print(f"saved {tot / 2**30:.1f} GiB, {len(index)} tensors, "
          f"{len(quant)} quantized modules ({time.time() - t0:.0f}s)", flush=True)
    return 0


def _wanted(k, model_keys, model_mods):
    if k in model_keys:
        return True
    if "." not in k:                      # top-level tensor (image_pad, ...)
        return False
    mod, leaf = k.rsplit(".", 1)
    return leaf in ("scales", "biases") and mod in model_mods


def _header(p):
    import struct
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return [k for k in json.loads(fh.read(n)) if k != "__metadata__"]


def _quant_from_keys(keys, a):
    return {k[: -len(".scales")]: {"group_size": a.expert_group, "bits": a.bits,
                                   "mode": a.mode}
            for k in keys if k.endswith(".scales")}


if __name__ == "__main__":
    sys.exit(main())
