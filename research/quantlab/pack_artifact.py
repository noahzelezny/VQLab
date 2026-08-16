#!/usr/bin/env python
"""Pack an existing (unpacked) VQ codes artifact into the sub-byte format.

    ./pack_artifact.py --src <unpacked-dir> --out <packed-dir>

This is a pure REPRESENTATION change: every decoded weight is bit-identical
to the source artifact's, so the packed model must referee to exactly the
same perplexity. Only `codes` tensors change (uint8/uint16 -> uint32 words);
codebooks, scales, and every non-VQ tensor are copied through untouched.

Sizes this unlocks (see vq_pack.py's table): d4 K128 100.1 GiB, d4 K2048
142.9 GiB — the points between the byte-aligned sizes, which is the whole
reason the format exists.

Writes model.py + config (model_file, vq_modules with pack_bits) via
add_model_file.py, so the result is still zero-patch loadable by stock
mlx_lm.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import mlx.core as mx
import numpy as np

import vq_pack

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--group", type=int, default=64)
args = ap.parse_args()

SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

idx_path = SRC / "model.safetensors.index.json"
weight_map = json.load(open(idx_path))["weight_map"]
shards = sorted(set(weight_map.values()))

# non-tensor files (config, tokenizer, template, README...) come along
for f in SRC.iterdir():
    if f.is_file() and f.suffix not in (".safetensors",):
        shutil.copy2(f, OUT / f.name)

vq_meta = {}
for si, sh in enumerate(shards, 1):
    data = mx.load(str(SRC / sh))
    out_data = {}
    n_packed = 0
    for key, val in data.items():
        if not key.endswith(".codes"):
            out_data[key] = val
            continue
        mod = key[:-len(".codes")]
        k = data[mod + ".codebook"].shape[0]
        dim = data[mod + ".codebook"].shape[1]
        bits = vq_pack.bits_for_k(k)
        codes = np.array(val, copy=False)
        nsub = codes.shape[2]
        packed = vq_pack.pack(codes.astype(np.uint16), bits)
        # verify THIS tensor round-trips before we let it out of the process:
        # a silent packing error decodes to plausible garbage, not an error.
        back = vq_pack.unpack(packed, nsub, bits)
        if not np.array_equal(back.astype(np.uint16), codes.astype(np.uint16)):
            sys.exit(f"FATAL: {mod} did not round-trip through the packer")
        out_data[key] = mx.array(packed)
        vq_meta[mod] = {"pack_bits": bits, "in": nsub * dim}
        n_packed += 1
    mx.save_safetensors(str(OUT / sh), out_data, metadata={"format": "mlx"})
    print(f"[{si}/{len(shards)}] {sh}: packed {n_packed} code tensors",
          flush=True)
    del data, out_data

shutil.copy2(idx_path, OUT / idx_path.name)

# seed config with pack_bits/in so add_model_file can read them back (it
# refuses to guess a bit width, by design)
cfg = json.load(open(OUT / "config.json"))
merged = cfg.get("vq_modules", {})
for mod, meta in vq_meta.items():
    merged.setdefault(mod, {}).update(meta)
cfg["vq_modules"] = merged
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)

subprocess.run([sys.executable, str(pathlib.Path(__file__).parent /
                                    "add_model_file.py"),
                "--artifact", str(OUT), "--group", str(args.group)], check=True)

src_gib = sum(f.stat().st_size for f in SRC.glob("*.safetensors")) / 1024 ** 3
out_gib = sum(f.stat().st_size for f in OUT.glob("*.safetensors")) / 1024 ** 3
print(f"\npacked {len(vq_meta)} tensors: {src_gib:.1f} -> {out_gib:.1f} GiB "
      f"({out_gib / src_gib:.3f}x)")
