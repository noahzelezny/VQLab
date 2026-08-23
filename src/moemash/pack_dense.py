#!/usr/bin/env python3
"""Pack a DENSE VQ artifact's codes in place (new output dir).

pack_artifact.py speaks the expert format ([E, OUT, NSUB]); dense e4b
artifacts carry 2D codes for VQLinear and row-table codes for VQEmbedding.
Same vq_pack layout (32 codes/block, NSUB %% 32 == 0), so the packed words
are identical to what the expert packer would emit — only the walking
differs. Loader shim reads pack_bits/in from config, so the artifact stays
self-contained.

    ./pack_dense.py --src <unpacked artifact> --out <packed artifact>
"""
import argparse
import json
import pathlib
import shutil

import mlx.core as mx
import numpy as np

import vq_pack

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(SRC / "config.json"))
vq_l, vq_e = cfg.get("vq_linear", {}), cfg.get("vq_embed", {})
idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
targets = {m + ".codes": m for m in list(vq_l) + list(vq_e)}

new_map, shard_no = {}, 0
packed_n = skipped = 0
for sh in sorted(set(idx.values())):
    # MATERIALISE ON THE CPU STREAM (FINDINGS IV.1). mx.load is LAZY, and
    # every tensor that is NOT packed below passes through still-lazy — its
    # read is then paid inside a GPU command buffer when save_safetensors
    # forces evaluation, which can silently yield ZEROS. That is the exact
    # mechanism confirmed in E123 (fitter, read returned zeros to one consumer
    # and correct bytes to the next) and the cause of the L60 zeroed splice in
    # build_dense_vq (013d2bb); pack_artifact had the same defect. This is the
    # THIRD file in the family, found 2026-08-21 before its first use on a
    # real dense rung. Do NOT remove the eval.
    with mx.stream(mx.cpu):
        data = dict(mx.load(str(SRC / sh)))
        mx.eval(list(data.values()))
    for k in list(data):
        m = targets.get(k)
        if m is None:
            continue
        meta = vq_l.get(m) or vq_e.get(m)
        K = meta["k"]
        bits = int(K - 1).bit_length()
        nsub = meta["in"] // meta["dim"]
        if nsub % 32:
            print(f"  SKIP {m}: NSUB={nsub} not %32");  skipped += 1
            continue
        if bits % 8 == 0:
            # byte-aligned: packing saves zero bytes and costs bit-field
            # extraction at decode (E70-era finding on the 397B). Copy through.
            continue
        codes = np.array(data[k]).astype(np.uint16)
        data[k] = mx.array(vq_pack.pack(codes[None], bits)[0])
        meta["pack_bits"] = bits
        packed_n += 1
    shard_no += 1
    name = f"model-{shard_no:05d}.safetensors"
    mx.save_safetensors(str(OUT / name), data)
    # Read back what we just wrote and refuse all-zero tensors. The write-side
    # twin of the assertion above; build_dense_vq carries the same check.
    with mx.stream(mx.cpu):
        back = mx.load(str(OUT / name))
        mx.eval(list(back.values()))
    zeroed = [k for k, v in back.items()
              if v.size and bool(mx.all(v == 0))]
    if zeroed:
        raise SystemExit(f"FATAL: {name} has {len(zeroed)} all-zero tensor(s) "
                         f"after write, first={zeroed[0]} — do not ship this "
                         f"artifact.")
    del back
    for k in data:
        new_map[k] = name
    del data
    mx.clear_cache()

json.dump({"metadata": {}, "weight_map": new_map},
          open(OUT / "model.safetensors.index.json", "w"), indent=1)
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)
for f in SRC.iterdir():
    if f.suffix in (".jinja",) or f.name.endswith(("tokenizer.json",
            "tokenizer_config.json", "generation_config.json",
            "processor_config.json", "preprocessor_config.json", "model.py")):
        shutil.copy(f, OUT / f.name)
tot = sum(p.stat().st_size for p in OUT.glob("*.safetensors"))
print(f"packed {packed_n} code tensors ({skipped} skipped) -> {OUT}")
print(f"  {tot/2**30:.2f} GiB of tensors")
