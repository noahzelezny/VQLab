#!/usr/bin/env python3
"""Rebuild Qwen3.8-Flash-Next-VQ-2.1bpw with its DENSE (non-expert) tensors
at 6-bit/g64 instead of 8-bit/g64. Everything else byte-identical.

PROVENANCE (verified, not assumed — see research/flash-next-recipe/LEDGER.md):

  The shipped 2.1bpw artifact's 726 dense modules are BYTE-IDENTICAL to
  `Qwen--Qwen3.8-Flash-Next-VQ-BASE`, the struct base produced by
  `vqlab stream-convert --struct --family qwen4_exp --protect-bits 8`
  from the bf16 Qwen3.8-Flash-Next checkpoint (sha256 spot-check on
  in_proj_qkv / out_proj.scales / lm_head / embed_tokens: identical).
  So the dense side was quantized DIRECTLY FROM bf16 at 8-bit/g64.

  The bf16 checkpoint is no longer on the SSD (only the affine ladder
  rungs and the VQ base survive). The 6-bit dense tensors are therefore
  taken from `Qwen--Qwen3.8-Flash-Next-6bit`, the affine q6 ladder rung
  (137 GiB, Aug 28 11:25, i.e. BEFORE the VQ base was cut) — itself a
  direct bf16 -> 6-bit/g64 affine conversion by the same tooling, with an
  identical per-module recipe (873-key quantization config, group 64 on
  the body, bf16 router/norms). This is a bf16-sourced requantization,
  NOT an 8-bit -> 6-bit requantization.

WHAT CHANGES: the 2178 tensors (weight/scales/biases) of the 726 dense
modules, in the 9 trunk shards that hold them, and their bits: 8 -> 6 in
config.json's per-module quantization map.

WHAT DOES NOT: every VQ expert code/codebook, every PLE shard, the bf16
vision graft, model.py, the tokenizer, the index's weight_map, and the
config's vq_modules / vq_ple / vision_config blocks. Untouched tensors are
copied as raw bytes, so they are byte-identical by construction.
"""
import argparse
import json
import os
import pathlib
import struct
import sys

ART = "/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw"
Q6 = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.8-Flash-Next-6bit"
OUT = "/Volumes/Thunderbay SSD/Exo Models/flashnext_dense6_experiment"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=ART, help="shipped 2.1bpw artifact (READ-ONLY)")
ap.add_argument("--dense-src", default=Q6, help="bf16-derived affine q6 rung")
ap.add_argument("--out", default=OUT)
ap.add_argument("--bits", type=int, default=6)
a = ap.parse_args()

SRC, DSRC, DST = (pathlib.Path(a.src), pathlib.Path(a.dense_src), pathlib.Path(a.out))
DST.mkdir(parents=True, exist_ok=True)

DTSIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U16": 2, "U8": 1, "I32": 4}


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


cfg = json.loads((SRC / "config.json").read_text())
dense = {k for k, v in cfg["quantization"].items() if isinstance(v, dict)}
src_map = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
d_map = json.loads((DSRC / "model.safetensors.index.json").read_text())["weight_map"]

# shards that hold at least one dense tensor -> rewritten; the rest hardlinked
touched = sorted({s for k, s in src_map.items() if k.rsplit(".", 1)[0] in dense})
print(f"{len(dense)} dense modules; {len(touched)} shards to rewrite:", touched)

# --- pass 1: hardlink everything that is not a rewritten shard ----------
SKIP = {".DS_Store", "model.py.pre-e141", "mtp-head-q6.safetensors", "__pycache__",
        "config.json", "model.safetensors.index.json"}
for p in sorted(SRC.iterdir()):
    if p.name in SKIP or p.name in touched or p.is_dir():
        continue
    q = DST / p.name
    if q.exists():
        continue
    os.link(p, q)          # same volume: free, and proves byte-identity
print("hardlinked", len(list(DST.iterdir())), "files")

# --- pass 2: rewrite the trunk shards ----------------------------------
open_d = {}


def dsrc_bytes(key):
    """raw tensor bytes + dtype/shape from the q6 rung"""
    sh = d_map[key]
    if sh not in open_d:
        open_d[sh] = (read_header(DSRC / sh), open(DSRC / sh, "rb"))
    (hdr, off), fh = open_d[sh]
    v = hdr[key]
    s, e = v["data_offsets"]
    fh.seek(off + s)
    return fh.read(e - s), v["dtype"], v["shape"]


total = 0
for sh in touched:
    hdr, off = read_header(SRC / sh)
    meta = hdr.get("__metadata__", {"format": "mlx"})
    keys = [k for k in hdr if k != "__metadata__"]
    new_hdr, cursor, plan = {}, 0, []
    for k in keys:
        if k.rsplit(".", 1)[0] in dense:
            blob, dt, shape = dsrc_bytes(k)
            plan.append(("new", blob))
            n = len(blob)
        else:
            v = hdr[k]
            dt, shape = v["dtype"], v["shape"]
            s, e = v["data_offsets"]
            plan.append(("old", (s, e)))
            n = e - s
        new_hdr[k] = {"dtype": dt, "shape": shape,
                      "data_offsets": [cursor, cursor + n]}
        cursor += n
    new_hdr["__metadata__"] = meta
    blob_hdr = json.dumps(new_hdr, separators=(",", ":")).encode()
    pad = (-(len(blob_hdr)) % 8)
    blob_hdr += b" " * pad
    tmp = DST / (sh + ".tmp")
    with open(SRC / sh, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob_hdr)))
        fout.write(blob_hdr)
        for kind, payload in plan:
            if kind == "new":
                fout.write(payload)
            else:
                s, e = payload
                fin.seek(off + s)
                left = e - s
                while left:
                    chunk = fin.read(min(left, 1 << 24))
                    fout.write(chunk)
                    left -= len(chunk)
    tmp.rename(DST / sh)
    total += cursor
    print(f"  {sh}: {cursor / 2**30:.2f} GiB", flush=True)

for (_, fh) in open_d.values():
    fh.close()

# --- pass 3: config + index -------------------------------------------
for k in dense:
    cfg["quantization"][k] = {"group_size": 64, "bits": a.bits}
    if "quantization_config" in cfg and isinstance(cfg["quantization_config"], dict) \
            and k in cfg["quantization_config"]:
        cfg["quantization_config"][k] = {"group_size": 64, "bits": a.bits}
(DST / "config.json").write_text(json.dumps(cfg, indent=2))

idx = json.loads((SRC / "model.safetensors.index.json").read_text())
tot = 0
for p in sorted(set(src_map.values())):
    h, o = read_header(DST / p)
    tot += max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
idx["metadata"]["total_size"] = tot
(DST / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
print(f"total_size {tot / 2**30:.3f} GiB -> {DST}")
