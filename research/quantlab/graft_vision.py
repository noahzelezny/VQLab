#!/usr/bin/env python
"""Graft the vision tower into a VQ artifact.

    ./graft_vision.py --artifact <dir> [--src <bf16 dir>]

Copies the 333 `model.visual.*` tensors (0.85 GiB bf16) from the source
model into a new shard and registers them in the artifact's index. The
config already carries `vision_config` / `image_token_id` (inherited from
the base), so after this the artifact is vision-complete ON DISK.

Who can use it (measured 2026-08-15, see EXPERIMENTS.md "Vision"):
- **exo**: works immediately, zero patches — its VisionCardConfig loads the
  tower from the artifact's own directory.
- **mlx_lm**: unaffected either way — `sanitize()` drops `model.visual.*`
  by construction, so text-only behavior is IDENTICAL before and after.
  (This is also why the graft cannot break existing referee numbers.)
- **mlx-vlm**: needs the `model_file` loader hook (upstream PR) to reach
  the VQ language model; the vision tensors themselves are stock.

Idempotent: re-running replaces the graft shard and its index entries.
"""
import argparse
import json
import pathlib

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/"
                                 "Qwen--Qwen3.5-397B-A17B-bf16")
args = ap.parse_args()

ART, SRC = pathlib.Path(args.artifact), pathlib.Path(args.src)
GRAFT_SHARD = "model-vision-graft.safetensors"

src_map = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
vis = {k: sh for k, sh in src_map.items()
       if k.startswith("model.visual") or k.startswith("vision_tower")}
if not vis:
    raise SystemExit(f"source {SRC} has no vision tensors")

idx_path = ART / "model.safetensors.index.json"
idx = json.load(open(idx_path))
art_map = idx["weight_map"]

# collect from however many source shards hold vision tensors
out = {}
for sh in sorted(set(vis.values())):
    data = mx.load(str(SRC / sh))
    for k in (k for k, s in vis.items() if s == sh):
        out[k] = data[k]
    del data
total = sum(v.size * v.dtype.size for v in out.values())
mx.save_safetensors(str(ART / GRAFT_SHARD), out, metadata={"format": "mlx"})

for k in vis:
    art_map[k] = GRAFT_SHARD
idx["weight_map"] = art_map
if "metadata" in idx and "total_size" in idx["metadata"]:
    idx["metadata"]["total_size"] += total
json.dump(idx, open(idx_path, "w"), indent=1)

cfg = json.load(open(ART / "config.json"))
missing = [k for k in ("vision_config", "image_token_id") if k not in cfg]
print(f"grafted {len(out)} vision tensors ({total / 1024**3:.2f} GiB) -> "
      f"{ART / GRAFT_SHARD}")
if missing:
    print(f"WARNING: config lacks {missing} — exo will not build a "
          "VisionCardConfig without them; copy from the source config.")
else:
    print("config already carries vision_config + image_token_id: "
          "exo-ready as-is.")
