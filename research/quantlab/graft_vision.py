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
ap.add_argument("--prefixes", default="model.visual,vision_tower",
                help="comma-separated key prefixes to graft. Default is the "
                     "Qwen set. gemma-4 needs "
                     "'vision_tower,embed_vision' (the tower is split across "
                     "two prefixes; grafting only one yields a broken tower).")
ap.add_argument("--copy-config-keys", default="vision_config,image_token_id",
                help="comma-separated config keys to copy from --src when the "
                     "artifact lacks them. mlx_lm.convert drops these for "
                     "text-only artifacts. DEFAULTS ON: this was opt-in until "
                     "2026-08-21 and no chain script ever passed it, so every "
                     "chain-built 397B artifact shipped without vision_config "
                     "and needed a hand-graft to be exo-loadable. Copies only "
                     "keys the artifact LACKS, so it never overwrites.")
ap.add_argument("--replace-config-keys", action="store_true",
                help="overwrite --copy-config-keys that are already present "
                     "and DISAGREE with --src. Off by default so a graft "
                     "never silently rewrites a config; required to repair an "
                     "artifact carrying another family's vision_config (every "
                     "35B packed before 2026-08-24 carried the 397B's).")
args = ap.parse_args()

ART, SRC = pathlib.Path(args.artifact), pathlib.Path(args.src)
GRAFT_SHARD = "model-vision-graft.safetensors"

src_map = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
# Qwen keeps the whole tower under one prefix; gemma-4 splits it across
# vision_tower.* AND embed_vision.* (356 tensors total), so a single-prefix
# filter silently grafts an incomplete tower. --prefixes is additive and the
# default reproduces the original Qwen behaviour exactly.
_PREFIXES = tuple(p for p in args.prefixes.split(",") if p)
vis = {k: sh for k, sh in src_map.items() if k.startswith(_PREFIXES)}
if not vis:
    raise SystemExit(f"source {SRC} has no vision tensors")

# BASE-IDENTITY CHECK, before anything is read or written.
#
# The first version of this compared the source's vision out_hidden_size to
# the artifact's hidden_size and called itself a family check. It is a WIDTH
# check. It passed a Qwen3.5 tower into a Qwen3.6 artifact on 2026-08-24
# because both project to 2048 — and 3.6 keys its tower `vision_tower.*`
# while 3.5 uses `model.visual.*`, so the graft also landed in a namespace
# the model does not read. Nothing downstream would have caught it:
# check_vision.py counts tensors, and the outlier gate never looks at the
# tower. `model_type` does not settle it either — a 3.6 artifact still
# reports qwen3_5_moe.
#
# The only thing that settles which model a source is: does it share a
# BYTE-IDENTICAL non-vision tensor with this artifact. Norms and biases are
# copied through the fit untouched, so the true base always matches on
# several; a different model, release, or family matches on none.
_artcfg = json.load(open(ART / "config.json"))
src_cfg = json.load(open(SRC / "config.json"))
_th = (_artcfg.get("text_config") or _artcfg).get("hidden_size")
_oh = (src_cfg.get("vision_config") or {}).get("out_hidden_size")
if _th is not None and _oh is not None and _th != _oh:
    raise SystemExit(
        f"FAIL: --src {SRC.name} has a tower projecting to {_oh}, but this "
        f"artifact's hidden_size is {_th}. Wrong model family — graft from "
        f"THIS model's base. Nothing written. (Re-pack from the fit rather "
        f"than rewriting config keys in place.)")

_art_map0 = json.load(open(ART / "model.safetensors.index.json"))["weight_map"]
_shared = [k for k in _art_map0
           if k in src_map and not k.startswith(_PREFIXES)]
if not _shared:
    raise SystemExit(
        f"FAIL: --src {SRC.name} shares NO tensor key with this artifact. "
        f"Either it is a different model or it uses a different key "
        f"namespace (Qwen3.5 keys text as model.language_model.*, Qwen3.6 as "
        f"language_model.model.*). This is not this artifact's base. "
        f"Nothing written.")

# prefer small 1-D tensors: norms/biases pass through a fit unchanged.
_probe = sorted(_shared, key=lambda k: ("norm" not in k, len(k)))[:6]
_hits = []
for _k in _probe:
    with mx.stream(mx.cpu):
        _a = mx.load(str(ART / _art_map0[_k]))[_k]
        _b = mx.load(str(SRC / src_map[_k]))[_k]
        mx.eval(_a, _b)
    if _a.shape == _b.shape and bool(
            mx.all(_a.astype(mx.float32) == _b.astype(mx.float32)).item()):
        _hits.append(_k)
if not _hits:
    raise SystemExit(
        f"FAIL: --src {SRC.name} shares {len(_shared)} tensor keys with this "
        f"artifact but NOT ONE is byte-identical (probed {len(_probe)}). "
        f"Same architecture, different model or release — its tower does not "
        f"belong to these weights. Nothing written.")
print(f"base-identity OK: {len(_hits)}/{len(_probe)} probed tensors are "
      f"byte-identical to {SRC.name} (e.g. {_hits[0]})")

idx_path = ART / "model.safetensors.index.json"
idx = json.load(open(idx_path))
art_map = idx["weight_map"]

# collect from however many source shards hold vision tensors
out = {}
for sh in sorted(set(vis.values())):
    # CPU-STREAM EAGER READ, per FINDINGS IV.1. These reads used to stay LAZY
    # across every source shard AND a `del data`, materialising only inside
    # save_safetensors below — a deferred read paid in a GPU command buffer,
    # which E123 proved can silently return ZEROS. This script runs on EVERY
    # published artifact, and check_vision.py verifies tensors are PRESENT,
    # not non-zero, so a zeroed graft would have passed every gate we own.
    # Third sibling of the same defect (build_dense_vq, pack_artifact,
    # pack_dense); found by sweeping for the pattern rather than by hitting it.
    with mx.stream(mx.cpu):
        data = mx.load(str(SRC / sh))
        picked = {k: data[k] for k, s in vis.items() if s == sh}
        mx.eval(list(picked.values()))
    out.update(picked)
    del data, picked
_dead = [k for k, v in out.items()
         if float(mx.max(mx.abs(v.astype(mx.float32))).item()) == 0.0]
if _dead:
    raise SystemExit(f"FAIL: {len(_dead)} vision tensors read as ALL ZERO "
                     f"(e.g. {_dead[:3]}) — deferred-read fault. Not writing.")
total = sum(v.size * v.dtype.size for v in out.values())
mx.save_safetensors(str(ART / GRAFT_SHARD), out, metadata={"format": "mlx"})
_rb = mx.load(str(ART / GRAFT_SHARD))
_bad = [k for k, v in _rb.items()
        if float(mx.max(mx.abs(v.astype(mx.float32))).item()) == 0.0]
if _bad:
    raise SystemExit(f"FAIL: {len(_bad)} vision tensors are ALL ZERO in the "
                     f"shard just written (e.g. {_bad[:3]}) — do not ship.")
del _rb

for k in vis:
    art_map[k] = GRAFT_SHARD
idx["weight_map"] = art_map
if "metadata" in idx and "total_size" in idx["metadata"]:
    idx["metadata"]["total_size"] += total
json.dump(idx, open(idx_path, "w"), indent=1)

cfg = json.load(open(ART / "config.json"))
if args.copy_config_keys:
    copied, stale = [], []
    for k in args.copy_config_keys.split(","):
        if not k or k not in src_cfg:
            continue
        if k not in cfg:
            cfg[k] = src_cfg[k]
            copied.append(k)
        elif cfg[k] != src_cfg[k]:
            if args.replace_config_keys:
                cfg[k] = src_cfg[k]
                copied.append(k + " (replaced)")
            else:
                stale.append(k)
    if stale:
        raise SystemExit(
            f"FAIL: artifact already carries {stale} and they DISAGREE with "
            f"{SRC.name}. The tower just written is {SRC.name}'s, so shipping "
            f"this config would describe a tower the artifact does not have. "
            f"Re-run with --replace-config-keys to repair. (The graft shard "
            f"is written and correct; only the config is unresolved.)")
    if copied:
        json.dump(cfg, open(ART / "config.json", "w"), indent=1)
        print(f"copied config keys from source: {copied}")
missing = [k for k in ("vision_config", "image_token_id") if k not in cfg]
print(f"grafted {len(out)} vision tensors ({total / 1024**3:.2f} GiB) -> "
      f"{ART / GRAFT_SHARD}")
if missing:
    print(f"WARNING: config lacks {missing} — exo will not build a "
          "VisionCardConfig without them; copy from the source config.")
else:
    print("config already carries vision_config + image_token_id: "
          "exo-ready as-is.")
