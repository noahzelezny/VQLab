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
ap.add_argument("--src", required=True,
                help="bf16 source model dir holding the vision tower")
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
ap.add_argument("--dest-prefix", default=None,
                help="rewrite the grafted keys' leading prefix to this. Needed "
                     "when the source is in HF layout (model.visual.*) and the "
                     "artifact is mlx-layout (vision_tower.*), e.g. the "
                     "Qwen3.8-27B bf16 against our 27B rungs. Verify the "
                     "rewritten names against the official mlx index before "
                     "trusting them — this flag renames, it does not check.")
ap.add_argument("--permute-conv5", action="store_true",
                help="apply transpose(0,2,3,4,1) to every 5-D tensor. mlx "
                     "stores the vision patch_embed conv CHANNELS-LAST: HF "
                     "(out,C,T,H,W) -> mlx (out,T,H,W,C). Measured on the "
                     "Qwen3.5-35B pair, where both layouts of the SAME model "
                     "exist: 332/333 tensors are identical as-is and exactly "
                     "one, patch_embed.proj.weight, needs this permutation. "
                     "Without it the tower loads with 333 tensors present and "
                     "a silently wrong patch embedding.")
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

# CROSS-LAYOUT KEY MAPPING. HF and mlx name the same tensor differently:
# HF `model.language_model.X` vs mlx `language_model.model.X`. A cross-layout
# graft (HF tower into an mlx artifact, --dest-prefix) is legitimate, but the
# identity probe still has to run — comparing raw names would find nothing
# shared and refuse a correct source. So try the direct intersection first,
# then the mapped one. The probe is never SKIPPED for cross-layout grafts;
# it is the only thing standing between us and a wrong tower.
def _pairs(art_keys):
    direct = [(k, k) for k in art_keys
              if k in src_map and not k.startswith(_PREFIXES)]
    if direct:
        return direct, "same-layout"
    mapped = []
    for k in art_keys:
        if k.startswith("language_model.model."):
            h = "model.language_model." + k[len("language_model.model."):]
            if h in src_map:
                mapped.append((k, h))
    return mapped, "HF<->mlx mapped"

_pair_list, _how = _pairs(list(_art_map0))
_shared = [a for a, _ in _pair_list]
_srckey = dict(_pair_list)
if _shared and _how != "same-layout":
    print(f"base-identity: comparing via {_how} key names "
          f"({len(_shared)} tensors)")
if not _shared:
    raise SystemExit(
        f"FAIL: --src {SRC.name} shares NO tensor key with this artifact. "
        f"Either it is a different model or it uses a different key "
        f"namespace (Qwen3.5 keys text as model.language_model.*, Qwen3.6 as "
        f"language_model.model.*). This is not this artifact's base. "
        f"Nothing written.")

# Prefer small 1-D tensors: norms/biases pass through a fit unchanged.
# SAMPLE WIDELY, and do not let one tensor FAMILY dominate the probe. The
# first version sorted norms first and took the six shortest names, which on
# Qwen3.8-27B drew six layer-norms and reported 0/6 against the artifact's
# actual base — that family stores RMSNorm as (1+w), so every layer-norm
# differs by exactly 1.0 while `linear_attn.norm` matches bit-for-bit. A probe
# that samples one shape of tensor inherits that shape's conventions.
_norms = [k for k in _shared if "norm" in k]
_other = [k for k in _shared if "norm" not in k]
_probe = ([_norms[i] for i in range(0, len(_norms), max(1, len(_norms) // 6))][:6]
          + sorted(_other, key=len)[:4])
_hits, _offset = [], []
for _k in _probe:
    with mx.stream(mx.cpu):
        _a = mx.load(str(ART / _art_map0[_k]))[_k]
        _sk = _srckey[_k]
        _b = mx.load(str(SRC / src_map[_sk]))[_sk]
        mx.eval(_a, _b)
    if _a.shape != _b.shape:
        continue
    _af, _bf = _a.astype(mx.float32), _b.astype(mx.float32)
    if bool(mx.all(_af == _bf).item()):
        _hits.append(_k)
    elif float(mx.max(mx.abs(_af - (_bf + 1.0))).item()) <= 0.01:
        # known (1+w) RMSNorm convention — evidence FOR the base, not against
        _offset.append(_k)
if _offset and not _hits:
    print(f"base-identity: {len(_offset)} tensors match under the (1+w) norm "
          f"convention, none bit-exact — treating as the base")
    _hits = _offset
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
if args.permute_conv5:
    _n5 = [k for k, v in out.items() if v.ndim == 5]
    for k in _n5:
        out[k] = mx.transpose(out[k], (0, 2, 3, 4, 1))
    mx.eval(list(out.values()))
    print(f"permuted {len(_n5)} 5-D tensor(s) to mlx channels-last: {_n5}")

if args.dest_prefix:
    _src_pref = sorted({k.split(".")[0] if not k.startswith("model.visual")
                        else "model.visual" for k in out}, key=len)[-1]
    out = {args.dest_prefix + k[len(_src_pref):]: v for k, v in out.items()}
    vis = {args.dest_prefix + k[len(_src_pref):]: v for k, v in vis.items()}
    print(f"renamed graft keys: {_src_pref}.* -> {args.dest_prefix}.*")

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
