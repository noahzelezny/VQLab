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

def _assert_tower_belongs(src_cfg, art_cfg, src_name, art_name):
    """Refuse a vision tower that cannot project into THIS model.

    Nothing in an artifact's config records which base it came from, so no
    smarter default can rescue a wrong --src: the only durable check is a
    CORRESPONDENCE assertion on whatever source is actually passed.

    This exists because a lab default pointed one family's packer at another
    family's bf16 dir, and five 35B artifacts were built carrying a 397B
    vision_config — 333 tensors present, presence checks green, and a tower
    dimensionally incapable of projecting into the model shipping it
    (out_hidden_size 4096 into hidden_size 2048). A gate that checks
    PRESENCE rather than CORRESPONDENCE always has this hole.
    """
    vc = (src_cfg or {}).get("vision_config") or {}
    out_h = vc.get("out_hidden_size")
    txt = (art_cfg or {}).get("hidden_size")
    if txt is None:
        txt = ((art_cfg or {}).get("text_config") or {}).get("hidden_size")
    if out_h is None or txt is None:
        return  # nothing to compare; say nothing rather than pretend
    if int(out_h) != int(txt):
        raise SystemExit(
            f"FAIL: refusing to graft. The tower in {src_name} projects to "
            f"out_hidden_size {out_h}, but {art_name} has hidden_size {txt}. "
            f"This source belongs to a different model family — nothing has "
            f"been written. Pass the --src that matches THIS artifact. If the "
            f"artifact ALREADY carries another family's vision block, re-pack "
            f"it from the fit: copy-if-absent cannot correct a key that is "
            f"present and wrong.")


def _assert_same_model(src_map, art_map, src_dir, art_dir, vis_prefixes,
                       probe_n=6):
    """Prove the source IS this artifact's base, by shared byte-identical tensors.

    The width assertion above is a correspondence check and is still NOT an
    identity check: two model generations can share a BYTE-IDENTICAL
    vision_config (Qwen3.5 and 3.6 both project to 2048), so width cannot
    separate them even in principle. Measured consequence: a 3.5 tower was
    grafted into a 3.6 artifact, and because 3.6 keys its tower
    `vision_tower.*` while 3.5 uses `model.visual.*`, 333 tensors landed in a
    namespace the model never reads — and the presence gate reported
    333/333 PASS.

    What actually settles identity is that non-vision tensors — norms and
    biases especially — pass through a fit UNTOUCHED. The true base shares
    several byte-identical ones with the artifact; a different model,
    release, or family shares none. Zero shared KEYS is decisive on its own,
    since a different generation does not share a key namespace.

    `config.model_type` is NOT the base and must not be used for this: a 3.6
    artifact reports `qwen3_5_moe`, which actively invites the wrong source.
    """
    vis = tuple(p for p in vis_prefixes if p)
    shared = [k for k in src_map
              if k in art_map and not k.startswith(vis)]
    if not shared:
        raise SystemExit(
            f"FAIL: refusing to graft. {src_dir} and {art_dir} share ZERO "
            f"non-vision tensor names. Different key namespaces mean a "
            f"different model generation or family — this is not this "
            f"artifact's base. Nothing written.")
    # norms first: small, untouched by any fit, and present in every model
    shared.sort(key=lambda k: (0 if "norm" in k else 1 if "bias" in k else 2, k))
    probe = shared[:probe_n]
    matches, checked = [], []
    for k in probe:
        try:
            with mx.stream(mx.cpu):
                a = mx.load(str(SRC / src_map[k]))[k]
                b = mx.load(str(ART / art_map[k]))[k]
                mx.eval(a, b)
                same = (a.shape == b.shape and a.dtype == b.dtype
                        and bool(mx.array_equal(a, b)))
        except Exception as e:                     # unreadable => not evidence
            checked.append(f"{k}: unreadable ({type(e).__name__})")
            continue
        checked.append(f"{k}: {'identical' if same else 'DIFFERS'}")
        if same:
            matches.append(k)
    if not matches:
        raise SystemExit(
            f"FAIL: refusing to graft. {src_dir} shares tensor NAMES with "
            f"{art_dir} but not one probed tensor is byte-identical:\n  "
            + "\n  ".join(checked)
            + f"\nThe source is not this artifact's base (a different "
              f"release will do this). Nothing written. Note config."
              f"model_type does NOT identify the base.")
    print(f"identity: {len(matches)}/{len(probe)} probed non-vision tensors "
          f"byte-identical -> source is this artifact's base")


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
args = ap.parse_args()

ART, SRC = pathlib.Path(args.artifact), pathlib.Path(args.src)
GRAFT_SHARD = "model-vision-graft.safetensors"

_src_cfg = json.load(open(SRC / "config.json")) if (SRC / "config.json").exists() else {}
_art_cfg = json.load(open(ART / "config.json")) if (ART / "config.json").exists() else {}
# BEFORE the shard is read or written: an abort placed after the write would
# leave the wrong family's tensors on disk and then fail.
_assert_tower_belongs(_src_cfg, _art_cfg, str(SRC), str(ART))

src_map = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
_art_map_early = json.load(open(ART / "model.safetensors.index.json"))["weight_map"]
# IDENTITY, before any tower read or write. Width above is a cheap early
# filter; this is what actually proves the source is this artifact's base.
_assert_same_model(src_map, _art_map_early, str(SRC), str(ART),
                   tuple(p for p in args.prefixes.split(",") if p))
# Qwen keeps the whole tower under one prefix; gemma-4 splits it across
# vision_tower.* AND embed_vision.* (356 tensors total), so a single-prefix
# filter silently grafts an incomplete tower. --prefixes is additive and the
# default reproduces the original Qwen behaviour exactly.
_PREFIXES = tuple(p for p in args.prefixes.split(",") if p)
vis = {k: sh for k, sh in src_map.items() if k.startswith(_PREFIXES)}
if not vis:
    raise SystemExit(f"source {SRC} has no vision tensors")

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
    src_cfg = json.load(open(SRC / "config.json"))
    copied = []
    for k in args.copy_config_keys.split(","):
        if k and k not in cfg and k in src_cfg:
            cfg[k] = src_cfg[k]
            copied.append(k)
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
