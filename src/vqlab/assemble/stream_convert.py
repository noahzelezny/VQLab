#!/usr/bin/env python3
"""Streaming affine convert for models bigger than RAM (or nearly).

Two lessons from the Flash-Next arc, both fatal to stock convert there:
- the Metal watchdog kills whole-shard eval of lazy quant graphs (even one
  MoE layer's), so tensors are eval'd ONE per command buffer;
- accumulating every evaluated tensor before save gets the process jetsam'd
  (q8 at ~186 GiB on a 96 GB box), so shards are written and RELEASED from
  the module tree as they fill.
Shard boundaries never split a module (weight/scales/biases co-located) —
vq_397b_codes.py's shard-oriented splice depends on it.

Modes:
  default        uniform --bits everywhere the recipe quantizes
  --struct       STRUCT BASE for the MoE fitter: expert modules (matched by
                 the family's target_substr) marked 2-bit placeholders,
                 everything else at --protect-bits

Recipe (measured on qwen4_exp; review per family): router (mlp.gate) bf16;
ngram/PLE tables at the global group (32, divides 160-wide rows); body at
group 64 via per-module override.

    python -m vqlab.stream_convert --src <hf snapshot> --out <dir> --bits 4
    python -m vqlab.stream_convert --src ... --out ... --struct --family qwen4_exp
"""
import argparse
import gc
import glob
import json
import pathlib
import shutil
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
# load_model is NOT imported here — every model load routes through
# runtime_load.load_for_family. quantize_model/save_config/load_tokenizer
# are runtime-agnostic (generic nn-tree walk / json / transformers) and are
# used for BOTH runtimes; quantize_model on an mlx_vlm-loaded model is
# PROVISIONAL until a glm5_next struct base is built.
from mlx_lm.utils import load_tokenizer, quantize_model, save_config

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
from families import FAMILY
import runtime_load
from vqlab.family import alias_fused, fused_projections  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--struct", action="store_true",
                help="build a fitter struct base: experts -> 2-bit markers")
ap.add_argument("--family", default="qwen4_exp", choices=sorted(FAMILY))
ap.add_argument("--protect-bits", type=int, default=8)
ap.add_argument("--shard-gib", type=float, default=5.0)
ap.add_argument("--skeleton-from", metavar="ARTIFACT",
                help="MATCHED-SKELETON comparator: copy every non-expert "
                     "module's quantization (bits/group, or bf16 when the "
                     "artifact leaves it unquantized) from ARTIFACT's config, "
                     "and quantize the expert modules affine at --bits. The "
                     "result differs from ARTIFACT only in how the experts are "
                     "stored -- the like-for-like comparison for a VQ build")
a = ap.parse_args()

SRC, DST = pathlib.Path(a.src), pathlib.Path(a.out)
SHARD = int(a.shard_gib * 2**30)
target = FAMILY[a.family]["target_substr"]

# Runtime per family registry (design decision 2026-08-29: provision for either
# mlx-lm or mlx-vlm, never fork). mlx_lm families behave exactly as before;
# an mlx_vlm family (glm5_next) loads the WHOLE model, vision tower
# included — see the predicate's vision branch.
model, config = runtime_load.load_for_family(a.family, SRC, lazy=True)
print(runtime_load.resolved_runtime_note(model), flush=True)   # III.13

SKEL = None
if a.skeleton_from:
    _ref = pathlib.Path(a.skeleton_from)
    _c = json.load(open(_ref / "config.json"))
    _q = _c.get("quantization") or _c.get("text_config", {}).get("quantization") or {}
    # A module is quantized in the reference iff its .scales tensor exists
    # there; its width is the per-module entry, else the config default. (A
    # uniform base -- the dense 27B's -- has NO per-module entries, so the
    # config map alone would read every skeleton module as bf16.)
    _default = {"bits": _q.get("bits", 4), "group_size": _q.get("group_size", 64),
                "mode": _q.get("mode", "affine")}
    _idx = json.load(open(_ref / "model.safetensors.index.json"))["weight_map"]
    # language_model.* in a VL checkpoint; the lazily loaded text model's
    # paths may or may not carry the prefix, so key both spellings.
    SKEL = {}
    for k in _idx:
        if k.endswith(".scales"):
            m = k[: -len(".scales")]
            v = _q.get(m) if isinstance(_q.get(m), dict) else _default
            SKEL[m] = v
            SKEL[m.removeprefix("language_model.")] = v
    # FUSED ALIASES: a runtime whose sanitize fuses two projections that share
    # their input (deepseek_v4: attn.wq_a + attn.wkv -> attn.wqkv_a) meets a
    # reference that stores them apart. The table lives in the family plugin
    # (vqlab.family.<name>.SPEC.fused); see family.alias_fused for the rule.
    try:
        alias_fused(SKEL, fused_projections(a.family))
    except ValueError as e:
        sys.exit(f"FAIL: {e}")
    SKEL_HITS = {"skeleton": 0, "expert": 0, "bf16": 0}


def predicate(path, module):
    if SKEL is not None:
        if "vision" in path or "visual" in path:
            return False
        if target in path:
            SKEL_HITS["expert"] += 1
            return {"group_size": 64, "bits": a.bits, "mode": "affine"}
        if path in SKEL:
            SKEL_HITS["skeleton"] += 1
            v = SKEL[path]
            return {"group_size": v.get("group_size", 64), "bits": v["bits"],
                    "mode": v.get("mode", "affine")}
        SKEL_HITS["bf16"] += 1
        return False                     # unquantized in the reference too
    if path.endswith("mlp.gate"):
        return False                                     # router bf16 (E7)
    if "vision" in path or "visual" in path:
        # Under an mlx_vlm runtime the tower is part of the loaded model
        # (mlx_lm's sanitize drops it, so this branch never fires there).
        # Keep it bf16: 1.05 GiB on glm5_next, and a bf16 tower in the
        # struct base means graft_vision is NOT needed for this family —
        # the tower rides through the whole pipeline. PROVISIONAL until a
        # glm5_next struct base is actually built and its tower verified
        # non-zero (the IV deferred-read family of faults).
        return False
    if a.struct and target in path:
        return {"group_size": 64, "bits": 2}             # fitter marker
    if "ngram_embedding" in path:
        return True                                      # global (32, bits)
    b = a.protect_bits if a.struct else a.bits
    return {"group_size": 64, "bits": b}

# Experts that ARRIVE quantized (deepseek_v4's teacher keeps the release's
# mxfp4 experts) are never offered to the predicate by mlx's quantizer, so
# they cannot count as hits; they ride through unchanged.
PREQ = sum(1 for n, m in model.named_modules()
           if target in n and "Quantized" in type(m).__name__)
gbits = a.protect_bits if a.struct else a.bits
model, config = quantize_model(model, config, 32, gbits, mode="affine",
                               quant_predicate=predicate)
if SKEL is not None:
    print(f"skeleton-from {a.skeleton_from}: {SKEL_HITS}"
          + (f"; {PREQ} expert modules arrived already quantized and are kept as-is" if PREQ else ""),
          flush=True)
    if (SKEL_HITS["expert"] == 0 and not PREQ) or SKEL_HITS["skeleton"] == 0:
        sys.exit("FAIL: --skeleton-from matched no expert or no skeleton "
                 "module; path spellings differ between the artifact and the "
                 "loaded model")

DST.mkdir(parents=True, exist_ok=True)
leaves = tree_flatten(model.parameters())
total = len(leaves)
index, shard, size, si, t0 = {}, {}, 0, 1, time.time()
prev_mod = None

def flush(si):
    fn = f"model-{si:05d}.safetensors"
    mx.save_safetensors(str(DST / fn), shard, metadata={"format": "mlx"})
    for k in shard:
        index[k] = fn
    model.update(tree_unflatten([(k, mx.zeros((0,))) for k in shard]))
    shard.clear()
    gc.collect(); mx.clear_cache()

for i in range(total):
    name, arr = leaves[i]
    leaves[i] = None            # the list must not keep evaluated arrays alive
    mod = name.rsplit(".", 1)[0]
    if size >= SHARD and shard and mod != prev_mod:
        flush(si)
        print(f"shard {si} at leaf {i}/{total} ({time.time()-t0:.0f}s)",
              flush=True)
        si += 1; size = 0
    prev_mod = mod
    mx.eval(arr)
    shard[name] = arr
    size += arr.nbytes
    del arr
if shard:
    flush(si)

tot = sum((DST / f).stat().st_size for f in set(index.values()))
(DST / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {"total_size": tot}, "weight_map": index}, indent=1))
save_config(config, config_path=DST / "config.json")
# mlx-vlm families: quantize_model/save_config can drop top-level keys the
# VLM loader requires (vision_config, observed on glm5_next 2026-08-30 —
# load_model then dies in VisionConfig.from_dict). Re-merge anything the
# source config carried that the written one lost; source values are never
# overwritten onto keys the convert legitimately changed.
_src_cfg = json.load(open(SRC / "config.json"))
_dst_cfg = json.load(open(DST / "config.json"))
_missing = {k: v for k, v in _src_cfg.items() if k not in _dst_cfg}
if _missing:
    _dst_cfg.update(_missing)
    (DST / "config.json").write_text(json.dumps(_dst_cfg, indent=2))
    print(f"re-merged config keys dropped by save path: {sorted(_missing)}",
          flush=True)
load_tokenizer(SRC).save_pretrained(DST)
for pat in ("*.py", "generation_config.json"):
    for f in glob.glob(str(SRC / pat)):
        shutil.copy(f, DST)
print(f"saved {tot/2**30:.1f} GiB in {len(set(index.values()))} shards "
      f"({time.time()-t0:.0f}s)", flush=True)
