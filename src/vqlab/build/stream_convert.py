#!/usr/bin/env python3
"""Streaming affine convert for models bigger than RAM (or nearly).

Two lessons from the Flash-Next arc, both fatal to stock convert there:
- the Metal watchdog kills whole-shard eval of lazy quant graphs (even one
  MoE layer's), so tensors are eval'd ONE per command buffer;
- accumulating every evaluated tensor before save gets the process jetsam'd
  (q8 at ~186 GiB on a 96 GB box), so shards are written and RELEASED from
  the module tree as they fill.
Shard boundaries never split a module (weight/scales/biases co-located) —
fit_moe.py's shard-oriented splice depends on it.

Modes:
  default        uniform --bits/--group everywhere the recipe quantizes;
                 with --protect-bits P, MIXED affine: experts at --bits/--group,
                 every other quantized module at P/64 (the shape of e.g. a
                 2-bit-experts / 8-bit-rest community build)
  --struct       STRUCT BASE for the MoE fitter: expert modules (matched by
                 the family's target_substr) marked 2-bit placeholders,
                 everything else at --protect-bits

Recipe (measured on qwen4_exp; review per family): the family's keep_fp modules
(family.keep_fp; the router everywhere) bf16;
ngram/PLE tables at the global group (32, divides 160-wide rows); body at
group 64 via per-module override.

    vqlab stream-convert --src <hf snapshot> --out <dir> --bits 4
    vqlab stream-convert --src ... --out ... --struct --family qwen4_exp
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

from vqlab.core.artifacts.artifact import (
    Artifact,
    copy_config_files,
    copy_tokenizer_files,
    write_index,
)
from vqlab.family.registry import FAMILY
from vqlab.family import model_loader as runtime_load
from vqlab.family import FamilyPlugin, alias_fused, fused_projections, keep_fp
from vqlab.util.fs import GIB, read_json

def body_spec(path, *, target, bits, group=64, protect_bits=None, struct=False,
              keep=FamilyPlugin.keep_fp):
    """Quantization of one module outside --skeleton-from mode: False (bf16),
    True (the global default), or a per-module {group_size, bits}. `keep` is
    the family's full-precision suffixes (family.keep_fp)."""
    if path.endswith(keep):
        return False                         # the family's gates stay bf16 (E7)
    if "vision" in path or "visual" in path:
        # Under an mlx_vlm runtime the tower is part of the loaded model
        # (mlx_lm's sanitize drops it, so this branch never fires there).
        # Keep it bf16: 1.05 GiB on glm5_next, and a bf16 tower in the
        # struct base means graft_vision is NOT needed for this family —
        # the tower rides through the whole pipeline. PROVISIONAL until a
        # glm5_next struct base is actually built and its tower verified
        # non-zero (the IV deferred-read family of faults).
        return False
    if struct and target in path:
        return {"group_size": 64, "bits": 2}             # fitter marker
    if "ngram_embedding" in path:
        return True                                      # global (32, bits)
    if struct:
        return {"group_size": 64, "bits": protect_bits}
    if protect_bits is not None and target not in path:
        return {"group_size": 64, "bits": protect_bits}   # mixed affine
    return {"group_size": group, "bits": bits}


# An expert stack is one 3-D tensor (397B: 512 x 1024 x 4096 bf16, 4 GiB per
# projection). Evaluated as one lazy graph, its read and quantization ran in
# one GPU command buffer and past the Metal watchdog (2026-10-13: 4.7-5.0 s per
# stack, then a GPU Timeout; slabs alone did not help, since slicing a lazy
# load still reads the whole tensor inside the GPU's work). So the write loop
# first evaluates the stack's SOURCE on its own (with --cpu-load the read is
# bound to the CPU stream and never enters a GPU buffer), then quantizes it a
# slab of experts per eval. Affine groups lie along the last axis, so slabs
# are exact: byte-identical to the one-kernel quantization.
SLAB_BYTES = 512 << 20


class _SlabQuantizer:
    """wrap(quantize) quantizes a stack bigger than SLAB_BYTES a slab of
    experts at a time; pop(arr) hands the write loop, for any one of a
    stack's outputs (weight, scales, biases), the (source stack, slab
    outputs) to evaluate first, or None. pop drops the handles of all three
    outputs at once: one left behind keeps the read stack alive (a 35B ran
    to 89 GiB)."""

    def __init__(self):
        self._pending = {}     # id(joined output) -> (source stack, slab outputs, every id)

    def wrap(self, quantize):
        def q(w, *args, **kw):
            if w.ndim != 3 or w.nbytes <= SLAB_BYTES:
                return quantize(w, *args, **kw)
            n = max(1, SLAB_BYTES // (w.nbytes // w.shape[0]))
            parts = [quantize(w[i:i + n], *args, **kw) for i in range(0, w.shape[0], n)]
            out = tuple(mx.concatenate([p[j] for p in parts], axis=0)
                        for j in range(len(parts[0])))
            entry = (w, parts, [id(o) for o in out])
            for o in out:
                self._pending[id(o)] = entry
            return out
        return q

    def pop(self, arr):
        entry = self._pending.pop(id(arr), None)
        if entry is None:
            return None
        src, parts, ids = entry
        for i in ids:
            self._pending.pop(i, None)
        return src, parts


def _eval_slabs(src, parts):
    mx.eval(src)                  # the read alone (CPU stream with --cpu-load)
    for part in parts:
        mx.eval(part)             # one slab of experts per eval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="Loads the teacher through Knurlogic's architectures and refuses without them "
               "(stock mlx-lm folds some norms differently); VQLAB_VENDORED_ARCH=0 builds on "
               "stock mlx-lm deliberately.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--struct", action="store_true",
                    help="build a fitter struct base: experts -> 2-bit markers")
    ap.add_argument("--family", default="qwen4_exp", choices=sorted(FAMILY))
    ap.add_argument("--group", type=int, default=64, choices=(32, 64, 128),
                    help="affine group size of the body (default mode; the "
                         "expert modules when --protect-bits is given)")
    ap.add_argument("--protect-bits", type=int, default=None,
                    help="non-expert width. --struct: default 8. Default mode: "
                         "only when given -- experts (the family's target_substr) "
                         "at --bits/--group, every other quantized module at "
                         "PROTECT/64 (mixed affine); router stays bf16")
    ap.add_argument("--shard-gib", type=float, default=5.0)
    ap.add_argument("--cpu-load", action="store_true",
                    help="bind the teacher's lazy reads to the CPU stream at load "
                         "(model_loader: a cold multi-GB read inside a GPU command "
                         "buffer trips the Metal watchdog -- the 397B teacher)")
    ap.add_argument("--skeleton-from", metavar="ARTIFACT",
                    help="MATCHED-SKELETON comparator: copy every non-expert "
                         "module's quantization (bits/group, or bf16 when the "
                         "artifact leaves it unquantized) from ARTIFACT's config, "
                         "and quantize the expert modules affine at --bits. The "
                         "result differs from ARTIFACT only in how the experts are "
                         "stored -- the like-for-like comparison for a VQ build")
    a = ap.parse_args(argv)
    if a.struct and a.protect_bits is None:
        a.protect_bits = 8

    SRC, DST = pathlib.Path(a.src), pathlib.Path(a.out)
    SHARD = int(a.shard_gib * GIB)
    target = FAMILY[a.family]["target_substr"]
    KEEP = keep_fp(a.family)

    # Runtime per family registry (design decision 2026-08-29: provision for either
    # mlx-lm or mlx-vlm, never fork). mlx_lm families behave exactly as before;
    # an mlx_vlm family (glm5_next) loads the WHOLE model, vision tower
    # included — see the predicate's vision branch.
    runtime_load.require_vendored_arch()      # no stock-mlx-lm build by accident
    # Freed buffers go to MLX's cache, which grows to the device's memory by
    # default: on the 397B that pushed tens of GiB into swap (2026-10-13).
    mx.set_cache_limit(2 * GIB)
    model, config = runtime_load.load_for_family(a.family, SRC, lazy=True, cpu_stream=a.cpu_load)
    print(runtime_load.resolved_runtime_note(model), flush=True)   # III.13

    SKEL = None
    if a.skeleton_from:
        _ref = pathlib.Path(a.skeleton_from)
        _c = read_json(_ref / "config.json")
        _q = _c.get("quantization") or _c.get("text_config", {}).get("quantization") or {}
        # A module is quantized in the reference iff its .scales tensor exists
        # there; its width is the per-module entry, else the config default. (A
        # uniform base -- the dense 27B's -- has NO per-module entries, so the
        # config map alone would read every skeleton module as bf16.)
        _default = {"bits": _q.get("bits", 4), "group_size": _q.get("group_size", 64),
                    "mode": _q.get("mode", "affine")}
        _idx = Artifact(_ref).index
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
                return {"group_size": a.group, "bits": a.bits, "mode": "affine"}
            if path in SKEL:
                SKEL_HITS["skeleton"] += 1
                v = SKEL[path]
                return {"group_size": v.get("group_size", 64), "bits": v["bits"],
                        "mode": v.get("mode", "affine")}
            SKEL_HITS["bf16"] += 1
            return False                     # unquantized in the reference too
        return body_spec(path, target=target, bits=a.bits, group=a.group,
                         protect_bits=a.protect_bits, struct=a.struct, keep=KEEP)

    # Experts that ARRIVE quantized (deepseek_v4's teacher keeps the release's
    # mxfp4 experts) are never offered to the predicate by mlx's quantizer, so
    # they cannot count as hits; they ride through unchanged.
    PREQ = sum(1 for n, m in model.named_modules()
               if target in n and "Quantized" in type(m).__name__)
    gbits = a.protect_bits if a.struct else a.bits
    slabs = _SlabQuantizer()
    whole, mx.quantize = mx.quantize, slabs.wrap(mx.quantize)
    try:
        model, config = quantize_model(model, config, 32, gbits, mode="affine",
                                       quant_predicate=predicate)
    finally:
        mx.quantize = whole
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
        slab = slabs.pop(arr)
        if slab is not None:
            _eval_slabs(*slab)
            del slab
            mx.clear_cache()              # a freed 4 GiB stack must not sit in MLX's cache
        mx.eval(arr)
        shard[name] = arr
        size += arr.nbytes
        del arr
    if shard:
        flush(si)

    tot = sum((DST / f).stat().st_size for f in set(index.values()))
    # total_size = tensor data bytes (HF's definition), not file sizes: the
    # affine chain ships this index as-is, and check-release reads it
    write_index(DST, index)
    save_config(config, config_path=DST / "config.json")
    # mlx-vlm families: quantize_model/save_config can drop top-level keys the
    # VLM loader requires (vision_config, observed on glm5_next 2026-08-30 —
    # load_model then dies in VisionConfig.from_dict). Re-merge anything the
    # source config carried that the written one lost; source values are never
    # overwritten onto keys the convert legitimately changed.
    _src_cfg = read_json(SRC / "config.json")
    _dst_cfg = read_json(DST / "config.json")
    _missing = {k: v for k, v in _src_cfg.items() if k not in _dst_cfg}
    if _missing:
        _dst_cfg.update(_missing)
        (DST / "config.json").write_text(json.dumps(_dst_cfg, indent=2))
        print(f"re-merged config keys dropped by save path: {sorted(_missing)}",
              flush=True)
    # save_pretrained still writes what the teacher lacks (a tokenizer.json
    # for a sentencepiece-only source); every file the teacher HAS is its own
    load_tokenizer(SRC).save_pretrained(DST)
    copy_tokenizer_files(SRC, DST)
    copy_config_files(SRC, DST)
    for f in glob.glob(str(SRC / "*.py")):
        shutil.copy(f, DST)
    print(f"saved {tot/GIB:.1f} GiB in {len(set(index.values()))} shards "
          f"({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
