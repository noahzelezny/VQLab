#!/usr/bin/env python3
"""Assemble a RUNNABLE dense VQ artifact from a quantized base + our VQ fits.

Generalized from build_e4b_vq.py, same contract: start from an already
quantized incumbent, DROP the affine tensors for the modules we are
replacing, splice in the VQ codes/codebook/scales, and keep everything else
byte-identical to the base — so any measured difference is attributable to
the VQ swap alone. Writes model.py so mlx_lm loads it with no user code.

fit_dense_vq.py alone is NOT a model: it emits only the 192 VQ'd mlp
tensors. This script is the other half of that pair, and the reason E95
could not be scored without it. --embed is the other build_e4b_vq.py path:
an embedding table fitted by fit-ple (e4b's per-layer table) spliced as
vq_embed, which pack-dense packs and the shim loads as a VQEmbedding.

NAME REMAP (the part that is not obvious): the fitter names modules with the
SOURCE checkpoint's convention, because that is what it read. The mlx base
uses a different one. For qwen3_8:

    source (fitter output):  model.language_model.layers.0.mlp.gate_proj
    base   (mlx rung):       language_model.model.layers.0.mlp.gate_proj

They are not the same string and nothing downstream would catch the mismatch
except a strict load failure at score time. FIT_TO_BASE below is the map, and
--dry-run asserts every fitted module lands on a real base module BEFORE any
bytes are written.
"""
import argparse
import sys
import pathlib
import re

from vqlab import _layout

import mlx.core as mx

from vqlab.family.registry import DENSE_REMAP as FAMILIES  # shared registry
from vqlab.util.fs import GIB, read_json, write_json
from vqlab.core.artifacts.artifact import (
    Artifact,
    copy_config_files,
    copy_tokenizer_files,
    write_index,
)

PROJS = ("gate_proj", "up_proj", "down_proj")
# the KV-shared layers' tensors mlx_lm's gemma4 never instantiates (has_kv is
# False for layer >= num_hidden_layers - num_kv_shared_layers): an older
# mlx-community e4b-8bit shipped them and failed a strict load ("126
# parameters not in model"); build_e4b_vq.py dropped them, so does this
DEAD_KV = (".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.k_norm.")


def drop_replaced_quantization(cfg: dict) -> None:
    """A base quantized per module (stream-convert writes one entry per
    module) names the modules just replaced; the loader would then try to
    affine-quantize a VQLinear ("Unable to quantize model of type
    VQLinear"). Those entries describe bytes this artifact no longer has."""
    mods = [*cfg.get("vq_linear", {}), *cfg.get("vq_embed", {})]
    for key in ("quantization", "quantization_config"):
        if isinstance(cfg.get(key), dict):
            for mod in mods:
                cfg[key].pop(mod, None)
            # an entry under a replaced module would still reach the loader
            left = sorted(q for q in cfg[key] for mod in mods
                          if q == mod or q.startswith(mod + "."))
            if left:
                raise SystemExit(f"FAIL: {key} still names {len(left)} entr(y/ies) under VQ "
                                 f"modules (first: {left[0]}); the loader would affine-"
                                 "quantize a VQ module")


def dead_kv(base_idx, cfg: dict) -> list[str]:
    """The base's tensors for modules of KV-shared layers (DEAD_KV); empty
    when the family shares no KV or the base no longer carries them."""
    t = cfg.get("text_config", cfg)
    shared = t.get("num_kv_shared_layers") or 0
    if not shared:
        return []
    first = t["num_hidden_layers"] - shared
    rx = re.compile(r"\.layers\.(\d+)\.")
    return [k for k in base_idx if any(d in k for d in DEAD_KV)
            and (m := rx.search(k)) and int(m.group(1)) >= first]


def zero_tensors(tensors: dict) -> list[str]:
    """Names of the tensors that are ALL ZERO: the signature of a deferred-
    read fault (FINDINGS IV.1), refused before and after every write."""
    return [k for k, v in tensors.items()
            if float(mx.max(mx.abs(v.astype(mx.float32))).item()) == 0.0]


def load_eager(paths) -> dict:
    """Every tensor of `paths`, read on the CPU stream and refused if any is
    all zero.

    CPU-STREAM EAGER LOAD, per FINDINGS IV.1. This read used to stay LAZY
    until save_safetensors() forced it ~1300 tensors later, across the whole
    base-carry loop and its mx.clear_cache() — i.e. the deferred read was
    paid inside a GPU command buffer under memory pressure. On 2026-08-21
    that produced an artifact whose L60 up_proj codes/scales/codebook were
    ALL ZERO while the source file was provably clean, and a rebuild from
    identical inputs came out correct — the signature of a transient
    deferred read, not of a bad disk. eval INSIDE the block: creation-binding
    alone is measured-insufficient [E70 add. 5-6]."""
    out = {}
    with mx.stream(mx.cpu):
        for p in paths:
            out.update(mx.load(str(p)))
        mx.eval(list(out.values()))
    dead = zero_tensors(out)
    if dead:
        raise SystemExit(f"FAIL: {len(dead)} tensors read as ALL ZERO (e.g. {dead[:3]}). "
                         "Deferred-read fault or a corrupt fit output — do not write "
                         "an artifact.")
    return out


def narrow_codes(mod: str, t, k: int):
    """Codes at the width the runtime shim allocates (uint8 for K<=256): a
    uint16 tensor both doubles the bytes on disk and mismatches the dtype the
    loader expects. fit_dense_vq.py hardcoded uint16 until 2026-08-21; narrow
    here so artifacts built from OLD fit outputs are still correct. Only
    narrows when it is provably lossless."""
    want = mx.uint8 if k <= 256 else mx.uint16
    if t.dtype == want:
        return t
    hi = int(mx.max(t).item())
    if want == mx.uint8 and hi > 255:
        raise SystemExit(f"FAIL: {mod} codes reach {hi} but k<=256 "
                         f"implies uint8 — codebook/codes disagree.")
    print(f"  narrowing {mod} codes {t.dtype} -> {want}")
    return t.astype(want)


def mlp_remap(family: str, mlp_cfg: dict, base_mods: set) -> dict:
    """fitted module -> base module, PROVEN to land on real base modules
    before any bytes are written (the module docstring's NAME REMAP)."""
    fit_tmpl, base_tmpl = FAMILIES[family]
    fit_to_base, missing = {}, []
    for fit_mod in mlp_cfg:
        li = int(fit_mod.split("layers.")[1].split(".")[0])
        key = fit_mod.rsplit(".", 1)[-1]
        base_mod = base_tmpl.format(li=li, key=key)
        if fit_mod != fit_tmpl.format(li=li, key=key):
            raise SystemExit(f"FAIL: fitted module {fit_mod} does not match the "
                             f"{family} fitter template — wrong --family?")
        if base_mod not in base_mods:
            missing.append((fit_mod, base_mod))
        fit_to_base[fit_mod] = base_mod
    if missing:
        print(f"FAIL: {len(missing)} fitted modules have no counterpart in the base.")
        for f, b in missing[:5]:
            print(f"    {f}\n      -> {b}   NOT IN BASE")
        raise SystemExit(2)
    print(f"remap OK: {len(fit_to_base)}/{len(mlp_cfg)} fitted modules land on real "
          f"base modules")
    return fit_to_base


def embed_plan(manifest: dict, base_mods: set) -> dict:
    """{base module: vq_embed record} for every table a fit-ple output holds
    (its keys are source tensor names; the module is the name less
    ".weight"). A fitted table that names no base module is refused: a
    mapping bug, caught before anything is written."""
    geo, tensors = manifest.get("geometry") or {}, manifest.get("tensors") or {}
    if not tensors:
        raise SystemExit("FAIL: the fit-ple manifest lists no tensors; nothing written")
    plan, missing = {}, []
    for key, meta in tensors.items():
        mod = key.removesuffix(".weight")
        if mod not in base_mods:
            missing.append(mod)
        rows, cols = meta["shape"]
        plan[mod] = {"rows": rows, "in": cols, "k": geo["k"], "dim": geo["dim"],
                     "group": geo["group"]}
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} fitted tables name no tensor in the base, "
                         f"e.g. {missing[0]}; nothing written")
    print(f"embed OK: {len(plan)} fitted table(s) land on real base modules")
    return plan


def spliced(tensors: dict, renamed: dict, meta: dict) -> dict:
    """Fit tensors under the base's module names, 2D, codes narrowed.
    Codes/scales are written [1, OUT, *] by the mlp fitter so verify_artifact
    (expert format) can read them; a DENSE module wants 2D, and 2D also keeps
    the venv's expert-shaped VQ hook off them — squeeze here, same as
    build_e4b_vq.py. `renamed`: fitted module -> base module."""
    out = {}
    for k, v in tensors.items():
        mod, leaf = k.rsplit(".", 1)
        base = renamed[mod]
        t = v[0] if (leaf in ("codes", "vq_scales") and v.ndim == 3) else v
        if leaf == "codes":
            t = narrow_codes(base, t, meta[base]["k"])
        out[f"{base}.{leaf}"] = t
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--base", required=True,
                    help="quantized incumbent to splice into; its other bytes "
                         "are carried through UNCHANGED and define what the "
                         "comparison means")
    ap.add_argument("--mlp", help="fit_dense_vq.py output dir")
    ap.add_argument("--embed", help="fit-ple output dir: embedding table(s) to splice "
                                    "as vq_embed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the name remap and print the plan; write nothing")
    args = ap.parse_args(argv)
    if not (args.mlp or args.embed):
        ap.error("give --mlp, --embed or both")

    BASE, OUT = pathlib.Path(args.base), pathlib.Path(args.out)
    MLP = pathlib.Path(args.mlp) if args.mlp else None
    EMB = pathlib.Path(args.embed) if args.embed else None

    base_idx = Artifact(BASE).index
    base_cfg = read_json(BASE / "config.json")
    base_mods = {k.rsplit(".", 1)[0] for k in base_idx}

    # ---- every fitted module -> its base module, PROVEN before touching bytes
    mlp_cfg = read_json(MLP / "config.json")["vq_modules"] if MLP else {}
    FIT_TO_BASE = mlp_remap(args.family, mlp_cfg, base_mods) if MLP else {}
    manifest = read_json(EMB / "ple_manifest.json") if EMB else {}
    vq_embed = embed_plan(manifest, base_mods) if EMB else {}
    vq_linear = {FIT_TO_BASE[m]: dict(v) for m, v in mlp_cfg.items()}

    DROP = tuple(f"{b}." for b in [*vq_linear, *vq_embed])
    DEAD = set(dead_kv(base_idx, base_cfg))

    def is_dropped(k):
        return k.startswith(DROP) or k in DEAD

    dropped = [k for k in base_idx if is_dropped(k)]
    print(f"will drop {len(dropped)} base tensors ({len(DEAD)} of KV-shared layers' "
          f"modules the runtime never builds), carry {len(base_idx)-len(dropped)}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        raise SystemExit(0)

    OUT.mkdir(parents=True, exist_ok=True)
    # Read the fits FIRST, eagerly, and refuse a silently-zeroed read HERE,
    # loudly, not after it reaches disk and waits for the post-hoc gate.
    vq = {}
    if MLP:
        vq.update(spliced(load_eager([MLP / "model-00001-of-00001.safetensors"]),
                          FIT_TO_BASE, vq_linear))
    if EMB:
        shards = sorted({m["shard"] for m in manifest["tensors"].values()})
        # fit-ple names its tensors <source key>.codes, the key ending .weight
        vq.update(spliced(load_eager([EMB / s for s in shards]),
                          {f"{m}.weight": m for m in vq_embed},
                          vq_embed))

    # ---- carry through everything we are NOT replacing, shard by shard
    new_map, shard_no, carried = {}, 0, 0
    for sh in sorted(set(base_idx.values())):
        data = mx.load(str(BASE / sh))
        keep = {k: v for k, v in data.items() if not is_dropped(k)}
        carried += len(keep)
        if keep:
            shard_no += 1
            name = f"model-{shard_no:05d}.safetensors"
            mx.save_safetensors(str(OUT / name), keep)
            for k in keep:
                new_map[k] = name
        del data, keep
        mx.clear_cache()
    print(f"carried {carried} tensors from the base")

    shard_no += 1
    name = f"model-{shard_no:05d}.safetensors"
    mx.save_safetensors(str(OUT / name), vq)
    # READ-BACK ZERO SCAN. The gate catches this later, but the gate is a
    # separate invocation someone can forget; this makes the builder itself
    # refuse to emit a zeroed splice.
    bad = zero_tensors(mx.load(str(OUT / name)))
    if bad:
        raise SystemExit(f"FAIL: {len(bad)} tensors are ALL ZERO in the shard "
                         f"just written (e.g. {bad[:3]}). Write-side fault — "
                         f"artifact is incomplete, do not gate or score it.")
    for k in vq:
        new_map[k] = name
    print(f"spliced {len(vq)} VQ tensors ({len(vq_linear)} mlp, {len(vq_embed)} embed "
          f"module(s); base names, codes 2D)")

    # metadata.total_size from the shards just written: loaders size and
    # place a model by it, and check-release refuses an index without it
    write_index(OUT, new_map)

    cfg = base_cfg
    cfg["vq_linear"] = vq_linear
    cfg["vq_embed"] = vq_embed
    drop_replaced_quantization(cfg)
    cfg["model_file"] = "model.py"
    write_json(OUT / "config.json", cfg, indent=1)

    # BUNDLE = both runtimes + the loader shim, in that order.
    # vq_dense's fused path needs _dense_fused/_fused, which live in
    # vq_switch.py. Historically the bundle carried vq_dense ONLY and reached
    # into `mlx_lm.models.vq_switch` at call time, so a dense artifact's fast
    # path required a VQ-PATCHED mlx-lm and raised ModuleNotFoundError on a
    # stock install — the artifact scored fine and could not serve, which is
    # exactly the failure class III.11 exists for. Concatenating both puts
    # _dense_fused in the bundle's own globals (verified: the two modules share
    # zero top-level names), and vq_dense prefers that copy over the import.
    _here = pathlib.Path(__file__).parent
    from vqlab.runtime.dense_shim import SHIM
    if "class Model" not in SHIM or "VQLinear" not in SHIM:
        raise SystemExit("FAIL: shim is missing class Model / VQLinear.")
    model_py = (_layout.runtime_file("vq_switch.py").read_text()
                + _layout.runtime_file("vq_dense.py").read_text() + SHIM)
    compile(model_py, "model.py", "exec")   # never ship a model.py that cannot parse
    (OUT / "model.py").write_text(model_py)

    copy_tokenizer_files(BASE, OUT)
    copy_config_files(BASE, OUT)

    tot = sum(p.stat().st_size for p in OUT.glob("*.safetensors"))
    print(f"\nartifact -> {OUT}\n  {tot/GIB:.2f} GiB of tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
