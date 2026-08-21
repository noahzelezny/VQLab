#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Assemble a RUNNABLE dense VQ artifact from a quantized base + our VQ fits.

Generalized from build_e4b_vq.py, same contract: start from an already
quantized incumbent, DROP the affine tensors for the modules we are
replacing, splice in the VQ codes/codebook/scales, and keep everything else
byte-identical to the base — so any measured difference is attributable to
the VQ swap alone. Writes model.py so mlx_lm loads it with no user code.

fit_dense_vq.py alone is NOT a model: it emits only the 192 VQ'd mlp
tensors. This script is the other half of that pair, and the reason E95
could not be scored without it.

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
import json
import pathlib
import shutil

import mlx.core as mx

# family -> (fitter module template, base module template)
FAMILIES = {
    "qwen3_8": ("model.language_model.layers.{li}.mlp.{key}",
                "language_model.model.layers.{li}.mlp.{key}"),
    "gemma4_e4b": ("language_model.model.layers.{li}.mlp.{key}",
                   "language_model.model.layers.{li}.mlp.{key}"),
}
PROJS = ("gate_proj", "up_proj", "down_proj")

ap = argparse.ArgumentParser()
ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
ap.add_argument("--base", required=True,
                help="quantized incumbent to splice into; its non-MLP bytes "
                     "are carried through UNCHANGED and define what the "
                     "comparison means")
ap.add_argument("--mlp", required=True, help="fit_dense_vq.py output dir")
ap.add_argument("--out", required=True)
ap.add_argument("--dry-run", action="store_true",
                help="validate the name remap and print the plan; write nothing")
args = ap.parse_args()

FIT_TMPL, BASE_TMPL = FAMILIES[args.family]
BASE, MLP = pathlib.Path(args.base), pathlib.Path(args.mlp)
OUT = pathlib.Path(args.out)

base_idx = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
mlp_cfg = json.load(open(MLP / "config.json"))["vq_modules"]
base_mods = {k.rsplit(".", 1)[0] for k in base_idx}

# ---- build the fit->base remap and PROVE it lands, before touching bytes
FIT_TO_BASE, missing = {}, []
for fit_mod in mlp_cfg:
    li = int(fit_mod.split("layers.")[1].split(".")[0])
    key = fit_mod.rsplit(".", 1)[-1]
    base_mod = BASE_TMPL.format(li=li, key=key)
    if fit_mod != FIT_TMPL.format(li=li, key=key):
        raise SystemExit(f"FAIL: fitted module {fit_mod} does not match the "
                         f"{args.family} fitter template — wrong --family?")
    if base_mod not in base_mods:
        missing.append((fit_mod, base_mod))
    FIT_TO_BASE[fit_mod] = base_mod

if missing:
    print(f"FAIL: {len(missing)} fitted modules have no counterpart in the base.")
    for f, b in missing[:5]:
        print(f"    {f}\n      -> {b}   NOT IN BASE")
    raise SystemExit(2)
print(f"remap OK: {len(FIT_TO_BASE)}/{len(mlp_cfg)} fitted modules land on real "
      f"base modules")

DROP = {f"{b}." for b in FIT_TO_BASE.values()}


def is_dropped(k):
    return any(k.startswith(d) for d in DROP)


dropped = [k for k in base_idx if is_dropped(k)]
print(f"will drop {len(dropped)} affine tensors from the base "
      f"({len(dropped)//len(FIT_TO_BASE)} per module), carry "
      f"{len(base_idx)-len(dropped)}")

if args.dry_run:
    print("\nDRY RUN — nothing written.")
    raise SystemExit(0)

OUT.mkdir(parents=True, exist_ok=True)
mlp_w = mx.load(str(MLP / "model-00001-of-00001.safetensors"))

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

# ---- splice VQ mlp, renaming fitter -> base convention.
# Codes/scales are written [1, OUT, *] so verify_artifact (expert format) can
# read them; a DENSE module wants 2D, and 2D also keeps the venv's
# expert-shaped VQ hook off them — squeeze here, same as build_e4b_vq.py.
mlp_2d = {}
for k, v in mlp_w.items():
    mod, leaf = k.rsplit(".", 1)
    nk = f"{FIT_TO_BASE[mod]}.{leaf}"
    t = v[0] if (leaf in ("codes", "vq_scales") and v.ndim == 3) else v
    # Defensive width fix: the runtime shim allocates uint8 for K<=256, so a
    # uint16 codes tensor both doubles the bytes on disk and mismatches the
    # dtype the loader expects. fit_dense_vq.py hardcoded uint16 until
    # 2026-08-21; narrow here so artifacts built from OLD fit outputs are still
    # correct. Only narrows when it is provably lossless.
    if leaf == "codes":
        want = mx.uint8 if mlp_cfg[mod]["k"] <= 256 else mx.uint16
        if t.dtype != want:
            hi = int(mx.max(t).item())
            if want == mx.uint8 and hi > 255:
                raise SystemExit(f"FAIL: {mod} codes reach {hi} but k<=256 "
                                 f"implies uint8 — codebook/codes disagree.")
            print(f"  narrowing {mod} codes {t.dtype} -> {want}")
            t = t.astype(want)
    mlp_2d[nk] = t
shard_no += 1
name = f"model-{shard_no:05d}.safetensors"
mx.save_safetensors(str(OUT / name), mlp_2d)
for k in mlp_2d:
    new_map[k] = name
print(f"spliced {len(mlp_2d)} VQ mlp tensors (renamed to base convention, "
      f"codes squeezed to 2D)")

json.dump({"metadata": {}, "weight_map": new_map},
          open(OUT / "model.safetensors.index.json", "w"), indent=1)

cfg = json.load(open(BASE / "config.json"))
cfg["vq_linear"] = {FIT_TO_BASE[m]: dict(v) for m, v in mlp_cfg.items()}
cfg["vq_embed"] = {}
cfg["model_file"] = "model.py"
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)

# Reuse build_e4b_vq.py's loader shim verbatim rather than forking a second
# copy that can drift. Extraction is by string split, so it is guarded: a
# refactor of that file must fail HERE and loudly, not emit a model.py with a
# silently empty shim that fails at load time with an opaque error.
_src = pathlib.Path("build_e4b_vq.py").read_text()
try:
    SHIM = _src.split("SHIM = '''")[1].split("'''")[0]
except IndexError:
    raise SystemExit("FAIL: could not extract the loader shim from "
                     "build_e4b_vq.py — its SHIM literal moved. Fix the "
                     "extraction here rather than shipping a shim-less model.py.")
if "class Model" not in SHIM or "VQLinear" not in SHIM:
    raise SystemExit("FAIL: extracted shim is missing class Model / VQLinear.")
model_py = pathlib.Path("vq_dense.py").read_text() + SHIM
compile(model_py, "model.py", "exec")   # never ship a model.py that cannot parse
(OUT / "model.py").write_text(model_py)

for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "processor_config.json",
          "preprocessor_config.json"):
    if (BASE / f).exists():
        shutil.copy(BASE / f, OUT / f)

tot = sum(p.stat().st_size for p in OUT.glob("*.safetensors"))
print(f"\nartifact -> {OUT}\n  {tot/2**30:.2f} GiB of tensors")
