#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Build a 397B variant off the spicymirror allocation, with a byte budget.

Baseline = spicymirror (E24, PPL 3.1775, 121G): 2-bit experts, 8-bit
structure, 4-bit linear_attn in_proj_qkv/z, routers/gates/in_proj_a,b bf16.

Variants change the allocation and PAY FOR IT, so comparisons are at matched
size (the only honest way to read a PPL delta):

  --structure-bits 6      demote 8-bit structure (frees bytes)
  --tail-expert-bits 3    promote the last --tail-layers expert layers
  --tail-layers N         how many layers from the end

Size is PREDICTED from the source parameter counts before the build starts —
a variant that overruns the baseline is a different point on the curve, not a
comparison, so the script prints the delta and refuses (unless --force) if it
grows by more than --tolerance-gb.

Usage:
  ./convert_variant.py --name struct6-tail3 --structure-bits 6 \
      --tail-expert-bits 3 --tail-layers 2
"""
import argparse
import json
import pathlib

import mlx.core as mx

mx.set_default_device(mx.cpu)

SPICY = "/Users/noahzelezny/.exo/models/spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit"
SRC = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16"
OUT_ROOT = "/Volumes/Thunderbay SSD/Exo Models"
# spicymirror was deleted in the post-release cleanup (08-19); the published
# VQ-2.4bpw carries the same bf16 vision sidecar, towers verified at debut
# (E61). Same tensors either way — every build shares one bf16 tower.
VISION_SRC = ("/Volumes/Thunderbay SSD/Exo Models/"
              "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw")

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="suffix: …-A17B-<name>")
ap.add_argument("--structure-bits", type=int, default=8)
ap.add_argument("--tail-expert-bits", type=int, default=2)
ap.add_argument("--tail-layers", type=int, default=0)
ap.add_argument("--promote-every", type=int, default=0,
                help="instead of a contiguous tail, promote every Nth expert "
                     "layer (spread across depth). E25 control: separates "
                     "'late layers matter' from 'more expert bits anywhere "
                     "matter'. Same layer COUNT as --tail-layers = same size.")
ap.add_argument("--qkv-bits", type=int, default=4,
                help="linear_attn in_proj_qkv/z bits (t2.1 died at 2)")
ap.add_argument("--expert-schedule", default=None,
                help="GRADED expert allocation, overrides --tail-layers/"
                     "--tail-expert-bits. Format 'lo-hi:bits,…' over layer "
                     "indices, e.g. '0-19:2,20-39:3,40-59:4'. Exists because "
                     "the tail dial can only promote off a fixed 2-bit base, "
                     "which cannot express the interesting test at a bigger "
                     "budget: at ~164 GiB, is a 2/3/4 RAMP better than a flat "
                     "3-bit at the SAME bytes? (E25 found position real — "
                     "tail10 3.0157 beat spread10 3.0490 at matched size — so "
                     "the ramp should win, and a flat build spends the tail "
                     "discovery instead of using it.) Layers not covered by "
                     "any range keep their source bits.")
ap.add_argument("--low-bit-group-size", type=int, default=None,
                help="E33: group size for expert tensors that end up at "
                     "2 bits (finer grid; gs32 = 2.5->3.0 effective bpw "
                     "there). Other tensors keep their source group size.")
ap.add_argument("--tolerance-gb", type=float, default=1.0)
ap.add_argument("--force", action="store_true")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

cfg = json.load(open(SPICY + "/config.json"))["quantization"]
overrides = {k: v for k, v in cfg.items() if isinstance(v, dict)}
default = {"group_size": cfg["group_size"], "bits": cfg["bits"],
           "mode": cfg.get("mode", "affine")}
wmap = json.load(open(SPICY + "/model.safetensors.index.json"))["weight_map"]
quantized = {k[: -len(".scales")] for k in wmap if k.endswith(".scales")}

n_layers = json.load(open(SRC + "/config.json"))["text_config"]["num_hidden_layers"]
tail_start = n_layers - args.tail_layers


def layer_index(path):
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return int(p)
    return -1


def _parse_schedule(spec):
    """'0-19:2,20-39:3,40-59:4' -> [(lo, hi, bits), …]. Ranges are inclusive.

    Validated eagerly (before a 25-minute build) — a typo here silently
    builds a DIFFERENT point on the curve, which reads as a result.
    """
    out = []
    for chunk in spec.split(","):
        rng, _, bits = chunk.strip().partition(":")
        lo, _, hi = rng.partition("-")
        lo, hi, bits = int(lo), int(hi or lo), int(bits)
        assert 0 <= lo <= hi < n_layers, f"range {lo}-{hi} outside 0-{n_layers-1}"
        assert bits in (2, 3, 4, 5, 6, 8), f"unsupported bits {bits}"
        out.append((lo, hi, bits))
    covered = sorted((lo, hi) for lo, hi, _ in out)
    for (a_lo, a_hi), (b_lo, _) in zip(covered, covered[1:]):
        assert a_hi < b_lo, f"overlapping ranges {a_lo}-{a_hi} and {b_lo}-…"
    return out


schedule = _parse_schedule(args.expert_schedule) if args.expert_schedule else None
if schedule:
    print("expert schedule: " + "  ".join(
        f"L{lo}-{hi}@{b}b" for lo, hi, b in schedule))


def plan_bits(path):
    """Bits this variant assigns, or None if the tensor stays bf16."""
    if path not in quantized:
        return None
    base = overrides[path]["bits"] if path in overrides else default["bits"]
    if "switch_mlp" in path and schedule:
        idx = layer_index(path)
        for lo, hi, bits in schedule:
            if lo <= idx <= hi:
                return bits
        return base
    if "switch_mlp" in path and (args.tail_layers or args.promote_every):
        idx = layer_index(path)
        if args.promote_every:
            if idx % args.promote_every == 0:
                return args.tail_expert_bits
        elif idx >= tail_start:
            return args.tail_expert_bits
        return base
    if "in_proj_qkv" in path or "in_proj_z" in path:
        return args.qkv_bits
    if base == 8:
        return args.structure_bits
    return base


# --- predict size from the bf16 source's parameter counts ---
src_index = json.load(open(SRC + "/model.safetensors.index.json"))["weight_map"]
import re

shapes = {}
try:
    from safetensors import safe_open
    seen_files = {}
    for key, fname in src_index.items():
        seen_files.setdefault(fname, []).append(key)
    for fname, keys in seen_files.items():
        with safe_open(SRC + "/" + fname, framework="numpy") as f:
            for k in keys:
                shapes[k] = f.get_slice(k).get_shape()
except Exception as e:  # pragma: no cover
    raise SystemExit(f"could not read source shapes: {e}")


def norm(path):
    """map a module path to its bf16 weight key (spicy naming -> src naming)"""
    for cand in (path + ".weight", path):
        if cand in shapes:
            return cand
    alt = path.replace("language_model.model", "model.language_model")
    for cand in (alt + ".weight", alt):
        if cand in shapes:
            return cand
    alt2 = path.replace("language_model.", "")
    for cand in (alt2 + ".weight", alt2):
        if cand in shapes:
            return cand
    # switch_mlp.{gate,up,down}_proj are mlx_lm's post-sanitize names for the
    # source's fused expert stacks: experts.gate_up_proj (holds BOTH gate and
    # up, so each accounts for half) and experts.down_proj.
    m = re.match(r"(.*)\.mlp\.switch_mlp\.(gate|up|down)_proj$", path)
    if m:
        prefix = m.group(1).replace("language_model.model", "model.language_model")
        if m.group(2) == "down":
            return (prefix + ".mlp.experts.down_proj", 1.0)
        return (prefix + ".mlp.experts.gate_up_proj", 0.5)
    return None


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def group_size_for(path, bits):
    if (args.low_bit_group_size and bits == 2 and "switch_mlp" in path):
        return args.low_bit_group_size
    return overrides.get(path, {}).get("group_size", default["group_size"])


def bytes_for(nparams, bits, group_size=64):
    # affine: packed weights + fp16 scale & bias per group
    return nparams * bits / 8 + (nparams / group_size) * 2 * 2


base_total = var_total = 0.0
missing = []
for path in sorted(quantized):
    key = norm(path)
    if key is None:
        missing.append(path)
        continue
    frac = 1.0
    if isinstance(key, tuple):
        key, frac = key
        if key not in shapes:
            missing.append(path)
            continue
    n = numel(shapes[key]) * frac
    b_base = overrides[path]["bits"] if path in overrides else default["bits"]
    b_var = plan_bits(path)
    base_total += bytes_for(n, b_base,
                            overrides.get(path, {}).get("group_size", 64))
    var_total += bytes_for(n, b_var, group_size_for(path, b_var))

GB = 1024 ** 3
delta = (var_total - base_total) / GB
print(f"layers={n_layers} tail_start={tail_start} unmatched_shapes={len(missing)}")
print(f"quantized bytes  baseline {base_total / GB:7.2f} GiB   "
      f"variant {var_total / GB:7.2f} GiB   delta {delta:+.2f} GiB")
if missing[:3]:
    print("  (sample unmatched:", missing[:3], ")")

if delta > args.tolerance_gb and not args.force:
    raise SystemExit(
        f"REFUSING: variant is {delta:.2f} GiB larger than baseline "
        f"(tolerance {args.tolerance_gb}). Not a matched-size comparison. "
        f"Re-run with --force to build it anyway.")
if args.dry_run:
    raise SystemExit("dry run — nothing built")

hits = {"skip": 0}


def predicate(path, module):
    b = plan_bits(path)
    if b is None:
        hits["skip"] += 1
        return False
    hits[b] = hits.get(b, 0) + 1
    return {"group_size": group_size_for(path, b), "bits": b,
            "mode": "affine"}


OUT = f"{OUT_ROOT}/TheDrainFlorist--Qwen3.5-397B-A17B-{args.name}"
from mlx_lm.convert import convert

convert(SRC, mlx_path=OUT, quantize=True,
        q_group_size=default["group_size"], q_bits=default["bits"],
        quant_predicate=predicate)
print("bit histogram:", hits)

# vision sidecar + preprocessor configs (same bf16 tower every build shares)
out = pathlib.Path(OUT)
(out / "optiq").mkdir(exist_ok=True)
src = pathlib.Path(VISION_SRC)
import shutil

try:
    shutil.copy2(src / "optiq" / "optiq_vision.safetensors", out / "optiq")
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy2(src / f, out / f)
    print(f"done: {OUT} (+vision)")
except FileNotFoundError as e:
    # The optiq-layout vision source was deleted in the 08-19 cleanup; the
    # published artifacts carry vision only in GRAFTED form
    # (model-vision-graft.safetensors), which this copy step does not speak.
    # A base variant without the sidecar is still a valid FIT BASE — the
    # fitter touches expert layers only — but it must be re-grafted
    # (graft_vision.py) and pass check_vision.py before any release (E61).
    print(f"done: {OUT} (NO VISION SIDECAR — {e}; re-graft before release)")
