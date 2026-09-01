#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Quantize our bf16 397B with spicyneuron's EXACT per-tensor bit map.

mlx_lm's current mixed_2_6 recipe is llama.cpp-style (6-bit down_proj/v_proj
on ~1/3 of layers, experts included → 3.16 bpw, 157G) and is NOT what built
spicyneuron (2.61 bpw, 120G: 2-bit everything + 8-bit structure + 4-bit
linear_attn in_proj). Rather than guess the generator, mirror the artifact:
read spicyneuron's config.json quantization dict and return each tensor's
exact setting from a predicate. Anything not in their map gets their default
(2-bit / group 64 / affine). Misses are counted and printed — expect ~0
besides never-quantized modules (norms, conv1d).
"""
import json
import pathlib

import mlx.core as mx

mx.set_default_device(mx.cpu)

SPICY = "/Users/noahzelezny/.exo/models/spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit/config.json"
SRC = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-397B-A17B-bf16"
OUT = "/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--Qwen3.5-397B-A17B-spicymirror"

q = json.load(open(SPICY))["quantization"]
overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
default = {"group_size": q["group_size"], "bits": q["bits"],
           "mode": q.get("mode", "affine")}
# v2: quantize EXACTLY the tensors spicyneuron quantized. Their config's
# override list is not the full story — tensors absent from it are either
# quantized at the default (experts) or NOT QUANTIZED AT ALL (routers,
# gates, in_proj_a/b stay bf16; mlx_lm's model quant_predicate excludes
# them). v1 forced 2-bit routers and scored 34.85 vs their 3.18. The
# artifact's own weight index is the ground truth: path is quantized iff
# "<path>.scales" exists in their weight_map.
spicy_dir = str(pathlib.Path(SPICY).parent)
wmap = json.load(open(spicy_dir + "/model.safetensors.index.json"))["weight_map"]
quantized_paths = {k[: -len(".scales")] for k in wmap if k.endswith(".scales")}
print(f"spicy map: {len(overrides)} overrides, "
      f"{len(quantized_paths)} quantized tensors, default {default}")

hits = {"override": 0, "default": 0, "skip": 0}


def predicate(path, module):
    if path not in quantized_paths:
        hits["skip"] += 1
        return False
    if path in overrides:
        hits["override"] += 1
        o = overrides[path]
        return {"group_size": o.get("group_size", 64), "bits": o["bits"],
                "mode": o.get("mode", "affine")}
    hits["default"] += 1
    return dict(default)


from mlx_lm.convert import convert

convert(SRC, mlx_path=OUT, quantize=True,
        q_group_size=default["group_size"], q_bits=default["bits"],
        quant_predicate=predicate)
print(f"done. predicate hits: {hits} (expected overrides: {len(overrides)})")
