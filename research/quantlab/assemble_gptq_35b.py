#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E34 assembly: swap GPTQ-solved expert tensors into an RTN-built artifact.

Takes the struct6 RTN build (identical everywhere else) and replaces every
switch_mlp gate/up/down weight/scales/biases with the packed GPTQ versions
from the per-layer checkpoints. Output loads with stock mlx-lm — same
format, same size, different rounding.
"""
import json
import pathlib
import shutil

import numpy as np
import mlx.core as mx

from gptq_solver import pack_mlx

mx.set_default_device(mx.cpu)

BASE = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models/rotlab-35B-base-struct6")
CKPT = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models/rotlab-gptq-ckpt")
OUT = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models/rotlab-35B-gptq-struct6")

INTER = 512  # gate/up rows per expert in the stacked [1024, 2048] solve

OUT.mkdir(exist_ok=True)
idx = json.load(open(BASE / "model.safetensors.index.json"))
wmap = idx["weight_map"]

# per-layer replacement tensors, built lazily per shard visit
loaded = {}


def layer_arrays(li):
    if li not in loaded:
        z = np.load(CKPT / f"layer{li:02d}.npz")
        loaded.clear()          # keep one layer resident
        loaded[li] = {k: z[k] for k in z.files}
    return loaded[li]


def replacement(name):
    # language_model.model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
    if "switch_mlp" not in name:
        return None
    parts = name.split(".")
    li = int(parts[3])
    proj = parts[6]          # gate_proj / up_proj / down_proj
    kind = parts[7]          # weight / scales / biases
    z = layer_arrays(li)
    if proj in ("gate_proj", "up_proj"):
        sl = slice(0, INTER) if proj == "gate_proj" else slice(INTER, None)
        qi, s, b = z["qi_gu"][:, sl], z["s_gu"][:, sl], z["b_gu"][:, sl]
    else:
        qi, s, b = z["qi_d"], z["s_d"], z["b_d"]
    if kind == "weight":
        return mx.array(np.stack([pack_mlx(qi[e]) for e in range(qi.shape[0])]))
    arr = s if kind == "scales" else b
    return mx.array(arr).astype(mx.bfloat16)


n_swapped = 0
for fname in sorted(set(wmap.values())):
    shard = mx.load(str(BASE / fname))
    out_shard = {}
    for k, v in shard.items():
        r = replacement(k)
        if r is not None:
            assert r.shape == v.shape, (k, r.shape, v.shape)
            assert r.dtype == v.dtype, (k, r.dtype, v.dtype)
            out_shard[k] = r
            n_swapped += 1
        else:
            out_shard[k] = v
    mx.save_safetensors(str(OUT / fname), out_shard)
    del shard, out_shard

for extra in BASE.iterdir():
    if extra.suffix != ".safetensors" and extra.is_file():
        shutil.copy2(extra, OUT / extra.name)
print(f"swapped {n_swapped} expert tensors -> {OUT}")
