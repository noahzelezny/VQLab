"""The selftest's tiny synthetic models: a dense checkpoint, a qwen3_5-shaped
MoE teacher + VQ base (geo-build), an affine MoE base (fit-moe), an
unknown-family teacher (family-profile / onboard), and a decoder that turns
any VQ artifact back into dense weights."""
from __future__ import annotations

import json
import pathlib

import mlx.core as mx

# tiny geometry: IN divisible by group(64) and dim; OUT small.
LAYERS, OUT_D, IN_D, G = 2, 64, 128, 64
KEY = "model.language_model.layers.{li}.mlp.{key}.weight"
PROJS = ("gate_proj", "up_proj", "down_proj")


def make_source(d: pathlib.Path):
    """A tiny checkpoint shaped like a dense qwen MLP stack."""
    mx.random.seed(7)
    w, wm = {}, {}
    # Draw subvectors from a small set of centres so the tensors are actually
    # VQ-compressible, the way real weights are. Pure gaussian noise fits at
    # relerr ~0.34 even when healthy, which would leave a COLLAPSED tensor
    # (relerr 1.0) sitting below 3x the median — invisible to the relative
    # gate. A fixture has to be realistic enough for the gate under test to
    # be able to fire.
    centres = mx.random.normal((16, 2)) * 0.05
    for li in range(LAYERS):
        for proj in PROJS:
            k = KEY.format(li=li, key=proj)
            pick = mx.random.randint(0, centres.shape[0], (OUT_D * IN_D // 2,))
            sub = centres[pick] + mx.random.normal((OUT_D * IN_D // 2, 2)) * 0.002
            w[k] = sub.reshape(OUT_D, IN_D).astype(mx.bfloat16)
            wm[k] = "model-00001-of-00001.safetensors"
    d.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"), w)
    json.dump({"metadata": {}, "weight_map": wm},
              open(d / "model.safetensors.index.json", "w"), indent=1)
    json.dump({"model_type": "qwen3", "quantization": {"group_size": G, "bits": 4}},
              open(d / "config.json", "w"), indent=1)


def geo_fixture(tmp: pathlib.Path):
    """A qwen3_5-shaped MoE teacher + a VQ base artifact for geo-build: one
    module, E=2 experts, IN=128 so d4 gives nsub=32 (geo-build refuses
    ragged packing). The base carries its OWN build record so the child's
    lineage link can be checked."""
    from vqlab.records import provenance
    E, I, H = 2, 8, 128
    teacher, base = tmp / "geo-teacher", tmp / "geo-base"
    teacher.mkdir(); base.mkdir()
    mx.random.seed(11)
    tk = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    mx.save_safetensors(str(teacher / "t.safetensors"),
                        {tk: (mx.random.normal((E, 2 * I, H)) * .05).astype(mx.bfloat16)})
    json.dump({"weight_map": {tk: "t.safetensors"}},
              open(teacher / "model.safetensors.index.json", "w"))
    mod = "model.language_model.layers.0.mlp.switch_mlp.gate_proj"
    w = {mod + ".codes": mx.zeros((E, I, H // 2), mx.uint8),
         mod + ".codebook": mx.zeros((256, 2), mx.float16),
         mod + ".vq_scales": mx.ones((E, I, H // G), mx.float16)}
    mx.save_safetensors(str(base / "model-00001-of-00001.safetensors"), w)
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in w}},
              open(base / "model.safetensors.index.json", "w"))
    json.dump({"model_type": "qwen3_5_moe", "vq_modules": {mod: {
        "experts": E, "out": I, "in": H, "group": G, "dim": 2, "k": 256}}},
        open(base / "config.json", "w"))
    brec = provenance.write_build_record(base, tool="selftest-fixture")
    gm = tmp / "geomap.json"
    json.dump({mod: {"dim": 4, "k": 16}}, open(gm, "w"))
    return {"teacher": teacher, "base": base, "map": gm, "module": mod,
            "out": tmp / "geo-out", "base_id": brec["id"]}


def moe_fixture(mt: pathlib.Path, mb: pathlib.Path):
    """A one-module MoE teacher + an affine (2-bit) base for fit-moe.
    Returns the base module name."""
    mt.mkdir(); mb.mkdir()
    E_, I_, H_ = 2, 8, 128
    tk = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    mx.save_safetensors(str(mt / "t.safetensors"),
                        {tk: (mx.random.normal((E_, 2 * I_, H_)) * .05).astype(mx.bfloat16)})
    json.dump({"weight_map": {tk: "t.safetensors"}},
              open(mt / "model.safetensors.index.json", "w"))
    bm = "model.language_model.layers.0.mlp.switch_mlp.gate_proj"
    bw = {bm + ".weight": mx.zeros((E_, I_, H_ * 2 // 32), mx.uint32),
          bm + ".scales": mx.ones((E_, I_, H_ // G), mx.float16),
          bm + ".biases": mx.zeros((E_, I_, H_ // G), mx.float16)}
    mx.save_safetensors(str(mb / "model-00001-of-00001.safetensors"), bw)
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in bw}},
              open(mb / "model.safetensors.index.json", "w"))
    json.dump({"model_type": "qwen3_5_moe",
               "quantization": {"group_size": G, "bits": 4,
                                bm: {"group_size": G, "bits": 2}}},
              open(mb / "config.json", "w"))
    return bm


def novel_teacher(U: pathlib.Path):
    """A teacher of a family nothing knows (model_type novelmoe)."""
    U.mkdir()
    wn = {f"transformer.blocks.{li}.moe.experts.{p_}.weight":
          mx.zeros((4, 64, 128), mx.bfloat16)
          for li in range(2) for p_ in ("w1", "w2", "w3")}
    mx.save_safetensors(str(U / "model.safetensors"), wn)
    json.dump({"model_type": "novelmoe", "num_hidden_layers": 2,
               "num_experts": 4, "hidden_size": 128}, open(U / "config.json", "w"))


def decode_all(art: pathlib.Path):
    """Decode every VQ tensor in an artifact to dense weights."""
    from vqlab.runtime import vq_pack
    cfg = json.load(open(art / "config.json"))
    mods = cfg.get("vq_modules") or cfg.get("vq_linear") or {}
    idx = json.load(open(art / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for m, meta in mods.items():
        shard = art / idx[m + ".codes"]
        with mx.stream(mx.cpu):
            data = mx.load(str(shard))
            mx.eval(list(data.values()))
        codes, cb = data[m + ".codes"], data[m + ".codebook"]
        sc = data[m + ".vq_scales"]
        D = int(cb.shape[1])
        nsub = meta["in"] // D
        c = codes.reshape(-1, codes.shape[-2], codes.shape[-1])   # [E, OUT, NSUB or words]
        if meta.get("pack_bits"):
            c = mx.array(vq_pack.unpack(np_of(c), nsub, meta["pack_bits"]))
        c = c.reshape(-1, nsub)
        w = cb.astype(mx.float32)[c.reshape(-1).astype(mx.int32)]
        w = w.reshape(c.shape[0], nsub * D)
        w = w.reshape(c.shape[0], meta["in"] // meta["group"], meta["group"])
        w = w * sc.reshape(-1, meta["in"] // meta["group"])[..., None].astype(mx.float32)
        out[m] = w.reshape(c.shape[0], meta["in"])
        mx.eval(out[m])
    return out


def np_of(a):
    import numpy as np
    return np.array(a)
