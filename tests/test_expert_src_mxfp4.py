"""DeepSeek-V4 FP4 expert source (families deepseek_v4, expert_src mxfp4 path).

The official release stores each expert as an I8 weight (two e2m1 codes per
byte) with an F8_E8M0 `.scale` sibling, one per 32 inputs. mx.load refuses a
shard holding F8_E8M0, so expert_src reads these pairs raw. The stack it
returns must equal mx.dequantize(mode="mxfp4") of the same bytes, bit for
bit, with experts in index order across shards.
"""
import json
import struct

import numpy as np
import mlx.core as mx
import pytest

from vqlab.core import expert_src as ES
from vqlab.core.families import FAMILY

FAM = FAMILY["deepseek_v4"]


def _write(path, tensors):
    """tensors: {key: (dtype_str, np.uint8 array)} -> safetensors file."""
    hdr, blobs, off = {}, [], 0
    for k, (dt, a) in tensors.items():
        b = a.tobytes()
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    js = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(js)) + js + b"".join(blobs))


def _checkpoint(tmp_path, E=5, OUT=64, IN=256, li=3):
    rng = np.random.default_rng(1234)
    ref, shards, wmap = {}, ({}, {}), {}
    for proj, (name, _) in FAM["proj"].items():
        out, inn = (IN, OUT) if proj == "down_proj" else (OUT, IN)
        stack = []
        for e in range(E):
            w = mx.array(rng.standard_normal((out, inn)).astype(np.float32))
            q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
            stack.append(mx.dequantize(q, s, group_size=32, bits=4,
                                       mode="mxfp4").astype(mx.bfloat16))
            k = FAM["src_key"].format(li=li, key=name, e=e)
            sk = k[: -len(".weight")] + ".scale"
            sh = 0 if e < 3 else 1
            shards[sh][k] = ("I8", np.array(mx.view(q, dtype=mx.uint8)))
            shards[sh][sk] = ("F8_E8M0", np.array(s))
            f = f"model-0000{sh + 1}-of-00002.safetensors"
            wmap[k] = wmap[sk] = f
        ref[proj] = mx.stack(stack)
    for i, t in enumerate(shards):
        _write(tmp_path / f"model-0000{i + 1}-of-00002.safetensors", t)
    return ref, wmap


def test_mxfp4_stack_equals_dequantize(tmp_path):
    ref, wmap = _checkpoint(tmp_path)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        T = ES.load_expert_stack(tmp_path, wmap, FAM, 3, proj)
        assert T.dtype == mx.bfloat16 and T.shape == ref[proj].shape
        assert np.array_equal(np.array(T.view(mx.uint16)),
                              np.array(ref[proj].view(mx.uint16)))
    assert ES.count_experts(FAM, wmap, 3) == 5


def test_mxfp4_experts_subset(tmp_path):
    ref, wmap = _checkpoint(tmp_path)
    T = ES.load_expert_stack(tmp_path, wmap, FAM, 3, "gate_proj", experts=2)
    assert np.array_equal(np.array(T.view(mx.uint16)),
                          np.array(ref["gate_proj"][:2].view(mx.uint16)))


def test_mxfp4_refuses_mismatched_scale(tmp_path):
    k = FAM["src_key"].format(li=0, key="w1", e=0)
    sk = k[: -len(".weight")] + ".scale"
    d = FAM["src_key"].format(li=0, key="w2", e=0)        # experts are counted on down
    _write(tmp_path / "s.safetensors",
           {k: ("I8", np.zeros((8, 64), np.uint8)),
            sk: ("F8_E8M0", np.zeros((8, 3), np.uint8)),
            d: ("I8", np.zeros((8, 64), np.uint8))})
    with pytest.raises(ValueError, match="not an mxfp4 pair"):
        ES.load_expert_stack(tmp_path, {x: "s.safetensors" for x in (k, sk, d)},
                             FAM, 0, "gate_proj")
