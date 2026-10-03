"""graft-extras: byte-copies the preset's tensors under the runtime's names,
refuses a source that is not the artifact's base, is idempotent."""
import json
import struct

import numpy as np
import pytest

from vqlab.assemble import graft_extras as GE
from vqlab.core.artifact import Artifact


def _st(path, t):
    hdr, off, blobs = {}, 0, []
    dt = {np.dtype("float32"): "F32", np.dtype("uint16"): "BF16", np.dtype("int32"): "I32"}
    for k in sorted(t):
        b = t[k].tobytes()
        hdr[k] = {"dtype": dt[t[k].dtype], "shape": list(t[k].shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _pair(tmp, same=True):
    r = np.random.default_rng(1234)
    src, art = tmp / "src", tmp / "art"
    src.mkdir(); art.mkdir()
    norms = {f"layers.{i}.attn_norm.weight": r.standard_normal(8).astype(np.float32) for i in range(4)}
    s = dict(norms)
    s.update({"vision.blocks.0.w": r.integers(0, 9, 6).astype(np.uint16),
              "image_start": r.integers(0, 9, 4).astype(np.uint16),
              **{f"layers.{i}.ffn.gate.bias_vl": r.standard_normal(3).astype(np.float32) for i in range(4)},
              "layers.0.ffn.gate.bias": r.standard_normal(3).astype(np.float32),
              "layers.3.ffn.gate.bias": r.standard_normal(3).astype(np.float32),
              "mtp.0.ffn.gate.bias_vl": r.standard_normal(3).astype(np.float32)})
    _st(src / "s.safetensors", s)
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "s.safetensors" for k in s}}))
    (src / "config.json").write_text(json.dumps({"vision_n_layers": 2}))
    a = {"model." + k: (v if same else v + 1) for k, v in norms.items()}
    _st(art / "a.safetensors", a)
    (art / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "a.safetensors" for k in a}}))
    (art / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    return src, art, s


def test_graft(tmp_path):
    src, art, s = _pair(tmp_path)
    for _ in range(2):                                     # idempotent
        GE.main(["--artifact", str(art), "--src", str(src), "--preset", "deepseek-v4-vision"])
    A = Artifact.open(art)
    got = {k for k, f in A.index.items() if f == GE.SHARD}
    assert got == {"vision.blocks.0.w", "model.image_start", "model.layers.0.ffn.gate.e_score_correction_bias",
                   *{f"model.layers.{i}.ffn.gate.bias_vl" for i in range(4)}}
    h = A.header(GE.SHARD)
    raw = (art / GE.SHARD).read_bytes()
    start = 8 + struct.unpack("<Q", raw[:8])[0]
    a, b = h["model.layers.2.ffn.gate.bias_vl"]["data_offsets"]
    assert raw[start + a:start + b] == s["layers.2.ffn.gate.bias_vl"].tobytes()
    assert h["model.layers.2.ffn.gate.bias_vl"]["dtype"] == "F32"
    assert A.config["vision_n_layers"] == 2


def test_wrong_base_refused(tmp_path):
    src, art, _ = _pair(tmp_path, same=False)
    with pytest.raises(SystemExit, match="not this artifact's base"):
        GE.main(["--artifact", str(art), "--src", str(src), "--preset", "deepseek-v4-vision"])
    assert not (art / GE.SHARD).exists()
