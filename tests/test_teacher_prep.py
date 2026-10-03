"""teacher-prep (CPU): the E8M0 relabel is header-only and idempotent, the
HF-cache guard refuses symlinked blobs, and sanitize-stream's key filter keeps
model parameters plus quantization siblings while dropping unknown tensors."""
import json
import struct

import numpy as np
import pytest

from vqlab.assemble import teacher_prep as TP
from vqlab.assemble import sanitize_stream as SS


def _st(path, tensors):
    hdr, off, blob = {}, 0, b""
    for k, (dt, arr) in tensors.items():
        b = arr.tobytes()
        hdr[k] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b); blob += b
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + blob)


def test_relabel_header_only_and_idempotent(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    p = tmp_path / "a.safetensors"
    _st(p, {"w": ("I8", np.arange(8, dtype=np.int8)),
            "s": ("F8_E8M0", np.full(4, 127, np.uint8))})
    before = p.read_bytes()
    assert TP.relabel_e8m0(tmp_path) == 1
    after = p.read_bytes()
    assert len(after) == len(before)
    n = struct.unpack("<Q", after[:8])[0]
    assert after[8 + n:] == before[8 + n:]                 # tensor bytes untouched
    assert json.loads(after[8:8 + n])["s"]["dtype"] == "U8"
    assert TP.relabel_e8m0(tmp_path) == 0                  # idempotent


def test_guard_refuses_cache_symlinks(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    blob = tmp_path / "blob"; blob.write_bytes(b"x")
    (tmp_path / "m.safetensors").symlink_to(blob)
    with pytest.raises(SystemExit):
        TP.guard_source(tmp_path)


def test_key_filter():
    keys = {"layers.0.ffn.gate.weight", "layers.0.ffn.switch_mlp.gate_proj.weight", "embed.weight"}
    mods = {k.rsplit(".", 1)[0] for k in keys}
    assert SS._wanted("layers.0.ffn.gate.weight", keys, mods)
    assert SS._wanted("layers.0.ffn.switch_mlp.gate_proj.scales", keys, mods)
    assert not SS._wanted("layers.0.ffn.gate.e_score_correction_bias_vl", keys, mods)
    assert not SS._wanted("image_pad", keys, mods)
    assert not SS._wanted("vision.blocks.0.attn.wo.weight", keys, mods)
