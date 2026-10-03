"""`vqlab slice`: a real slice of a teacher, CPU only, tiny fixture.

The fixture is adversarial where the writer chooses: layers are spread across
shards so some shards are partial (byte-copied) and one holds only top-level
tensors (symlinked); each tensor has DIFFERENT bytes so a copy from the wrong
offset or layer cannot pass; a bf16-labelled tensor checks the header entry
is carried verbatim; a vision tensor and an MTP tensor must be left out."""
from __future__ import annotations

import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest

from vqlab.core.artifact import read_header

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
N = 6
LAYER = "model.language_model.layers.{i}.mlp.down_proj.weight"


def _write(path, tensors):
    """Raw safetensors writer: {name: (dtype, shape, bytes)}."""
    hdr, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for k, (dt, shape, b) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def _tensor_bytes(path, key):
    h = read_header(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        a, b = h[key]["data_offsets"]
        f.seek(8 + n + a)
        return f.read(b - a)


def _teacher(d: pathlib.Path):
    d.mkdir()
    rng = np.random.default_rng(1234)
    t = lambda i: ("F32", [4, 8], rng.standard_normal((4, 8)).astype(np.float32).tobytes())  # noqa: E731
    shards = {"model-00001-of-00003.safetensors": {LAYER.format(i=i): t(i) for i in (0, 1, 2)},
              "model-00002-of-00003.safetensors": {LAYER.format(i=i): t(i) for i in (3, 4, 5)},
              "model-00003-of-00003.safetensors": {
                  "model.language_model.embed_tokens.weight": ("BF16", [4, 8], bytes(range(64))),
                  "model.language_model.norm.weight": t(-1), "lm_head.weight": t(-1)}}
    shards["model-00002-of-00003.safetensors"]["model.visual.blocks.0.attn.qkv.weight"] = t(0)
    shards["model-00002-of-00003.safetensors"]["mtp.layers.0.mlp.down_proj.weight"] = t(0)
    wm = {}
    for f, ts in shards.items():
        _write(d / f, ts)
        wm.update({k: f for k in ts})
    json.dump({"metadata": {"total_size": 1}, "weight_map": wm},
              open(d / "model.safetensors.index.json", "w"))
    json.dump({"model_type": "qwen3_5_moe", "text_config": {
        "num_hidden_layers": N, "mtp_num_hidden_layers": 1,
        "layer_types": [f"t{i}" for i in range(N)], "architectures": ["X"]},
        "quantization": {"group_size": 64, "bits": 8,
                         **{LAYER.format(i=i)[:-7]: {"bits": i} for i in range(N)}}},
        open(d / "config.json", "w"))
    (d / "tokenizer.json").write_text("{}")
    return shards


@pytest.fixture
def env(tmp_path):
    return {**os.environ, "PYTHONPATH": str(SRC), "VQLAB_SCRATCH": str(tmp_path),
            "VQLAB_CONFIG": str(tmp_path / "none.toml"), "VQLAB_LOG_DIR": str(tmp_path / "log"),
            "VQLAB_ALLOW_SYSTEM_DISK": "1"}


def _slice(env, *args):
    return subprocess.run([sys.executable, "-m", "vqlab.cli", "slice", *map(str, args)],
                          capture_output=True, text=True, env=env)


def test_slice_renumbers_copies_bytes_and_adjusts_config(tmp_path, env):
    teacher = tmp_path / "teacher"
    shards = _teacher(teacher)
    out = tmp_path / "slice-2-3"
    p = _slice(env, teacher, "--layers", "2-3", "--out", out)
    assert p.returncode == 0, p.stdout + p.stderr
    idx = json.load(open(out / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    assert set(wm) == {LAYER.format(i=0), LAYER.format(i=1),
                       "model.language_model.embed_tokens.weight",
                       "model.language_model.norm.weight", "lm_head.weight"}
    # layer 2 -> 0 (from shard 1), layer 3 -> 1 (from shard 2): bytes exactly
    assert _tensor_bytes(out / wm[LAYER.format(i=0)], LAYER.format(i=0)) == \
        shards["model-00001-of-00003.safetensors"][LAYER.format(i=2)][2]
    assert _tensor_bytes(out / wm[LAYER.format(i=1)], LAYER.format(i=1)) == \
        shards["model-00002-of-00003.safetensors"][LAYER.format(i=3)][2]
    # top-level shard taken whole: a symlink, bf16 entry untouched
    top = out / "model-00003-of-00003.safetensors"
    assert top.is_symlink()
    assert read_header(top)["model.language_model.embed_tokens.weight"]["dtype"] == "BF16"
    assert idx["metadata"]["total_size"] == sum(
        v["data_offsets"][1] - v["data_offsets"][0]
        for f in set(wm.values()) for v in read_header(out / f).values())
    cfg = json.load(open(out / "config.json"))
    tc = cfg["text_config"]
    assert tc["num_hidden_layers"] == 2 and tc["layer_types"] == ["t2", "t3"]
    assert tc["architectures"] == ["X"]
    q = cfg["quantization"]
    assert q[LAYER.format(i=0)[:-7]] == {"bits": 2} and q[LAYER.format(i=1)[:-7]] == {"bits": 3}
    assert LAYER.format(i=2)[:-7] not in q and q["bits"] == 8
    assert (out / "tokenizer.json").exists()
    assert json.load(open(out / "vqlab_provenance.json"))["tool"]["name"] == "slice"


def test_slice_from_zero_copy_clones(tmp_path, env):
    teacher = tmp_path / "teacher"
    _teacher(teacher)
    out = tmp_path / "slice-0-2"
    p = _slice(env, teacher, "--layers", "0-2", "--out", out, "--copy")
    assert p.returncode == 0, p.stdout + p.stderr
    s1 = out / "model-00001-of-00003.safetensors"
    assert s1.exists() and not s1.is_symlink()     # whole shard, cloned
    assert s1.read_bytes() == (teacher / s1.name).read_bytes()
    assert not (out / "model-00002-of-00003.safetensors").exists()


def test_slice_refusals(tmp_path, env):
    teacher = tmp_path / "teacher"
    _teacher(teacher)
    p = _slice(env, teacher, "--layers", "4-9", "--out", tmp_path / "s")
    assert p.returncode != 0 and "layers 0-5" in (p.stdout + p.stderr)
    outside = pathlib.Path(env["VQLAB_SCRATCH"]).parent / "not-storage-slice"
    p = _slice(env, teacher, "--layers", "0-1", "--out", outside)
    assert p.returncode != 0 and "outside the configured storage" in (p.stdout + p.stderr)
    assert not outside.exists()
    sysdisk = {k: v for k, v in env.items() if k != "VQLAB_ALLOW_SYSTEM_DISK"}
    if os.stat(tmp_path).st_dev == os.stat("/").st_dev:
        p = _slice(sysdisk, teacher, "--layers", "0-1", "--out", tmp_path / "s2")
        assert p.returncode != 0 and "system disk" in (p.stdout + p.stderr)
