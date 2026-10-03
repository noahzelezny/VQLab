"""`fit-moe --pack` writes, byte for byte, what fit-moe followed by
`vqlab pack` writes (devlist #13: unpacked K>256 codes never land on disk).

CPU only. Each tool runs as its own process with mlx's default device set to
the CPU before the script starts, so the seeded k-means is deterministic and
no GPU is touched. The fixture is selftest 5b's fit-moe fixture with two
modules at different geometries, so the packer has to choose between them:
gate_proj d4 K512 (9 bits, packed) and up_proj d4 K256 (8 bits, the
byte-aligned copy-through).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
E, I, H, G = 2, 8, 128, 64
MOD = "model.language_model.layers.0.mlp.switch_mlp.{p}"
CPU_RUN = ("import sys, runpy, mlx.core as mx; mx.set_default_device(mx.cpu); "
           "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")


def _fixture(tmp: pathlib.Path):
    from safetensors.numpy import save_file
    rng = np.random.default_rng(1234)
    teacher, base = tmp / "teacher", tmp / "base"
    teacher.mkdir()
    base.mkdir()
    tk = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    # numpy has no bf16; the fitter casts to float32, so fp16 is a legal source
    save_file({tk: (rng.standard_normal((E, 2 * I, H)) * .05).astype(np.float16)},
              str(teacher / "t.safetensors"))
    json.dump({"weight_map": {tk: "t.safetensors"}},
              open(teacher / "model.safetensors.index.json", "w"))
    w, q = {}, {"group_size": G, "bits": 4}
    for p in ("gate_proj", "up_proj"):
        m = MOD.format(p=p)
        w[m + ".weight"] = np.zeros((E, I, H * 2 // 32), np.uint32)
        w[m + ".scales"] = np.ones((E, I, H // G), np.float16)
        w[m + ".biases"] = np.zeros((E, I, H // G), np.float16)
        q[m] = {"group_size": G, "bits": 2}
    save_file(w, str(base / "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in w}},
              open(base / "model.safetensors.index.json", "w"))
    json.dump({"model_type": "qwen3_5_moe", "quantization": q},
              open(base / "config.json", "w"))
    return teacher, base


def _run(script, *args, env):
    p = subprocess.run([sys.executable, "-c", CPU_RUN, str(script), *map(str, args)],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    return p


def test_fit_pack_is_fit_then_pack(tmp_path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("safetensors")
    sys.path.insert(0, str(SRC))
    from vqlab._layout import find
    teacher, base = _fixture(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(SRC),
           "VQLAB_CONFIG": str(tmp_path / "no-config.toml"),
           "VQLAB_FIT_STORE": str(tmp_path / "fits"),
           "VQLAB_SCRATCH": str(tmp_path), "VQLAB_SKIP_DISK_CHECK": "1"}
    fit = [find("fit_moe.py"), "--base", base, "--src", teacher,
           "--vq-layers", "0", "--geom", "gate_proj=d4k512,up_proj=d4k256",
           "--iters", "2", "--sample", "1000", "--family", "qwen3_5",
           "--relerr-abort", "1.0"]
    unpacked, packed_after, packed_inflight = (tmp_path / n for n in
                                               ("fit", "fit-then-pack", "fit-pack"))
    _run(*fit, "--out", unpacked, env=env)
    _run(find("pack.py"), "--src", unpacked, "--out", packed_after, env=env)
    _run(*fit, "--out", packed_inflight, "--pack", env=env)

    from safetensors import safe_open
    gate, up = MOD.format(p="gate_proj"), MOD.format(p="up_proj")
    with safe_open(str(unpacked / "model-00001-of-00001.safetensors"), "numpy") as f:
        assert f.get_slice(gate + ".codes").get_dtype() == "U16"   # the bytes --pack avoids
    with safe_open(str(packed_inflight / "model-00001-of-00001.safetensors"), "numpy") as f:
        assert f.get_slice(gate + ".codes").get_dtype() == "U32"   # packed, 9 bits
        assert f.get_slice(up + ".codes").get_dtype() == "U8"      # K256 stays byte-aligned

    for name in ("model-00001-of-00001.safetensors", "model.safetensors.index.json",
                 "config.json", "model.py"):
        a = (packed_after / name).read_bytes()
        b = (packed_inflight / name).read_bytes()
        assert a == b, f"{name} differs between fit-then-pack and fit --pack"
    cfg = json.loads((packed_inflight / "config.json").read_text())
    assert cfg["vq_modules"][gate]["pack_bits"] == 9
    assert "pack_bits" not in cfg["vq_modules"][up]
