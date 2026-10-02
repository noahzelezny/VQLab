"""Pin that skipzero is refused for codebook dims the runtime cannot serve.

The runtime row-table switch serves compact rows at d4 and packed d2 / d8;
its decode raises NotImplementedError for unpacked d2 / d8 and its prefill for
any other dim. sz-pack once packed d8
modules before the d8 switch existed, every bundle gate passed, and the
artifact crashed on its first prompt. Two guards: sz-pack leaves unservable
modules untouched, and check-bundle FAILS a config whose vq_skipzero lists one.
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "src" / "vqlab" / "gate" / "check_bundle.py"
SZ_DIR = ROOT / "src" / "vqlab" / "experimental" / "skipzero"
sys.path.insert(0, str(SZ_DIR))
import sz_pack  # noqa: E402


def _write_st(path, tensors):
    """Minimal safetensors writer: {name: (dtype, np.ndarray)}."""
    hdr, off, blobs = {"__metadata__": {"format": "mlx"}}, 0, []
    for name, (dt, a) in tensors.items():
        b = a.tobytes()
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    js = json.dumps(hdr).encode()
    js += b" " * ((8 - len(js) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(js)) + js + b"".join(blobs))


def _module(E=2, OUT=8, W=4, G=2):
    sc = np.ones((E, OUT, G), np.float32)
    sc[0, :3] = 0                            # three dead rows
    return {"codes": ("U8", np.zeros((E, OUT, W), np.uint8)),
            "vq_scales": ("F32", sc)}


def _artifact(tmp_path, dims, pack_bits=None):
    art = tmp_path / "src_art"
    art.mkdir()
    vqm, tens = {}, {}
    for i, d in enumerate(dims):
        p = f"model.layers.{i}.mlp.switch_mlp.up_proj"
        vqm[p] = {"dim": d, "k": 256, "group": 64}
        if pack_bits:
            vqm[p]["pack_bits"] = pack_bits
        for leaf, t in _module().items():
            tens[f"{p}.{leaf}"] = t
    _write_st(art / "model-00001-of-00001.safetensors", tens)
    (art / "config.json").write_text(json.dumps({"vq_modules": vqm}))
    (art / "model.py").write_text("# model\n")
    return art


def test_sz_pack_skips_unsupported_dim(tmp_path, capsys):
    art = _artifact(tmp_path, [4, 2])
    out = tmp_path / "out"
    assert sz_pack.main([str(art), "--out", str(out), "--allow-any-out"]) == 0
    cfg = json.loads((out / "config.json").read_text())
    mods = cfg["vq_skipzero"]["modules"]
    assert list(mods) == ["model.layers.0.mlp.switch_mlp.up_proj"]
    err = capsys.readouterr().out
    assert "skipped 1 module(s)" in err and "dim=2 unpacked: 1" in err


@pytest.mark.parametrize("dim", [2, 8])
def test_sz_pack_packs_packed_d2_d8(tmp_path, dim):
    art = _artifact(tmp_path, [4, dim], pack_bits=8)
    out = tmp_path / "out"
    assert sz_pack.main([str(art), "--out", str(out), "--allow-any-out"]) == 0
    mods = json.loads((out / "config.json").read_text())["vq_skipzero"]["modules"]
    assert sorted(mods) == [f"model.layers.{i}.mlp.switch_mlp.up_proj" for i in (0, 1)]


def test_sz_pack_skips_unpacked_d8(tmp_path, capsys):
    art = _artifact(tmp_path, [4, 8])
    out = tmp_path / "out"
    assert sz_pack.main([str(art), "--out", str(out), "--allow-any-out"]) == 0
    mods = json.loads((out / "config.json").read_text())["vq_skipzero"]["modules"]
    assert list(mods) == ["model.layers.0.mlp.switch_mlp.up_proj"]
    assert "dim=8 unpacked: 1" in capsys.readouterr().out


def test_sz_pack_refuses_when_nothing_qualifies(tmp_path, capsys):
    art = _artifact(tmp_path, [2])
    assert sz_pack.main([str(art), "--dry-run"]) == 1
    assert "REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize("dim,pack_bits", [(2, None), (8, None), (6, 10)])
def test_check_bundle_fails_on_unservable_skipzero_module(tmp_path, dim, pack_bits):
    art = tmp_path / "art"
    art.mkdir()
    p = "model.layers.0.mlp.switch_mlp.up_proj"
    spec = {"dim": dim, "k": {2: 1024, 8: 16384}.get(dim, 256), "group": 64}
    if pack_bits:
        spec["pack_bits"] = pack_bits
    cfg = {"vq_modules": {p: spec},
           "vq_skipzero": {"loader": "runtime", "modules": {p: {}}}}
    (art / "config.json").write_text(json.dumps(cfg))
    (art / "model.py").write_text("_SZ_MODS = None\n")
    r = subprocess.run([sys.executable, str(GATE), "--artifact", str(art)],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no skipzero kernel" in r.stdout and p in r.stdout and f"dim={dim}" in r.stdout
