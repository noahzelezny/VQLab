"""vqlab mix (CPU): bands pick whole shards by the layers of their VQ modules,
config maps follow the bytes, a straddling band and a foreign skeleton are
refused."""
import json
import struct

import pytest

from vqlab.assemble import mix


def _shard(p, keys):
    hdr = {k: {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]} for i, k in enumerate(keys)}
    h = json.dumps(hdr).encode()
    p.write_bytes(struct.pack("<Q", len(h)) + h + bytes(len(keys)))


def _art(d, k, shards, skel_shape=1):
    d.mkdir()
    wm, vq = {}, {}
    for f, layers in shards.items():
        keys = []
        for L in layers:
            m = f"layers.{L}.mlp.experts"
            keys += [m + ".codes", f"layers.{L}.norm.weight"]
            vq[m] = {"k": k, "dim": 4}
        _shard(d / f, keys)
        wm.update({x: f for x in keys})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"vq_modules": vq, "quantization": {"bits": 4}}))
    (d / "model.py").write_text("# runtime\n")
    return d


def test_band_and_maps(tmp_path):
    lay = {"a.safetensors": [0, 1], "b.safetensors": [2, 3]}
    base = _art(tmp_path / "base", 256, lay)
    hi = _art(tmp_path / "hi", 2048, {"b.safetensors": [2, 3]})       # partial source
    out = tmp_path / "out"
    mix.build(out, base, [(hi, 2, 3)])
    cfg = json.load(open(out / "config.json"))
    assert cfg["vq_modules"]["layers.0.mlp.experts"]["k"] == 256
    assert cfg["vq_modules"]["layers.3.mlp.experts"]["k"] == 2048
    assert (out / "b.safetensors").resolve() == (hi / "b.safetensors").resolve()


def test_straddle_refused(tmp_path):
    base = _art(tmp_path / "base", 256, {"a.safetensors": [0, 1]})
    hi = _art(tmp_path / "hi", 2048, {"a.safetensors": [0, 1]})
    with pytest.raises(SystemExit, match="straddles"):
        mix.plan(base, [(hi, 1, 1)])


def test_runtime_mismatch_refused(tmp_path):
    base = _art(tmp_path / "base", 256, {"a.safetensors": [0]})
    hi = _art(tmp_path / "hi", 2048, {"a.safetensors": [0]})
    (hi / "model.py").write_text("# other runtime\n")
    with pytest.raises(SystemExit, match="one runtime"):
        mix.build(tmp_path / "out", base, [(hi, 0, 0)])


def test_minibase(tmp_path):
    from vqlab.assemble import minibase
    base = _art(tmp_path / "base", 256, {"a.safetensors": [0, 1], "b.safetensors": [2, 3]})
    minibase.main(["--base", str(base), "--layers", "2-3", "--out", str(tmp_path / "mb")])
    wm = json.load(open(tmp_path / "mb" / "model.safetensors.index.json"))["weight_map"]
    assert set(wm.values()) == {"b.safetensors"}
    assert (tmp_path / "mb" / "config.json").exists()
