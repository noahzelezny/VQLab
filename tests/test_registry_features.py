"""records/registry.entry: what the GUI reads about a build comes from its
own config.json, skip-zero included."""
import json

from vqlab.records import registry


def _art(tmp_path, name, cfg):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", **cfg}))
    (d / "model-00001.safetensors").write_bytes(b"\0" * 64)
    return d


def test_skipzero_comes_from_the_config(tmp_path):
    sz = _art(tmp_path, "Org--Sz", {"vq_skipzero": {"format": "vq-skipzero", "version": 1}})
    plain = _art(tmp_path, "Org--Plain", {})
    assert registry.entry(sz)["skipzero"] is True
    assert registry.entry(plain)["skipzero"] is False
