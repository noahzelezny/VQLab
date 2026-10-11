"""vqlab lab scan: finds models in the standard places, tags them, dedupes
symlinked copies, and never reads past config.json and file lists."""
import json

import pytest

from vqlab import config as C
from vqlab.agents import model_scan as M


def _model(d, cfg, files=("model.safetensors",)):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    for f in files:
        (d / f).write_bytes(b"x" * 16)
    return d


@pytest.fixture
def box(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for v in (*C._ENV.values(), "HF_HOME", "HF_HUB_CACHE", "EXO_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(M, "_home", lambda: home)
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.setenv("VQLAB_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VQLAB_TEACHERS_DIR", str(tmp_path / "teachers"))
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path / "rec"))
    C._file_paths.cache_clear(); C._file_table.cache_clear()
    (tmp_path / "models").mkdir()
    return tmp_path, home


def test_finds_and_tags_every_source(box):
    tmp, home = box
    _model(tmp / "teachers" / "Qwen--Qwen3.6-35B-A3B-bf16",
           {"model_type": "qwen3_5_moe", "torch_dtype": "bfloat16"})
    snap = home / ".cache/huggingface/hub/models--mlx-community--X-4bit/snapshots/abc"
    _model(snap, {"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}})
    part = home / ".cache/huggingface/hub/models--org--Partial/snapshots/def"
    part.mkdir(parents=True)
    (part / "README.md").write_text("")
    _model(home / ".lmstudio/models/pub/Some-MLX", {"model_type": "llama", "torch_dtype": "float16"})
    _model(home / ".exo/models/deepseek-ai--DS",
           {"model_type": "deepseek_v4", "quantization_config": {"quant_method": "fp8"}})
    g = home / ".lmstudio/models/pub/Some-GGUF"
    g.mkdir(parents=True)
    (g / "m.gguf").write_bytes(b"x")
    o = home / ".ollama/models"
    (o / "manifests/registry/lib/llama3").mkdir(parents=True)
    (o / "manifests/registry/lib/llama3/latest").write_text("{}")
    (o / "blobs").mkdir()
    rows = {r["name"]: r for r in M.refresh()}
    assert rows["Qwen/Qwen3.6-35B-A3B-bf16"]["use"] == "original"
    assert rows["Qwen/Qwen3.6-35B-A3B-bf16"]["family"] == "qwen3_5"
    assert rows["mlx-community/X-4bit"]["use"] == "quantized" and rows["mlx-community/X-4bit"]["bits"] == 4
    assert rows["org/Partial"]["use"] == "incomplete"
    assert rows["pub/Some-MLX"]["source"] == "LM Studio" and rows["pub/Some-MLX"]["family"] is None
    assert rows["deepseek-ai/DS"]["use"] == "original"          # fp8 as released
    assert rows["pub/Some-GGUF"]["use"] == "unreadable"
    assert any(r["source"] == "Ollama" and r["count"] == 1 for r in rows.values())
    assert M.load() and (tmp / "rec" / "registry" / "found.jsonl").exists()


def test_symlinked_copy_listed_once(box):
    tmp, home = box
    real = _model(tmp / "models" / "a--B", {"model_type": "qwen3_5", "torch_dtype": "bfloat16"})
    (home / ".exo").mkdir(parents=True)
    (home / ".exo" / "models").symlink_to(tmp / "models")
    rows = [r for r in M.scan() if r["name"] == "a/B"]
    assert len(rows) == 1 and rows[0]["source"] == "vqlab models" and real.exists()
