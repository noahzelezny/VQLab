"""Knurlogic arm of vision-smoke: routes via the page, waits for ready, passes roles through (F205)."""
import argparse
import types

import pytest

from vqlab.gate import vision_smoke as vs


class FakeK:
    def __init__(self):
        self.loaded = None
        self.unloaded = None

    def load(self, **kw):
        self.loaded = kw
        return {"job": "j1"}

    def unload(self, job):
        self.unloaded = job


def test_wait_loading_then_ready():
    seq = iter([{"data": []},
                {"data": [{"id": "mdl", "status": "loading"}]},
                {"data": [{"id": "mdl", "status": "ready"}]}])
    sleeps = []
    mid = vs._wait_ready("http://p", "mdl", 100, 1, sleeps.append,
                         get=lambda u: next(seq))
    assert mid == "mdl" and len(sleeps) == 2


def test_routes_via_page(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[boxes.a]\nmachine = "MacBook Pro"\n')
    monkeypatch.setenv("VQLAB_CONFIG", str(cfg))
    art = tmp_path / "mdl"
    art.mkdir()
    md = tmp_path / "models"
    md.mkdir()
    (md / "mdl").symlink_to(art)
    monkeypatch.setenv("KNURLOGIC_MODELS", str(md))
    img = tmp_path / "i.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(vs, "_wait_ready", lambda page, name, *a, **k: name)
    posts = []

    def post(url, doc):
        posts.append((url, doc["model"]))
        n = 20 if isinstance(doc["messages"][0]["content"], str) else 90
        return {"usage": {"prompt_tokens": n},
                "choices": [{"message": {"content": "a circle"}}]}

    K = FakeK()
    a = argparse.Namespace(knurlogic=["fit"], knurlogic_split="", knurlogic_link="",
                           image=str(img), max_tokens=8)
    assert vs._knurlogic(art, a, K=K, SPK=types.SimpleNamespace(_post=post),
                         sleep=lambda s: None) == 0
    assert K.loaded["machines"] == ["fit"] and K.unloaded == "j1"
    assert posts and all(u == "http://127.0.0.1:8899" and m == "mdl" for u, m in posts)


def test_fit_stands_alone(tmp_path):
    a = argparse.Namespace(knurlogic=["fit", "peers"], knurlogic_split="",
                           knurlogic_link="", image=None, max_tokens=8)
    with pytest.raises(SystemExit, match="stands alone"):
        vs._knurlogic(tmp_path, a, K=FakeK(), SPK=types.SimpleNamespace())
