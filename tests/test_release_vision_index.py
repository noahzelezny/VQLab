"""check-release's index total_size gate and the image-smoke arms (CPU, no model).

check-release must FAIL an index whose metadata.total_size is missing or
stale, through the same function `vqlab size --check-index` uses. Any tower
tensor makes an image smoke part of the gate. vision-smoke's HTTP arms
(exo --cluster, Knurlogic --knurlogic) share one request/verdict function,
tested here against fake servers; the live Knurlogic image path is GPU-owed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import struct
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = REPO / "src/vqlab/gate"


def _vs():
    spec = importlib.util.spec_from_file_location("vision_smoke_t", GATE / "vision_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shard(path, tensors):
    """A safetensors file of zero-filled F16 tensors; returns its tensor bytes."""
    hdr, off = {}, 0
    for k, shape in tensors.items():
        n = 2
        for d in shape:
            n *= d
        hdr[k] = {"dtype": "F16", "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * off)
    return off


def _artifact(root, total_size="right", tower=False):
    t = {"model.embed_tokens.weight": (8, 4)}
    if tower:
        t["vision.blocks.0.attn.weight"] = (4, 4)
    n = _shard(root / "model-00001.safetensors", t)
    md = {} if total_size is None else {"total_size": n if total_size == "right" else n * 2}
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": md, "weight_map": {k: "model-00001.safetensors" for k in t}}))
    (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    return root


@pytest.mark.parametrize("ts,state", [("right", "ok"), (None, "missing"), ("double", "stale")])
def test_index_total_size(tmp_path, ts, state):
    from vqlab.records.size_cmd import index_total_size
    have, want, got = index_total_size(_artifact(tmp_path, ts))
    assert got.startswith(state)
    assert want == 64


def _check_release(art, *extra):
    return subprocess.run([sys.executable, str(GATE / "check_release.py"),
                           "--artifact", str(art), *extra],
                          capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(REPO / "src")))


def test_check_release_fails_a_stale_total_size(tmp_path):
    r = _check_release(_artifact(tmp_path, "double"), "--no-smoke")
    assert r.returncode == 1
    assert "index metadata.total_size" in r.stdout and "stale by" in r.stdout


def test_check_release_quiet_on_a_right_total_size(tmp_path):
    r = _check_release(_artifact(tmp_path, "right"), "--no-smoke")
    assert "index metadata.total_size" not in r.stdout


def test_check_release_notes_unverified_tower_under_no_smoke(tmp_path):
    r = _check_release(_artifact(tmp_path, "right", tower=True), "--no-smoke")
    assert "1 tower tensors, image path NOT verified" in r.stdout


def test_check_release_routes_the_image_smoke_by_arm():
    src = (GATE / "check_release.py").read_text()
    assert 'tensor_class as _tc' in src and '_find("vision_smoke.py")' in src
    blk = src[src.index("# IMAGE SMOKE."):src.index("# BUILD RECORD")]
    assert '"--knurlogic"' in blk and '"--cluster"' in blk


def test_vision_smoke_counts_a_flat_deepseek_tower(tmp_path):
    vs = _vs()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert vs._n_tower_tensors(_artifact(tmp_path / "a", tower=True)) == 1
    assert vs._n_tower_tensors(_artifact(tmp_path / "b", tower=False)) == 0


def _probe(tmp_path, vs):
    return vs._probe_image(tmp_path / "probe.png")


def _server(n_text=12, n_img=300, caption="A yellow circle on blue.", fail_image=False):
    sent = []

    def post(doc):
        sent.append(doc)
        content = doc["messages"][0]["content"]
        if isinstance(content, str):
            return {"usage": {"prompt_tokens": n_text},
                    "choices": [{"message": {"content": "x"}}]}
        if fail_image:
            raise RuntimeError("HTTP 400: image_url not supported")
        return {"usage": {"prompt_tokens": n_img},
                "choices": [{"message": {"content": caption}}]}
    return post, sent


def test_image_request_is_openai_shaped(tmp_path):
    pytest.importorskip("PIL")
    vs = _vs()
    post, sent = _server()
    assert vs._image_request_check(post, "m", _probe(tmp_path, vs), 32) == []
    text, img = sent
    assert text["max_tokens"] == 1 and isinstance(text["messages"][0]["content"], str)
    parts = img["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["type"] == "text"
    assert img["temperature"] == 0 and img["max_tokens"] == 32 and img["model"] == "m"


@pytest.mark.parametrize("kw,needle", [
    ({"n_img": 12}, "NO prompt tokens"),
    ({"caption": ""}, "no caption"),
    ({"fail_image": True}, "refused the image request"),
])
def test_image_request_failures(tmp_path, kw, needle):
    pytest.importorskip("PIL")
    vs = _vs()
    post, _ = _server(**kw)
    probs = vs._image_request_check(post, "m", _probe(tmp_path, vs), 8)
    assert any(needle in p for p in probs)


class _FakeK:
    def __init__(self):
        self.loaded, self.unloaded = [], []

    def load(self, **kw):
        self.loaded.append(kw)
        return {"job": "j1", "url": "http://fake"}

    def state(self):
        return {"machines": []}

    def unload(self, job):
        self.unloaded.append(job)


class _FakeSPK:
    def __init__(self, post):
        self.post = post

    def _entry(self, K, job):
        return {"phase": "ready"}

    def _ready(self, e):
        return True

    def _post(self, url, doc, timeout=1800):
        assert url == "http://fake"
        return self.post(doc)


def _knurlogic_args(img):
    return argparse.Namespace(knurlogic=["here", "peers"], knurlogic_split="pipeline",
                              knurlogic_link="", image=str(img), max_tokens=16)


def test_knurlogic_arm_loads_this_artifact_and_unloads(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    vs = _vs()
    vs._wait_ready = lambda page, name, *a, **k: name
    art = _artifact(tmp_path, tower=True)
    md = tmp_path.parent / "kmodels"
    md.mkdir(exist_ok=True)
    link = md / art.resolve().name
    if not link.exists():
        link.symlink_to(art.resolve())
    monkeypatch.setenv("KNURLOGIC_MODELS", str(md))
    K, (post, sent) = _FakeK(), _server()
    assert vs._knurlogic(art, _knurlogic_args(_probe(tmp_path, vs)), K=K,
                         SPK=_FakeSPK(post), page="http://fake") == 0
    assert K.loaded[0]["machines"] == ["here", "peers"] and K.loaded[0]["split"] == "pipeline"
    assert K.unloaded == ["j1"] and len(sent) == 2


def test_knurlogic_arm_fails_and_still_unloads(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    vs = _vs()
    vs._wait_ready = lambda page, name, *a, **k: name
    art = _artifact(tmp_path, tower=True)
    md = tmp_path.parent / "kmodels2"
    md.mkdir(exist_ok=True)
    link = md / art.resolve().name
    if not link.exists():
        link.symlink_to(art.resolve())
    monkeypatch.setenv("KNURLOGIC_MODELS", str(md))
    K, (post, _) = _FakeK(), _server(n_img=12)
    with pytest.raises(SystemExit, match="NO prompt tokens"):
        vs._knurlogic(art, _knurlogic_args(_probe(tmp_path, vs)), K=K, SPK=_FakeSPK(post), page="http://fake")
    assert K.unloaded == ["j1"]


def test_knurlogic_arm_refuses_a_models_dir_pointing_elsewhere(tmp_path, monkeypatch):
    vs = _vs()
    art = _artifact(tmp_path, tower=True)
    monkeypatch.setenv("KNURLOGIC_MODELS", str(tmp_path / "nowhere"))
    K = _FakeK()
    with pytest.raises(SystemExit, match="does not resolve"):
        vs._knurlogic(art, _knurlogic_args("x.png"), K=K, SPK=_FakeSPK(None))
    assert K.loaded == []
