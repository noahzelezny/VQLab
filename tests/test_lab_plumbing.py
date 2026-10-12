"""The small shared pieces the GUI and the queue both lean on: host identity,
atomic JSON, JSON lines, hf:// expansion, what a step writes, past-run ETAs."""
import json
import os

import pytest

from vqlab import config
from vqlab.queue import eta, steps
from vqlab.util import fs


def test_host_identity_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(config.socket, "gethostname", lambda: "Box-A.example.lan")
    assert config.short_host() == "Box-A"
    assert config.is_this_host("box-a") and config.is_this_host("BOX-A")
    assert not config.is_this_host("box-b") and not config.is_this_host(None)


def test_write_json_atomic_replaces_whole_and_leaves_no_temp(tmp_path):
    p = tmp_path / "deep" / "state.json"
    fs.write_json_atomic(p, {"a": 1})
    fs.write_json_atomic(p, {"b": 2})
    assert json.loads(p.read_text()) == {"b": 2} and p.read_text() == '{\n "b": 2\n}'
    assert [x.name for x in p.parent.iterdir()] == ["state.json"]
    assert oct(p.stat().st_mode & 0o777) == oct(0o666 & ~_umask())


def _umask():
    m = os.umask(0)
    os.umask(m)
    return m


def test_read_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    assert fs.read_jsonl(p) == []
    p.write_text('{"a": 1}\n\nnot json\n{"a": 2}\n{"a": 3}\n')
    assert fs.read_jsonl(p) == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert fs.read_jsonl(p, limit=2) == [{"a": 2}, {"a": 3}]
    with pytest.raises(ValueError):
        fs.read_jsonl(p, strict=True)


def test_hf_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snaps = tmp_path / "models--org--m" / "snapshots"
    (snaps / ("abc1234" + "0" * 33)).mkdir(parents=True)
    (snaps / ("abc1234" + "1" * 33)).mkdir()
    (snaps / ("fed4321" + "0" * 33)).mkdir()
    assert config.hf_snapshot("org", "m", "abc1234") is None              # ambiguous prefix
    assert config.hf_snapshot("org", "m", "fed4321") == snaps / ("fed4321" + "0" * 33)
    assert config.hf_snapshot("org", "other", "fed4321") is None
    assert config.expand("hf://org/m@0000000") == "hf://org/m@0000000"   # not held: left as is
    assert config.expand("--src=hf://org/m@fed4321") == f"--src={snaps / ('fed4321' + '0' * 33)}"
    assert config.expand("hf://org/m@fed432") == "hf://org/m@fed432"     # under 7: not a ref


def _st(pf, args=("--out", "x")):
    return {"name": "s", "cmd": "fit-moe", "args": list(args), "preflight": pf}


def test_preflight_problem_writes_gib_rule():
    assert steps._preflight_problem(_st({"args": ["--small"], "writes_gib": 2})) is None
    assert steps._preflight_problem(_st({"full": "why", "writes_gib": 2})) is None
    assert "but skip" in steps._preflight_problem(_st({"skip": "why", "writes_gib": 2}))
    assert "one of" in steps._preflight_problem(_st({"writes_gib": 2}))
    assert "identical" in steps._preflight_problem(_st({"args": ["--out", "x"]}))


def test_preflight_writes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.safetensors").write_bytes(b"x" * 100)
    args = ["--src", str(src), "--out", str(tmp_path / "out")]
    stated = _st({"args": ["--small"], "writes_gib": 1.5}, args)
    assert steps.preflight_writes(stated, args, tmp_path, preflight=True) == int(1.5 * 2**30)
    assert steps.preflight_writes(stated, args, tmp_path, preflight=False) == int(1.5 * 2**30)
    small = _st({"args": ["--small"]}, args)
    assert steps.preflight_writes(small, args, tmp_path, preflight=True) == 0
    assert steps.preflight_writes(small, args, tmp_path, preflight=False) == 100
    full = _st({"full": "no partial mode"}, args)
    assert steps.preflight_writes(full, args, tmp_path, preflight=True) == 100


def _q(root, name, secs, preflight=False, args=("--a",)):
    d = root / name
    d.mkdir()
    (d / "queue.json").write_text(json.dumps({"steps": [{"name": "s", "cmd": "c", "args": list(args)}]}))
    (d / "state.json").write_text(json.dumps({"preflight": preflight, "steps": [
        {"name": "s", "status": "pass" if secs else "fail", "seconds": secs}]}))
    return d


def test_past_runs_and_median(tmp_path):
    _q(tmp_path, "1", 100)
    _q(tmp_path, "2", 300)
    _q(tmp_path, "3", 900, preflight=True)            # a preflight's time is not a full run's
    _q(tmp_path, "4", None)                           # failed: no time
    _q(tmp_path, "5", 50, args=("--b",))              # another step
    me = _q(tmp_path, "6", 1000)
    past = eta.past_runs(tmp_path, exclude=me)
    st = {"name": "s", "cmd": "c", "args": ["--a"]}
    assert past[eta._key(st)] == [100.0, 300.0]
    assert eta.median_past(past, st) == 200.0
    assert eta.median_past(past, {"cmd": "c", "args": ["--z"]}) is None
    assert eta.history(tmp_path, st, exclude=me) == [100.0, 300.0]


def test_step_eta_uses_the_given_index(tmp_path):
    st = {"name": "s", "cmd": "c", "args": ["--a"]}
    past = {eta._key(st): [60.0, 120.0]}
    e = eta.step_eta(st, {"status": "pending"}, tmp_path / "none", tmp_path, past=past)
    assert e["eta_s"] == 90 and "median of 2" in e["basis"]
    e = eta.step_eta(st, {"status": "pending"}, tmp_path / "none", tmp_path, past={})
    assert e["eta_s"] is None                         # an empty index is not "read the disk"
    assert eta.step_eta(st, {"status": "pass"}, tmp_path, tmp_path, past={})["eta_s"] == 0.0
