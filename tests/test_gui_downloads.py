"""agents/downloads.py: Hub downloads into <teachers>, one process each,
judged by its exit file; nothing here deletes."""
import fcntl
import json
import os
import signal
import types

import pytest

from vqlab.agents import downloads as D


class FakeProc:
    def __init__(self, argv, **k):
        self.argv, self.pid, self.rc = argv, 4242, None

    def poll(self):
        return self.rc


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_TEACHERS_DIR", str(tmp_path / "teachers"))
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path / "records"))
    monkeypatch.setattr(D, "_PROCS", {})
    monkeypatch.setattr(D, "_plan", lambda repo: ([*D.PATTERNS, "model.safetensors"], 1000))
    procs = []
    monkeypatch.setattr(D.subprocess, "Popen", lambda argv, **k: procs.append(FakeProc(argv)) or procs[-1])
    scans = []
    monkeypatch.setattr(D, "_rescan", lambda: scans.append(1))
    return tmp_path, procs, scans


def test_start_runs_one_fetch_into_teachers(lab):
    root, procs, _ = lab
    assert D.start({"repo": "org/model"})["ok"]
    assert D.start({"repo": "org/model"})["ok"] and len(procs) == 1      # already running
    argv = procs[0].argv
    assert argv[3:5] == ["org/model", str(root / "teachers" / "org--model")]
    assert "model.safetensors" in argv and "*.safetensors" not in argv and "*.gguf" not in argv
    [row] = D.downloads()
    assert row["status"] == "downloading" and row["total"] == 1000
    assert row["dest"] == "<teachers>/org--model"


@pytest.mark.parametrize("code, status, scanned", [("0", "done", 1), ("1", "failed", 0), (None, "stopped", 0)])
def test_how_a_fetch_ended(lab, code, status, scanned):
    root, procs, scans = lab
    D.start({"repo": "org/model"})
    procs[0].rc = 0 if code == "0" else -15
    if code is not None:
        (root / "teachers" / "org--model" / ".vqlab-download-rc").write_text(code)
    assert D.downloads()[0]["status"] == status and len(scans) == scanned
    assert json.loads((root / "records" / "downloads.json").read_text())["org/model"]["status"] == status


def test_refusals(lab, monkeypatch):
    assert D.start({"repo": "not a repo"})["ok"] is False
    assert D.start({"repo": "../../etc"})["ok"] is False
    assert D.stop({"repo": "org/none"})["ok"] is False

    def missing(repo):
        raise RuntimeError("404 Repository Not Found\nlong detail")
    monkeypatch.setattr(D, "_plan", missing)
    r = D.start({"repo": "org/gone"})
    assert r["ok"] is False and "long detail" not in r["error"]
    assert not D.dest("org/gone").exists()            # a refused repo leaves no folder


def test_stop_signals_the_fetch_group(lab, monkeypatch):
    _, procs, _ = lab
    sent = []
    monkeypatch.setattr(D.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    D.start({"repo": "org/model"})
    assert D.stop({"repo": "org/model"})["ok"] and sent == [(4242, signal.SIGTERM)]
    assert D.downloads()[0]["status"] == "stopped"


def _restarted(lab, monkeypatch, pid):
    """A download an earlier server started: on record, not our child."""
    root, _, _ = lab
    sent = []
    monkeypatch.setattr(D.os, "killpg", lambda p, sig: sent.append((p, sig)))
    d = root / "teachers" / "org--model"
    d.mkdir(parents=True)
    (root / "records").mkdir()
    (root / "records" / "downloads.json").write_text(json.dumps(
        {"org/model": {"dest": str(d), "pid": pid, "status": "downloading"}}))
    return d, sent


def test_stale_pid_without_the_lock_is_never_signalled(lab, monkeypatch):
    d, sent = _restarted(lab, monkeypatch, os.getpid())     # a live pid, reused
    (d / D.LOCKFILE).write_text(str(os.getpid()))           # but nobody holds the lock
    assert D.stop({"repo": "org/model"})["ok"] and sent == []
    assert D.downloads()[0]["status"] == "stopped"


def test_held_lock_with_the_recorded_pid_is_signalled(lab, monkeypatch):
    d, sent = _restarted(lab, monkeypatch, 4343)
    with open(d / D.LOCKFILE, "w") as f:
        f.write("4343")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_EX)
        # another holder's file lock: flock is per open file description
        assert D._lock_pid(d) == 4343
        assert D.stop({"repo": "org/model"})["ok"] and sent == [(4343, signal.SIGTERM)]


def test_held_lock_with_another_pid_is_not_ours(lab, monkeypatch):
    d, sent = _restarted(lab, monkeypatch, 4343)
    with open(d / D.LOCKFILE, "w") as f:
        f.write("9999")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_EX)
        assert D.stop({"repo": "org/model"})["ok"] and sent == []


def test_plan_allows_only_the_indexed_top_level_shards(tmp_path, monkeypatch):
    import huggingface_hub as hf
    files = {"model.safetensors.index.json": 10, "model-00001-of-00002.safetensors": 100,
             "model-00002-of-00002.safetensors": 200, "original/model.safetensors": 999,
             "tokenizer.model": 5, "vocab.bpe": 3, "model.gguf": 777}
    info = types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=n, size=b) for n, b in files.items()])
    monkeypatch.setattr(hf, "HfApi", lambda: types.SimpleNamespace(model_info=lambda r, **k: info))

    def fetch(repo, name):
        p = tmp_path / name
        p.write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                                "b": "model-00002-of-00002.safetensors"}}))
        return str(p)
    monkeypatch.setattr(hf, "hf_hub_download", fetch)
    allow, total = D._plan("org/model")
    assert not (tmp_path / "teachers").exists()
    assert "model-00002-of-00002.safetensors" in allow and "*.safetensors" not in allow
    assert total == 10 + 100 + 200 + 5 + 3


def test_plan_without_an_index_takes_top_level_safetensors(tmp_path, monkeypatch):
    import huggingface_hub as hf
    files = {"model.safetensors": 100, "config.json": 1, "sub/model.safetensors": 999,
             "consolidated.safetensors": 500}
    info = types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=n, size=b) for n, b in files.items()])
    monkeypatch.setattr(hf, "HfApi", lambda: types.SimpleNamespace(model_info=lambda r, **k: info))
    allow, total = D._plan("org/model")
    assert allow[-1:] == ["model.safetensors"] and "consolidated.safetensors" not in allow
    assert total == 101


@pytest.mark.parametrize("code, status, scanned", [("0", "done", 1), ("1", "failed", 0), (None, "stopped", 0)])
def test_stop_of_an_ended_fetch_reads_its_exit_file(lab, monkeypatch, code, status, scanned):
    d, sent = _restarted(lab, monkeypatch, 4343)
    if code is not None:
        (d / ".vqlab-download-rc").write_text(code)
    r = D.stop({"repo": "org/model"})
    assert r["ok"] and r["status"] == status and sent == [] and len(lab[2]) == scanned
