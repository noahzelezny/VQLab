"""agents/downloads.py: Hub downloads into <teachers>, one process each,
judged by its exit file; nothing here deletes."""
import json
import signal

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
    monkeypatch.setattr(D, "_total", lambda repo: 1000)
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
    assert "*.safetensors" in argv and "*.gguf" not in argv
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
    monkeypatch.setattr(D, "_total", missing)
    r = D.start({"repo": "org/gone"})
    assert r["ok"] is False and "long detail" not in r["error"]


def test_stop_signals_the_fetch_group(lab, monkeypatch):
    _, procs, _ = lab
    sent = []
    monkeypatch.setattr(D.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    D.start({"repo": "org/model"})
    assert D.stop({"repo": "org/model"})["ok"] and sent == [(4242, signal.SIGTERM)]
    assert D.downloads()[0]["status"] == "stopped"
