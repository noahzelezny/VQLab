"""agents/downloads.py: Hub downloads into <teachers>, one process each,
judged by its exit file; nothing here deletes."""
import fcntl
import json
import os
import signal
import types

import pytest

from vqlab.agents import _fetch
from vqlab.agents import downloads as D
from vqlab.util.errors import UserError


class FakeProc:
    def __init__(self, argv, **k):
        self.argv, self.pid, self.rc = argv, 4242, None

    def poll(self):
        return self.rc


@pytest.fixture
def lab(lab_dirs, monkeypatch):
    tmp_path = lab_dirs.root
    monkeypatch.setattr(D, "_PROCS", {})
    monkeypatch.setattr(D, "_plan", lambda repo: ([*D.PATTERNS, "model.safetensors"], 1000))
    procs = []
    monkeypatch.setattr(D.subprocess, "Popen", lambda argv, **k: procs.append(FakeProc(argv)) or procs[-1])
    scans = []
    monkeypatch.setattr(D.model_scan, "refresh_in_background", lambda: scans.append(1))
    return tmp_path, procs, scans


def test_start_runs_one_fetch_into_teachers(lab):
    root, procs, _ = lab
    assert D.start({"repo": "org/model"})["ok"]
    assert D.start({"repo": "org/model"})["ok"] and len(procs) == 1      # already running
    argv = procs[0].argv
    assert argv[1:5] == ["-m", "vqlab.agents._fetch", "org/model", str(root / "teachers" / "org--model")]
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
        (root / "teachers" / "org--model" / D.RCFILE).write_text(code)
    assert D.downloads()[0]["status"] == status and len(scans) == scanned
    assert json.loads((root / "records" / "downloads.json").read_text())["org/model"]["status"] == status


def test_refusals(lab, monkeypatch):
    for repo in ("not a repo", "../../etc"):
        with pytest.raises(UserError, match="org/name"):
            D.start({"repo": repo})
    with pytest.raises(UserError, match="not running"):
        D.stop({"repo": "org/none"})

    def missing(repo):
        raise OSError("404 Repository Not Found\nlong detail")
    monkeypatch.setattr(D, "_plan", missing)
    with pytest.raises(UserError) as e:
        D.start({"repo": "org/gone"})
    assert "404" in str(e.value) and "long detail" not in str(e.value)
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
    files = {"model.safetensors.index.json": 10, "model-00001-of-00002.safetensors": 100,
             "model-00002-of-00002.safetensors": 200, "original/model.safetensors": 999,
             "tokenizer.model": 5, "vocab.bpe": 3, "model.gguf": 777}
    info = types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=n, size=b) for n, b in files.items()])
    monkeypatch.setattr(D, "HfApi", lambda: types.SimpleNamespace(model_info=lambda r, **k: info))

    def fetch(repo, name):
        p = tmp_path / name
        p.write_text(json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                                "b": "model-00002-of-00002.safetensors"}}))
        return str(p)
    monkeypatch.setattr(D, "hf_hub_download", fetch)
    allow, total = D._plan("org/model")
    assert not (tmp_path / "teachers").exists()
    assert "model-00002-of-00002.safetensors" in allow and "*.safetensors" not in allow
    assert total == 10 + 100 + 200 + 5 + 3


def test_plan_without_an_index_takes_top_level_safetensors(tmp_path, monkeypatch):
    files = {"model.safetensors": 100, "config.json": 1, "sub/model.safetensors": 999,
             "consolidated.safetensors": 500}
    info = types.SimpleNamespace(siblings=[types.SimpleNamespace(rfilename=n, size=b) for n, b in files.items()])
    monkeypatch.setattr(D, "HfApi", lambda: types.SimpleNamespace(model_info=lambda r, **k: info))
    allow, total = D._plan("org/model")
    assert allow[-1:] == ["model.safetensors"] and "consolidated.safetensors" not in allow
    assert total == 101


@pytest.mark.parametrize("code, status, scanned", [("0", "done", 1), ("1", "failed", 0), (None, "stopped", 0)])
def test_stop_of_an_ended_fetch_reads_its_exit_file(lab, monkeypatch, code, status, scanned):
    d, sent = _restarted(lab, monkeypatch, 4343)
    if code is not None:
        (d / D.RCFILE).write_text(code)
    r = D.stop({"repo": "org/model"})
    assert r["ok"] and r["status"] == status and sent == [] and len(lab[2]) == scanned


@pytest.mark.parametrize("fails", [False, True])
def test_fetch_records_its_pid_and_how_it_ended(tmp_path, monkeypatch, fails):
    got = []

    def snapshot(repo, local_dir, allow_patterns):
        got.append((repo, local_dir, allow_patterns))
        if fails:
            raise OSError("gated")
    monkeypatch.setattr(_fetch, "snapshot_download", snapshot)
    argv = ["org/model", str(tmp_path), D.LOCKFILE, D.RCFILE, "*.json", "model.safetensors"]
    if fails:
        with pytest.raises(OSError):
            _fetch.main(argv)
    else:
        assert _fetch.main(argv) == 0
    assert got == [("org/model", tmp_path, ["*.json", "model.safetensors"])]
    assert (tmp_path / D.LOCKFILE).read_text() == str(os.getpid())
    assert (tmp_path / D.RCFILE).read_text() == ("1" if fails else "0")


def test_fetch_leaves_a_running_fetchs_lock_and_rc_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(_fetch, "snapshot_download", lambda *a, **k: pytest.fail("ran twice"))
    with open(tmp_path / D.LOCKFILE, "w") as f:
        f.write("4343")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_EX)
        assert _fetch.main(["org/model", str(tmp_path), D.LOCKFILE, D.RCFILE]) == 2
    assert (tmp_path / D.LOCKFILE).read_text() == "4343" and not (tmp_path / D.RCFILE).exists()


def test_an_interrupted_fetch_records_nothing(tmp_path, monkeypatch):
    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(_fetch, "snapshot_download", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _fetch.main(["org/model", str(tmp_path), D.LOCKFILE, D.RCFILE])
    assert not (tmp_path / D.RCFILE).exists()


def test_plan_says_size_folder_room_and_fit(lab, monkeypatch):
    monkeypatch.setattr(D.config, "free_bytes", lambda p: 7 * D.GIB)
    p = D.plan({"repo": "org/model"})
    assert p["bytes"] == p["need"] == 1000 and p["free"] == 7 * D.GIB and p["fits"]
    assert p["dest"] == str(D.dest("org/model"))
    with pytest.raises(UserError, match="org/name"):
        D.plan({"repo": "nope"})


def test_a_download_the_disk_cannot_hold_is_refused(lab, monkeypatch):
    """A resume counts only the bytes still to fetch; plan and start agree."""
    _, procs, _ = lab
    D.dest("org/model").mkdir(parents=True)
    (D.dest("org/model") / "model.safetensors").write_bytes(b"x" * 400)
    asked = []
    monkeypatch.setattr(D.config, "free_shortfall",
                        lambda path, need: asked.append(need) or 10 * D.GIB)
    assert D.plan({"repo": "org/model"})["fits"] is False
    with pytest.raises(UserError, match="not enough room"):
        D.start({"repo": "org/model"})
    assert asked == [600, 600] and not procs
