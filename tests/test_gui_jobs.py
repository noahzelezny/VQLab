"""Jobs Start / Stop / Resume from the GUI (agents/jobs.py): every start goes
through `vqlab queue run`, once, with the token of the plan the operator
confirmed."""
import json
import signal
import socket
import subprocess

import pytest

from vqlab.agents import gui, jobs

STEP = {"name": "s1", "cmd": "smoke", "args": ["--max-tokens", "8"],
        "preflight": {"args": ["--max-tokens", "1"]}}
HERE = socket.gethostname().split(".")[0]


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path / "records"))
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path / "queues"))
    monkeypatch.setenv("VQLAB_SCRATCH", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    (tmp_path / "queues").mkdir()
    files = tmp_path / "records" / "queue-files"
    files.mkdir(parents=True)
    (files / "night.json").write_text(json.dumps({"name": "night", "steps": [STEP]}))
    monkeypatch.setattr(jobs.exo, "exo_instances", lambda: None)
    monkeypatch.setattr(jobs, "lease_holder", lambda: None)
    monkeypatch.setattr(jobs, "_TOKENS", {})
    calls = []

    class R:
        returncode, stderr = 0, ""
        stdout = f"detached (pid 1); vqlab queue status {tmp_path}/queues/20260101-000000-night"

    def run(argv, **k):
        if argv[0] == "ps":
            return type("P", (), {"stdout": runner_cmd[0]})()
        calls.append(argv)
        return R()
    runner_cmd = [""]
    monkeypatch.setattr(jobs.subprocess, "run", run)
    return tmp_path, calls, runner_cmd


def _queue(root, name, **state):
    d = root / "queues" / name
    d.mkdir()
    (d / "queue.json").write_text(json.dumps({"name": "night", "steps": [STEP]}))
    (d / "state.json").write_text(json.dumps({"name": "night", "host": HERE, "commit": "c0ffee",
                                              "steps": [{"name": "s1", "status": "pending"}], **state}))
    return d


def _run_args(argv):
    return argv[argv.index("run") + 1:]


def test_lists_queue_files(lab):
    assert jobs.queue_files() == [{"file": "night.json", "name": "night"}]


def test_start_needs_the_confirmed_token_once(lab):
    _root, calls, _ = lab
    p = jobs.plan({"file": "night.json"})
    assert p["preflight"] is True and [s["name"] for s in p["steps"]] == ["s1"]
    assert p["problems"] == [] and p["eta_s"] is None          # no past run: unknown, not 0
    assert jobs.start({"file": "night.json", "token": "nope"})["ok"] is False and calls == []
    r = jobs.start({"file": "night.json", "token": p["token"]})
    assert r["ok"] is True and r["id"] == "20260101-000000-night"
    assert _run_args(calls[0]) == [str(jobs._queue_file("night.json")), "--preflight", "--detach"]
    assert jobs.start({"file": "night.json", "token": p["token"]})["ok"] is False   # spent
    assert len(calls) == 1


def test_an_edited_file_must_be_confirmed_again(lab):
    root, calls, _ = lab
    tok = jobs.plan({"file": "night.json"})["token"]
    (root / "records/queue-files/night.json").write_text(json.dumps({"name": "night", "steps": [STEP, STEP]}))
    assert jobs.start({"file": "night.json", "token": tok})["ok"] is False and calls == []


def test_plan_sums_past_runs(lab):
    root, _, _ = lab
    _queue(root, "old", status="passed", steps=[{"name": "s1", "status": "pass", "seconds": 600}])
    _queue(root, "older", status="passed", steps=[{"name": "s1", "status": "pass", "seconds": 400}])
    p = jobs.plan({"file": "night.json"})
    assert p["eta_s"] == 500 and p["steps"][0]["eta_s"] == 500


def test_full_run_once_from_a_passed_preflight_at_its_commit(lab):
    root, calls, _ = lab
    _queue(root, "a-preflight", preflight=True, status="failed")
    with pytest.raises(ValueError):
        jobs.plan({"id": "a-preflight"})
    d = _queue(root, "b-preflight", preflight=True, status="passed")
    p = jobs.plan({"id": "b-preflight"})
    assert p["preflight"] is False
    assert jobs.start({"id": "b-preflight", "token": p["token"]})["ok"] is True
    assert _run_args(calls[0]) == [str(d / "queue.json"), "--commit", "c0ffee", "--detach"]
    _queue(root, "c-full", status="running", source=str((d / "queue.json").resolve()))
    with pytest.raises(ValueError, match="already started"):
        jobs.plan({"id": "b-preflight"})


def test_cli_timeout_is_reported_not_raised(lab, monkeypatch):
    def slow(argv, **k):
        raise subprocess.TimeoutExpired(argv, 120)
    monkeypatch.setattr(jobs.subprocess, "run", slow)
    tok = jobs.plan({"file": "night.json"})["token"]
    r = jobs.start({"file": "night.json", "token": tok})
    assert r["ok"] is False and "queue list" in r["error"]


def test_paths_outside_the_lab_are_refused(lab):
    for body in ({"file": "../queues/x.json"}, {"id": ".."}, {"id": "/etc"}):
        with pytest.raises(ValueError):
            jobs.plan(body)


def test_stop_signals_only_this_queues_runner(lab, monkeypatch):
    root, _, runner_cmd = lab
    sent = []
    monkeypatch.setattr(jobs.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(jobs, "pid_alive", lambda pid: True)
    _queue(root, "far", status="running", pid=42, host="elsewhere")
    assert jobs.stop({"id": "far"})["ok"] is False
    _queue(root, "here", status="running", pid=42)
    runner_cmd[0] = "python -m some.other.thing"           # the pid was reused
    assert jobs.stop({"id": "here"})["ok"] is False and sent == []
    runner_cmd[0] = f"python -m vqlab.cli queue run --resume {root}/queues/here"
    assert jobs.stop({"id": "here"})["ok"] is True and sent == [(42, signal.SIGTERM)]

    def gone(pid, sig):
        raise ProcessLookupError(pid)
    monkeypatch.setattr(jobs.os, "kill", gone)
    assert jobs.stop({"id": "here"})["ok"] is False


def test_resume_only_a_halted_queue_of_this_machine(lab):
    root, calls, _ = lab
    _queue(root, "done", status="passed")
    _queue(root, "theirs", status="stopped", host="elsewhere")
    assert jobs.resume({"id": "done"})["ok"] is False
    assert jobs.resume({"id": "theirs"})["ok"] is False and calls == []
    d = _queue(root, "lost", status="running", pid=None)     # runner died
    assert jobs.resume({"id": "lost"})["ok"] is True
    assert _run_args(calls[0]) == ["--resume", str(d), "--detach"]


def test_routes_are_wired():
    assert gui.ROUTES["/api/jobs/files"] is gui.api_jobs_files
    assert {gui.POSTS[f"/api/jobs/{k}"] for k in ("plan", "start", "stop", "resume")} == \
        {jobs.plan, jobs.start, jobs.stop, jobs.resume}


def test_a_preflight_without_a_commit_is_refused_plainly(lab):
    root, _, _ = lab
    _queue(root, "p-preflight", preflight=True, status="passed", commit="")
    with pytest.raises(ValueError, match="no recorded commit"):
        jobs.plan({"id": "p-preflight"})


def test_tokens_expire(lab, monkeypatch):
    _, calls, _ = lab
    tok = jobs.plan({"file": "night.json"})["token"]
    later = jobs.time.time() + jobs.TOKEN_TTL + 1
    monkeypatch.setattr(jobs.time, "time", lambda: later)
    assert jobs.start({"file": "night.json", "token": tok})["ok"] is False and calls == []
