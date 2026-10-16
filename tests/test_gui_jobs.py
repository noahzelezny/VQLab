"""Jobs Start / Stop / Resume from the GUI (agents/jobs.py): every start goes
through `vqlab queue run`, once, with the token of the plan the operator
confirmed."""
import json
import signal
import subprocess

import pytest

from vqlab import config
from vqlab.agents import gui, jobs
from vqlab.util.errors import UserError
from vqlab.queue import report as qreport
from vqlab.queue import runner as qrunner

STEP = {"name": "s1", "cmd": "smoke", "args": ["--max-tokens", "8"],
        "preflight": {"args": ["--max-tokens", "1"]}}
HERE = config.short_host()


@pytest.fixture
def lab(lab_dirs, monkeypatch):
    files = config.queue_files_dir()
    files.mkdir(parents=True)
    (files / "night.json").write_text(json.dumps({"name": "night", "steps": [STEP]}))
    monkeypatch.setattr(jobs.exo, "exo_instances", lambda: None)
    monkeypatch.setattr(jobs, "lease_holder", lambda: None)
    monkeypatch.setattr(jobs, "_TOKENS", jobs.ConfirmTokens())
    calls = []

    def run_cli(*args, timeout):
        calls.append(list(args))
        out = "pinning...\n" + json.dumps({"qdir": f"{lab_dirs.queues}/20260101-000000-night",
                                            "pid": 1}) + "\n"
        return subprocess.CompletedProcess(args, 0, out, "")
    monkeypatch.setattr(jobs.qf, "run_cli", run_cli)
    runner_cmd = [""]
    monkeypatch.setattr(qreport, "pid_cmdline", lambda pid: runner_cmd[0])
    return lab_dirs.root, calls, runner_cmd


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
    assert p["writes"] == 0                                    # a small preflight: not measured
    with pytest.raises(UserError, match="not confirmed"):
        jobs.start({"file": "night.json", "token": "nope"})
    assert calls == []
    r = jobs.start({"file": "night.json", "token": p["token"]})
    assert r["ok"] is True and r["id"] == "20260101-000000-night"
    assert _run_args(calls[0]) == [str(jobs._queue_file("night.json")), "--preflight", "--detach", "--json"]
    with pytest.raises(UserError):                             # spent
        jobs.start({"file": "night.json", "token": p["token"]})
    assert len(calls) == 1


def test_an_edited_file_must_be_confirmed_again(lab):
    root, calls, _ = lab
    tok = jobs.plan({"file": "night.json"})["token"]
    (root / "records/queue-files/night.json").write_text(json.dumps({"name": "night", "steps": [STEP, STEP]}))
    with pytest.raises(UserError):
        jobs.start({"file": "night.json", "token": tok})
    assert calls == []


def test_plan_sums_past_runs(lab):
    root, _, _ = lab
    _queue(root, "old", status="passed", steps=[{"name": "s1", "status": "pass", "seconds": 600}])
    _queue(root, "older", status="passed", steps=[{"name": "s1", "status": "pass", "seconds": 400}])
    p = jobs.plan({"file": "night.json"})
    assert p["eta_s"] == 500 and p["steps"][0]["eta_s"] == 500


def test_a_full_run_of_a_compiled_recipe_writes_its_estimate(lab):
    root, _, _ = lab
    q = {"name": "night", "estimate": {"artifact_gib": 1.0, "scratch_gib": 2.5},
         "steps": [{**STEP, "preflight": {"args": ["--max-tokens", "1"], "writes_gib": 1}}]}
    d = _queue(root, "e-preflight", preflight=True, status="passed")
    (d / "queue.json").write_text(json.dumps(q))
    assert jobs.plan({"file": "night.json"})["writes"] == 0
    assert jobs.plan({"id": "e-preflight"})["writes"] == int(2.5 * 2**30)


def test_full_run_once_from_a_passed_preflight_at_its_commit(lab):
    root, calls, _ = lab
    _queue(root, "a-preflight", preflight=True, status="failed")
    with pytest.raises(UserError):
        jobs.plan({"id": "a-preflight"})
    d = _queue(root, "b-preflight", preflight=True, status="passed")
    p = jobs.plan({"id": "b-preflight"})
    assert p["preflight"] is False
    assert jobs.start({"id": "b-preflight", "token": p["token"]})["ok"] is True
    assert _run_args(calls[0]) == [str(d / "queue.json"), "--commit", "c0ffee", "--detach", "--json"]
    _queue(root, "c-full", status="running", source=str((d / "queue.json").resolve()))
    with pytest.raises(UserError, match="already started"):
        jobs.plan({"id": "b-preflight"})


def test_cli_timeout_is_a_plain_refusal(lab, monkeypatch):
    def slow(*args, timeout):
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(jobs.qf, "run_cli", slow)
    tok = jobs.plan({"file": "night.json"})["token"]
    with pytest.raises(UserError, match="queue list"):
        jobs.start({"file": "night.json", "token": tok})


def test_paths_outside_the_lab_are_refused(lab):
    for body in ({"file": "../queues/x.json"}, {"file": "night.txt"}, {"id": ".."}, {"id": "/etc"},
                 {"id": ""}):
        with pytest.raises(UserError):
            jobs.plan(body)


def test_child_stays_in_its_root(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    assert jobs._child(tmp_path, "a.json", ".json") == (tmp_path / "a.json").resolve()
    assert jobs._child(tmp_path, "a.json", ".txt") is None
    for name in ("..", "", "../x", "/etc", "a/../../x"):
        assert jobs._child(tmp_path, name) is None


def test_stop_signals_only_this_queues_runner(lab, monkeypatch):
    root, _, runner_cmd = lab
    sent = []
    monkeypatch.setattr(qrunner.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(qreport, "pid_alive", lambda pid: True)
    _queue(root, "far", status="running", pid=42, host="elsewhere")
    with pytest.raises(UserError):
        jobs.stop({"id": "far"})
    _queue(root, "here", status="running", pid=42)
    runner_cmd[0] = "python -m some.other.thing"           # the pid was reused
    with pytest.raises(UserError):
        jobs.stop({"id": "here"})
    runner_cmd[0] = "python -m vqlab.cli queue run night.json"    # foreground: names no dir
    with pytest.raises(UserError):
        jobs.stop({"id": "here"})
    assert sent == []
    runner_cmd[0] = f"python -m vqlab.cli queue run --resume {root}/queues/here"
    assert jobs.stop({"id": "here"})["ok"] is True and sent == [(42, signal.SIGTERM)]

    def gone(pid, sig):
        raise ProcessLookupError(pid)
    monkeypatch.setattr(qrunner.os, "kill", gone)
    with pytest.raises(UserError, match="could not stop"):
        jobs.stop({"id": "here"})


def test_queue_stop_cli_is_the_same_stop(lab, monkeypatch, capsys):
    """`vqlab queue stop DIR` signals exactly what the Jobs page's Stop does,
    and fails (exit 1) with nothing to stop."""
    root, _, runner_cmd = lab
    sent = []
    monkeypatch.setattr(qrunner.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(qreport, "pid_alive", lambda pid: True)
    d = _queue(root, "here", status="running", pid=42)
    runner_cmd[0] = "python -m some.other.thing"
    assert qrunner.main(["stop", str(d)]) == 1 and sent == []
    assert "FAIL" in capsys.readouterr().out
    runner_cmd[0] = f"python -m vqlab.cli queue run --resume {d}"
    assert qrunner.main(["stop", str(d)]) == 0 and sent == [(42, signal.SIGTERM)]
    assert "--resume" in capsys.readouterr().out


def test_resume_only_a_halted_queue_of_this_machine(lab, monkeypatch):
    root, calls, runner_cmd = lab
    _queue(root, "done", status="passed")
    _queue(root, "theirs", status="stopped", host="elsewhere")
    with pytest.raises(UserError):
        jobs.resume({"id": "done"})
    with pytest.raises(UserError, match="ran on elsewhere"):
        jobs.resume({"id": "theirs"})
    assert calls == []
    _queue(root, "fg", status="running", pid=42)
    monkeypatch.setattr(qreport, "pid_alive", lambda pid: pid == 42)
    runner_cmd[0] = "python -m vqlab.cli queue run night.json"      # a foreground runner owns it
    with pytest.raises(UserError):
        jobs.resume({"id": "fg"})
    d = _queue(root, "lost", status="running", pid=None)     # runner died
    assert jobs.resume({"id": "lost"})["ok"] is True
    assert _run_args(calls[0]) == ["--resume", str(d), "--detach", "--json"]


def test_a_host_recorded_in_another_case_is_this_machine(lab):
    root, calls, _ = lab
    _queue(root, "cased", status="stopped", host=HERE.swapcase())
    assert jobs.resume({"id": "cased"})["ok"] is True


def test_routes_are_wired():
    assert gui.ROUTES["/api/jobs/files"] is gui.api_jobs_files
    assert {gui.POSTS[f"/api/jobs/{k}"] for k in ("plan", "start", "stop", "resume")} == \
        {jobs.plan, jobs.start, jobs.stop, jobs.resume}


def test_a_preflight_without_a_commit_is_refused_plainly(lab):
    root, _, _ = lab
    _queue(root, "p-preflight", preflight=True, status="passed", commit="")
    with pytest.raises(UserError, match="no recorded commit"):
        jobs.plan({"id": "p-preflight"})


def test_tokens_expire(lab, monkeypatch):
    _, calls, _ = lab
    tok = jobs.plan({"file": "night.json"})["token"]
    later = jobs.time.time() + jobs.TOKEN_TTL + 1
    monkeypatch.setattr(jobs.time, "time", lambda: later)
    with pytest.raises(UserError):
        jobs.start({"file": "night.json", "token": tok})
    assert calls == []


def test_confirm_tokens_are_single_use_and_bound():
    t = jobs.ConfirmTokens(ttl=60)
    a = t.issue("digest-a")
    assert t.spend(a, "digest-b") is False                # wrong digest: spent anyway
    assert t.spend(a, "digest-a") is False
    b = t.issue("digest-a")
    assert t.spend(b, "digest-a") is True and t.spend(b, "digest-a") is False


def test_preview_compiles_without_writing(lab, monkeypatch):
    q = {"name": "r", "estimate": {"artifact_gib": 3.25, "scratch_gib": 7.5}, "steps": [STEP]}
    monkeypatch.setattr(jobs.recipe_queue, "compile_recipe", lambda r: (q, ["a note"]))
    p = jobs.preview({"recipe": {"name": "r"}})
    assert p == {"ok": True, "steps": [{"name": "s1", "cmd": "smoke", "gpu": p["steps"][0]["gpu"],
                                        "eta_s": None, "writes": 0}],
                 "eta_s": None, "writes": int(7.5 * 2**30), "artifact_gib": 3.25, "warnings": ["a note"]}
    assert list(config.queue_files_dir().glob("r.json")) == []


def test_preview_refuses_a_bad_recipe(lab):
    with pytest.raises(jobs.recipe_queue.RecipeError):
        jobs.preview({"recipe": {"name": "Bad Name"}})


def test_write_recipe_never_overwrites(lab, monkeypatch):
    q = {"name": "r", "estimate": {"artifact_gib": 1.0, "scratch_gib": 1.0}, "steps": [STEP]}
    monkeypatch.setattr(jobs.recipe_queue, "compile_recipe", lambda r: (q, []))
    assert jobs.write_recipe({"recipe": {}}) == {"ok": True, "file": "r.json", "warnings": []}
    with pytest.raises(UserError, match="already exists"):
        jobs.write_recipe({"recipe": {}})


def test_preview_over_http(lab, monkeypatch):
    """One real round trip through gui.Handler: the Host check, a refused
    request as 400 {ok: false, error}, a good one as 200."""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def post(host, body):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("POST", "/api/jobs/preview", body=body,
                  headers={"Host": host, "Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")
    try:
        assert post(f"evil.example:{port}", "{}")[0] == 403
        code, body = post(f"127.0.0.1:{port}", json.dumps({"recipe": {"name": "Bad Name"}}))
        assert code == 400 and body["ok"] is False and body["error"].startswith("name:")
        assert post(f"127.0.0.1:{port}", "{not json")[0] == 400
        def bug(r):
            raise ValueError("a server bug, not a refusal")
        monkeypatch.setattr(jobs.recipe_queue, "compile_recipe", bug)
        assert post(f"127.0.0.1:{port}", json.dumps({"recipe": {}}))[0] == 500
        q = {"name": "r", "estimate": {"artifact_gib": 1.0, "scratch_gib": 1.0}, "steps": [STEP]}
        monkeypatch.setattr(jobs.recipe_queue, "compile_recipe", lambda r: (q, []))
        code, body = post(f"127.0.0.1:{port}", json.dumps({"recipe": {"name": "r"}}))
        assert code == 200 and body["ok"] is True and body["steps"][0]["name"] == "s1"
    finally:
        srv.shutdown()


def test_detach_json_is_one_parseable_line(tmp_path, monkeypatch, capsys):
    """`queue run --detach --json` ends with {"qdir", "pid"}: the line the
    Jobs page reads, in place of regexing the runner's prose."""
    import argparse
    from vqlab.queue import runner
    monkeypatch.setattr(runner, "cli_launch", lambda: (["vqlab"], None, None))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: argparse.Namespace(pid=4242))
    a = argparse.Namespace(lease_wait=1, force=False, override_reservation=False,
                           keep_preflight=False, json=True)
    (tmp_path / "state.json").write_text(json.dumps({"status": "created", "steps": []}))
    assert runner._detach(tmp_path, a) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line) == {"qdir": str(tmp_path), "pid": 4242}
    with pytest.raises(SystemExit):
        runner.main(["run", "x.json", "--json"])
