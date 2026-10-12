"""/api/queues: shape of queue runs read from VQLAB_QUEUE_DIR."""
import json

from vqlab.agents import gui


def test_api_queues_shape(tmp_path, monkeypatch):
    qd = tmp_path / "20260101-000000-demo"
    qd.mkdir()
    (qd / "state.json").write_text(json.dumps({
        "schema": "vqlab.queue/1", "name": "demo", "status": "passed", "preflight": False,
        "created": "2026-01-01T00:00:00+00:00", "host": "h", "commit": "abc123",
        "source": "/some/where/q.json",
        "steps": [{"name": "s1", "status": "pass", "seconds": 1.5, "attempts": 1, "rc": 0,
                   "reasons": [], "warnings": []}]}))
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path))
    out = gui.api_queues({"n": ["5"]})
    assert [r["id"] for r in out] == ["20260101-000000-demo"]
    r = out[0]
    assert r["name"] == "demo" and r["status"] == "passed" and r["live"] is False
    assert r["preflight"] is False and r["commit"] == "abc123"
    assert r["steps"][0]["name"] == "s1" and r["steps"][0]["seconds"] == 1.5


def test_eta_only_while_a_runner_owns_the_queue(lab_dirs, monkeypatch):
    from vqlab import config
    from vqlab.queue import report
    qd = lab_dirs.queues / "20260101-000000-live"
    qd.mkdir()
    (qd / "queue.json").write_text(json.dumps({"name": "live", "steps": [{"name": "s1"}]}))
    (qd / "state.json").write_text(json.dumps({
        "name": "live", "status": "running", "host": config.short_host(), "pid": 42,
        "steps": [{"name": "s1", "status": "running"}]}))
    monkeypatch.setattr(report, "pid_alive", lambda pid: pid == 42)
    monkeypatch.setattr(report, "pid_cmdline", lambda pid: "python -m pytest")     # reused pid
    [r] = gui.queues_snapshot()
    assert r["live"] is False and r["eta"] is None
    monkeypatch.setattr(report, "pid_cmdline", lambda pid: "python -m vqlab.cli queue run q.json")
    [r] = gui.queues_snapshot()
    assert r["live"] is True and set(r["eta"]) >= {"eta_s", "steps"}
