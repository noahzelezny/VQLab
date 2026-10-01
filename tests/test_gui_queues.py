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
