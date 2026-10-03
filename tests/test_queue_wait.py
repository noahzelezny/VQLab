"""queue wait / MCP queue_status (CPU): terminal states come from state.json,
a running queue whose runner pid is gone counts as died, exit codes."""
import json
import os
import sys

from vqlab import _layout  # noqa: F401
sys.path.insert(0, str(_layout.SRC / "vqlab" / "agents"))
import run_queue as rq  # noqa: E402
import mcp_server as ms  # noqa: E402


def _q(tmp, name, status, pid=None):
    d = tmp / name
    d.mkdir()
    (d / "state.json").write_text(json.dumps({"status": status, "pid": pid,
                                             "steps": [{"name": "s", "status": "pass"}]}))
    return d


def test_wait_exit_codes(tmp_path):
    a = _q(tmp_path, "a", "passed")
    b = _q(tmp_path, "b", "failed")
    assert rq.wait([a], poll=0.01) == 0
    assert rq.wait([a, b], poll=0.01) == 4


def test_dead_runner_is_terminal(tmp_path):
    d = _q(tmp_path, "c", "running", pid=2**22 + 12345)     # no such pid
    assert rq.wait([d], poll=0.01) == 4


def test_running_times_out(tmp_path):
    d = _q(tmp_path, "d", "running", pid=os.getpid())
    assert rq.wait([d], timeout=0.05, poll=0.01) == 2


def test_mcp_queue_status(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path))
    _q(tmp_path, "20261002-a", "running", pid=2**22 + 12345)
    r = ms.t_queue_status()
    assert r["queues"][0]["status"] == "died" and r["queues"][0]["terminal"]
