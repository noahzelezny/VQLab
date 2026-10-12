"""queue wait / MCP queue_status (CPU): terminal states come from state.json,
a running queue whose runner pid is gone counts as died, exit codes."""
import json
import os
import sys

from vqlab import _layout  # noqa: F401
sys.path.insert(0, str(_layout.SRC / "vqlab" / "agents"))
from vqlab.queue import report as rq
from vqlab.agents import mcp_server as ms


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


def test_running_times_out(tmp_path, monkeypatch):
    d = _q(tmp_path, "d", "running", pid=os.getpid())
    monkeypatch.setattr(rq, "pid_cmdline", lambda pid: f"python -m vqlab.cli queue run --resume {d}")
    assert rq.wait([d], timeout=0.05, poll=0.01) == 2


def test_a_reused_pid_is_not_the_runner(tmp_path, monkeypatch):
    d = _q(tmp_path, "e", "running", pid=os.getpid())     # alive, but not a queue runner
    monkeypatch.setattr(rq, "pid_cmdline", lambda pid: "python -m pytest")
    assert rq.wait([d], poll=0.01) == 4
    assert rq.runner_pid(d, json.loads((d / "state.json").read_text()), strict=False) is None


def test_mcp_queue_status(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path))
    _q(tmp_path, "20261002-a", "running", pid=2**22 + 12345)
    r = ms.t_queue_status()
    assert r["queues"][0]["status"] == "died" and r["queues"][0]["terminal"]


def test_remote_queue_not_judged_by_local_pid(tmp_path):
    d = _q(tmp_path, "r", "running", pid=2**22 + 12345)
    s = json.loads((d / "state.json").read_text()); s["host"] = "some-other-box"
    (d / "state.json").write_text(json.dumps(s))
    assert rq.wait([d], timeout=0.05, poll=0.01) == 2      # still running, not died


def test_written_paths_are_outputs():
    """stream-score --kl-per-position and runtime-equiv --keep-dir WRITE their
    path: the queue must not demand it exists. (--save-topk is the top-k COUNT;
    the cache dir is --out.)"""
    from vqlab.queue import steps
    assert "--save-topk" not in steps.OUT_FLAGS
    for f in ("--kl-per-position", "--keep-dir"):
        assert f in steps.OUT_FLAGS


def test_a_failed_ps_never_reads_as_a_dead_runner(tmp_path, monkeypatch):
    d = _q(tmp_path, "f", "running", pid=os.getpid())
    s = json.loads((d / "state.json").read_text())
    monkeypatch.setattr(rq, "pid_cmdline", lambda pid: None)      # ps timed out / failed
    assert rq.live_status(d, s) == "running"
    assert rq.runner_pid(d, s, strict=False) == os.getpid()
    assert rq.runner_pid(d, s, strict=True) is None               # never signal on a guess


def test_strict_match_is_the_whole_queue_dir(tmp_path, monkeypatch):
    d = _q(tmp_path, "x", "running", pid=os.getpid())
    s = json.loads((d / "state.json").read_text())
    monkeypatch.setattr(rq, "pid_cmdline", lambda pid: f"python -m vqlab.cli queue run --resume {d}-preflight")
    assert rq.runner_pid(d, s, strict=True) is None
    monkeypatch.setattr(rq, "pid_cmdline", lambda pid: f"python -m vqlab.cli queue run --resume {d} --force")
    assert rq.runner_pid(d, s, strict=True) == os.getpid()


def test_pid_cmdline_tells_no_process_from_a_failed_ps(monkeypatch):
    from vqlab.core.box import gpu_lease
    assert gpu_lease.pid_cmdline(os.getpid()) != ""
    gone = next(p for p in range(99990, 30000, -1) if not gpu_lease.pid_alive(p))
    assert gpu_lease.pid_cmdline(gone) == ""

    def broken(*a, **k):
        raise OSError("no ps")
    monkeypatch.setattr(gpu_lease.subprocess, "run", broken)
    assert gpu_lease.pid_cmdline(os.getpid()) is None


def test_a_host_reservation_matches_in_any_case(tmp_path, monkeypatch):
    from vqlab import config
    from vqlab.queue import reserve
    monkeypatch.setenv("VQLAB_RESERVATIONS", str(tmp_path / "r.json"))
    monkeypatch.setattr(reserve, "local_path", lambda: tmp_path / "local.json")
    monkeypatch.setattr(config, "this_box", lambda: None)
    monkeypatch.setattr(reserve, "who", lambda: "me")
    reserve.reserve(config.short_host().swapcase(), "someone", "+1h")
    assert reserve.get_by_host()["for"] == "someone"
    assert reserve.refusal("here") is not None
