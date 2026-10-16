"""queue runner (CPU): a resumed queue's state is the new runner's, not the
last attempt's -- for `queue wait` and for `queue status`."""
import argparse
import json
import os
import sys

from vqlab import config
from vqlab.queue import report as rq
from vqlab.queue import runner


def _died_queue(tmp):
    d = tmp / "q"
    d.mkdir()
    state = {"status": "running", "pid": 2**22 + 12345,     # no such pid
             "steps": [{"name": "s", "status": "pending"}]}
    (d / "state.json").write_text(json.dumps(state))
    return d


def test_detach_claims_a_resumed_queue(tmp_path, monkeypatch):
    """`run --resume --detach` then `wait` at once: the old runner's dead pid
    must not read as `died` before the new runner writes its state."""
    d = _died_queue(tmp_path)
    monkeypatch.setattr(runner, "cli_launch",
                        lambda: ([sys.executable], dict(os.environ), str(tmp_path)))
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda *a, **k: argparse.Namespace(pid=os.getpid()))
    monkeypatch.setattr(rq, "pid_cmdline",
                        lambda pid: f"python -m vqlab.cli queue run --resume {d}")
    a = argparse.Namespace(lease_wait=1, force=False, override_reservation=False,
                           keep_preflight=False, json=False)
    runner._detach(d, a)
    assert json.loads((d / "state.json").read_text())["host"] == config.short_host()
    assert rq.wait([d], timeout=0.05, poll=0.01) == 2      # running, not died


def test_a_rerun_step_drops_the_last_attempts_verdict(tmp_path, monkeypatch):
    rec = {"name": "s", "status": "stopped", "rc": -15, "seconds": 588.3, "attempts": 2,
           "finished": "then", "reasons": ["exit code -15"], "warnings": []}
    state = {"tree": str(tmp_path), "qdir": str(tmp_path), "steps": [rec]}
    seen = {}

    def spawn(*a):
        seen.update(rec)
        return 0
    monkeypatch.setattr(runner, "_spawn", spawn)
    monkeypatch.setattr(runner._steps, "env", lambda state: {})
    runner._attempts({}, rec, tmp_path, state, ["true"], {}, 0)
    assert seen["status"] == "running"
    assert not {"rc", "seconds", "attempts", "finished", "reasons", "warnings"} & seen.keys()
