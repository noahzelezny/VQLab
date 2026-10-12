"""`vqlab queue run` end to end, in a subprocess, as an operator or the GUI
runs it: a preflight, status, wait, a resume and the full run, on one
GPU-free step. A broken import or attribute anywhere on the CLI path fails
here (2026-10-12: a renamed helper crashed every `queue run`)."""
import json
import pathlib
import subprocess

import pytest

from vqlab.queue import files as qf

pytestmark = pytest.mark.skipif(qf.installed(), reason="pins a git worktree of the checkout")


def _vqlab(env, *args):
    pre, _, cwd = qf.cli_launch()
    return subprocess.run([*pre, *args], env=env, cwd=cwd, capture_output=True, text=True,
                          timeout=300)


def test_queue_run_preflight_status_wait_resume_full(lab_dirs, tmp_path):
    _, env, _ = qf.cli_launch()
    env.update({"VQLAB_RECORDS": str(lab_dirs.records), "VQLAB_QUEUE_DIR": str(lab_dirs.queues),
                "VQLAB_SCRATCH": str(lab_dirs.scratch), "VQLAB_GPU_LEASE": str(tmp_path / "lease"),
                "VQLAB_RESERVATIONS": str(tmp_path / "reservations.json"),
                "VQLAB_CONFIG": str(tmp_path / "none.toml"), "VQLAB_NO_SETUP": "1"})
    env.pop("VQLAB_QUEUE", None)
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps({"name": "e2e", "steps": [
        {"name": "help", "cmd": "recipe", "args": ["--help"], "gpu": False,
         "preflight": {"args": ["compile", "--help"]}, "timeout_s": 120}]}))
    trees = []
    try:
        r = _vqlab(env, "queue", "run", str(qfile), "--preflight", "--allow-dirty")
        assert r.returncode == 0, r.stdout + r.stderr
        [pf] = [p for p in lab_dirs.queues.iterdir() if (p / "state.json").exists()]
        s = json.loads((pf / "state.json").read_text())
        trees.append(s["tree"])
        assert s["status"] == "passed" and s["preflight"] is True
        assert [x["status"] for x in s["steps"]] == ["pass"]
        r = _vqlab(env, "queue", "status", str(pf))
        assert r.returncode == 0 and "passed" in r.stdout, r.stdout + r.stderr
        r = _vqlab(env, "queue", "wait", str(pf), "--poll", "0.1", "--timeout", "30")
        assert r.returncode == 0, r.stdout + r.stderr
        r = _vqlab(env, "queue", "run", "--resume", str(pf))          # nothing left: still passed
        assert r.returncode == 0, r.stdout + r.stderr
        r = _vqlab(env, "queue", "run", str(pf / "queue.json"), "--commit", s["commit"])
        assert r.returncode == 0, r.stdout + r.stderr
        [full] = [p for p in lab_dirs.queues.iterdir() if p != pf and (p / "state.json").exists()]
        f = json.loads((full / "state.json").read_text())
        trees.append(f["tree"])
        assert f["status"] == "passed" and f.get("tree_removed") is True
    finally:
        for t in trees:
            if pathlib.Path(t).exists():
                subprocess.run(["git", "worktree", "remove", "--force", t], cwd=qf.REPO,
                               capture_output=True)
