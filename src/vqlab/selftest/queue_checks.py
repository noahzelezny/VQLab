"""Selftest: the job queue (vqlab.queue): pinned tree, loud failure,
preflight."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess

from vqlab import _layout as L
from vqlab.queue import files as QF
from vqlab.queue import runner as QR
from vqlab.queue import steps as QS
from vqlab.selftest import check

_ENV = ("VQLAB_QUEUE_DIR", "VQLAB_GPU_LEASE", "VQLAB_PREFLIGHT_DIR")


def check_queue(ctx):
    print("[6a4/7] queue: pinned tree, loud failure, preflight")
    tmp = ctx.tmp
    _saved = {k: os.environ.get(k) for k in _ENV}
    os.environ["VQLAB_QUEUE_DIR"] = str(tmp / "queues")
    os.environ["VQLAB_PREFLIGHT_DIR"] = str(tmp / "pf")
    os.environ["VQLAB_GPU_LEASE"] = str(tmp / "gpu.lease")
    qdirs = []
    try:
        _check_run_and_pin(tmp, qdirs)
        _check_preflight(tmp, qdirs)
    finally:
        for qd in qdirs:
            subprocess.run(["git", "worktree", "remove", "--force", str(qd / "tree")],
                           cwd=str(L.SRC.parent), capture_output=True)
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _check_run_and_pin(tmp, qdirs):
    qf = tmp / "q.json"
    qf.write_text(json.dumps({"name": "st", "steps": [
        {"name": "ok", "cmd": "runs", "args": ["-n", "1"], "preflight": {"args": ["-n", "2"]}},
        {"name": "gone", "cmd": "fits", "args": ["list", "--root", str(tmp / "no" / "such")]},
        {"name": "after", "cmd": "runs", "args": ["-n", "1"]}]}))
    with contextlib.redirect_stdout(io.StringIO()):
        qd = QF.create(qf, allow_dirty=True)
        qdirs.append(qd)
        qrc = QR.run(qd)
    qs = json.load(open(qd / "state.json"))
    if QF.installed():
        check("queue: an installed package is pinned by version + file hash",
              qs.get("installed") and qs["commit"].startswith("installed:")
              and qs["commit"].endswith(":" + QF.package_hash()))
    else:
        check("queue: steps run from a worktree pinned at the queue's commit",
              QF.git("rev-parse", "HEAD", cwd=qd / "tree") == qs["commit"]
              and str(qd / "tree") in (qd / "steps" / "00-ok" / "cmd").read_text() + qs["tree"])
    check("queue: a missing input path FAILS the step and stops the queue",
          qrc == 4 and [r["status"] for r in qs["steps"]] == ["pass", "fail", "pending"]
          and "does not exist" in qs["steps"][1]["reasons"][0])
    with contextlib.redirect_stdout(io.StringIO()):
        pd = QF.create(qf, allow_dirty=True, preflight=True)
        qdirs.append(pd)
        QR.run(pd)
    ps_ = json.load(open(pd / "state.json"))
    check("queue --preflight: reports every step; no preflight block is a FAIL",
          [r["status"] for r in ps_["steps"]] == ["pass", "fail", "fail"]
          and "no preflight defined" in ps_["steps"][2]["reasons"][0])


def _check_preflight(tmp, qdirs):
    om = {}
    r1 = QS.redirect_outs(["--teacher", "/T", "--out", "/V/fit"], tmp / "pf" / "a", om)
    r2 = QS.redirect_outs(["--rung", "r1=/V/fit", "/V/fit/x", "--out=/V/pin"],
                          tmp / "pf" / "b", om)
    check("queue --preflight: outputs redirected off the real paths; chained inputs follow",
          r1[3] == str(tmp / "pf" / "a" / "fit")
          and r2[:3] == ["--rung", f"r1={tmp}/pf/a/fit", f"{tmp}/pf/a/fit/x"]
          and r2[3] == f"--out={tmp}/pf/b/pin")
    check("queue: name=path args are checked as paths",
          QS.path_args(["--cache", "prose=/V/c", "--out", "/o"]) == (["/V/c"], ["/o"]))
    check("queue: a preflight identical to the real step is refused (full-size in scratch)",
          any("identical" in e for e in QS.validate({"steps": [
              {"name": "c", "cmd": "stream-convert", "args": ["x"],
               "preflight": {"append": []}}]})))
    qf2 = tmp / "q2.json"
    qf2.write_text(json.dumps({"name": "st2", "steps": [
        {"name": "ok", "cmd": "runs", "args": ["-n", "1"],
         "preflight": {"args": ["-n", "2"]}}]}))
    with contextlib.redirect_stdout(io.StringIO()):
        pd2 = QF.create(qf2, allow_dirty=True, preflight=True)
        qdirs.append(pd2)
        QR.run(pd2)
    check("queue: a PASSED preflight leaves nothing on scratch",
          not (tmp / "pf" / pd2.name).exists()
          and json.load(open(pd2 / "state.json"))["preflight_outputs"] == "removed after pass")
    check("queue: publish can never be queued",
          any("publish" in e for e in QS.validate({"steps": [{"name": "p", "cmd": "publish"}]})))
