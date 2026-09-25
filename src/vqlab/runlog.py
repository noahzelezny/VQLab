"""Run log: one JSONL line when any `vqlab <cmd>` starts, one when it ends.

Every invocation, on every machine that installs vqlab, is recorded with its
full argv, the code it ran (commit + dirty files) and the library versions --
so "which parameters did that fit use?" is a grep, not an archaeology dig
through shell history and EXPERIMENTS.md. The START line is written before
the tool runs, so a run that is killed (watchdog, OOM, closed laptop) still
leaves a record; an unmatched start IS the record that it died.

    ~/.vqlab/runs.jsonl            (override the dir with VQLAB_LOG_DIR)

Cost is two appends and two `git` calls per run. Logging must never break or
slow a run: every failure here is swallowed. Tools that write a build record
(provenance.py) pick up VQLAB_RUN_ID, so an artifact points at its log line.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import time
import uuid

_T0 = {}


def _path() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("VQLAB_LOG_DIR", pathlib.Path.home() / ".vqlab"))
    d.mkdir(parents=True, exist_ok=True)
    return d / "runs.jsonl"


def _append(rec) -> None:
    line = json.dumps(rec, default=str) + "\n"
    # one write() on an O_APPEND fd: concurrent sessions do not interleave
    fd = os.open(_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def start(cmd, argv):
    run_id = os.environ.get("VQLAB_RUN_ID") or uuid.uuid4().hex[:16]
    os.environ["VQLAB_RUN_ID"] = run_id          # children + build records
    _T0[run_id] = time.time()
    try:
        import provenance
        rec = {"event": "start", "run_id": run_id, "time": _now(), "cmd": cmd,
               "argv": list(argv), "cwd": os.getcwd(), "pid": os.getpid(),
               "code": provenance.code_state(), "env": provenance.env_state()}
        rec["code"].pop("repo", None)
        _append(rec)
    except Exception:
        pass
    return run_id


def end(run_id, rc):
    try:
        _append({"event": "end", "run_id": run_id, "time": _now(), "rc": rc,
                 "seconds": round(time.time() - _T0.pop(run_id, time.time()), 1)})
    except Exception:
        pass
