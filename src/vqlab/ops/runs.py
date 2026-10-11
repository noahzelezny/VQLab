"""Launching and watching jobs on THIS box: launch (allowlisted, detached,
under the GPU lease), status, stop, list_runs. The allowlist and the GPU-free
set come from the command registry (vqlab/commands.py); publish is never
launchable. The detached process is vqlab.ops.supervisor."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vqlab import commands as _commands
from vqlab import config
from vqlab.core.box.exo import exo_instances as _exo_instances
from vqlab.core.box.gpu_lease import lease_holder as _lease_holder
from vqlab.core.box.gpu_lease import lease_path
from vqlab.core.box.gpu_lease import pid_alive as _pid_alive
from vqlab.ops.errors import LabError
from vqlab.ops.roots import REPO, _under, lab_roots

# Subcommands an agent may launch, and the subset that never touches the GPU
# (header reads, index queries, records, HTTP clients of a server holding its
# own memory): not gated on an exo placement or the GPU lease, and they do
# not take the lease. Both come from the command registry (vqlab/commands.py,
# the `mcp` and `gpu` fields). publish is deliberately never mcp=True.
RUN_ALLOWLIST = _commands.mcp_allowlist()
GPU_FREE = _commands.mcp_gpu_free()
# Commands that checkpoint and resume: a GPU-timeout crash is retried.
RESUMABLE = _commands.RESUMABLE
DEFAULT_RETRIES = _commands.DEFAULT_RETRIES

# Commands that need the box's memory to themselves. When an external
# scheduler records in lab-state.json (scratch) that this box hosts a
# long-lived manager model, these are refused here and belong on `fit_host`.
HEAVY = {"fit-moe", "fit-dense", "geo-build", "alloc-sweep", "layer-leverage",
         "score", "kl", "validate", "smoke", "verify",
         "probe-init", "kl-ladder", "tasks", "decode-timeline"}


def runs_dir() -> Path:
    env = os.environ.get("VQLAB_RUNS_DIR")
    if env:
        return Path(env)
    return config.scratch() / "mcp-runs" / socket.gethostname().split(".")[0]


def lab_state_path() -> Path:
    env = os.environ.get("VQLAB_LAB_STATE")
    return Path(env) if env else config.scratch() / "lab-state.json"


def _lab_state() -> Dict[str, Any]:
    try:
        return json.loads(lab_state_path().read_text())
    except Exception:  # no record ⇒ no rule in force
        return {}


def _this_host() -> str:
    return os.environ.get("VQLAB_HOSTNAME") or socket.gethostname().split(".")[0]


def _read_meta(run_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((run_dir / "meta.json").read_text())
    except Exception:
        return {}


def _refuse(cmd: str, args: List[str], force: bool) -> None:
    """Raise LabError if `vqlab <cmd> <args>` may not start on this box now."""
    if cmd not in RUN_ALLOWLIST:
        raise LabError("NOT_ALLOWED", f"{cmd!r} is not launchable from the MCP",
                       allowlist=sorted(RUN_ALLOWLIST),
                       hint="publish is a human action; other commands run via the CLI")
    roots = lab_roots()
    for a in args:
        if a.startswith("/") and not _under(Path(a), roots) and not _under(Path(a), [REPO]):
            raise LabError("OUTSIDE_ROOTS", f"path argument {a} is outside the lab roots",
                           roots=[str(r) for r in roots],
                           hint="artifacts never go on the internal disk")
    if cmd not in ("selftest", *GPU_FREE) and not force:
        placed = _exo_instances()
        if placed:
            raise LabError("EXO_PLACED", "an exo instance is placed on this cluster; "
                           "it pins GPU memory on this box", instances=placed,
                           hint="DELETE /instance/<id> on the exo API, kill orphaned "
                                "multiprocessing.spawn runners, then retry (or force=true)")
    if cmd in HEAVY and not force:
        st = _lab_state()
        if st.get("manager_hostname") and st["manager_hostname"] == _this_host():
            raise LabError("LAB_MANAGER_HERE",
                           f"this box ({_this_host()}) hosts the manager model "
                           f"({st.get('manager_service')}); heavy jobs run on {st.get('fit_host')}",
                           lab_state=st,
                           hint=f"run it through {st.get('fit_host')}'s vqlab server, "
                                "or move the manager model off this box")
    holder = None if cmd in GPU_FREE else _lease_holder()
    if holder:
        raise LabError("GPU_BUSY", "the GPU lease on this box is held", holder=holder,
                       lease=str(lease_path()), hint="deferring is normal; poll and retry")


def _new_run_dir(cmd: str, tag: str) -> Path:
    # Second-granularity ids collide when two launches land in the same second
    # (two agents, or one retrying): make the id unique rather than crashing.
    stem = time.strftime("%Y%m%d-%H%M%S") + "-" + cmd + (("-" + re.sub(r"[^\w.-]", "_", tag)[:24]) if tag else "")
    base = runs_dir()
    for suffix in ("", *(f"-{i}" for i in range(2, 100))):
        rd = base / (stem + suffix)
        try:
            rd.mkdir(parents=True, exist_ok=False)
            return rd
        except FileExistsError:
            continue
    raise LabError("BUSY", "100 runs in one second on this box; something is looping")


def launch(cmd: str, args: Optional[List[str]] = None, tag: str = "",
           retries: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
    """Launch an allowlisted vqlab subcommand, detached, under the GPU lease.
    Returns the launcher's record: run_id, run_dir, supervisor_pid, command."""
    args = list(args or [])
    _refuse(cmd, args, force)
    rd = _new_run_dir(cmd, tag)
    run_id = rd.name
    n_retries = DEFAULT_RETRIES if retries is None else int(retries)
    if cmd not in RESUMABLE:
        n_retries = 0
    meta = {"run_id": run_id, "cmd": cmd, "args": args, "tag": tag,
            "host": socket.gethostname().split(".")[0], "python": sys.executable,
            "repo": str(REPO), "retries": n_retries, "status": "launching",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    env = dict(os.environ)
    if (REPO / "src" / "vqlab").is_dir():            # a checkout; an installed wheel needs no path
        env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path.home() / ".pycache-vqlab"))
    sup = subprocess.Popen(
        [sys.executable, "-m", "vqlab.ops.supervisor", "--supervise", str(rd)],
        cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
        stdout=open(rd / "supervisor.log", "ab"), stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group: survives the caller
    )
    meta.update(supervisor_pid=sup.pid, status="running")
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"run_id": run_id, "run_dir": str(rd), "supervisor_pid": sup.pid,
            "command": " ".join(shlex.quote(x) for x in ["vqlab", cmd, *args]),
            "note": "Detached. This returns immediately; poll status(run_id). "
                    "Long jobs take hours."}


def status(run_id: str, tail: int = 40) -> Dict[str, Any]:
    rd = runs_dir() / run_id
    if not rd.is_dir():
        raise LabError("NOT_FOUND", f"no run {run_id} on this box", runs_dir=str(runs_dir()))
    meta = _read_meta(rd)
    alive = _pid_alive(meta.get("supervisor_pid"))
    if meta.get("status") == "running" and not alive:
        meta["status"] = "lost"  # supervisor died without writing a terminal status
    log = rd / "job.log"
    lines: List[str] = []
    if log.is_file():
        try:
            data = log.read_bytes()[-64000:]
            lines = data.decode("utf-8", "replace").split("\n")[-max(1, tail):]
        except OSError:
            pass
    return {**meta, "supervisor_alive": alive, "log_tail": "\n".join(lines),
            "result_json": str(rd / "result.json") if (rd / "result.json").is_file() else None}


def stop(run_id: str) -> Dict[str, Any]:
    rd = runs_dir() / run_id
    meta = _read_meta(rd)
    pid = meta.get("supervisor_pid")
    if not pid:
        raise LabError("NOT_FOUND", f"no supervisor pid recorded for {run_id}")
    if not _pid_alive(pid):
        return {"run_id": run_id, "stopped": False, "note": "already finished", "status": meta.get("status")}
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as exc:
        raise LabError("KILL_FAILED", str(exc))
    meta.update(status="stopped", stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"run_id": run_id, "stopped": True}


def list_runs(n: int = 20) -> Dict[str, Any]:
    base = runs_dir()
    if not base.is_dir():
        return {"host": socket.gethostname().split(".")[0], "runs": [], "runs_dir": str(base)}
    dirs = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)[:max(1, n)]
    runs = []
    for d in dirs:
        m = _read_meta(d)
        if m.get("status") == "running" and not _pid_alive(m.get("supervisor_pid")):
            m["status"] = "lost"
        runs.append({k: m.get(k) for k in ("run_id", "cmd", "status", "exit_code", "created", "finished", "tag")})
    return {"host": socket.gethostname().split(".")[0], "runs": runs, "runs_dir": str(base)}
