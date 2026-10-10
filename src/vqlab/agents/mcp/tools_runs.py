"""Launching and watching jobs on THIS box: run (allowlisted, detached, under
the GPU lease), status, stop, list_runs, queue_status, gpu_state, and the
supervisor process `run` detaches. The allowlist and the GPU-free set come
from the command registry (vqlab/commands.py); publish is never runnable."""

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

from vqlab import config
from vqlab import commands as _commands
from vqlab.agents.mcp.common import REPO, ToolError, _read_meta, _under, lab_roots
from vqlab.core.exo import exo_instances as _exo_instances
from vqlab.core.gpu_lease import acquire_lease, release_lease
from vqlab.core.gpu_lease import lease_holder as _lease_holder
from vqlab.core.gpu_lease import lease_path
from vqlab.core.gpu_lease import pid_alive as _pid_alive
from vqlab.queue import eta as _queue_eta
from vqlab.queue import reserve as _resv


def runs_dir() -> Path:
    env = os.environ.get("VQLAB_RUNS_DIR")
    if env:
        return Path(env)
    return config.scratch() / "mcp-runs" / socket.gethostname().split(".")[0]


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

def t_run(cmd: str, args: Optional[List[str]] = None, tag: str = "",
          retries: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
    """Launch an allowlisted vqlab subcommand, detached, under the GPU lease."""
    args = list(args or [])
    if cmd not in RUN_ALLOWLIST:
        raise ToolError("NOT_ALLOWED", f"{cmd!r} is not launchable from the MCP",
                        allowlist=sorted(RUN_ALLOWLIST),
                        hint="publish is a human action; other commands run via the CLI")
    roots = lab_roots()
    for a in args:
        if a.startswith("/") and not _under(Path(a), roots) and not _under(Path(a), [REPO]):
            raise ToolError("OUTSIDE_ROOTS", f"path argument {a} is outside the lab roots",
                            roots=[str(r) for r in roots],
                            hint="artifacts never go on the internal disk")
    if cmd not in ("selftest", *GPU_FREE) and not force:
        placed = _exo_instances()
        if placed:
            raise ToolError("EXO_PLACED", "an exo instance is placed on this cluster; "
                            "it pins GPU memory on this box", instances=placed,
                            hint="DELETE /instance/<id> on the exo API, kill orphaned "
                                 "multiprocessing.spawn runners, then retry (or force=true)")
    if cmd in HEAVY and not force:
        st = _lab_state()
        if st.get("manager_hostname") and st["manager_hostname"] == _this_host():
            raise ToolError("LAB_MANAGER_HERE",
                            f"this box ({_this_host()}) hosts the manager model "
                            f"({st.get('manager_service')}); heavy jobs run on {st.get('fit_host')}",
                            lab_state=st,
                            hint=f"run it through {st.get('fit_host')}'s vqlab server, "
                                 "or move the manager model off this box")
    holder = None if cmd in GPU_FREE else _lease_holder()
    if holder:
        raise ToolError("GPU_BUSY", "the GPU lease on this box is held", holder=holder,
                        lease=str(lease_path()), hint="deferring is normal; poll and retry")
    # Second-granularity ids collide when two launches land in the same second
    # (two agents, or one retrying): make the id unique rather than crashing.
    stem = time.strftime("%Y%m%d-%H%M%S") + "-" + cmd + (("-" + re.sub(r"[^\w.-]", "_", tag)[:24]) if tag else "")
    base = runs_dir()
    for suffix in ("", *(f"-{i}" for i in range(2, 100))):
        run_id = stem + suffix
        rd = base / run_id
        try:
            rd.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise ToolError("BUSY", "100 runs in one second on this box; something is looping")
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
        [sys.executable, "-m", "vqlab.agents.mcp.server", "--supervise", str(rd)],
        cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
        stdout=open(rd / "supervisor.log", "ab"), stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group: survives the MCP client
    )
    meta.update(supervisor_pid=sup.pid, status="running")
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"run_id": run_id, "run_dir": str(rd), "supervisor_pid": sup.pid,
            "command": " ".join(shlex.quote(x) for x in ["vqlab", cmd, *args]),
            "note": "Detached. This returns immediately; poll status(run_id). "
                    "Long jobs take hours."}


def t_status(run_id: str, tail: int = 40) -> Dict[str, Any]:
    rd = runs_dir() / run_id
    if not rd.is_dir():
        raise ToolError("NOT_FOUND", f"no run {run_id} on this box", runs_dir=str(runs_dir()))
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


def t_stop(run_id: str) -> Dict[str, Any]:
    rd = runs_dir() / run_id
    meta = _read_meta(rd)
    pid = meta.get("supervisor_pid")
    if not pid:
        raise ToolError("NOT_FOUND", f"no supervisor pid recorded for {run_id}")
    if not _pid_alive(pid):
        return {"run_id": run_id, "stopped": False, "note": "already finished", "status": meta.get("status")}
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as exc:
        raise ToolError("KILL_FAILED", str(exc))
    meta.update(status="stopped", stopped_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"run_id": run_id, "stopped": True}


def t_list_runs(n: int = 20) -> Dict[str, Any]:
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


def _queues_dir() -> Path:
    # same resolution as vqlab.queue.files.queues_dir
    return Path(os.environ.get("VQLAB_QUEUE_DIR") or Path.home() / ".vqlab" / "queues")


def t_queue_status(name: Optional[str] = None, n: int = 5) -> Dict[str, Any]:
    """Queue state from state.json (what `vqlab queue wait` blocks on): never
    from log wording. A 'running' queue whose runner pid is gone is 'died'."""
    qs = sorted(p for p in _queues_dir().glob("*") if (p / "state.json").exists())
    if name:
        qs = [p for p in qs if name in p.name]
    out = []
    for q in qs[-max(1, int(n)):]:
        s = json.loads((q / "state.json").read_text())
        st = s["status"]
        here = socket.gethostname().split(".")[0]
        if st == "running" and s.get("host", here) == here and not _pid_alive(s.get("pid")):
            st = "died"
        live = st in ("running", "created")
        try:
            eta = _queue_eta.queue_eta(q, _queues_dir())
        except Exception as e:  # an ETA must never hide the state
            eta = {"steps": [{"eta_s": None, "basis": f"eta failed: {e}"}] * len(s["steps"]),
                   "eta_s": None, "finish": None}
        out.append({"queue": str(q), "status": st, "terminal": st in (
            "passed", "failed", "stopped", "deferred", "died"),
            "eta_s": eta["eta_s"] if live else None,
            "eta": _queue_eta.fmt(eta["eta_s"]) if live else None,
            "finish": eta["finish"] if live else None,
            "reservation_override": s.get("reservation_override"),
            "steps": [{"name": r["name"], "status": r["status"], "seconds": r.get("seconds"),
                       "reasons": r.get("reasons", []),
                       **({"eta_s": e["eta_s"], "eta_basis": e["basis"]}
                          if r["status"] in ("running", "pending") else {})}
                      for r, e in zip(s["steps"], eta["steps"])]})
    return {"queues_dir": str(_queues_dir()), "queues": out}

def _box_state(name: str, b: Dict[str, Any]) -> Dict[str, Any]:
    """Another box's lease holder and GPU-heavy processes, over ssh, from its
    own clone and config (the [boxes.NAME] table). Never raises."""
    remote = (f"cd {shlex.quote(b['repo'])} && VQLAB_CONFIG={shlex.quote(b['config'])} "
              f"PYTHONPATH=src {shlex.quote(b['python'])} -c "
              + shlex.quote("import json; from vqlab.agents import mcp_server as m; "
                            "from vqlab.bench import box_quiet as q; s=q.state(); "
                            "print(json.dumps({'lease_holder': m._lease_holder(), 'load1': s['load1'], "
                            "'heavy': s['heavy'][:5]}))"))
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", b["ssh"], remote],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else \
            {"error": (r.stderr or r.stdout).strip()[-300:]}
    except Exception as e:
        return {"error": str(e)[:300]}


def t_gpu_state(boxes: bool = True) -> Dict[str, Any]:
    out = {"host": socket.gethostname().split(".")[0], "lease": str(lease_path()),
           "lease_holder": _lease_holder(), "exo_instances": _exo_instances(),
           "lab_state": _lab_state(), "this_host": _this_host()}
    try:   # every box's reservation (`vqlab reserve`), from the shared file
        out["reservations"] = [dict(r, text=_resv.describe(r)) for r in _resv.active()]
        out["reservations_file"] = str(_resv.shared_path())
    except Exception as e:
        out["reservations"] = {"error": str(e)[:300]}
    if boxes and config.boxes():
        out["boxes"] = {n: _box_state(n, b) for n, b in config.boxes().items()}
    return out

# --------------------------------------------------------------------------- #
# Supervisor (the detached process that owns the lease and runs the job)
# --------------------------------------------------------------------------- #

def supervise(run_dir: Path) -> int:
    meta = _read_meta(run_dir)
    # A GPU-free job locks a private file: same supervisor code path, no
    # contention with the shared GPU lease.
    lp = run_dir / "no-gpu.lock" if meta.get("cmd") in GPU_FREE else lease_path()
    fd = acquire_lease(lp, f"vqlab:{meta.get('cmd')}", meta.get("run_id"), wait_s=1800)
    if fd is None:
        meta.update(status="deferred", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    note="GPU lease held for 30 min; deferred, not failed")
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return 0

    argv = [sys.executable, "-m", "vqlab.cli", meta["cmd"], *meta.get("args", [])]
    attempts, rc = 0, 1
    stopping = {"flag": False}

    def _term(_s, _f):
        stopping["flag"] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        with open(run_dir / "job.log", "ab") as log:
            while attempts <= int(meta.get("retries", 0)):
                attempts += 1
                log.write(f"\n=== attempt {attempts} {time.strftime('%Y-%m-%dT%H:%M:%S')}: {' '.join(shlex.quote(a) for a in argv)}\n".encode())
                log.flush()
                meta.update(status="running", attempt=attempts, started=meta.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"))
                (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
                try:
                    proc = subprocess.Popen(argv, cwd=meta.get("repo", str(REPO)), stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT)
                    meta["job_pid"] = proc.pid
                    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
                    rc = proc.wait()
                except KeyboardInterrupt:
                    try:
                        proc.terminate()
                        proc.wait(30)
                    except Exception:
                        pass
                    rc = -15
                    break
                if rc == 0:
                    break
                log.write(f"=== exit {rc}\n".encode())
                log.flush()
                if attempts <= int(meta.get("retries", 0)):
                    time.sleep(60)
    finally:
        status = "stopped" if stopping["flag"] else ("completed" if rc == 0 else "failed")
        meta.update(status=status, exit_code=rc, attempts=attempts,
                    finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        release_lease(fd)
    return 0 if rc == 0 else 1

