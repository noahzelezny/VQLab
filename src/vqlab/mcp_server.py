#!/usr/bin/env python3
"""vqlab mcp — the lab's MCP server (stdio JSON-RPC, stdlib only).

Nobody runs the lab by hand any more; agents do. This is the interface they
get. One server per box (it mirrors exo): every tool acts on THIS machine,
and a client that wants the other box talks to the other box's server.

    PYTHONPATH=src python -m vqlab.cli mcp            # serve on stdio
    PYTHONPATH=src python -m vqlab.cli mcp --list     # print the tool table

Design rules (each one paid for):

* ``where_is`` answers deterministically. A model that searches directories
  by hand declared a teacher cache "gone" four times in one night
  (2026-09-15) while it sat in ``Exo Models/``. This tool walks the lab's
  roots and returns the path or NOT_FOUND *with the roots it searched*. It
  cannot editorialize.
* ``run`` launches ALLOWLISTED ``vqlab`` subcommands only, detached (nohup +
  its own process group, so it survives the client), under the box's GPU
  lease (an advisory flock that dies with its holder — see Scout's
  ``gpu_lease``; we lock the SAME file when Scout's cache dir exists), with a
  retry supervisor for the commands that resume from checkpoints. It refuses
  any absolute path outside the lab's roots (artifacts never go on the
  internal disk) and refuses to start while an exo instance is placed.
* ``findings_append`` is the only way the measured record is written from
  here. It allocates the next F-number, requires the pre-registered
  prediction, and takes the verdict from a fixed vocabulary. A falsified
  prediction is recorded as falsified.
* ``publish`` is NOT exposed. Publishing is irreversible and is a human's
  action.

No third-party packages: the exo envs this runs in must not have anything
pip-installed into them.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vqlab"
SERVER_VERSION = "0.1.0"

PKG = Path(__file__).resolve().parent
REPO = PKG.parents[1]

# --------------------------------------------------------------------------- #
# Lab layout
# --------------------------------------------------------------------------- #

DEFAULT_ROOTS = [
    "/Volumes/Thunderbay SSD/vqlab-scratch",
    "/Volumes/Thunderbay SSD/vqlab-dogfood",
    "/Volumes/Thunderbay SSD/Exo Models",
    "/Volumes/Thunderbay HDD/vqlab-fits",
    "/Volumes/Thunderbay HDD/Teacher Models",
]


def lab_roots() -> List[Path]:
    env = os.environ.get("VQLAB_ROOTS")
    raw = env.split(os.pathsep) if env else DEFAULT_ROOTS
    return [Path(r) for r in raw if r]


def runs_dir() -> Path:
    env = os.environ.get("VQLAB_RUNS_DIR")
    if env:
        return Path(env)
    return Path(DEFAULT_ROOTS[0]) / "mcp-runs" / socket.gethostname().split(".")[0]


def lease_path() -> Path:
    """Lock the SAME file Scout's gpu_lease uses when Scout is on this box."""
    env = os.environ.get("VQLAB_GPU_LEASE") or os.environ.get("SCOUT_GPU_LEASE")
    if env:
        return Path(env)
    scout_cache = REPO.parent / "cache"
    if scout_cache.is_dir():
        return scout_cache / "gpu.lease"
    return Path.home() / ".vqlab" / "gpu.lease"


def exo_api() -> str:
    return os.environ.get("VQLAB_EXO_API", "http://127.0.0.1:52415")


# Subcommands an agent may launch. publish is deliberately absent.
RUN_ALLOWLIST = {
    "fit-moe", "fit-dense", "geo-build", "alloc-sweep", "score", "kl",
    "layer-leverage", "validate", "check-release", "check-bundle", "check",
    "smoke", "selftest", "price", "verify", "coverage", "manifest",
}
# Commands that checkpoint and resume: a GPU-timeout crash is retried.
RESUMABLE = {"fit-moe", "fit-dense", "geo-build", "alloc-sweep", "validate"}
DEFAULT_RETRIES = 2

# Commands that need the box's memory to themselves. On the box that hosts
# the manager model (recorded by scout.ops.lab_residency in lab-state.json on
# the shared SSD) these are refused: the manager lives on the box NOT fitting.
HEAVY = {"fit-moe", "fit-dense", "geo-build", "alloc-sweep", "layer-leverage",
         "score", "kl", "validate", "smoke", "verify"}


def lab_state_path() -> Path:
    env = os.environ.get("VQLAB_LAB_STATE")
    return Path(env) if env else Path(DEFAULT_ROOTS[0]) / "lab-state.json"


def _lab_state() -> Dict[str, Any]:
    try:
        return json.loads(lab_state_path().read_text())
    except Exception:  # noqa: BLE001 — no record ⇒ no rule in force
        return {}


def _this_host() -> str:
    return os.environ.get("VQLAB_HOSTNAME") or socket.gethostname().split(".")[0]


DOC_ALLOW = ("docs", "AGENTS.md", "README.md", "METHODOLOGY.md",
             "REPRODUCING.md", "research/quantlab/FINDINGS.md",
             "research/quantlab/EXPERIMENTS.md")
FINDINGS_LOG = REPO / "docs" / "FINDINGS-LOG.md"
VERDICTS = ("CONFIRMED", "FALSIFIED", "VOID", "CORRECTS", "NULL")


class ToolError(Exception):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def as_result(self) -> Dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.extra}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _under(path: Path, roots: List[Path]) -> bool:
    try:
        rp = path.resolve()
    except OSError:
        rp = path
    for r in roots:
        try:
            rp.relative_to(r.resolve() if r.exists() else r)
            return True
        except ValueError:
            continue
    return False


def _dir_bytes(d: Path, suffixes=(".safetensors",)) -> int:
    total = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file(follow_symlinks=False) and e.name.endswith(suffixes):
                    total += e.stat(follow_symlinks=False).st_size
    except OSError:
        pass
    return total


def _walk(roots: List[Path], max_depth: int):
    for root in roots:
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                yield root, e, depth
                if e.is_dir(follow_symlinks=False) and depth + 1 < max_depth \
                        and e.name != "__pycache__":
                    stack.append((Path(e.path), depth + 1))


def _exo_instances() -> Optional[List[str]]:
    """Instance ids currently placed on this cluster, or None if exo is down."""
    try:
        with urllib.request.urlopen(exo_api() + "/state", timeout=3) as r:
            st = json.load(r)
    except Exception:  # noqa: BLE001 — exo down means nothing is placed here
        return None
    inst = st.get("instances") or {}
    return list(inst.keys()) if isinstance(inst, dict) else [str(i) for i in inst]


def _lease_holder() -> Optional[Dict[str, Any]]:
    p = lease_path()
    if not p.exists():
        return None
    fd = os.open(p, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                try:
                    return json.loads(Path(p).read_text() or "{}")
                except Exception:  # noqa: BLE001
                    return {"holder": "unknown"}
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def _read_meta(run_dir: Path) -> Dict[str, Any]:
    try:
        return json.loads((run_dir / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def t_where_is(name: str, kind: str = "any", max_depth: int = 3) -> Dict[str, Any]:
    """Find a thing by (sub)name across the lab roots. Deterministic."""
    if not name or len(name) < 3:
        raise ToolError("INVALID_ARG", "name must be at least 3 characters")
    needle = name.lower()
    roots = lab_roots()
    hits: List[Dict[str, Any]] = []
    for root, e, _ in _walk(roots, max_depth):
        if needle not in e.name.lower():
            continue
        is_dir = e.is_dir(follow_symlinks=False)
        if kind == "dir" and not is_dir or kind == "file" and is_dir:
            continue
        rec: Dict[str, Any] = {"path": e.path, "kind": "dir" if is_dir else "file",
                               "root": str(root)}
        if is_dir:
            p = Path(e.path)
            rec["has_config"] = (p / "config.json").is_file()
            rec["safetensors_gib"] = round(_dir_bytes(p) / 2**30, 3)
        else:
            try:
                rec["bytes"] = e.stat(follow_symlinks=False).st_size
            except OSError:
                pass
        hits.append(rec)
        if len(hits) >= 50:
            break
    hits.sort(key=lambda h: h["path"])
    if not hits:
        return {"found": False, "status": "NOT_FOUND", "name": name,
                "roots_searched": [str(r) for r in roots],
                "roots_missing": [str(r) for r in roots if not r.is_dir()],
                "max_depth": max_depth,
                "note": "Searched every lab root to this depth. If you expected it, "
                        "the name is different, not the location."}
    return {"found": True, "count": len(hits), "matches": hits,
            "roots_searched": [str(r) for r in roots]}


def t_list_artifacts(root: str = "", max_depth: int = 3) -> Dict[str, Any]:
    """Directories with a config.json under the lab roots (or one root)."""
    roots = [Path(root)] if root else lab_roots()
    if root and not _under(Path(root), lab_roots()):
        raise ToolError("OUTSIDE_ROOTS", f"{root} is not under a lab root",
                        roots=[str(r) for r in lab_roots()])
    out = []
    for r, e, _ in _walk(roots, max_depth):
        if e.is_dir(follow_symlinks=False) and (Path(e.path) / "config.json").is_file():
            p = Path(e.path)
            out.append({"path": e.path, "safetensors_gib": round(_dir_bytes(p) / 2**30, 3),
                        "has_model_py": (p / "model.py").is_file(),
                        "has_readme": (p / "README.md").is_file()})
    out.sort(key=lambda a: a["path"])
    return {"count": len(out), "artifacts": out, "roots": [str(r) for r in roots]}


def t_artifact_config(path: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """The shipped config.json — the record of what actually shipped."""
    p = Path(path)
    if not _under(p, lab_roots()):
        raise ToolError("OUTSIDE_ROOTS", f"{path} is not under a lab root")
    cfg = p / "config.json" if p.is_dir() else p
    if not cfg.is_file():
        raise ToolError("NOT_FOUND", f"no config.json at {p}")
    data = json.loads(cfg.read_text())
    if keys:
        data = {k: data.get(k) for k in keys}
    text = json.dumps(data, indent=1)
    if len(text) > 20000:
        text = text[:20000] + "\n… (truncated; pass keys=[...] to narrow)"
    return {"path": str(cfg), "config": text}


def t_read_doc(path: str, start: int = 1, lines: int = 200) -> Dict[str, Any]:
    """Read a slice of a lab doc (docs/, AGENTS.md, the law book, …)."""
    rel = path.lstrip("/")
    if not any(rel == a or rel.startswith(a.rstrip("/") + "/") for a in DOC_ALLOW):
        raise ToolError("NOT_ALLOWED", f"{path} is not a readable lab doc",
                        allowed=list(DOC_ALLOW))
    p = REPO / rel
    if not p.is_file():
        raise ToolError("NOT_FOUND", f"{p} does not exist")
    all_lines = p.read_text(errors="replace").split("\n")
    start = max(1, start)
    chunk = all_lines[start - 1:start - 1 + max(1, min(lines, 1000))]
    return {"path": str(p), "start": start, "end": start + len(chunk) - 1,
            "total_lines": len(all_lines), "text": "\n".join(chunk)}


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
    if cmd != "selftest" and not force:
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
                            hint="talk to the other box's vqlab server, or flip residency: "
                                 f"python -m scout.ops.lab_residency --fit-on {st.get('manager_host')}")
    holder = _lease_holder()
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
    env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path.home() / ".pycache-vqlab"))
    sup = subprocess.Popen(
        [sys.executable, str(PKG / "mcp_server.py"), "--supervise", str(rd)],
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


def t_gpu_state() -> Dict[str, Any]:
    return {"host": socket.gethostname().split(".")[0], "lease": str(lease_path()),
            "lease_holder": _lease_holder(), "exo_instances": _exo_instances(),
            "lab_state": _lab_state(), "this_host": _this_host()}


_F_HEAD = re.compile(r"^## F(\d+)\b")


def _findings_entries(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = path or FINDINGS_LOG  # resolved at call time (tests redirect it)
    out = []
    if not path.is_file():
        return out
    for i, line in enumerate(path.read_text(errors="replace").split("\n"), 1):
        m = _F_HEAD.match(line)
        if m:
            out.append({"f": int(m.group(1)), "line": i, "headline": line[3:].strip()})
    return out


def t_next_f_number() -> Dict[str, Any]:
    ents = _findings_entries()
    top = max((e["f"] for e in ents), default=0)
    return {"next": top + 1, "highest": top, "log": str(FINDINGS_LOG)}


def t_findings_tail(n: int = 10) -> Dict[str, Any]:
    ents = _findings_entries()
    ents.sort(key=lambda e: e["f"])
    return {"count": len(ents), "entries": ents[-max(1, n):], "log": str(FINDINGS_LOG)}


def t_findings_append(headline: str, artifact: str, instrument: str, prediction: str,
                      measured: str, verdict: str, body: str = "",
                      corrects: str = "") -> Dict[str, Any]:
    """Append the next F-entry. The prediction is required and is written as given."""
    missing = [k for k, v in dict(headline=headline, artifact=artifact, instrument=instrument,
                                  prediction=prediction, measured=measured).items() if not str(v).strip()]
    if missing:
        raise ToolError("INCOMPLETE", "every field is required; a finding without them is not a measurement",
                        missing=missing)
    verdict = verdict.upper().strip()
    if verdict not in VERDICTS:
        raise ToolError("BAD_VERDICT", f"verdict must be one of {VERDICTS}")
    if verdict == "CORRECTS" and not corrects:
        raise ToolError("INCOMPLETE", "verdict CORRECTS needs corrects='F<n>'")
    n = t_next_f_number()["next"]
    today = date.today().isoformat()
    head = f"## F{n} ({today}) — {headline.strip()}"
    if verdict == "CORRECTS":
        head = f"## F{n} ({today}) — CORRECTS {corrects}: {headline.strip()}"
    entry = "\n".join([
        "", head, "",
        f"ARTIFACT:   {artifact.strip()}",
        f"INSTRUMENT: {instrument.strip()}",
        f"PREDICTION (pre-registered): {prediction.strip()}",
        f"MEASURED:   {measured.strip()}",
        f"VERDICT:    {verdict}",
        "", body.rstrip(), "",
    ])
    with open(FINDINGS_LOG, "a", encoding="utf-8") as fh:
        fh.write(entry)
    return {"f": n, "headline": head, "log": str(FINDINGS_LOG),
            "note": "Written by tool: the prediction was recorded as given. "
                    "Corrections edit an entry in place and say CORRECTED; never delete."}


# --------------------------------------------------------------------------- #
# Tool table (schema is what the model sees — keep descriptions honest)
# --------------------------------------------------------------------------- #

def _schema(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


S = lambda d="": {"type": "string", "description": d}  # noqa: E731
I = lambda d="": {"type": "integer", "description": d}  # noqa: E731
B = lambda d="": {"type": "boolean", "description": d}  # noqa: E731
L = lambda d="": {"type": "array", "items": {"type": "string"}, "description": d}  # noqa: E731

TOOLS: Dict[str, Dict[str, Any]] = {
    "where_is": {
        "fn": t_where_is, "readonly": True,
        "description": "Find an artifact, fit, teacher cache, corpus or file by (sub)name across the lab's "
                       "storage roots. Deterministic: returns matches, or NOT_FOUND with the roots searched. "
                       "Use this instead of guessing or listing directories.",
        "schema": _schema({"name": S("substring of the directory/file name, ≥3 chars"),
                           "kind": {"type": "string", "enum": ["any", "dir", "file"]},
                           "max_depth": I("directory depth under each root (default 3)")}, ["name"]),
    },
    "list_artifacts": {
        "fn": t_list_artifacts, "readonly": True,
        "description": "Directories carrying a config.json under the lab roots (or under one root), with packed size.",
        "schema": _schema({"root": S("optional: one lab root or sub-directory"), "max_depth": I()}, []),
    },
    "artifact_config": {
        "fn": t_artifact_config, "readonly": True,
        "description": "Read a shipped artifact's config.json — the authoritative record of what shipped "
                       "(authority order: config → FINDINGS-LOG → EXPERIMENTS.md).",
        "schema": _schema({"path": S("artifact dir or config.json path"), "keys": L("optional key subset")}, ["path"]),
    },
    "read_doc": {
        "fn": t_read_doc, "readonly": True,
        "description": "Read a slice of a lab document: docs/*, AGENTS.md, METHODOLOGY.md, REPRODUCING.md, "
                       "research/quantlab/FINDINGS.md (the law book) or EXPERIMENTS.md.",
        "schema": _schema({"path": S("repo-relative path"), "start": I("1-based first line"), "lines": I("max 1000")}, ["path"]),
    },
    "run": {
        "fn": t_run, "readonly": False,
        "description": "Launch an allowlisted vqlab subcommand on THIS box, detached, under the GPU lease. "
                       "Returns immediately with a run_id; poll status. Refuses paths outside the lab roots, "
                       "refuses while an exo instance is placed, refuses while the lease is held (deferring is normal). "
                       "publish is not available here.",
        "schema": _schema({"cmd": {"type": "string", "enum": sorted(RUN_ALLOWLIST)},
                           "args": L("arguments exactly as for `vqlab <cmd>`"),
                           "tag": S("short label for the run id"),
                           "retries": I("checkpoint-resume retries (resumable cmds only)"),
                           "force": B("skip the exo-placement check (you know why)")}, ["cmd"]),
    },
    "status": {
        "fn": t_status, "readonly": True,
        "description": "State of a run on this box: status, exit code, attempts, log tail, result path.",
        "schema": _schema({"run_id": S(), "tail": I("log lines (default 40)")}, ["run_id"]),
    },
    "stop": {
        "fn": t_stop, "readonly": False,
        "description": "SIGTERM a run's process group on this box.",
        "schema": _schema({"run_id": S()}, ["run_id"]),
    },
    "list_runs": {
        "fn": t_list_runs, "readonly": True,
        "description": "Recent runs launched on this box.",
        "schema": _schema({"n": I()}, []),
    },
    "gpu_state": {
        "fn": t_gpu_state, "readonly": True,
        "description": "Who holds this box's GPU lease and which exo instances are placed.",
        "schema": _schema({}, []),
    },
    "next_f_number": {
        "fn": t_next_f_number, "readonly": True,
        "description": "The next free F-number in docs/FINDINGS-LOG.md.",
        "schema": _schema({}, []),
    },
    "findings_tail": {
        "fn": t_findings_tail, "readonly": True,
        "description": "The last n finding headlines with their line numbers (read the body with read_doc).",
        "schema": _schema({"n": I()}, []),
    },
    "findings_append": {
        "fn": t_findings_append, "readonly": False,
        "description": "Append the next F-entry to FINDINGS-LOG.md. Every field is required; the pre-registered "
                       "prediction is written verbatim and a falsified prediction stays falsified. "
                       "Verdict ∈ CONFIRMED | FALSIFIED | VOID | NULL | CORRECTS (needs corrects='F<n>').",
        "schema": _schema({"headline": S(), "artifact": S("artifact name/path the number belongs to"),
                           "instrument": S("scoring path + batching, e.g. 'vqlab score, prose corpus, resident'"),
                           "prediction": S("what was pre-registered before the run"),
                           "measured": S("the number(s)"),
                           "verdict": {"type": "string", "enum": list(VERDICTS)},
                           "body": S("narrative / tables"), "corrects": S("F<n> when verdict=CORRECTS")},
                          ["headline", "artifact", "instrument", "prediction", "measured", "verdict"]),
    },
}


def tool_list() -> List[Dict[str, Any]]:
    return [{"name": n, "description": t["description"], "inputSchema": t["schema"],
             "annotations": {"readOnlyHint": t["readonly"], "destructiveHint": not t["readonly"]}}
            for n, t in TOOLS.items()]


def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Dispatch; returns an MCP tools/call result (never raises for tool errors)."""
    t = TOOLS.get(name)
    if t is None:
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(
            {"error": "UNKNOWN_TOOL", "message": name, "tools": list(TOOLS)})}]}
    try:
        out = t["fn"](**(arguments or {}))
        return {"isError": False, "content": [{"type": "text", "text": json.dumps(out, default=str)}]}
    except ToolError as te:
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(te.as_result(), default=str)}]}
    except TypeError as exc:  # bad/missing arguments
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(
            {"error": "INVALID_ARG", "message": str(exc), "schema": t["schema"]})}]}
    except Exception as exc:  # noqa: BLE001 — surface, never crash the server
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(
            {"error": "INTERNAL", "message": f"{type(exc).__name__}: {exc}"})}]}


# --------------------------------------------------------------------------- #
# JSON-RPC over stdio (newline-delimited, per the MCP stdio transport)
# --------------------------------------------------------------------------- #

def handle(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method == "initialize":
        return _ok(mid, {"protocolVersion": PROTOCOL_VERSION,
                         "capabilities": {"tools": {"listChanged": False}},
                         "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION,
                                        "host": socket.gethostname().split(".")[0]},
                         "instructions": ("vqlab lab server for THIS box. Use where_is before claiming "
                                          "anything is missing. Read the law book (read_doc "
                                          "research/quantlab/FINDINGS.md) before proposing an experiment. "
                                          "Pre-register a prediction before you run; record it with "
                                          "findings_append. publish is a human action.")})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": tool_list()})
    if method == "tools/call":
        return _ok(mid, call_tool(params.get("name", ""), params.get("arguments")))
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def _ok(mid: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(inp=None, out=None) -> int:
    inp = inp or sys.stdin.buffer
    out = out or sys.stdout.buffer
    for raw in inp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            try:
                resp = handle(msg)
            except Exception as exc:  # noqa: BLE001
                resp = {"jsonrpc": "2.0", "id": msg.get("id"),
                        "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        if resp is not None:
            out.write((json.dumps(resp, default=str) + "\n").encode("utf-8"))
            out.flush()
    return 0


# --------------------------------------------------------------------------- #
# Supervisor (the detached process that owns the lease and runs the job)
# --------------------------------------------------------------------------- #

def supervise(run_dir: Path) -> int:
    meta = _read_meta(run_dir)
    lp = lease_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + 1800
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise
            if time.time() > deadline:
                meta.update(status="deferred", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            note="GPU lease held for 30 min; deferred, not failed")
                (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
                return 0
            time.sleep(15)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps({"holder": f"vqlab:{meta.get('cmd')}", "pid": os.getpid(),
                             "run_id": meta.get("run_id"),
                             "since": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode())
    os.fsync(fd)

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
                    except Exception:  # noqa: BLE001
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
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return 0 if rc == 0 else 1


# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="vqlab mcp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the tool table and exit")
    ap.add_argument("--call", nargs=2, metavar=("TOOL", "JSON_ARGS"), help="call one tool and print the result")
    ap.add_argument("--supervise", metavar="RUN_DIR", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.supervise:
        return supervise(Path(a.supervise))
    if a.list:
        for t in tool_list():
            print(f"{t['name']:16s} {'ro ' if t['annotations']['readOnlyHint'] else 'RW '} {t['description']}")
        return 0
    if a.call:
        print(json.dumps(call_tool(a.call[0], json.loads(a.call[1])), indent=1))
        return 0
    return serve()


if __name__ == "__main__":
    sys.exit(main())
