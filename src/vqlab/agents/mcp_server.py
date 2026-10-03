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
  lease (an advisory flock that dies with its holder; the file is
  ``vqlab.config.gpu_lease``, shareable with other GPU tenants), with a
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
import shutil
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
from vqlab import config  # noqa: E402
from vqlab.agents import queue_eta as _queue_eta  # noqa: E402
from vqlab.agents import reserve as _resv  # noqa: E402

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vqlab"
SERVER_VERSION = "0.1.0"

AGENTS_DIR = Path(__file__).resolve().parent
PKG = AGENTS_DIR.parent          # src/vqlab
REPO = PKG.parents[1]

# --------------------------------------------------------------------------- #
# Lab layout
# --------------------------------------------------------------------------- #

def lab_roots() -> List[Path]:
    return config.roots()


def runs_dir() -> Path:
    env = os.environ.get("VQLAB_RUNS_DIR")
    if env:
        return Path(env)
    return config.scratch() / "mcp-runs" / socket.gethostname().split(".")[0]


def lease_path() -> Path:
    """The GPU lease file (`vqlab.config.gpu_lease`). Point every GPU tenant
    on the machine at the same file and they take turns."""
    return config.gpu_lease()


def exo_api() -> str:
    return os.environ.get("VQLAB_EXO_API", "http://127.0.0.1:52415")


# Subcommands an agent may launch. publish is deliberately absent.
RUN_ALLOWLIST = {
    "fit-moe", "fit-dense", "geo-build", "alloc-sweep", "score", "kl",
    "layer-leverage", "validate", "check-release", "check-bundle", "check",
    "smoke", "selftest", "price", "verify", "coverage", "manifest",
    # the onboarding + reuse loop, so a new family needs no human at the CLI:
    # profile (headers only) -> init sweep -> leverage -> fits from the store
    # -> KL gate -> task benchmarks; plus the records that make it auditable
    "onboard", "family-profile", "probe-init", "fits", "kl-ladder", "kl-pair", "tasks",
    "provenance", "runs", "registry", "active-bytes", "decode-timeline",
    "stream-score",
}
# Commands that never touch the GPU (header reads, index queries, records).
# They are not gated on an exo placement or the GPU lease, and do not take
# the lease: profiling a teacher must not wait on, or block, a fit.
GPU_FREE = {"onboard", "family-profile", "fits", "provenance", "runs", "price",
            "manifest", "check-bundle", "active-bytes", "registry"}
# Commands that checkpoint and resume: a GPU-timeout crash is retried.
RESUMABLE = {"fit-moe", "fit-dense", "geo-build", "alloc-sweep", "validate"}
DEFAULT_RETRIES = 2

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
    except Exception:  # noqa: BLE001 — no record ⇒ no rule in force
        return {}


def _this_host() -> str:
    return os.environ.get("VQLAB_HOSTNAME") or socket.gethostname().split(".")[0]


# Read-only doc access: the routing layer (CONTEXT.md at the root and in
# every stage folder), family data, the docs and the law book, plus a
# private `lab/` notebook when the checkout has one (it is gitignored).
DOC_ALLOW = ("docs", "families", "CONTEXT.md", "AGENTS.md",
             "README.md", "METHODOLOGY.md", "REPRODUCING.md", "lab",
             "research/CONTEXT.md")
# The F-numbered findings log is the lab's own record: private (lab/ is
# gitignored) unless VQLAB_FINDINGS_LOG points elsewhere.
FINDINGS_LOG = Path(os.environ.get("VQLAB_FINDINGS_LOG") or REPO / "lab" / "FINDINGS-LOG.md")
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
    # Registered teacher caches match on their TEACHER or corpus too, not
    # only their directory name: "397B" finds q397_prose_12k.
    try:
        from vqlab.core.families import teacher_caches
        known = {h["path"] for h in hits}
        for c in teacher_caches():
            if needle in (c["teacher"] + " " + c.get("corpus", "") + " " + c["path"]).lower() \
                    and c["path"] not in known and kind in ("any", "dir"):
                hits.append({"path": c["path"], "kind": "teacher_cache",
                             "teacher": c["teacher"], "corpus": c.get("corpus"),
                             "tokens": c.get("tokens"), "full_vocab": c.get("full_vocab"),
                             "exists": Path(c["path"]).is_dir()})
    except Exception:  # noqa: BLE001  the registry is a bonus, never a failure
        pass
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
    # Stage contracts (src/vqlab/<stage>/CONTEXT.md) are docs; the source
    # beside them is not -- read_doc never serves code.
    is_contract = (rel.startswith("src/vqlab/") and rel.endswith("/CONTEXT.md")
                   and ".." not in rel)
    if not is_contract and not any(rel == a or rel.startswith(a.rstrip("/") + "/")
                                   for a in DOC_ALLOW):
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
    env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONPYCACHEPREFIX", str(Path.home() / ".pycache-vqlab"))
    sup = subprocess.Popen(
        [sys.executable, str(AGENTS_DIR / "mcp_server.py"), "--supervise", str(rd)],
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
    # same resolution as run_queue.queues_dir (run_queue imports this module)
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
        except Exception as e:  # noqa: BLE001 -- an ETA must never hide the state
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


def t_disk_free() -> Dict[str, Any]:
    """Free space on every configured storage root (writers run out mid-run)."""
    seen, out = set(), []
    for r in lab_roots():
        try:
            u = shutil.disk_usage(r)
        except OSError:
            continue
        key = (u.total, u.free)
        out.append({"root": str(r), "free_gib": round(u.free / 2**30, 1),
                    "total_gib": round(u.total / 2**30, 1), "same_volume_as_previous": key in seen})
        seen.add(key)
    return {"roots": out}


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
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:300]}


def t_gpu_state(boxes: bool = True) -> Dict[str, Any]:
    out = {"host": socket.gethostname().split(".")[0], "lease": str(lease_path()),
           "lease_holder": _lease_holder(), "exo_instances": _exo_instances(),
           "lab_state": _lab_state(), "this_host": _this_host()}
    try:   # every box's reservation (`vqlab reserve`), from the shared file
        out["reservations"] = [dict(r, text=_resv.describe(r)) for r in _resv.active()]
        out["reservations_file"] = str(_resv.shared_path())
    except Exception as e:  # noqa: BLE001
        out["reservations"] = {"error": str(e)[:300]}
    if boxes and config.boxes():
        out["boxes"] = {n: _box_state(n, b) for n, b in config.boxes().items()}
    return out


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


_FLOCK_DIR = FINDINGS_LOG.parent / ".f-locks"
_FLOCK_TTL = 6 * 3600          # a crashed session must not burn a number forever


def _live_reservations(published: Optional[set] = None) -> Dict[int, float]:
    """Claimed F-numbers that have not expired, as {number: claimed_at}.

    Self-healing on two conditions. A lock is dropped when it is STALE (past
    the TTL, so the session holding it is gone) and when its number is
    ALREADY IN THE LOG -- a published number cannot meaningfully be
    reserved, and a lock left behind by an append that has since written its
    entry is pure noise. The second case was real: before append released
    its own lock, every tool-written entry left one behind, and a fresh
    `next_f_number` reported a run of published numbers as "held".
    """
    out: Dict[int, float] = {}
    if not _FLOCK_DIR.is_dir():
        return out
    if published is None:
        published = set(e["f"] for e in _findings_entries())
    now = time.time()
    for f in _FLOCK_DIR.glob("F*.lock"):
        try:
            n = int(f.stem[1:])
            age = now - f.stat().st_mtime
        except (ValueError, OSError):
            continue
        if age > _FLOCK_TTL or n in published:
            try:
                f.unlink()          # stale, or already written to the log
            except OSError:
                pass
            continue
        out[n] = f.stat().st_mtime
    return out


def t_next_f_number(reserve: bool = False, owner: str = "") -> Dict[str, Any]:
    """The next free F-number; with reserve=True, CLAIM it atomically.

    WHY RESERVE EXISTS (2026-09-19). This used to be `max(log) + 1`, a pure
    read. Two sessions working the same repo both called it, both got 151,
    and both wrote an F151 -- neither had committed when the other looked.
    A read cannot prevent that no matter how careful either session is: the
    log is only updated at commit time, so the window is the whole length of
    the experiment. The fix has to be a WRITE, and it has to be atomic.

    O_CREAT|O_EXCL on a per-number lock file is that write: exactly one
    caller can create `F<n>.lock`, and the loser simply advances to n+1.
    Locks carry a TTL so a session that dies mid-experiment releases its
    number instead of holding it forever, and `findings_append` releases
    the lock once the entry is actually in the log.

    reserve=False stays the default so a plain "what number am I on?" costs
    nothing and burns nothing.
    """
    ents = _findings_entries()
    top = max((e["f"] for e in ents), default=0)
    pub = set(e["f"] for e in ents)
    taken = pub | set(_live_reservations(pub))
    n = top + 1
    while n in taken:
        n += 1
    if not reserve:
        held = sorted(_live_reservations())
        return {"next": n, "highest": top, "log": str(FINDINGS_LOG),
                "reserved": False, "reservations_held": held,
                "note": ("call with reserve=true before a long experiment; a "
                         "bare read races another session (F151/F152 vs "
                         "F153/F154, 2026-09-19)")}
    _FLOCK_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(_FLOCK_DIR / f"F{n}.lock"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            n += 1                  # someone claimed it between the scan and now
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"owner": owner or "unknown",
                                 "claimed": datetime.now().isoformat()}))
        return {"next": n, "highest": top, "log": str(FINDINGS_LOG),
                "reserved": True, "owner": owner or "unknown",
                "expires_in_s": _FLOCK_TTL,
                "note": "release happens automatically on findings_append"}


def t_release_f_number(n: int) -> Dict[str, Any]:
    """Give a reserved number back (abandoned experiment)."""
    f = _FLOCK_DIR / f"F{int(n)}.lock"
    existed = f.exists()
    if existed:
        f.unlink()
    return {"released": int(n), "was_held": existed}


def t_findings_tail(n: int = 10) -> Dict[str, Any]:
    ents = _findings_entries()
    ents.sort(key=lambda e: e["f"])
    return {"count": len(ents), "entries": ents[-max(1, n):], "log": str(FINDINGS_LOG)}


def t_findings_append(headline: str, artifact: str, instrument: str, prediction: str,
                      measured: str, verdict: str, body: str = "",
                      corrects: str = "", use_reserved: int = 0) -> Dict[str, Any]:
    """Append the next F-entry. The prediction is required and is written as given.

    `use_reserved` CONSUMES a number this caller already claimed with
    `next_f_number(reserve=true)`. Without it, append would reserve a FRESH
    number -- because `_live_reservations()` cannot tell the caller's own
    hold from another session's and skips both -- so a reserve-then-append
    in one session silently burned a number (reported by vqlab-28, who
    reserved 162 and got an entry at 163). Passing the reserved number back
    is the fix; the lock is then released as part of writing the entry.

    Append also RELEASES whatever lock it used. The first version left it
    behind for the full 6 h TTL, so every tool-written entry leaked a
    reservation that then had to age out -- harmless for correctness, since
    both the log and the lock mark the number taken, but it meant the
    held-reservations list filled with numbers that were already published.
    """
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
    if use_reserved:
        n = int(use_reserved)
        ents = _findings_entries()
        if n in set(e["f"] for e in ents):
            raise ValueError(f"F{n} is already in the log; it cannot be "
                             "reserved or reused")
        if not (_FLOCK_DIR / f"F{n}.lock").exists():
            raise ValueError(
                f"F{n} was not reserved (no lock held). Either call "
                "next_f_number(reserve=true) first, or omit use_reserved "
                "and let append allocate.")
    else:
        n = t_next_f_number(reserve=True, owner="findings_append")["next"]
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
    # the number is now IN the log, so the lock has done its job
    try:
        (_FLOCK_DIR / f"F{n}.lock").unlink()
    except OSError:
        pass
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
                       "docs/FINDINGS.md (the law book) or EXPERIMENTS.md.",
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
        "description": "Who holds this box's GPU lease and which exo instances are placed; with "
                       "[boxes.*] in the vqlab config, the same for every other box over ssh "
                       "(lease holder, load, GPU-heavy processes), plus every box's reservation "
                       "(`vqlab reserve`: who has it, until when). Check it before loading a big "
                       "model on a box that might be fitting or is reserved.",
        "schema": _schema({"boxes": B("also report the other configured boxes (default true)")}, []),
    },
    "queue_status": {
        "fn": t_queue_status, "readonly": True,
        "description": "State of `vqlab queue` runs on this box from state.json: per-step status and "
                       "reasons, terminal=true once passed/failed/stopped/deferred/died, and an ETA "
                       "(eta_s, finish, per-step eta_basis) from the running step's own progress lines "
                       "or past runs of the same step -- null means unknown, not zero. Use this (or "
                       "`vqlab queue wait`) to know a queue finished; never grep its log.",
        "schema": _schema({"name": S("substring of the queue dir name"), "n": I("most recent n (default 5)")}, []),
    },
    "disk_free": {
        "fn": t_disk_free, "readonly": True,
        "description": "Free GiB on every configured storage root. Check before any multi-hour writer.",
        "schema": _schema({}, []),
    },
    "next_f_number": {
        "fn": t_next_f_number, "readonly": False,
        "description": "The next free F-number. Pass reserve=true to CLAIM it "
                       "atomically before a long experiment -- a bare read "
                       "races any concurrent session, because the log only "
                       "changes at commit time (two sessions both took 151 on "
                       "2026-09-19). Reservations expire after 6 h and are "
                       "released by findings_append.",
        "schema": _schema({"reserve": B(), "owner": S()}, []),
    },
    "release_f_number": {
        "fn": t_release_f_number, "readonly": False,
        "description": "Give back a reserved F-number whose experiment was abandoned.",
        "schema": _schema({"n": I()}, ["n"]),
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
        "schema": _schema({"use_reserved": I("an F-number THIS session already claimed "
                                            "with next_f_number(reserve=true); omit to "
                                            "allocate fresh. Without it a reserve-then-"
                                            "append burns a number"),
                           "headline": S(), "artifact": S("artifact name/path the number belongs to"),
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
                                          "docs/FINDINGS.md) before proposing an experiment. "
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

def acquire_lease(lp: Path, holder: str, run_id: Optional[str], wait_s: float = 1800,
                  poll_s: float = 15) -> Optional[int]:
    """Take an exclusive flock on `lp`, waiting up to wait_s; stamp the holder.
    Returns the fd (keep it open to hold the lease; release_lease to drop it)
    or None if it stayed held. Shared by the MCP supervisor and `vqlab queue`,
    so there is one lease protocol, not a pgrep script per campaign."""
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + wait_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                os.close(fd)
                raise
            if time.time() > deadline:
                os.close(fd)
                return None
            time.sleep(poll_s)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps({"holder": holder, "pid": os.getpid(), "run_id": run_id,
                             "since": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode())
    os.fsync(fd)
    return fd


def release_lease(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


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
        release_lease(fd)
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
