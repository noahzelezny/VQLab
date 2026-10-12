"""Start, stop and resume `vqlab queue` runs from the GUI.

Nothing here runs a step. Every start goes through the same CLI an operator
types (`vqlab queue run FILE --preflight --detach`, `--resume QDIR --detach`),
so the pinned tree, the dirty-tree refusal, reservations, the GPU lease and
the exo gate stay in one place (queue/runner.py).

A start must carry the token of the plan the operator confirmed. `plan()`
issues a single-use token bound to the queue file's bytes and the mode, and
`start()` spends it, so the confirmation is never skipped, a double click or
a second tab cannot start a queue twice, and a file edited after it was
shown is not run unseen.

Queue files the GUI offers live in <records>/queue-files/*.json. A full run
starts only from a preflight that passed, at the commit that preflight ran,
and only once.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import socket
import subprocess
import threading
import time

from vqlab import config
from vqlab.core.box import exo
from vqlab.core.box.gpu_lease import lease_holder, pid_alive
from vqlab.queue import eta as queue_eta
from vqlab.queue import files as qf
from vqlab.queue import steps as qsteps

RESUMABLE = ("stopped", "failed", "deferred", "running")   # running: only once its runner is dead
TOKEN_TTL = 15 * 60
_START = threading.Lock()             # one start / resume at a time (they spawn the CLI)
_TOK = threading.Lock()
_TOKENS: dict[str, tuple[str, float]] = {}    # token -> (digest of what it confirmed, issued)
_QDIR = re.compile(r"vqlab queue status (\S+)")


def files_dir() -> pathlib.Path:
    return config.records() / "queue-files"


def queue_files() -> list[dict]:
    d = files_dir()
    return [{"file": p.name, "name": _name(p)} for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def write_recipe(body: dict) -> dict:
    """A recipe from the Recipe page -> <records>/queue-files/<name>.json
    (plan/recipe_queue.py), which the Jobs page then offers to Start. A
    file of that name is never overwritten."""
    from vqlab.plan.recipe_queue import RecipeError, compile_recipe
    try:
        q, warns = compile_recipe(body.get("recipe") or {})
    except RecipeError as e:
        return {"ok": False, "error": str(e)}
    errs = qsteps.validate(q)
    if errs:
        return {"ok": False, "error": "compiled queue invalid: " + "; ".join(errs)}
    d = files_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{q['name']}.json"
    try:
        with open(p, "x") as f:
            json.dump(q, f, indent=1)
    except FileExistsError:
        return {"ok": False, "error": f"a queue file named {p.name} already exists; pick another name"}
    return {"ok": True, "file": p.name, "warnings": warns}


def _name(p: pathlib.Path) -> str:
    try:
        return json.loads(p.read_text()).get("name") or p.stem
    except (OSError, ValueError):
        return p.stem


def _qdir(qid: str) -> pathlib.Path:
    """A queue dir by its folder name; never a path from outside the queues dir."""
    root = qf.queues_dir().resolve()
    d = (root / qid).resolve()
    if d.parent != root or not (d / "state.json").is_file():
        raise ValueError(f"no queue run {qid!r}")
    return d


def _queue_file(name: str) -> pathlib.Path:
    root = files_dir().resolve()
    p = (root / name).resolve()
    if p.parent != root or p.suffix != ".json" or not p.is_file():
        raise ValueError(f"no queue file {name!r} in {files_dir()}")
    return p


def _source(body: dict) -> tuple[pathlib.Path, list[str]]:
    """(queue file, extra `queue run` flags) for a start: a file runs its
    preflight; a passed preflight's id runs that queue for real, pinned at the
    preflight's commit."""
    if body.get("file"):
        return _queue_file(body["file"]), ["--preflight"]
    d = _qdir(str(body.get("id") or ""))
    s = qf.load_state(d)
    if not (s.get("preflight") and s.get("status") == "passed"):
        raise ValueError("a full run starts from a preflight that passed")
    qfile = d / "queue.json"
    if any(qf.load_state(o).get("source") == str(qfile.resolve()) for o in qf.queue_dirs()):
        raise ValueError("this preflight's full run was already started")
    if s.get("installed"):
        return qfile, []
    if not s.get("commit"):
        raise ValueError("that preflight has no recorded commit")
    return qfile, ["--commit", s["commit"]]


def _digest(qfile: pathlib.Path, flags: list[str]) -> str:
    return hashlib.sha256(qfile.read_bytes() + "\0".join(["", *flags]).encode()).hexdigest()


def _writes(st: dict, preflight: bool) -> int | None:
    """Bytes a step writes to scratch, as the runner sizes it, or None."""
    pf = st.get("preflight") or {}
    if "writes_gib" in pf:
        return int(pf["writes_gib"] * 2**30)
    if preflight and "full" not in pf:
        return None           # a first-module run: small, and not measured
    args = [config.expand(str(a)) for a in st.get("args", [])]
    return qsteps.input_bytes(args, qf.REPO) or None


def _rows(q: dict, preflight: bool) -> tuple[list[dict], float | None, int]:
    """(steps with the full run's ETA and what each writes, total ETA or None
    when any step has no past run, bytes written)."""
    past = queue_eta.past_runs(qf.queues_dir())
    rows, total, written = [], 0.0, 0
    for st in q.get("steps", []):
        med = queue_eta.median_past(past, st)
        eta = None if med is None else round(med)
        w = _writes(st, preflight)
        rows.append({"name": st.get("name"), "cmd": st.get("cmd") or st.get("script"),
                     "gpu": qsteps.needs_gpu(st), "eta_s": eta, "writes": w})
        total = None if total is None or eta is None else total + eta
        written += w or 0
    return rows, total, written


def preview(body: dict) -> dict:
    """What a recipe would compile to, without writing it: the Recipe page's
    step list, notes and full-run estimate."""
    from vqlab.plan.recipe_queue import RecipeError, compile_recipe
    try:
        q, warns = compile_recipe(body.get("recipe") or {})
    except RecipeError as e:
        return {"ok": False, "error": str(e)}
    rows, total, _ = _rows(q, preflight=False)
    est = q["estimate"]
    return {"ok": True, "steps": rows, "eta_s": total, "writes": int(est["scratch_gib"] * 2**30),
            "artifact_gib": est["artifact_gib"], "warnings": warns}


def plan(body: dict) -> dict:
    """What a start would run, for the confirmation: steps with the FULL
    run's ETA (queue/eta.py: past runs of the same step, never a guess; a
    preflight's own time is not recorded), what they write, scratch free, and
    anything that would stop it on the GPU. Issues the start's token."""
    qfile, flags = _source(body)
    preflight = "--preflight" in flags
    q = json.loads(qfile.read_text())
    rows, total, written = _rows(q, preflight)
    if not preflight and q.get("estimate"):      # a compiled recipe knows its own size
        written = int(q["estimate"]["scratch_gib"] * 2**30)
    gpu = any(r["gpu"] for r in rows)
    scratch = config.scratch()
    token = secrets.token_hex(16)
    now = time.time()
    with _TOK:
        for t in [t for t, (_, at) in _TOKENS.items() if now - at > TOKEN_TTL]:
            del _TOKENS[t]
        _TOKENS[token] = (_digest(qfile, flags), now)
    return {"name": q.get("name") or qfile.stem, "preflight": preflight, "steps": rows,
            "eta_s": total, "writes": written,
            "scratch_free": shutil.disk_usage(scratch).free if scratch.exists() else None,
            "gpu_held_by": lease_holder() if gpu else None,
            "exo_placed": (exo.exo_instances() or []) if gpu else [],
            "problems": qsteps.validate(q), "token": token}


def _cli(*args: str) -> dict:
    """Run `vqlab queue run ... --detach`: it pins the queue and returns once
    the runner is detached. -> {ok, output, id (the new queue dir's name)}."""
    pre, env, cwd = qf.cli_launch()
    try:
        r = subprocess.run([*pre, "queue", "run", *args, "--detach"],
                           capture_output=True, text=True, env=env, cwd=cwd, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "the start is still setting up; check the queue list before trying again"}
    out = (r.stdout + r.stderr).strip()[-2000:]
    m = _QDIR.search(out)
    return {"ok": r.returncode == 0, "output": out, "id": pathlib.Path(m.group(1)).name if m else None}


def start(body: dict) -> dict:
    with _START:
        qfile, flags = _source(body)
        with _TOK:
            digest, at = _TOKENS.pop(str(body.get("token")), (None, 0.0))
        if digest != _digest(qfile, flags) or time.time() - at > TOKEN_TTL:
            return {"ok": False, "error": "this start was not confirmed, or the queue changed since; review it again"}
        errs = qsteps.validate(json.loads(qfile.read_text()))
        if errs:
            return {"ok": False, "error": "queue file invalid: " + "; ".join(errs)}
        return _cli(str(qfile), *flags)


def _here(s: dict) -> bool:
    return s.get("host") == socket.gethostname().split(".")[0]


def _cmdline(pid) -> str:
    return subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))],
                          capture_output=True, text=True).stdout


def _runner(qdir: pathlib.Path, s: dict) -> int | None:
    """The runner's pid, only while that pid is still THIS queue's runner on
    this machine (a dead runner's pid can be reused by anything)."""
    pid = s.get("pid")
    if not (_here(s) and pid_alive(pid)):
        return None
    cmd = _cmdline(pid)
    return int(pid) if "queue run" in cmd and qdir.name in cmd else None


def _runner_alive(s: dict) -> bool:
    """A runner may still own this queue: its pid lives and runs a queue. A
    foreground `queue run FILE` counts (its command line does not name the
    queue dir), so a queue is never resumed underneath one."""
    return pid_alive(s.get("pid")) and "queue run" in _cmdline(s["pid"])


def stop(body: dict) -> dict:
    """SIGTERM the runner: it stops the running step, records `stopped`, and
    the queue resumes from the step that did not pass."""
    d = _qdir(str(body.get("id") or ""))
    pid = _runner(d, qf.load_state(d))
    if pid is None:
        return {"ok": False, "error": "that queue has no runner started from here or by `--detach` "
                                      "on this machine; stop a foreground run in its terminal"}
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as e:
        return {"ok": False, "error": f"could not stop the runner: {e}"}
    return {"ok": True}


def resume(body: dict) -> dict:
    with _START:
        d = _qdir(str(body.get("id") or ""))
        s = qf.load_state(d)
        if not _here(s):
            return {"ok": False, "error": f"that queue ran on {s.get('host')}; resume it there"}
        if s.get("status") not in RESUMABLE or _runner_alive(s):
            return {"ok": False, "error": f"a {s.get('status')} queue does not resume"}
        return _cli("--resume", str(d))
