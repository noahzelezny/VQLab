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

Queue files the GUI offers live in config.queue_files_dir()
(<records>/queue-files/*.json). A full run starts only from a preflight that
passed, at the commit that preflight ran, and only once.

What a plan says a run writes is each step's own size (queue/steps.py
preflight_writes); for a full run of a compiled recipe, the recipe's
estimate wins over that sum, since it knows the intermediates the chain
leaves behind.

A refusal raises util/errors.UserError (the server answers 400); a start
or resume returns the CLI's own verdict, {ok, output, id}.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import secrets
import shutil
import signal
import subprocess
import threading
import time

from vqlab import config
from vqlab.util.errors import UserError
from vqlab.core.box import exo
from vqlab.core.box.gpu_lease import lease_holder
from vqlab.plan import recipe_queue
from vqlab.queue import eta as queue_eta
from vqlab.queue import files as qf
from vqlab.queue import report as qreport
from vqlab.queue import steps as qsteps
from vqlab.util.fs import GIB

RESUMABLE = ("stopped", "failed", "deferred", "running")   # running: only once its runner is dead
TOKEN_TTL = 15 * 60
_START = threading.Lock()             # one start / resume at a time (they spawn the CLI)


class ConfirmTokens:
    """Single-use tokens, each bound to the digest of what the operator was
    shown and good for `ttl` seconds."""

    def __init__(self, ttl: float = TOKEN_TTL):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._issued: dict[str, tuple[str, float]] = {}   # token -> (digest, issued at)

    def issue(self, digest: str) -> str:
        token, now = secrets.token_hex(16), time.time()
        with self._lock:
            for t in [t for t, (_, at) in self._issued.items() if now - at > self.ttl]:
                del self._issued[t]
            self._issued[token] = (digest, now)
        return token

    def spend(self, token, digest: str) -> bool:
        """Whether `token` confirmed exactly `digest` and is still fresh. A
        token is spent by trying it, right or wrong."""
        with self._lock:
            got, at = self._issued.pop(str(token), (None, 0.0))
        return got == digest and time.time() - at <= self.ttl


_TOKENS = ConfirmTokens()


def queue_files() -> list[dict]:
    d = config.queue_files_dir()
    return [{"file": p.name, "name": _name(p)} for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def write_recipe(body: dict) -> dict:
    """A recipe from the Recipe page -> <queue files>/<name>.json
    (plan/recipe_queue.py; its RecipeError is a 400), which the Jobs page
    then offers to Start. A file of that name is never overwritten."""
    q, warns = recipe_queue.compile_recipe(body.get("recipe") or {})
    errs = qsteps.validate(q)
    if errs:
        raise UserError("compiled queue invalid: " + "; ".join(errs))
    d = config.queue_files_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{q['name']}.json"
    try:
        with open(p, "x") as f:            # never overwrites a queue file of that name
            f.write(recipe_queue.queue_text(q))
    except FileExistsError:
        raise UserError(f"a queue file named {p.name} already exists; pick another name")
    return {"ok": True, "file": p.name, "warnings": warns}


def _name(p: pathlib.Path) -> str:
    try:
        return json.loads(p.read_text()).get("name") or p.stem
    except (OSError, ValueError):
        return p.stem


def _child(root: pathlib.Path, name: str, suffix: str | None = None) -> pathlib.Path | None:
    """root/name when that is an entry directly in root (never a path out of
    it, nor root itself) with the given suffix, else None."""
    root = root.resolve()
    p = (root / name).resolve()
    if p.parent != root or (suffix is not None and p.suffix != suffix):
        return None
    return p


def _qdir(qid: str) -> pathlib.Path:
    """A queue dir by its folder name."""
    d = _child(qf.queues_dir(), qid)
    if d is None or not (d / "state.json").is_file():
        raise UserError(f"no queue run {qid!r}")
    return d


def _queue_file(name: str) -> pathlib.Path:
    p = _child(config.queue_files_dir(), name, ".json")
    if p is None or not p.is_file():
        raise UserError(f"no queue file {name!r} in {config.queue_files_dir()}")
    return p


def _source(body: dict) -> tuple[pathlib.Path, list[str]]:
    """(queue file, extra `queue run` flags) for a start: a file runs its
    preflight; a passed preflight's id runs that queue for real, pinned at the
    preflight's commit."""
    if body.get("file"):
        return _queue_file(str(body["file"])), ["--preflight"]
    d = _qdir(str(body.get("id") or ""))
    s = qf.load_state(d)
    if not (s.get("preflight") and s.get("status") == "passed"):
        raise UserError("a full run starts from a preflight that passed")
    qfile = d / "queue.json"
    if any(qf.load_state(o).get("source") == str(qfile.resolve()) for o in qf.queue_dirs()):
        raise UserError("this preflight's full run was already started")
    if s.get("installed"):
        return qfile, []
    if not s.get("commit"):
        raise UserError("that preflight has no recorded commit")
    return qfile, ["--commit", s["commit"]]


def _digest(qfile: pathlib.Path, flags: list[str]) -> str:
    return hashlib.sha256(qfile.read_bytes() + "\0".join(["", *flags]).encode()).hexdigest()


def _rows(q: dict, preflight: bool) -> tuple[list[dict], float | None, int]:
    """(steps with the full run's ETA and what each writes, total ETA or None
    when any step has no past run, bytes written)."""
    past = queue_eta.past_runs(qf.queues_dir())
    rows, total, written = [], 0.0, 0
    for st in q.get("steps", []):
        med = queue_eta.median_past(past, st)
        eta = None if med is None else round(med)
        args = [config.expand(str(a)) for a in st.get("args", [])]
        w = qsteps.preflight_writes(st, args, qf.REPO, preflight)
        rows.append({"name": st.get("name"), "cmd": st.get("cmd") or st.get("script"),
                     "gpu": qsteps.needs_gpu(st), "eta_s": eta, "writes": w})
        total = None if total is None or eta is None else total + eta
        written += w
    return rows, total, written


def preview(body: dict) -> dict:
    """What a recipe would compile to, without writing it: the Recipe page's
    step list, notes and full-run estimate."""
    q, warns = recipe_queue.compile_recipe(body.get("recipe") or {})
    rows, total, _ = _rows(q, preflight=False)
    est = q["estimate"]
    return {"ok": True, "steps": rows, "eta_s": total, "writes": int(est["scratch_gib"] * GIB),
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
        written = int(q["estimate"]["scratch_gib"] * GIB)
    gpu = any(r["gpu"] for r in rows)
    scratch = config.scratch()
    return {"name": q.get("name") or qfile.stem, "preflight": preflight, "steps": rows,
            "eta_s": total, "writes": written,
            "scratch_free": shutil.disk_usage(scratch).free if scratch.exists() else None,
            "gpu_held_by": lease_holder() if gpu else None,
            "exo_placed": (exo.exo_instances() or []) if gpu else [],
            "problems": qsteps.validate(q), "token": _TOKENS.issue(_digest(qfile, flags))}


def _cli(*args: str) -> dict:
    """Run `vqlab queue run ... --detach --json`: it pins the queue and
    returns once the runner is detached, its last stdout line
    {"qdir", "pid"}. -> {ok, output, id (the new queue dir's name)}."""
    try:
        r = qf.run_cli("queue", "run", *args, "--detach", "--json", timeout=120)
    except subprocess.TimeoutExpired:
        raise UserError("the start is still setting up; check the queue list before trying again")
    lines = r.stdout.strip().splitlines()
    try:
        qdir = json.loads(lines[-1])["qdir"] if r.returncode == 0 else None
    except (IndexError, ValueError, KeyError, TypeError):
        qdir = None
    return {"ok": r.returncode == 0, "output": qf.output_tail(r, 40),
            "id": pathlib.Path(qdir).name if qdir else None}


def start(body: dict) -> dict:
    with _START:
        qfile, flags = _source(body)
        if not _TOKENS.spend(body.get("token"), _digest(qfile, flags)):
            raise UserError("this start was not confirmed, or the queue changed since; review it again")
        errs = qsteps.validate(json.loads(qfile.read_text()))
        if errs:
            raise UserError("queue file invalid: " + "; ".join(errs))
        return _cli(str(qfile), *flags)


def stop(body: dict) -> dict:
    """SIGTERM the runner: it stops the running step, records `stopped`, and
    the queue resumes from the step that did not pass. Only a runner whose
    command line names this queue is signalled (queue/report.py runner_pid)."""
    d = _qdir(str(body.get("id") or ""))
    pid = qreport.runner_pid(d, qf.load_state(d), strict=True)
    if pid is None:
        raise UserError("that queue has no runner started from here or by `--detach` "
                        "on this machine; stop a foreground run in its terminal")
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as e:
        raise UserError(f"could not stop the runner: {e}") from e
    return {"ok": True}


def resume(body: dict) -> dict:
    """Resume a halted queue of this machine, never one any `queue run`
    process may still own (a foreground run's included)."""
    with _START:
        d = _qdir(str(body.get("id") or ""))
        s = qf.load_state(d)
        if not qreport.ran_here(s):
            raise UserError(f"that queue ran on {s.get('host')}; resume it there")
        if s.get("status") not in RESUMABLE or qreport.runner_pid(d, s, strict=False) is not None:
            raise UserError(f"a {s.get('status')} queue does not resume")
        return _cli("--resume", str(d))
