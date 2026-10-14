"""`vqlab queue status | list | wait`: read queue state, never change it.

Judged from state.json (the runner writes it after each step's
verdict.json), never from log wording.
"""
from __future__ import annotations

import pathlib
import re
import time

from vqlab import config
from vqlab.core.box.gpu_lease import pid_alive, pid_cmdline
from vqlab.queue import eta as queue_eta
from vqlab.queue.files import latest, load_state, queue_dirs, queues_dir
from vqlab.util.clock import now_iso

TERMINAL = ("passed", "failed", "stopped", "deferred")


def ran_here(state: dict) -> bool:
    """A pid is only meaningful on the box that wrote it; a state without a
    host predates the field and was written here."""
    return "host" not in state or config.is_this_host(state["host"])


def _names_qdir(cmd: str, qdir, state: dict) -> bool:
    """Whether a command line runs exactly this queue dir: `--resume DIR`
    (as `--detach` relaunches it) with DIR as written or resolved, never a
    name that merely starts the same (X vs X-preflight)."""
    forms = {str(qdir), str(pathlib.Path(qdir).resolve())}
    if state.get("qdir"):
        forms.add(str(state["qdir"]))
    return any(re.search(r"(?:^|\s)--resume[\s=]" + re.escape(f) + r"(?:\s|$)", cmd) for f in forms)


def runner_pid(qdir, state: dict, *, strict: bool) -> int | None:
    """The pid of a live runner of this queue on this machine, else None (a
    dead runner's pid can be reused by anything, so a live pid alone is not
    enough). strict: its command line runs this queue dir, as a `--detach`
    or `--resume` runner's does -- the only kind safe to signal; a command
    line `ps` could not read is not. Loose: any `queue run` process, which
    includes a foreground `queue run FILE` (its command line names the file,
    not the dir), and a live pid whose command line `ps` could not read --
    loose is the safe side for "may it still be running"."""
    pid = state.get("pid")
    if not (ran_here(state) and pid_alive(pid)):
        return None
    cmd = pid_cmdline(pid)
    if cmd is None:
        return None if strict else int(pid)
    if "queue run" not in cmd or (strict and not _names_qdir(cmd, qdir, state)):
        return None
    return int(pid)


def live_status(qdir, state: dict) -> str:
    """state.json's status, except a `running` queue of this machine whose
    runner is gone is `died`: its pid is dead, or `ps` read its command line
    and it is not a `queue run` (another box's queue is judged by its state)."""
    st = state["status"]
    if st == "running" and ran_here(state) and runner_pid(qdir, state, strict=False) is None:
        return "died"
    return st


def list_queues() -> int:
    for p in queue_dirs():
        s = load_state(p)
        n = sum(r["status"] == "pass" for r in s["steps"])
        print(f"{p.name:48s} {s['status']:9s} {n}/{len(s['steps'])}  @ {s['commit'][:10]}")
    return 0


def _step_line(i, r, e) -> str:
    return (f"  {i:2d} {r['status']:8s} {r['name'][:40]:40s} "
            f"{'' if r.get('seconds') is None else str(r['seconds']) + ' s':>10s} "
            f"{'x' + str(r['attempts']) if (r.get('attempts') or 1) > 1 else ''}"
            + (f"  {queue_eta.fmt(e['eta_s'])} ({e['basis']})"
               if r["status"] in ("running", "pending") else ""))


def status(qdir) -> int:
    qdir = pathlib.Path(qdir) if qdir else latest()
    if not qdir:
        print(f"no queues under {queues_dir()}")
        return 1
    s = load_state(qdir)
    alive = runner_pid(qdir, s, strict=False) is not None
    eta = queue_eta.queue_eta(qdir, queues_dir())
    print(f"{s['name']}  {s['status']}{' (runner alive)' if alive else ''}  "
          f"@ {s['commit'][:10]}  {qdir}")
    if s.get("reservation_override"):
        print(f"  ran over a reservation: {s['reservation_override']}")
    for i, r in enumerate(s["steps"]):
        print(_step_line(i, r, eta["steps"][i]))
        for x in r.get("reasons", []) + r.get("warnings", []):
            print(f"       {x}")
    if s["status"] in ("running", "created"):
        print(f"  queue: {queue_eta.fmt(eta['eta_s'])}"
              + (f", done ~{eta['finish']}" if eta["finish"] else
                 " (a step has neither progress lines nor a past run)"))
    return 0


def wait(qdirs, timeout=None, poll=15.0) -> int:
    """Block until every queue reaches a terminal state. A runner that died
    mid-queue is terminal too. Exit 0 only if every queue passed; 4 if any
    failed/stopped/deferred/died; 2 on timeout."""
    qdirs = [pathlib.Path(q) for q in qdirs] or [latest()]
    t0, seen = time.time(), {}
    while True:
        states = {}
        for q in qdirs:
            s = load_state(q)
            st = states[q] = live_status(q, s)
            if seen.get(q) != st:
                n = sum(r["status"] == "pass" for r in s["steps"])
                print(f"{now_iso()}  {q.name}: {st} ({n}/{len(s['steps'])} passed)", flush=True)
                seen[q] = st
        if all(v in TERMINAL + ("died",) for v in states.values()):
            return 0 if all(v == "passed" for v in states.values()) else 4
        if timeout and time.time() - t0 > timeout:
            print(f"timed out after {timeout} s", flush=True)
            return 2
        time.sleep(poll)
