"""`vqlab queue status | list | wait`: read queue state, never change it.

Judged from state.json (the runner writes it after each step's
verdict.json), never from log wording.
"""
from __future__ import annotations

import pathlib
import socket
import time

from vqlab.core.gpu_lease import pid_alive
from vqlab.queue import eta as queue_eta
from vqlab.queue.files import _now, latest, load_state, queue_dirs, queues_dir

TERMINAL = ("passed", "failed", "stopped", "deferred")


def list_queues() -> int:
    for p in queue_dirs():
        s = load_state(p)
        n = sum(r["status"] == "pass" for r in s["steps"])
        print(f"{p.name:48s} {s['status']:9s} {n}/{len(s['steps'])}  @ {s['commit'][:10]}")
    return 0


def _step_line(i, r, e) -> str:
    return (f"  {i:2d} {r['status']:8s} {r['name'][:40]:40s} "
            f"{'' if r.get('seconds') is None else str(r['seconds']) + ' s':>10s} "
            f"{'x' + str(r['attempts']) if r.get('attempts', 1) > 1 else ''}"
            + (f"  {queue_eta.fmt(e['eta_s'])} ({e['basis']})"
               if r["status"] in ("running", "pending") else ""))


def status(qdir) -> int:
    qdir = pathlib.Path(qdir) if qdir else latest()
    if not qdir:
        print(f"no queues under {queues_dir()}")
        return 1
    s = load_state(qdir)
    alive = pid_alive(s.get("pid"))
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


def _live_status(q: pathlib.Path, s: dict) -> str:
    st = s["status"]
    # a pid is only meaningful on the box that wrote it: a queue run
    # --on another box is judged by its state.json alone
    here = socket.gethostname().split(".")[0]
    if st == "running" and s.get("host", here) == here and not pid_alive(s.get("pid")):
        st = "died"
    return st


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
            st = states[q] = _live_status(q, s)
            if seen.get(q) != st:
                n = sum(r["status"] == "pass" for r in s["steps"])
                print(f"{_now()}  {q.name}: {st} ({n}/{len(s['steps'])} passed)", flush=True)
                seen[q] = st
        if all(v in TERMINAL + ("died",) for v in states.values()):
            return 0 if all(v == "passed" for v in states.values()) else 4
        if timeout and time.time() - t0 > timeout:
            print(f"timed out after {timeout} s", flush=True)
            return 2
        time.sleep(poll)
