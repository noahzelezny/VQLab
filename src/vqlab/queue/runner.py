#!/usr/bin/env python3
"""vqlab queue — run a list of steps from PINNED code, under the GPU lease,
failing loudly.

    vqlab queue run <queue.json> [--commit REV] [--preflight] [--detach]
    vqlab queue run --resume <queue dir>      skip steps that already passed
    vqlab queue run <queue.json> --on m4      another box: reservation + remote-path check first
    (a box reserved for someone else by `vqlab reserve` is refused unless
    --override-reservation, which state.json records)
    vqlab queue status [<queue dir>]          default: the latest queue
    vqlab queue list

WHY (night 4, 2026-09-26, run by hand): every step imported vqlab from the
live checkout, so the tree was frozen all night; a speed step piped stderr to
/dev/null, died on a stale data path, and was logged "done: 0 runs"; the
lease was a pgrep script edited twice mid-night; builds had no GPU-timeout
retry. This runner fixes each:

* PINNED CODE. The queue resolves one commit (default HEAD, refusing a dirty
  src/ or scripts/ unless --allow-dirty) and checks out a detached git
  worktree of it in the queue dir. Every step runs from that tree
  (cwd + PYTHONPATH), so anyone may commit mid-run, and every step's line in
  the run log carries that commit. families/ writes go to the LIVE repo
  (VQLAB_FAMILIES_DIR), since that is shared state, not code.
* LOUD FAILURE. Each step's stdout and stderr are kept in steps/NN-name/ and
  judged by records/step_verdict.py: nonzero exit, a traceback, or missing
  expected output is a FAIL, and the default queue stops on it.
  stdout must be non-empty unless the step's `expect` says otherwise.
* ONE LEASE. The queue takes the SAME GPU lease MCP `run` uses (one
  protocol, core/gpu_lease.acquire_lease) for its whole length when any
  step needs the GPU, and re-checks for a placed exo instance before each
  GPU step.
* RETRY. Resumable builds (fit-moe, fit-dense, geo-build, alloc-sweep,
  validate) retry after a failure (GPU timeouts under disk contention); each
  attempt's output is kept.
* PINS. A scoring step refuses any artifact dir whose vqlab_pin.json does
  not allow it (gate/pin.py check_pin).
* PREFLIGHT. `--preflight` runs every step for real on its `preflight`
  arguments (the first cell / module / a few tokens), then stops. Before
  that it checks every input PATH a step names -- in its args, and string
  literals in a by-path script's source -- exists IN THE PINNED TREE. A
  step with no `preflight` block fails preflight: saying how to run a small
  real version is part of writing the step.

Queue file (JSON):

    {"name": "night5", "commit": "HEAD", "python": "/path/to/env/bin/python",
     "steps": [
       {"name": "kl-27b", "cmd": "kl-ladder", "args": ["--teacher-cache", "...", "--rung", "..."],
        "expect": {"stdout_regex": "paired", "files": ["<scratch>/out.json"]},
        "preflight": {"append": ["--preflight"]}},
       {"name": "speed", "script": "scripts/speed_pair.py", "args": ["--n", "3"],
        "preflight": {"args": ["--n", "1"]}, "retries": 0, "on_fail": "continue",
        "timeout_s": 3600}
     ]}

`cmd` is a vqlab subcommand, flat (`fit-moe`) or namespaced (`fit moe`; stored
flat); `script` is a path relative to the repo (run
from the pinned tree). `python` (queue or step) picks the interpreter, e.g.
the exo env. `preflight` is {"args": [...]} (replace), {"append": [...]} or
{"skip": "reason"}, or {"full": "reason"} to run the real step full-size on
purpose (checked against free scratch first -- its input size, or the
"writes_gib" it declares, e.g. 0 for a pin; an identical {"append": []} is
refused). Any non-skip preflight may declare "writes_gib" (a one-layer
fit-moe still rewrites every base shard), and scratch is checked against it. A passing preflight deletes its outputs unless --keep-preflight. geo-build and kl-ladder take `--preflight` (first
module / first cache x rung), so {"append": ["--preflight"]} is their block. In preflight, every OUTPUT flag's value (--out,
--out-dir, ...) is redirected to <scratch SSD>/queue-preflight/<queue>/<step>/<name>, so a
preflight never writes into a real step's output; and a later step that names
an earlier step's real output is pointed at that step's PREFLIGHT output, so
chained steps (pin <- build) preflight end to end. A `name=path` argument
(kl-ladder --cache prose=/..., --rung r39=/...) is checked as its path.

Module map (untangle step 12, 2026-10-10; was agents/run_queue.py):
  queue/runner.py  this file: the run loop, one step's attempts, the CLI
  queue/steps.py   step schema, preflight args, input-path and pin checks
  queue/files.py   queue dirs, state.json, creating (pinning) a queue
  queue/report.py  status / list / wait
  queue/remote.py  --on BOX
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import time

from vqlab import config as _config
from vqlab.core.box import exo
from vqlab.core.box import gpu_lease
from vqlab.queue import report
from vqlab.queue import reserve as resv
from vqlab.queue import steps as _steps
from vqlab.queue.files import (REPO, _now, cli_launch, create, git, load_queue, load_state,
                               package_hash, save_state)
from vqlab.queue.remote import run_on
from vqlab.records import step_verdict as sv
from vqlab.util.clock import slug

_stop = {"flag": False, "proc": None}


def _term(_s, _f):
    _stop["flag"] = True
    p = _stop["proc"]
    if p and p.poll() is None:
        p.terminate()


# ------------------------------------------------------------------ one step
def _step_args(st, state, preflight, outmap) -> tuple[list[str] | None, dict | None]:
    """(args to run with, None), or (None, verdict) when it must not run."""
    tree = pathlib.Path(state["tree"])
    # <teachers>/X: this box's root, so static_check sees the real path
    args = [_config.expand(str(a)) for a in st.get("args", [])]
    if preflight:
        return _steps.preflight_args(st, args, state, outmap)
    _steps.make_out_parents(args, tree)
    return args, None


def _gate(st, tree, args, force) -> tuple[dict | None, list[str]]:
    """(failing verdict or None, warnings): input paths, pins, exo placement."""
    probs = _steps.static_check(st, tree, args)
    refuse, warns = _steps.pin_check(st, args)
    if probs or refuse:
        return {"verdict": "fail", "reasons": probs + refuse, "warnings": warns}, warns
    if _steps.needs_gpu(st) and not force:
        placed = exo.exo_instances()
        if placed:
            return {"verdict": "fail", "reasons": [f"exo instance placed on this cluster: {placed}"],
                    "warnings": warns}, warns
    return None, warns


def _spawn(argv, tree, env, sdir, timeout_s, rec) -> int:
    """One attempt, stdout/stderr into sdir; its exit code."""
    with open(sdir / "stdout", "wb") as o, open(sdir / "stderr", "wb") as e:
        p = subprocess.Popen(argv, cwd=str(tree), env=env, stdin=subprocess.DEVNULL,
                             stdout=o, stderr=e, start_new_session=True)
        _stop["proc"] = p
        rec["pid"] = p.pid
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                rc = p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
            rc = rc or -9
        _stop["proc"] = None
    return rc


def _attempts(st, rec, sdir, state, argv, expect, retries) -> dict:
    """Run the step until it passes, is stopped, or its retries run out."""
    tree = pathlib.Path(state["tree"])
    env = _steps.env(state)
    verdict, attempt = None, 0
    while attempt <= retries and not _stop["flag"]:
        attempt += 1
        for f in ("stdout", "stderr"):
            if (sdir / f).exists():
                (sdir / f).rename(sdir / f"{f}.attempt{attempt - 1}")
        rec.update(status="running", attempt=attempt, started=rec.get("started") or _now())
        save_state(pathlib.Path(state["qdir"]), state)
        t0 = time.time()
        rc = _spawn(argv, tree, env, sdir, st.get("timeout_s"), rec)
        verdict = sv.step_verdict(rc, sdir / "stdout", sdir / "stderr", expect)
        verdict.update(rc=rc, seconds=round(time.time() - t0, 1), attempt=attempt)
        if st.get("timeout_s") and verdict["seconds"] >= st["timeout_s"]:
            verdict["reasons"].append(f"timed out after {st['timeout_s']} s")
            verdict.update(ok=False, verdict="fail")
        if verdict["ok"] or _stop["flag"]:
            break
        if attempt <= retries:
            time.sleep(int(os.environ.get("VQLAB_QUEUE_RETRY_SLEEP", "60")))
    return verdict


def run_step(q, st, rec, sdir, state, preflight, force, outmap=None) -> dict:
    """Check, then run, one step; its verdict dict."""
    tree = pathlib.Path(state["tree"])
    args, early = _step_args(st, state, preflight, outmap)
    if early:
        return early
    bad, warns = _gate(st, tree, args, force)
    if bad:
        return bad
    argv = _steps.argv(q, st, args, tree)
    (sdir / "cmd").write_text(" ".join(shlex.quote(a) for a in argv) + "\n")
    verdict = _attempts(st, rec, sdir, state, argv, _steps.expect(st, tree, preflight),
                        _steps.retries(st, preflight))
    verdict["warnings"] = warns
    if _stop["flag"] and not verdict.get("ok"):
        verdict["verdict"] = "stopped"
    return verdict


# ------------------------------------------------------------------ the queue
def _check_pin(state):
    """The pinned tree must still be what the queue was created at."""
    tree = pathlib.Path(state["tree"])
    if not tree.is_dir():
        raise SystemExit(f"pinned tree is gone: {tree}")
    if state.get("installed"):
        now = package_hash()
        if not state["commit"].endswith(":" + now):
            raise SystemExit(f"installed vqlab changed since the queue was created "
                             f"({state['commit']} -> {now}); create the queue again")
    else:
        head = git("rev-parse", "HEAD", cwd=tree)
        if head != state["commit"]:
            raise SystemExit(f"pinned tree moved: HEAD {head[:10]} != {state['commit'][:10]}")


def _print_step(st, v, status):
    print(f"  {status.upper():8s} {st['name']}"
          + (f"  ({v.get('seconds')} s)" if v.get("seconds") is not None else ""), flush=True)
    for r in v.get("reasons", []) + v.get("warnings", []):
        print(f"           {r}", flush=True)
    for ln in (v.get("stderr_tail") or [])[-4:] if status == "fail" else []:
        print(f"           | {ln}", flush=True)


def _drain(qdir, q, state, todo, preflight, force) -> bool:
    """Run the pending steps in order; True if any failed or was stopped."""
    failed = False
    outmap = {}                 # preflight: real output path -> preflight output path
    for i in todo:
        st, rec = q["steps"][i], state["steps"][i]
        sdir = qdir / "steps" / f"{i:02d}-{slug(st['name'])}"
        sdir.mkdir(parents=True, exist_ok=True)
        v = run_step(q, st, rec, sdir, state, preflight, force, outmap)
        (sdir / "verdict.json").write_text(json.dumps(v, indent=1))
        status = {"pass": "pass"}.get(v["verdict"], v["verdict"])
        rec.update(status=status, finished=_now(), rc=v.get("rc"),
                   seconds=v.get("seconds"), attempts=v.get("attempt"),
                   reasons=v.get("reasons", []), warnings=v.get("warnings", []))
        rec.pop("pid", None)
        save_state(qdir, state)
        _print_step(st, v, status)
        if status in ("fail", "stopped"):
            failed = True
            # preflight reports EVERY problem in one pass; a real run stops
            if status == "stopped" or (not preflight and st.get("on_fail", "stop") == "stop"):
                break
    return failed


def _tidy(qdir, state, done, preflight, keep_preflight):
    """After the drain: preflight outputs, and the pinned worktree."""
    pfroot = None
    if preflight:
        try:
            pfroot = _config.preflight_dir() / qdir.name
        except SystemExit:
            pass
    if preflight and done and pfroot and pfroot.is_dir() and not keep_preflight:
        # preflight outputs are proof the steps RUN, not artifacts: a passed
        # preflight leaves nothing on scratch (a failed one keeps its outputs
        # for inspection)
        shutil.rmtree(pfroot, ignore_errors=True)
        state["preflight_outputs"] = "removed after pass"
        save_state(qdir, state)
    elif preflight and pfroot and pfroot.is_dir():
        state["preflight_outputs"] = str(pfroot)
        save_state(qdir, state)
    if done and not preflight and not state.get("installed"):
        subprocess.run(["git", "worktree", "remove", "--force", state["tree"]], cwd=REPO,
                       capture_output=True)
        state["tree_removed"] = True
        save_state(qdir, state)


def run(qdir: pathlib.Path, force=False, lease_wait=1800, keep_preflight=False) -> int:
    """Drain one queue dir (the runner process). 0 all passed, 4 a step
    failed, 3 the GPU lease stayed held, 1 otherwise unfinished."""
    state, q = load_state(qdir), load_queue(qdir)
    state["qdir"] = str(qdir)
    _check_pin(state)
    preflight = state.get("preflight", False)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    todo = [i for i, r in enumerate(state["steps"]) if r["status"] not in ("pass", "skipped")]
    fd = None
    if any(_steps.needs_gpu(q["steps"][i]) for i in todo):
        fd = gpu_lease.acquire_lease(gpu_lease.lease_path(), f"vqlab:queue:{state['name']}",
                                     os.environ.get("VQLAB_RUN_ID"), wait_s=lease_wait)
        if fd is None:
            state.update(status="deferred", note=f"GPU lease held for {lease_wait} s; nothing ran")
            save_state(qdir, state)
            print(f"deferred: the GPU lease ({gpu_lease.lease_path()}) stayed held for {lease_wait} s")
            return 3
    state.update(status="running", started=state.get("started") or _now(),
                 queue_run_id=os.environ.get("VQLAB_RUN_ID"), pid=os.getpid())
    save_state(qdir, state)
    print(f"queue {state['name']} @ {state['commit'][:10]}  ({qdir})"
          + ("  PREFLIGHT" if preflight else ""), flush=True)
    try:
        failed = _drain(qdir, q, state, todo, preflight, force)
    finally:
        if fd is not None:
            gpu_lease.release_lease(fd)
    done = all(r["status"] in ("pass", "skipped") for r in state["steps"])
    state.update(status="stopped" if _stop["flag"] else ("passed" if done else "failed"),
                 finished=_now())
    state.pop("pid", None)
    save_state(qdir, state)
    _tidy(qdir, state, done, preflight, keep_preflight)
    print(f"queue {state['status']}: "
          f"{sum(r['status'] == 'pass' for r in state['steps'])}/{len(state['steps'])} passed", flush=True)
    return 0 if done else (4 if failed else 1)


# ------------------------------------------------------------------ CLI
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vqlab queue", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("file", nargs="?")
    pr.add_argument("--resume", help="an existing queue dir: re-run what did not pass")
    pr.add_argument("--commit", help="pin this revision (default: the file's `commit`, else HEAD)")
    pr.add_argument("--allow-dirty", action="store_true",
                    help="run HEAD even though src/ or scripts/ has uncommitted changes")
    pr.add_argument("--preflight", action="store_true",
                    help="run every step's small REAL version, then stop")
    pr.add_argument("--detach", action="store_true", help="run in its own session; survives the shell")
    pr.add_argument("--force", action="store_true", help="ignore a placed exo instance")
    pr.add_argument("--lease-wait", type=int, default=1800)
    pr.add_argument("--on", help="run on another box from [boxes.NAME] in the vqlab config")
    pr.add_argument("--override-reservation", action="store_true",
                    help="run on a box reserved for someone else (`vqlab reserve`); recorded in state.json")
    pr.add_argument("--keep-preflight", action="store_true",
                    help="keep preflight outputs after a passing preflight (default: removed)")
    ps = sub.add_parser("status")
    ps.add_argument("qdir", nargs="?")
    sub.add_parser("list")
    pw = sub.add_parser("wait", help="block until the queue(s) finish; exit 0 only if all passed")
    pw.add_argument("qdirs", nargs="*")
    pw.add_argument("--timeout", type=float)
    pw.add_argument("--poll", type=float, default=15.0)
    return ap


def _record_override(qdir, held):
    state = load_state(qdir)
    state["reservation_override"] = {**{k: held.get(k) for k in ("box", "for", "until", "note", "by")},
                                     "overridden_by": resv.who(), "at": _now()}
    save_state(qdir, state)
    print(f"overriding: {resv.describe(held)}", flush=True)


def _detach(qdir, a) -> int:
    """Re-launch this queue as `queue run --resume` in its own session."""
    pre, env, cwd = cli_launch()
    cmd = pre + ["queue", "run", "--resume", str(qdir), "--lease-wait", str(a.lease_wait)] \
        + (["--force"] if a.force else []) \
        + (["--override-reservation"] if a.override_reservation else []) \
        + (["--keep-preflight"] if a.keep_preflight else [])
    p = subprocess.Popen(cmd, cwd=cwd, env=env,
                         stdin=subprocess.DEVNULL, stdout=open(qdir / "queue.log", "ab"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    print(f"detached (pid {p.pid}); vqlab queue status {qdir}")
    return 0


def _cmd_run(ap, a) -> int:
    if bool(a.file) == bool(a.resume):
        ap.error("give a queue file, or --resume <queue dir>")
    if a.on:
        if not a.file:
            ap.error("--on takes a queue file")
        return run_on(a.on, a.file, a.preflight, a.force, a.override_reservation)
    why = resv.refusal("here", a.override_reservation)
    if why:
        raise SystemExit(why)
    held = resv.get("here") or resv.get_by_host()
    qdir = pathlib.Path(a.resume) if a.resume else create(a.file, a.commit, a.allow_dirty, a.preflight)
    if a.override_reservation and held and held.get("for") != resv.who():
        _record_override(qdir, held)
    if a.detach:
        return _detach(qdir, a)
    return run(qdir, force=a.force, lease_wait=a.lease_wait, keep_preflight=a.keep_preflight)


def main(argv=None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    if a.cmd == "list":
        return report.list_queues()
    if a.cmd == "status":
        return report.status(a.qdir)
    if a.cmd == "wait":
        return report.wait(a.qdirs, a.timeout, a.poll)
    return _cmd_run(ap, a)


if __name__ == "__main__":
    sys.exit(main())
