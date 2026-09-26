#!/usr/bin/env python3
"""vqlab queue — run a list of steps from PINNED code, under the GPU lease,
failing loudly.

    vqlab queue run <queue.json> [--commit REV] [--preflight] [--detach]
    vqlab queue run --resume <queue dir>      skip steps that already passed
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
  protocol, agents/mcp_server.acquire_lease) for its whole length when any
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

    {"name": "night5", "commit": "HEAD", "python": "/opt/anaconda3/envs/exo/bin/python",
     "steps": [
       {"name": "kl-27b", "cmd": "kl-ladder", "args": ["--teacher-cache", "...", "--rung", "..."],
        "expect": {"stdout_regex": "paired", "files": ["/Volumes/.../out.json"]},
        "preflight": {"append": ["--max-chunks", "1"]}},
       {"name": "speed", "script": "scripts/speed_pair.py", "args": ["--n", "3"],
        "preflight": {"args": ["--n", "1"]}, "retries": 0, "on_fail": "continue",
        "timeout_s": 3600}
     ]}

`cmd` is a vqlab subcommand; `script` is a path relative to the repo (run
from the pinned tree). `python` (queue or step) picks the interpreter, e.g.
the exo env. `preflight` is {"args": [...]} (replace), {"append": [...]} or
{"skip": "reason"}.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import mcp_server as ms  # noqa: E402
import step_verdict as sv  # noqa: E402
import pin as pinmod  # noqa: E402

REPO = _layout.SRC.parent
SCHEMA = "vqlab.queue/1"
SCORING = {"score", "kl", "kl-ladder", "kl-pair", "tasks", "decode-timeline",
           "decode-ladder", "active-bytes"}
# Flags whose value is an OUTPUT path: it need not exist yet, its parent must.
OUT_FLAGS = {"--out", "--out-dir", "--output", "-o", "--per-pos-dir", "--pool",
             "--parts-dir", "--save", "--log", "--json-out", "--cache-out"}
_PATHLIKE = re.compile(r"""["'](/(?:Volumes|Users|opt|tmp|private)/[^"'\n]+|(?:src|scripts|research|families)/[^"'\s\n]+)["']""")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def queues_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("VQLAB_QUEUE_DIR") or pathlib.Path.home() / ".vqlab" / "queues")


def _git(*a, cwd=REPO):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout.strip()


def _save(qdir, state):
    tmp = qdir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(qdir / "state.json")


def _slug(s):
    return re.sub(r"[^\w.-]", "_", s)[:40]


def validate(q) -> list[str]:
    errs = []
    if not isinstance(q.get("steps"), list) or not q["steps"]:
        return ["queue has no steps"]
    names = set()
    for i, st in enumerate(q["steps"]):
        tag = f"step {i} ({st.get('name', '?')})"
        if not st.get("name"):
            errs.append(f"{tag}: no name")
        elif st["name"] in names:
            errs.append(f"{tag}: duplicate name")
        names.add(st.get("name"))
        if bool(st.get("cmd")) == bool(st.get("script")):
            errs.append(f"{tag}: give exactly one of cmd / script")
        if st.get("cmd") == "publish":
            errs.append(f"{tag}: publish is a human action, never queued")
        if st.get("on_fail", "stop") not in ("stop", "continue"):
            errs.append(f"{tag}: on_fail must be stop or continue")
        pf = st.get("preflight")
        if pf is not None and not (isinstance(pf, dict) and len(pf) == 1
                                   and next(iter(pf)) in ("args", "append", "skip")):
            errs.append(f"{tag}: preflight must be one of {{args}}, {{append}}, {{skip}}")
    return errs


def _needs_gpu(st):
    return not (st.get("cmd") in ms.GPU_FREE or st.get("gpu") is False)


# ------------------------------------------------------------------ create
def create(qfile, commit=None, allow_dirty=False, preflight=False):
    q = json.loads(pathlib.Path(qfile).read_text())
    errs = validate(q)
    if errs:
        raise SystemExit("queue file invalid:\n  " + "\n  ".join(errs))
    rev = commit or q.get("commit") or "HEAD"
    if rev == "HEAD" and not allow_dirty:
        dirty = _git("status", "--porcelain", "--", "src", "scripts")
        if dirty:
            raise SystemExit(
                "uncommitted changes in src/ or scripts/ would NOT be in the pinned tree:\n"
                + dirty + "\ncommit them first, or pass --allow-dirty to run HEAD without them")
    sha = _git("rev-parse", "--verify", rev + "^{commit}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    qdir = queues_dir() / f"{stamp}-{_slug(q.get('name') or pathlib.Path(qfile).stem)}{'-preflight' if preflight else ''}"
    qdir.mkdir(parents=True, exist_ok=False)
    (qdir / "queue.json").write_text(json.dumps(q, indent=1))
    _git("worktree", "add", "--detach", str(qdir / "tree"), sha)
    state = {"schema": SCHEMA, "name": q.get("name"), "source": str(pathlib.Path(qfile).resolve()),
             "commit": sha, "tree": str(qdir / "tree"), "live_repo": str(REPO),
             "host": socket.gethostname().split(".")[0], "preflight": preflight,
             "created": _now(), "status": "created", "queue_run_id": None,
             "steps": [{"name": st["name"], "status": "pending"} for st in q["steps"]]}
    _save(qdir, state)
    return qdir


# ------------------------------------------------------------------ checks
def _path_args(args):
    """(inputs, outputs) among a step's args: values that look like paths."""
    ins, outs = [], []
    prev = None
    for a in args:
        if a.startswith("-"):
            prev = a.split("=", 1)[0]
            if "=" in a:
                v = a.split("=", 1)[1]
                (outs if prev in OUT_FLAGS else ins).append(v) if "/" in v else None
            continue
        if "/" in a and not a.startswith(("http:", "https:")):
            (outs if prev in OUT_FLAGS else ins).append(a)
        prev = None
    return ins, outs


def static_check(st, tree: pathlib.Path, args) -> list[str]:
    """Every input path a step names must exist IN THE PINNED TREE (or be
    absolute and exist). The night-4 miss was a DATA path inside a script, so
    a by-path script's string literals are checked too."""
    probs = []
    ins, outs = _path_args(args)

    def exists(p):
        pp = pathlib.Path(p)
        return (pp if pp.is_absolute() else tree / pp).exists()

    for p in ins:
        if not exists(p):
            probs.append(f"input path does not exist: {p}")
    for p in outs:
        pp = pathlib.Path(p)
        par = (pp if pp.is_absolute() else tree / pp).parent
        if not par.exists():
            probs.append(f"output's parent dir does not exist: {p}")
    if st.get("script"):
        sp = tree / st["script"]
        if not sp.is_file():
            probs.append(f"script not in the pinned tree: {st['script']}")
        else:
            for lit in sorted(set(_PATHLIKE.findall(sp.read_text(errors="replace")))):
                if "{" in lit or "*" in lit:
                    continue                       # a template, not a path
                if not exists(lit) and not pathlib.Path(lit).parent.exists():
                    probs.append(f"{st['script']} names a path that does not exist: {lit}")
    return probs


def pin_check(st, args) -> tuple[list[str], list[str]]:
    """(refusals, warnings) for the artifact dirs a scoring step names."""
    if st.get("cmd") not in SCORING:
        return [], []
    ref, warn = [], []
    for a in _path_args(args)[0]:
        d = pathlib.Path(a)
        if d.is_dir() and (d / "config.json").exists():
            ok, msg = pinmod.check_pin(d)
            if not ok:
                ref.append(msg)
            elif msg:
                warn.append(msg)
            elif not (d / pinmod.MARKER).exists():
                warn.append(f"NOTE {d}: not a pin (AGENTS.md: pin, then smoke, then measure)")
    return ref, warn


# ------------------------------------------------------------------ run
_stop = {"flag": False, "proc": None}


def _term(_s, _f):
    _stop["flag"] = True
    p = _stop["proc"]
    if p and p.poll() is None:
        p.terminate()


def _argv(q, st, args, tree):
    py = st.get("python") or q.get("python") or sys.executable
    if st.get("cmd"):
        return [py, "-m", "vqlab.cli", st["cmd"], *args]
    return [py, str(tree / st["script"]), *args]


def _env(state):
    env = dict(os.environ)
    tree = state["tree"]
    env["PYTHONPATH"] = str(pathlib.Path(tree) / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONPYCACHEPREFIX", str(pathlib.Path.home() / ".pycache-vqlab"))
    env.setdefault("VQLAB_FAMILIES_DIR", str(pathlib.Path(state["live_repo"]) / "families"))
    env["VQLAB_QUEUE"] = state["qdir"]
    env["VQLAB_QUEUE_COMMIT"] = state["commit"]
    return env


def _run_step(q, st, rec, sdir, state, preflight, force):
    tree = pathlib.Path(state["tree"])
    args = [str(a) for a in st.get("args", [])]
    if preflight:
        pf = st.get("preflight")
        if pf is None:
            return {"verdict": "fail", "reasons": [
                "no preflight defined: add preflight.args / preflight.append (a small REAL "
                "run: first cell, one module, a few tokens) or preflight.skip with a reason"]}
        if "skip" in pf:
            return {"verdict": "skipped", "reasons": [f"preflight skipped: {pf['skip']}"]}
        args = [str(a) for a in pf["args"]] if "args" in pf else args + [str(a) for a in pf["append"]]
    probs = static_check(st, tree, args)
    refuse, warns = pin_check(st, args)
    if probs or refuse:
        return {"verdict": "fail", "reasons": probs + refuse, "warnings": warns}
    if _needs_gpu(st) and not force:
        placed = ms._exo_instances()
        if placed:
            return {"verdict": "fail", "reasons": [f"exo instance placed on this cluster: {placed}"],
                    "warnings": warns}
    retries = st.get("retries")
    if retries is None:
        retries = ms.DEFAULT_RETRIES if st.get("cmd") in ms.RESUMABLE and not preflight else 0
    expect = dict(st.get("expect") or {"stdout_nonempty": True})
    expect["files"] = [str(f if pathlib.Path(f).is_absolute() else tree / f)
                       for f in expect.get("files", ())] if not preflight else []
    argv = _argv(q, st, args, tree)
    (sdir / "cmd").write_text(" ".join(shlex.quote(a) for a in argv) + "\n")
    env = _env(state)
    verdict, attempt = None, 0
    while attempt <= retries and not _stop["flag"]:
        attempt += 1
        for f in ("stdout", "stderr"):
            if (sdir / f).exists():
                (sdir / f).rename(sdir / f"{f}.attempt{attempt - 1}")
        rec.update(status="running", attempt=attempt, started=rec.get("started") or _now())
        _save(pathlib.Path(state["qdir"]), state)
        t0 = time.time()
        with open(sdir / "stdout", "wb") as o, open(sdir / "stderr", "wb") as e:
            p = subprocess.Popen(argv, cwd=str(tree), env=env, stdin=subprocess.DEVNULL,
                                 stdout=o, stderr=e, start_new_session=True)
            _stop["proc"] = p
            rec["pid"] = p.pid
            try:
                rc = p.wait(timeout=st.get("timeout_s"))
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    rc = p.wait(60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = p.wait()
                rc = rc or -9
            _stop["proc"] = None
        verdict = sv.step_verdict(rc, sdir / "stdout", sdir / "stderr", expect)
        verdict.update(rc=rc, seconds=round(time.time() - t0, 1), attempt=attempt)
        if st.get("timeout_s") and verdict["seconds"] >= st["timeout_s"]:
            verdict["reasons"].append(f"timed out after {st['timeout_s']} s")
            verdict.update(ok=False, verdict="fail")
        if verdict["ok"] or _stop["flag"]:
            break
        if attempt <= retries:
            time.sleep(int(os.environ.get("VQLAB_QUEUE_RETRY_SLEEP", "60")))
    verdict["warnings"] = warns
    if _stop["flag"] and not verdict.get("ok"):
        verdict["verdict"] = "stopped"
    return verdict


def run(qdir: pathlib.Path, force=False, lease_wait=1800) -> int:
    state = json.loads((qdir / "state.json").read_text())
    q = json.loads((qdir / "queue.json").read_text())
    state["qdir"] = str(qdir)
    tree = pathlib.Path(state["tree"])
    if not tree.is_dir():
        raise SystemExit(f"pinned tree is gone: {tree}")
    head = _git("rev-parse", "HEAD", cwd=tree)
    if head != state["commit"]:
        raise SystemExit(f"pinned tree moved: HEAD {head[:10]} != {state['commit'][:10]}")
    preflight = state.get("preflight", False)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    todo = [i for i, r in enumerate(state["steps"]) if r["status"] not in ("pass", "skipped")]
    fd = None
    if any(_needs_gpu(q["steps"][i]) for i in todo):
        fd = ms.acquire_lease(ms.lease_path(), f"vqlab:queue:{state['name']}",
                              os.environ.get("VQLAB_RUN_ID"), wait_s=lease_wait)
        if fd is None:
            state.update(status="deferred", note=f"GPU lease held for {lease_wait} s; nothing ran")
            _save(qdir, state)
            print(f"deferred: the GPU lease ({ms.lease_path()}) stayed held for {lease_wait} s")
            return 3
    state.update(status="running", started=state.get("started") or _now(),
                 queue_run_id=os.environ.get("VQLAB_RUN_ID"), pid=os.getpid())
    _save(qdir, state)
    print(f"queue {state['name']} @ {state['commit'][:10]}  ({qdir})"
          + ("  PREFLIGHT" if preflight else ""), flush=True)
    failed = False
    try:
        for i in todo:
            st, rec = q["steps"][i], state["steps"][i]
            sdir = qdir / "steps" / f"{i:02d}-{_slug(st['name'])}"
            sdir.mkdir(parents=True, exist_ok=True)
            v = _run_step(q, st, rec, sdir, state, preflight, force)
            (sdir / "verdict.json").write_text(json.dumps(v, indent=1))
            status = {"pass": "pass"}.get(v["verdict"], v["verdict"])
            rec.update(status=status, finished=_now(), rc=v.get("rc"),
                       seconds=v.get("seconds"), attempts=v.get("attempt"),
                       reasons=v.get("reasons", []), warnings=v.get("warnings", []))
            rec.pop("pid", None)
            _save(qdir, state)
            print(f"  {status.upper():8s} {st['name']}"
                  + (f"  ({v.get('seconds')} s)" if v.get("seconds") is not None else ""), flush=True)
            for r in v.get("reasons", []) + v.get("warnings", []):
                print(f"           {r}", flush=True)
            for ln in (v.get("stderr_tail") or [])[-4:] if status == "fail" else []:
                print(f"           | {ln}", flush=True)
            if status in ("fail", "stopped"):
                failed = True
                # preflight reports EVERY problem in one pass; a real run stops
                if status == "stopped" or (not preflight and st.get("on_fail", "stop") == "stop"):
                    break
    finally:
        if fd is not None:
            ms.release_lease(fd)
    done = all(r["status"] in ("pass", "skipped") for r in state["steps"])
    state.update(status="stopped" if _stop["flag"] else ("passed" if done else "failed"),
                 finished=_now())
    state.pop("pid", None)
    _save(qdir, state)
    if done and not preflight:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=REPO,
                       capture_output=True)
        state["tree_removed"] = True
        _save(qdir, state)
    print(f"queue {state['status']}: "
          f"{sum(r['status'] == 'pass' for r in state['steps'])}/{len(state['steps'])} passed", flush=True)
    return 0 if done else (4 if failed else 1)


# ------------------------------------------------------------------ report
def _latest():
    qs = sorted(p for p in queues_dir().glob("*") if (p / "state.json").exists())
    return qs[-1] if qs else None


def status(qdir):
    qdir = pathlib.Path(qdir) if qdir else _latest()
    if not qdir:
        print(f"no queues under {queues_dir()}")
        return 1
    s = json.loads((qdir / "state.json").read_text())
    alive = ms._pid_alive(s.get("pid"))
    print(f"{s['name']}  {s['status']}{' (runner alive)' if alive else ''}  "
          f"@ {s['commit'][:10]}  {qdir}")
    for i, r in enumerate(s["steps"]):
        print(f"  {i:2d} {r['status']:8s} {r['name'][:40]:40s} "
              f"{'' if r.get('seconds') is None else str(r['seconds']) + ' s':>10s} "
              f"{'x' + str(r['attempts']) if r.get('attempts', 1) > 1 else ''}")
        for x in r.get("reasons", []) + r.get("warnings", []):
            print(f"       {x}")
    return 0


def main(argv=None) -> int:
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
    ps = sub.add_parser("status")
    ps.add_argument("qdir", nargs="?")
    sub.add_parser("list")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for p in sorted(queues_dir().glob("*")):
            if (p / "state.json").exists():
                s = json.loads((p / "state.json").read_text())
                n = sum(r["status"] == "pass" for r in s["steps"])
                print(f"{p.name:48s} {s['status']:9s} {n}/{len(s['steps'])}  @ {s['commit'][:10]}")
        return 0
    if a.cmd == "status":
        return status(a.qdir)

    if bool(a.file) == bool(a.resume):
        ap.error("give a queue file, or --resume <queue dir>")
    qdir = pathlib.Path(a.resume) if a.resume else create(a.file, a.commit, a.allow_dirty, a.preflight)
    if a.detach:
        cmd = [sys.executable, "-m", "vqlab.cli", "queue", "run", "--resume", str(qdir),
               "--lease-wait", str(a.lease_wait)] + (["--force"] if a.force else [])
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
                             stdout=open(qdir / "queue.log", "ab"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        print(f"detached (pid {p.pid}); vqlab queue status {qdir}")
        return 0
    return run(qdir, force=a.force, lease_wait=a.lease_wait)


if __name__ == "__main__":
    sys.exit(main())
