"""`vqlab queue run --on BOX`: launch a queue file on another box over ssh,
after checking that every path it names READS there."""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess

from vqlab import config as _config
from vqlab.queue import reserve as resv
from vqlab.queue import steps as _steps
from vqlab.queue.files import REPO, git

# Runs ON the remote with that box's python: for each path, does it exist,
# list, and READ? An SMB mount can list directories while every read returns
# EIO (the M4, 2026-10-03), so one file per path is actually read.
_REMOTE_PROBE = r"""
import errno, json, os, sys
def probe(p):
    r = {"path": p}
    if not os.path.exists(p):
        a = p
        while a != os.path.dirname(a) and not os.path.exists(a):
            a = os.path.dirname(a)
        r.update(exists=False, ancestor=a)
        return r
    r["exists"] = True
    try:
        f = None
        if os.path.isdir(p):
            names = sorted(os.listdir(p))
            r["listed"] = True
            for n in names[:64]:
                q = os.path.join(p, n)
                if os.path.isfile(q) and not n.startswith("."):
                    f = q
                    break
        else:
            f = p
        if f:
            with open(f, "rb") as h:
                h.read(64)
            r["read"] = f
        r["readable"] = True
    except OSError as e:
        r.update(readable=False, errno=errno.errorcode.get(e.errno, str(e.errno)), error=str(e)[:200])
    return r
print(json.dumps([probe(p) for p in json.loads(sys.argv[1])]))
"""


def _mount(path: str):
    parts = pathlib.PurePosixPath(path).parts
    return str(pathlib.PurePosixPath(*parts[:3])) if len(parts) >= 3 and parts[1] == "Volumes" else None


def remote_paths(q, file, b, name, preflight=False) -> list[str]:
    """Every absolute path a queue names, as the box `name` will see it:
    the queue file, the clone, config and queue dir, each step's input paths
    and the nearest existing ancestor of each output."""
    out = [str(pathlib.Path(file).resolve()), b["repo"], b["config"],
           str(pathlib.Path(b["queue_dir"]).parent)]
    for st in q["steps"]:
        arglists = [st.get("args", [])]
        pf = st.get("preflight") or {}
        if preflight and "args" in pf:
            arglists.append(pf["args"])
        for args in arglists:
            ins, outs = _steps.path_args([_config.expand(str(a), box=name) for a in args])
            out += [p for p in ins if p.startswith("/")]
            for o in outs:
                if o.startswith("/"):
                    a = pathlib.Path(o).parent
                    while not a.exists() and a != a.parent:
                        a = a.parent
                    out.append(str(a))
    return list(dict.fromkeys(out))


def _ssh_probe(name, b, paths):
    """(results, None) from the probe on box `name`, or (None, problem)."""
    cmd = f"{shlex.quote(b['python'])} -c {shlex.quote(_REMOTE_PROBE)} {shlex.quote(json.dumps(paths))}"
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", b["ssh"], cmd],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, f"on {name}: the path check timed out after 120 s (a hung mount?)"
    if r.returncode:
        return None, (f"on {name}: could not run the path check over ssh {b['ssh']} "
                      f"(exit {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]), None
    except (ValueError, IndexError):
        return None, f"on {name}: path check printed no result: {r.stdout.strip()[-300:]}"


def remote_check(name, b, paths) -> list[str]:
    """Plain problems with `paths` on box `name` (empty: all readable)."""
    res, why = _ssh_probe(name, b, paths)
    if why:
        return [why]
    probs, said = [], set()
    for x in res:
        p, m = x["path"], _mount(x["path"])
        vol = pathlib.PurePosixPath(m).name if m else None
        if not x.get("exists"):
            anc = x.get("ancestor") or "/"
            if m and not (anc == m or anc.startswith(m + "/")):
                if m not in said:
                    said.add(m)
                    probs.append(f"on {name}: {m} is not mounted ({p} does not exist): "
                                 f"mount {vol} on {name}")
            else:
                probs.append(f"on {name}: {p} does not exist")
        elif not x.get("readable"):
            key = m or p
            if key not in said:
                said.add(key)
                probs.append(f"on {name}: {key} is listed but unreadable ({x.get('errno')}): "
                             + (f"remount {vol} on {name}" if m else f"check {p} on {name}"))
    return probs


def _fast_forward(clone: pathlib.Path) -> str:
    """Fast-forward the box's clone to this repo's HEAD; its new HEAD."""
    if git("status", "--porcelain", "--untracked-files=no", cwd=clone):
        raise SystemExit(f"{clone} has local changes; refusing to fast-forward it")
    br = git("rev-parse", "--abbrev-ref", "HEAD")
    r = subprocess.run(["git", "-C", str(clone), "pull", "-q", "--ff-only", str(REPO), br],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"fast-forward of {clone} failed: {r.stderr.strip()}")
    head = git("rev-parse", "HEAD", cwd=clone)
    if head != git("rev-parse", "HEAD"):
        raise SystemExit(f"{clone} is at {head[:10]}, not this repo's HEAD")
    return head


def _remote_command(name, b, file, clone, fwd, preflight, force, override_reservation) -> str:
    # VQLAB_BOX names the box there (its reservation, its `teachers` override);
    # the override is forwarded too, since that box's own config may not carry it
    ident = f"VQLAB_BOX={shlex.quote(name)} VQLAB_WHO={shlex.quote(resv.who())}" + (
        f" VQLAB_TEACHERS_DIR={shlex.quote(b['teachers'])}" if b.get("teachers") else "")
    return (f"cd {shlex.quote(str(clone))} && {fwd} {ident} VQLAB_CONFIG={shlex.quote(b['config'])} "
            f"VQLAB_QUEUE_DIR={shlex.quote(b['queue_dir'])} PYTHONPATH=src "
            f"{shlex.quote(b['python'])} -m vqlab.cli queue run "
            f"{shlex.quote(str(pathlib.Path(file).resolve()))} --detach"
            + (" --preflight" if preflight else "") + (" --force" if force else "")
            + (" --override-reservation" if override_reservation else ""))


def run_on(name, file, preflight=False, force=False, override_reservation=False) -> int:
    """Launch a queue file on another box: refuse a box reserved for someone
    else, check every path the queue names is READABLE there, fast-forward its
    clone (shared storage) to this repo's HEAD, then `queue run --detach` there
    over ssh with that box's config and a SHARED queue dir, so `vqlab queue
    wait <qdir>` works from here. The queue still pins its own worktree there."""
    b = _config.boxes().get(name)
    if not b:
        raise SystemExit(f"no [boxes.{name}] in {_config.config_file()}")
    missing = [k for k in ("repo", "python", "config", "queue_dir") if not b.get(k)]
    if missing:
        raise SystemExit(f"[boxes.{name}] cannot run queues: it names no {', '.join(missing)} "
                         "(a vqlab clone on that box, its python, config and a shared queue dir)")
    why = resv.refusal(name, override_reservation)
    if why:
        raise SystemExit(why)
    q = json.loads(pathlib.Path(file).read_text())
    errs = _steps.validate(q)
    if errs:
        raise SystemExit("queue file invalid:\n  " + "\n  ".join(errs))
    probs = remote_check(name, b, remote_paths(q, file, b, name, preflight))
    if probs:
        raise SystemExit("REFUSED, nothing launched:\n  " + "\n  ".join(probs))
    clone = pathlib.Path(b["repo"])
    head = _fast_forward(clone)
    # numerics switches must match on both boxes (runtime VQ_* flags, family
    # scorer variants such as VQLAB_DS4_SHARED_CLAMP): forward the ones set here
    fwd = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(os.environ.items())
                   if k.startswith(("VQ_", "VQLAB_DS4_")))
    remote = _remote_command(name, b, file, clone, fwd, preflight, force, override_reservation)
    print(f"{name}: paths readable, clone at {head[:10]}; launching over ssh {b['ssh']}"
          + (f" with {fwd}" if fwd else ""), flush=True)
    return subprocess.call(["ssh", "-o", "ConnectTimeout=10", b["ssh"], remote])
