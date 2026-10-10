"""Queue files on disk: where queues live, their state.json, and creating one.

A queue dir is `<records>/queues/<stamp>-<slug>[-preflight]/` holding
queue.json (the steps, as given), state.json (written after every step) and,
from a git checkout, tree/ (the detached worktree every step runs from).
"""
from __future__ import annotations

import json
import pathlib
import re
import socket
import subprocess
import time

from vqlab import _layout
from vqlab import config as _config
from vqlab.queue import steps as _steps
from vqlab.util.clock import now_iso, slug

REPO = _layout.SRC.parent
SCHEMA = "vqlab.queue/1"          # data written into every state.json; never rename

_now = now_iso
_slug = slug


def queues_dir() -> pathlib.Path:
    return _config.queues_dir()


def git(*a, cwd=REPO):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout.strip()


def load_state(qdir) -> dict:
    return json.loads((pathlib.Path(qdir) / "state.json").read_text())


def load_queue(qdir) -> dict:
    return json.loads((pathlib.Path(qdir) / "queue.json").read_text())


def save_state(qdir, state):
    qdir = pathlib.Path(qdir)
    tmp = qdir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(qdir / "state.json")


def queue_dirs() -> list[pathlib.Path]:
    """Every queue dir with a state.json, oldest first (names start with a stamp)."""
    return sorted(p for p in queues_dir().glob("*") if (p / "state.json").exists())


def latest():
    qs = queue_dirs()
    return qs[-1] if qs else None


def installed() -> bool:
    """True when vqlab runs from an installed package, not a git checkout.
    Then there is no tree to pin: the installed files ARE the pin, and the
    state records the version plus a hash of every installed .py file."""
    return not (REPO / ".git").exists()


def package_hash() -> str:
    import hashlib
    h = hashlib.sha256()
    pkg = _layout.SRC / "vqlab"
    for f in sorted(pkg.rglob("*.py")):
        h.update(str(f.relative_to(pkg)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _new_qdir(q, qfile, preflight) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = slug(q.get("name") or pathlib.Path(qfile).stem)
    qdir = queues_dir() / f"{stamp}-{name}{'-preflight' if preflight else ''}"
    qdir.mkdir(parents=True, exist_ok=False)
    (qdir / "queue.json").write_text(json.dumps(q, indent=1))
    return qdir


def _new_state(q, qfile, preflight, **pin) -> dict:
    return {"schema": SCHEMA, "name": q.get("name"), "source": str(pathlib.Path(qfile).resolve()),
            **pin, "host": socket.gethostname().split(".")[0], "preflight": preflight,
            "created": _now(), "status": "created", "queue_run_id": None,
            "steps": [{"name": st["name"], "status": "pending"} for st in q["steps"]]}


def _refuse_dirty():
    dirty = "\n".join(
        ln for ln in git("status", "--porcelain", "--", "src", "scripts").splitlines()
        # untracked junk (.DS_Store, __pycache__, notes) is not code the
        # queue would miss; untracked CODE is, and still blocks
        if not (ln.startswith("??") and not re.search(r"\.(py|sh|json|metal)/?$", ln)))
    if dirty:
        raise SystemExit(
            "uncommitted changes in src/ or scripts/ would NOT be in the pinned tree:\n"
            + dirty + "\ncommit them first, or pass --allow-dirty to run HEAD without them")


def create(qfile, commit=None, allow_dirty=False, preflight=False):
    """Validate a queue file and make its queue dir, pinned at one commit
    (a detached worktree) or, installed, at the package version + hash."""
    q = json.loads(pathlib.Path(qfile).read_text())
    errs = _steps.validate(q)
    if errs:
        raise SystemExit("queue file invalid:\n  " + "\n  ".join(errs))
    if installed():
        from vqlab import __version__
        qdir = _new_qdir(q, qfile, preflight)
        state = _new_state(q, qfile, preflight,
                           commit=f"installed:{__version__}:{package_hash()}", installed=True,
                           tree=str(_layout.SRC), live_repo=str(_layout.SRC))
        save_state(qdir, state)
        return qdir
    rev = commit or q.get("commit") or "HEAD"
    if rev == "HEAD" and not allow_dirty:
        _refuse_dirty()
    sha = git("rev-parse", "--verify", rev + "^{commit}")
    qdir = _new_qdir(q, qfile, preflight)
    git("worktree", "add", "--detach", str(qdir / "tree"), sha)
    state = _new_state(q, qfile, preflight, commit=sha, tree=str(qdir / "tree"),
                       live_repo=str(REPO))
    save_state(qdir, state)
    return qdir
