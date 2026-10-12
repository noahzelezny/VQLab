"""Hugging Face downloads for the GUI: a teacher's weights into this
machine's teachers folder.

    /api/downloads                       every download, with bytes on disk
    POST /api/downloads/start {repo}     start, or resume a stopped one
    POST /api/downloads/stop {repo}      stop (what is on disk stays; start resumes)

Each download is its own process (`huggingface_hub.snapshot_download` into
<teachers>/<org>--<name>, the folder layout the model scan and
`family-profile` read), so the server never blocks and Stop is a signal.
Progress is the bytes on disk against the repo's file sizes from the Hub,
never a guess. A finished download triggers a model scan; profiling it is
the Recipe page's Profile action. Nothing here deletes: removing a model's
files is the operator's own action.
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

from vqlab import config

REPO = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,95}$")
_LOCK = threading.Lock()
_PROCS: dict[str, subprocess.Popen] = {}     # this server's children, reaped by poll()
# what a teacher needs besides its weights; never the repo's other formats
# (gguf, onnx, pt copies). Weights are allowed by exact name (_plan).
PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.vocab", "*.bpe", "*.py", "*.md"]
INDEX = "model.safetensors.index.json"
LOCKFILE = ".vqlab-download.lock"
# The child holds a flock on LOCKFILE for its whole life, with its pid in it:
# a recorded pid is only ever signalled while that lock says it is still ours
# (a pid alone can be reused by an unrelated process after a restart). It
# records how it ended: 0 done, 1 failed (its log says why); a stopped child
# (SIGTERM) records nothing.
_FETCH = """\
import fcntl, os, pathlib, sys
from huggingface_hub import snapshot_download
d, rc = pathlib.Path(sys.argv[2]), 1
lk = open(d / ".vqlab-download.lock", "a+")
lk.truncate(0)
lk.write(str(os.getpid()))
lk.flush()
try:
    fcntl.flock(lk, fcntl.LOCK_EX)
    snapshot_download(sys.argv[1], local_dir=d, allow_patterns=sys.argv[3:])
    rc = 0
finally:
    (d / ".vqlab-download-rc").write_text(str(rc))
"""


def _state_path() -> pathlib.Path:
    return config.records() / "downloads.json"


def _load() -> dict:
    try:
        return json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)


def dest(repo: str) -> pathlib.Path:
    return config.teachers() / repo.replace("/", "--", 1)


def _on_disk(d: pathlib.Path) -> int:
    """Bytes of the repo's files under d, finished or partial (the hub keeps
    a partial file under .cache/huggingface/download as *.incomplete)."""
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())


def _plan(repo: str) -> tuple[list[str], int]:
    """(allow patterns, total bytes). Weights are the top-level shards the
    index names; without an index, model.safetensors alone when present
    (not consolidated.safetensors and the like), else the top-level
    *.safetensors; never a subfolder's copies. The total is summed over the
    same allow list the fetch uses. The index goes to the hub cache, so a
    refused repo leaves nothing in <teachers>."""
    from fnmatch import fnmatch

    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(repo, files_metadata=True)
    names = {s.rfilename for s in info.siblings}
    if INDEX in names:
        idx = hf_hub_download(repo, INDEX)
        weights = set(json.loads(pathlib.Path(idx).read_text())["weight_map"].values())
    elif "model.safetensors" in names:
        weights = {"model.safetensors"}
    else:
        weights = {n for n in names if "/" not in n and n.endswith(".safetensors")}
    total = sum(s.size or 0 for s in info.siblings
                if s.rfilename in weights or any(fnmatch(s.rfilename, p) for p in PATTERNS))
    return [*PATTERNS, *sorted(weights)], total


def _lock_pid(d: pathlib.Path) -> int | None:
    """The pid in d's lock file while a fetch holds the lock, else None."""
    try:
        with open(d / LOCKFILE) as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = f.read().strip()
                if not pid:          # the child writes it just before locking: read once more
                    time.sleep(0.05)
                    f.seek(0)
                    pid = f.read().strip()
                return int(pid or 0) or None
            fcntl.flock(f, fcntl.LOCK_UN)
            return None
    except (OSError, ValueError):
        return None


def _alive(repo: str, e: dict) -> bool:
    """A child this server started is judged by poll(), which also reaps it
    (an unreaped child still answers kill(pid, 0)); one an earlier server
    started is alive only while it holds its lock with the recorded pid."""
    p = _PROCS.get(repo)
    if p is not None:
        return p.poll() is None
    return e.get("pid") is not None and _lock_pid(pathlib.Path(e["dest"])) == e.get("pid")


def downloads() -> list[dict]:
    with _LOCK:
        state = _load()
        changed = False
        for repo, e in state.items():
            if e.get("status") == "downloading" and not _alive(repo, e):
                _ended(e)
                changed = True
        if changed:
            _save(state)
    return [{"repo": r, "dest": config.portable(e["dest"]), "status": e.get("status", "stopped"),
             "bytes": _on_disk(pathlib.Path(e["dest"])), "total": e.get("total"),
             "started": e.get("started")} for r, e in sorted(state.items())]


def _ended(e: dict) -> None:
    """Settle an entry whose process is gone: its exit file says done or
    failed; none = stopped. A finished one triggers a model scan."""
    rc = pathlib.Path(e["dest"]) / ".vqlab-download-rc"
    code = rc.read_text().strip() if rc.is_file() else None
    e["status"] = {"0": "done", None: "stopped"}.get(code, "failed")
    e.pop("pid", None)
    if e["status"] == "done":
        _rescan()


def _rescan() -> None:
    from vqlab.agents import model_scan
    threading.Thread(target=model_scan.refresh, daemon=True).start()


def start(body: dict) -> dict:
    repo = str(body.get("repo") or "")
    if not REPO.match(repo):
        return {"ok": False, "error": "a Hugging Face model id is org/name"}
    with _LOCK:
        e = _load().get(repo)
        if e and _alive(repo, e):
            return {"ok": True, "status": "downloading"}
    try:                                         # network: never under _LOCK
        allow, total = _plan(repo)
    except Exception as exc:                     # not found, gated, offline
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return {"ok": False, "error": f"cannot read {repo} on the Hub: {first}"}
    d = dest(repo)
    d.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        state = _load()
        e = state.get(repo)
        if e and _alive(repo, e):                # started by another request meanwhile
            return {"ok": True, "status": "downloading"}
        (d / ".vqlab-download-rc").unlink(missing_ok=True)
        with open(d / ".vqlab-download.log", "ab") as log:     # the child keeps its own copy
            p = subprocess.Popen([sys.executable, "-c", _FETCH, repo, str(d), *allow],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=log, start_new_session=True)
        _PROCS[repo] = p
        state[repo] = {"dest": str(d), "pid": p.pid, "status": "downloading", "total": total,
                       "started": (e or {}).get("started") or time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save(state)
    return {"ok": True, "status": "downloading"}


def stop(body: dict) -> dict:
    repo = str(body.get("repo") or "")
    with _LOCK:
        state = _load()
        e = state.get(repo)
        if not e or e.get("status") != "downloading":
            return {"ok": False, "error": "that download is not running"}
        if not _alive(repo, e):
            # ended, or a stale pid nobody holds the lock for: never signal it
            _ended(e)
            _save(state)
            return {"ok": True, "status": e["status"]}
        try:
            os.killpg(int(e["pid"]), signal.SIGTERM)    # the fetch and its worker threads
        except (ProcessLookupError, PermissionError) as exc:
            return {"ok": False, "error": f"could not stop it: {exc}"}
        e["status"] = "stopped"
        e.pop("pid", None)
        _save(state)
    return {"ok": True}
