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
from vqlab.core.box.gpu_lease import pid_alive

REPO = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,95}$")
_LOCK = threading.Lock()
_PROCS: dict[str, subprocess.Popen] = {}     # this server's children, reaped by poll()
# what a teacher needs; never the repo's other formats (gguf, onnx, pt copies)
PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py", "*.md"]
# The child records how it ended: 0 done, 1 failed (its log says why); a
# stopped child (SIGTERM) records nothing.
_FETCH = """\
import pathlib, sys
from huggingface_hub import snapshot_download
d, rc = pathlib.Path(sys.argv[2]), 1
try:
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


def _total(repo: str) -> int | None:
    from huggingface_hub import HfApi
    from fnmatch import fnmatch
    info = HfApi().model_info(repo, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings
               if any(fnmatch(s.rfilename.rsplit("/", 1)[-1], p) for p in PATTERNS))


def _alive(repo: str, e: dict) -> bool:
    """A child this server started is judged by poll(), which also reaps it
    (an unreaped child still answers kill(pid, 0)); one an earlier server
    started has been reparented and is judged by its pid."""
    p = _PROCS.get(repo)
    if p is not None:
        return p.poll() is None
    return pid_alive(e.get("pid"))


def downloads() -> list[dict]:
    with _LOCK:
        state = _load()
        changed = False
        for repo, e in state.items():
            if e.get("status") == "downloading" and not _alive(repo, e):
                # the process ended: its exit file says done or failed; none = stopped
                rc = pathlib.Path(e["dest"]) / ".vqlab-download-rc"
                code = rc.read_text().strip() if rc.is_file() else None
                e["status"] = {"0": "done", None: "stopped"}.get(code, "failed")
                e.pop("pid", None)
                changed = True
                if e["status"] == "done":
                    _rescan()
        if changed:
            _save(state)
    return [{"repo": r, "dest": config.portable(e["dest"]), "status": e.get("status", "stopped"),
             "bytes": _on_disk(pathlib.Path(e["dest"])), "total": e.get("total"),
             "started": e.get("started")} for r, e in sorted(state.items())]


def _rescan() -> None:
    try:
        from vqlab.agents import model_scan
        threading.Thread(target=model_scan.refresh, daemon=True).start()
    except ImportError:
        pass


def start(body: dict) -> dict:
    repo = str(body.get("repo") or "")
    if not REPO.match(repo):
        return {"ok": False, "error": "a Hugging Face model id is org/name"}
    with _LOCK:
        state = _load()
        e = state.get(repo)
        if e and _alive(repo, e):
            return {"ok": True, "status": "downloading"}
        try:
            total = _total(repo)
        except Exception as exc:                 # not found, gated, offline
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            return {"ok": False, "error": f"cannot read {repo} on the Hub: {first}"}
        d = dest(repo)
        d.mkdir(parents=True, exist_ok=True)
        (d / ".vqlab-download-rc").unlink(missing_ok=True)
        p = subprocess.Popen([sys.executable, "-c", _FETCH, repo, str(d), *PATTERNS],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=open(d / ".vqlab-download.log", "ab"), start_new_session=True)
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
        if not e or not _alive(repo, e):
            return {"ok": False, "error": "that download is not running"}
        try:
            os.killpg(int(e["pid"]), signal.SIGTERM)    # the fetch and its worker threads
        except (ProcessLookupError, PermissionError) as exc:
            return {"ok": False, "error": f"could not stop it: {exc}"}
        e["status"] = "stopped"
        e.pop("pid", None)
        _save(state)
    return {"ok": True}
