"""The GPU lease: one file every GPU tenant on the machine flocks in turn.

Lives in core so a bench (box_quiet), the queue runner (queue/) and the MCP
server (agents) share one lease protocol without importing agents.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from vqlab import config


def lease_path() -> Path:
    """The GPU lease file (`vqlab.config.gpu_lease`). Point every GPU tenant
    on the machine at the same file and they take turns."""
    return config.gpu_lease()


def lease_holder() -> Optional[Dict[str, Any]]:
    """The lease file's JSON if someone holds the lock, else None."""
    p = lease_path()
    if not p.exists():
        return None
    fd = os.open(p, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                try:
                    return json.loads(Path(p).read_text() or "{}")
                except Exception:
                    return {"holder": "unknown"}
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def acquire_lease(lp: Path, holder: str, run_id: Optional[str], wait_s: float = 1800,
                  poll_s: float = 15) -> Optional[int]:
    """Take an exclusive flock on `lp`, waiting up to wait_s; stamp the holder.
    Returns the fd (keep it open to hold the lease; release_lease to drop it)
    or None if it stayed held. Shared by the MCP supervisor and `vqlab queue`,
    so there is one lease protocol, not a pgrep script per campaign."""
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + wait_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                os.close(fd)
                raise
            if time.time() > deadline:
                os.close(fd)
                return None
            time.sleep(poll_s)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps({"holder": holder, "pid": os.getpid(), "run_id": run_id,
                             "since": time.strftime("%Y-%m-%dT%H:%M:%S")}).encode())
    os.fsync(fd)
    return fd


def release_lease(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def pid_alive(pid: Optional[int]) -> bool:
    """True if a process with this pid exists on THIS machine."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
