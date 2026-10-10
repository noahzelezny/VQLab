"""The GPU lease: one file every GPU tenant on the machine flocks in turn.

Lives in core so a bench (box_quiet) and the MCP server (agents) can both
ask who holds it without bench importing agents.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
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
