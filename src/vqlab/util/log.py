"""Progress lines for long-running tools: one print, flushed, so a detached
run's log file shows progress as it happens."""
from __future__ import annotations

import sys
import time


def log(*a, prefix: str | None = None, stamp: bool = False, file=None) -> None:
    """Print `a` flushed, after `[HH:MM:SS]` when stamp and/or `prefix`.
    Goes to stdout unless `file` is given."""
    head = []
    if stamp:
        head.append(time.strftime("[%H:%M:%S]"))
    if prefix:
        head.append(prefix)
    print(*head, *a, flush=True, file=file if file is not None else sys.stdout)


def make_log(prefix: str | None = None, stamp: bool = False):
    """A tool's own `_log(*a)`: `make_log("[sz-pack]")`, `make_log(stamp=True)`."""
    def _log(*a):
        log(*a, prefix=prefix, stamp=stamp)
    return _log
