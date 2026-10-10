"""Time stamps and filename-safe slugs."""
from __future__ import annotations

import datetime as dt
import re


def now() -> dt.datetime:
    """The current time, UTC, timezone-aware."""
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    """The current UTC time as ISO 8601 to the second (run logs, queues)."""
    return now().isoformat(timespec="seconds")


def slug(s: str, n: int = 40) -> str:
    """`s` with every character outside [A-Za-z0-9_.-] replaced by `_`,
    cut to `n` characters: safe as one path component."""
    return re.sub(r"[^\w.-]", "_", s)[:n]
