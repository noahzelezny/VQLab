"""Is a newer vqlab on PyPI? Asked at most once a day.

PyPI's JSON for the package names its latest release; when it is newer than
the one running, `vqlab doctor` and the GUI say so, with the upgrade command.
The answer is kept in <records>/update-check.json for a day, so the network
is asked rarely and never by an ordinary command. VQLAB_OFFLINE=1 (or
HF_HUB_OFFLINE=1) skips the check entirely.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

from vqlab import config

PYPI = "https://pypi.org/pypi/vqlab/json"
TIMEOUT_S = 5.0
MAX_AGE_S = 24 * 3600          # one PyPI request a day at most
COMMAND = "pip install -U vqlab"


def offline() -> bool:
    """The user asked for no network: VQLAB_OFFLINE or HF_HUB_OFFLINE."""
    return any(os.environ.get(v, "").strip().lower() in ("1", "true", "yes", "on")
               for v in ("VQLAB_OFFLINE", "HF_HUB_OFFLINE"))


def pypi_latest() -> str:
    """PyPI's latest release of vqlab (it skips pre-releases)."""
    with urllib.request.urlopen(PYPI, timeout=TIMEOUT_S) as r:
        return str(json.loads(r.read())["info"]["version"])


def version_key(v: str) -> tuple:
    """A release's order: its leading numbers (0.1.10 after 0.1.9); a
    version with none sorts first."""
    nums = re.match(r"\d+(?:\.\d+)*", v.strip())
    return tuple(int(x) for x in nums.group(0).split(".")) if nums else ()


def latest(ask=pypi_latest, now=None) -> str | None:
    """The latest release, from the day-old cache or one PyPI request; None
    when offline or PyPI did not answer (a failed ask is not cached)."""
    if offline():
        return None
    now = time.time() if now is None else now
    cache = config.records() / "update-check.json"
    try:
        c = json.loads(cache.read_text())
        if now - c["checked"] < MAX_AGE_S:
            return c["latest"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        v = ask()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"checked": now, "latest": v}))
    except OSError:
        pass
    return v


def release_doc(ask=pypi_latest) -> dict:
    """{"current", "latest", "update", "command"}: `latest` None when unknown,
    `update` True only when it is newer than the running vqlab."""
    import vqlab
    v = latest(ask)
    return {"current": vqlab.__version__, "latest": v,
            "update": bool(v) and version_key(v) > version_key(vqlab.__version__),
            "command": COMMAND}
