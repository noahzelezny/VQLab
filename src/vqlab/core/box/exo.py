"""The exo cluster's HTTP API, as the GPU gate reads it: is an instance placed?

Lives in core so the queue runner (queue/) and the MCP server (agents) ask
the same question the same way without the queue importing agents.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional


def exo_api() -> str:
    return os.environ.get("VQLAB_EXO_API", "http://127.0.0.1:52415")


def exo_instances() -> Optional[List[str]]:
    """Instance ids currently placed on this cluster, or None if exo is down."""
    try:
        with urllib.request.urlopen(exo_api() + "/state", timeout=3) as r:
            st = json.load(r)
    except Exception:  # exo down means nothing is placed here
        return None
    inst = st.get("instances") or {}
    return list(inst.keys()) if isinstance(inst, dict) else [str(i) for i in inst]
