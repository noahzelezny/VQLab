"""queue_status: `vqlab queue` runs on this box, from their state.json (what
`vqlab queue wait` blocks on) -- never from log wording."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

from vqlab.core.gpu_lease import pid_alive as _pid_alive
from vqlab.queue import eta as _queue_eta


def _queues_dir() -> Path:
    # same resolution as vqlab.queue.files.queues_dir
    return Path(os.environ.get("VQLAB_QUEUE_DIR") or Path.home() / ".vqlab" / "queues")


def queue_status(name: Optional[str] = None, n: int = 5) -> Dict[str, Any]:
    """Queue state from state.json. A 'running' queue whose runner pid is gone
    is 'died'."""
    qs = sorted(p for p in _queues_dir().glob("*") if (p / "state.json").exists())
    if name:
        qs = [p for p in qs if name in p.name]
    out = []
    for q in qs[-max(1, int(n)):]:
        s = json.loads((q / "state.json").read_text())
        st = s["status"]
        here = socket.gethostname().split(".")[0]
        if st == "running" and s.get("host", here) == here and not _pid_alive(s.get("pid")):
            st = "died"
        live = st in ("running", "created")
        try:
            eta = _queue_eta.queue_eta(q, _queues_dir())
        except Exception as e:  # an ETA must never hide the state
            eta = {"steps": [{"eta_s": None, "basis": f"eta failed: {e}"}] * len(s["steps"]),
                   "eta_s": None, "finish": None}
        out.append({"queue": str(q), "status": st, "terminal": st in (
            "passed", "failed", "stopped", "deferred", "died"),
            "eta_s": eta["eta_s"] if live else None,
            "eta": _queue_eta.fmt(eta["eta_s"]) if live else None,
            "finish": eta["finish"] if live else None,
            "reservation_override": s.get("reservation_override"),
            "steps": [{"name": r["name"], "status": r["status"], "seconds": r.get("seconds"),
                       "reasons": r.get("reasons", []),
                       **({"eta_s": e["eta_s"], "eta_basis": e["basis"]}
                          if r["status"] in ("running", "pending") else {})}
                      for r, e in zip(s["steps"], eta["steps"])]})
    return {"queues_dir": str(_queues_dir()), "queues": out}
