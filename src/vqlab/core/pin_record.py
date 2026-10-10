"""The pin record (`vqlab_pin.json`) and the check every scorer runs on it.

`vqlab pin` (gate/pin.py) WRITES the record after its smoke; scorers and
benches (score/kl_ladder, bench/speed_pair, the queue runner) READ it through
`check_pin` before measuring. The format and the read live here so a scorer
does not import a gate.
"""
from __future__ import annotations

import json
import pathlib

from vqlab.util.hashing import sha256_file

SCHEMA = "vqlab.pin/1"
MARKER = "vqlab_pin.json"
SMOKE_LOG = "vqlab_pin_smoke.log"


def _sha(p: pathlib.Path) -> str | None:
    return sha256_file(p) if p.exists() else None


def check_pin(d) -> tuple[bool, str]:
    """(allowed, message) for a directory a scorer is about to measure."""
    d = pathlib.Path(d)
    m = d / MARKER
    if not m.exists():
        return True, ""
    try:
        rec = json.loads(m.read_text())
    except ValueError:
        return False, f"REFUSE {d}: {MARKER} is unreadable"
    if rec.get("schema") != SCHEMA:
        return False, f"REFUSE {d}: unknown pin schema {rec.get('schema')!r}"
    if rec.get("model_py_sha256") != _sha(d / "model.py"):
        return False, (f"REFUSE {d}: model.py changed after the pin was smoked; "
                       "the smoke no longer describes it. Re-pin.")
    st = rec.get("state")
    if st == "ok":
        return True, ""
    if st == "deferred-ram":
        return True, (f"WARNING {d}: pin NOT smoked (too large for this box: "
                      f"{rec.get('reason')}). Smoke it on a cluster before citing a number.")
    return False, f"REFUSE {d}: pin state {st!r} ({rec.get('reason')})"
