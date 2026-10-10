"""MCP adapters for the findings log: next_f_number (reserve / release),
findings_tail, findings_append. Argument mapping only; the work is in
vqlab.ops.findings, reached through vqlab.api. The module-level names
(FINDINGS_LOG, _FLOCK_DIR, ...) are re-exported for old imports: patch
vqlab.ops.findings, not this module."""

from __future__ import annotations

from typing import Any, Dict

from vqlab import api
from vqlab.ops.findings import (_F_HEAD, _FLOCK_DIR, _FLOCK_TTL, FINDINGS_LOG,  # noqa: F401
                                VERDICTS, _findings_entries, _live_reservations)


def t_next_f_number(reserve: bool = False, owner: str = "") -> Dict[str, Any]:
    """The next free F-number; with reserve=True, CLAIM it atomically."""
    return api.next_f_number(reserve=reserve, owner=owner)


def t_release_f_number(n: int) -> Dict[str, Any]:
    """Give a reserved number back (abandoned experiment)."""
    return api.release_f_number(n=n)


def t_findings_tail(n: int = 10) -> Dict[str, Any]:
    return api.findings_tail(n=n)


def t_findings_append(headline: str, artifact: str, instrument: str, prediction: str,
                      measured: str, verdict: str, body: str = "",
                      corrects: str = "", use_reserved: int = 0) -> Dict[str, Any]:
    """Append the next F-entry. The prediction is required and is written as given."""
    return api.findings_append(headline=headline, artifact=artifact, instrument=instrument,
                               prediction=prediction, measured=measured, verdict=verdict,
                               body=body, corrects=corrects, use_reserved=use_reserved)
