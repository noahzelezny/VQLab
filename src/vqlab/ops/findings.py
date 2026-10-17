"""The F-numbered findings log: next_f_number (reserve / release),
findings_tail, findings_append -- the only tool-side writer of the log."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vqlab import config
from vqlab.ops.errors import LabError

# The F-numbered findings log is the lab's own record: private (lab/ is
# gitignored); it lives under the configured records location (vqlab.config).
FINDINGS_LOG = config.findings_log()   # $VQLAB_FINDINGS_LOG, else <records>/FINDINGS-LOG.md
VERDICTS = ("CONFIRMED", "FALSIFIED", "VOID", "CORRECTS", "NULL")

_F_HEAD = re.compile(r"^## F(\d+)\b")


def _refuse_moved_log() -> None:
    """Refuse to number from an empty log while the old one sits in a checkout:
    the records default moved (2026-10), and handing out F1 would restart
    the lab's numbering silently."""
    if FINDINGS_LOG.exists():
        return
    old = config.legacy_findings_log()
    if old:
        raise LabError(
            "RECORDS_MOVED",
            f"the findings log {FINDINGS_LOG} does not exist, but the old one does at {old} "
            f"(records moved to the `records` location). Keep using it: "
            f"`vqlab lab config set records {old.parent}` or export VQLAB_FINDINGS_LOG={old}. "
            f"Refusing to start numbering at F1.")


def findings_entries(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = path or FINDINGS_LOG  # resolved at call time (tests redirect it)
    out = []
    if not path.is_file():
        return out
    for i, line in enumerate(path.read_text(errors="replace").split("\n"), 1):
        m = _F_HEAD.match(line)
        if m:
            out.append({"f": int(m.group(1)), "line": i, "headline": line[3:].strip()})
    return out


_FLOCK_DIR = FINDINGS_LOG.parent / ".f-locks"
_FLOCK_TTL = 6 * 3600          # a crashed session must not burn a number forever


def _live_reservations(published: Optional[set] = None) -> Dict[int, float]:
    """Claimed F-numbers that have not expired, as {number: claimed_at}.

    Self-healing on two conditions. A lock is dropped when it is STALE (past
    the TTL, so the session holding it is gone) and when its number is
    ALREADY IN THE LOG -- a published number cannot meaningfully be
    reserved, and a lock left behind by an append that has since written its
    entry is pure noise. The second case was real: before append released
    its own lock, every tool-written entry left one behind, and a fresh
    `next_f_number` reported a run of published numbers as "held".
    """
    out: Dict[int, float] = {}
    if not _FLOCK_DIR.is_dir():
        return out
    if published is None:
        published = set(e["f"] for e in findings_entries())
    now = time.time()
    for f in _FLOCK_DIR.glob("F*.lock"):
        try:
            n = int(f.stem[1:])
            age = now - f.stat().st_mtime
        except (ValueError, OSError):
            continue
        if age > _FLOCK_TTL or n in published:
            try:
                f.unlink()          # stale, or already written to the log
            except OSError:
                pass
            continue
        out[n] = f.stat().st_mtime
    return out


def next_f_number(reserve: bool = False, owner: str = "") -> Dict[str, Any]:
    """The next free F-number; with reserve=True, CLAIM it atomically.

    WHY RESERVE EXISTS (2026-09-19). This used to be `max(log) + 1`, a pure
    read. Two sessions working the same repo both called it, both got 151,
    and both wrote an F151 -- neither had committed when the other looked.
    A read cannot prevent that no matter how careful either session is: the
    log is only updated at commit time, so the window is the whole length of
    the experiment. The fix has to be a WRITE, and it has to be atomic.

    O_CREAT|O_EXCL on a per-number lock file is that write: exactly one
    caller can create `F<n>.lock`, and the loser simply advances to n+1.
    Locks carry a TTL so a session that dies mid-experiment releases its
    number instead of holding it forever, and `findings_append` releases
    the lock once the entry is actually in the log.

    reserve=False stays the default so a plain "what number am I on?" costs
    nothing and burns nothing.
    """
    _refuse_moved_log()
    ents = findings_entries()
    top = max((e["f"] for e in ents), default=0)
    pub = set(e["f"] for e in ents)
    taken = pub | set(_live_reservations(pub))
    n = top + 1
    while n in taken:
        n += 1
    if not reserve:
        held = sorted(_live_reservations())
        return {"next": n, "highest": top, "log": str(FINDINGS_LOG),
                "reserved": False, "reservations_held": held,
                "note": ("call with reserve=true before a long experiment; a "
                         "bare read races another session (F151/F152 vs "
                         "F153/F154, 2026-09-19)")}
    _FLOCK_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(_FLOCK_DIR / f"F{n}.lock"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            n += 1                  # someone claimed it between the scan and now
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"owner": owner or "unknown",
                                 "claimed": datetime.now().isoformat()}))
        return {"next": n, "highest": top, "log": str(FINDINGS_LOG),
                "reserved": True, "owner": owner or "unknown",
                "expires_in_s": _FLOCK_TTL,
                "note": "release happens automatically on findings_append"}


def release_f_number(n: int) -> Dict[str, Any]:
    """Give a reserved number back (abandoned experiment)."""
    f = _FLOCK_DIR / f"F{int(n)}.lock"
    existed = f.exists()
    if existed:
        f.unlink()
    return {"released": int(n), "was_held": existed}


def findings_tail(n: int = 10) -> Dict[str, Any]:
    _refuse_moved_log()
    ents = findings_entries()
    ents.sort(key=lambda e: e["f"])
    return {"count": len(ents), "entries": ents[-max(1, n):], "log": str(FINDINGS_LOG)}


def findings_append(headline: str, artifact: str, instrument: str, prediction: str,
                    measured: str, verdict: str, body: str = "",
                    corrects: str = "", use_reserved: int = 0) -> Dict[str, Any]:
    """Append the next F-entry. The prediction is required and is written as given.

    `use_reserved` CONSUMES a number this caller already claimed with
    `next_f_number(reserve=true)`. Without it, append would reserve a FRESH
    number -- because `_live_reservations()` cannot tell the caller's own
    hold from another session's and skips both -- so a reserve-then-append
    in one session silently burned a number (reported by vqlab-28, who
    reserved 162 and got an entry at 163). Passing the reserved number back
    is the fix; the lock is then released as part of writing the entry.

    Append also RELEASES whatever lock it used. The first version left it
    behind for the full 6 h TTL, so every tool-written entry leaked a
    reservation that then had to age out -- harmless for correctness, since
    both the log and the lock mark the number taken, but it meant the
    held-reservations list filled with numbers that were already published.
    """
    _refuse_moved_log()
    missing = [k for k, v in dict(headline=headline, artifact=artifact, instrument=instrument,
                                  prediction=prediction, measured=measured).items() if not str(v).strip()]
    if missing:
        raise LabError("INCOMPLETE", "every field is required; a finding without them is not a measurement",
                        missing=missing)
    verdict = verdict.upper().strip()
    if verdict not in VERDICTS:
        raise LabError("BAD_VERDICT", f"verdict must be one of {VERDICTS}")
    if verdict == "CORRECTS" and not corrects:
        raise LabError("INCOMPLETE", "verdict CORRECTS needs corrects='F<n>'")
    if use_reserved:
        n = int(use_reserved)
        ents = findings_entries()
        if n in set(e["f"] for e in ents):
            raise ValueError(f"F{n} is already in the log; it cannot be "
                             "reserved or reused")
        if not (_FLOCK_DIR / f"F{n}.lock").exists():
            raise ValueError(
                f"F{n} was not reserved (no lock held). Either call "
                "next_f_number(reserve=true) first, or omit use_reserved "
                "and let append allocate.")
    else:
        n = next_f_number(reserve=True, owner="findings_append")["next"]
    today = date.today().isoformat()
    head = f"## F{n} ({today}) — {headline.strip()}"
    if verdict == "CORRECTS":
        head = f"## F{n} ({today}) — CORRECTS {corrects}: {headline.strip()}"
    entry = "\n".join([
        "", head, "",
        f"ARTIFACT:   {artifact.strip()}",
        f"INSTRUMENT: {instrument.strip()}",
        f"PREDICTION (pre-registered): {prediction.strip()}",
        f"MEASURED:   {measured.strip()}",
        f"VERDICT:    {verdict}",
        "", body.rstrip(), "",
    ])
    with open(FINDINGS_LOG, "a", encoding="utf-8") as fh:
        fh.write(entry)
    # the number is now IN the log, so the lock has done its job
    try:
        (_FLOCK_DIR / f"F{n}.lock").unlink()
    except OSError:
        pass
    return {"f": n, "headline": head, "log": str(FINDINGS_LOG),
            "note": "Written by tool: the prediction was recorded as given. "
                    "Corrections edit an entry in place and say CORRECTED; never delete."}

