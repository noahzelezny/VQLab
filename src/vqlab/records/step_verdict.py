"""step_verdict: did a queue step succeed? One rule, for every runner.

Paid for on night 4 (2026-09-26): a speed step piped stderr to /dev/null and
grep'd stdout; a data path had moved, the script died with a traceback, and the
queue logged "speed done: 0 runs" as if it had finished. A step is a FAILURE
unless it proves otherwise:

  * nonzero exit                                  -> fail
  * a Python traceback in stderr, even at rc 0    -> fail (a pipe can eat rc)
  * expect["stdout_nonempty"] and stdout is empty -> fail
  * expect["stdout_regex"] never matches stdout   -> fail
  * expect["min_lines"]: fewer matching lines     -> fail
  * expect["files"]: any listed path missing/empty-> fail

The verdict carries the stderr tail, so a runner's log says WHY without anyone
opening the step's files. `allow_traceback=True` in expect exempts a tool that
prints a handled traceback on purpose (none known today).
"""
from __future__ import annotations

import pathlib
import re

TAIL = 12


def _read(p) -> str:
    if p is None:
        return ""
    try:
        return pathlib.Path(p).read_text(errors="replace")
    except OSError:
        return ""


def step_verdict(rc: int, stdout_path=None, stderr_path=None, expect: dict | None = None) -> dict:
    expect = expect or {}
    out, err = _read(stdout_path), _read(stderr_path)
    reasons = []
    if rc != 0:
        reasons.append(f"exit code {rc}")
    if "Traceback (most recent call last)" in err and not expect.get("allow_traceback"):
        reasons.append("traceback in stderr")
    if expect.get("stdout_nonempty") and not out.strip():
        reasons.append("expected output, stdout is empty")
    rx = expect.get("stdout_regex")
    if rx:
        hits = [ln for ln in out.splitlines() if re.search(rx, ln)]
        need = int(expect.get("min_lines", 1))
        if len(hits) < need:
            reasons.append(f"{len(hits)} stdout line(s) match {rx!r}, need {need}")
    for f in expect.get("files", ()):
        fp = pathlib.Path(f)
        if not fp.exists() or (fp.is_file() and fp.stat().st_size == 0):
            reasons.append(f"expected file missing or empty: {f}")
    return {"ok": not reasons, "verdict": "pass" if not reasons else "fail",
            "reasons": reasons,
            "stderr_tail": err.strip().splitlines()[-TAIL:]}
