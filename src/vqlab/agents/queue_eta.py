"""ETAs for `vqlab queue status` / MCP queue_status, from the data.

WHY (2026-10-03): "free by 01:00", then "until morning", then "02:30" -- the
ETA came from someone's head. It now comes from two measured sources, in
order, and is "ETA unknown" when neither applies (never a guess):

1. the RUNNING step's own progress lines:
   * fit-moe prints ``[i/N] <shard>  (Ts)`` after each shard, T cumulative
     seconds (fit/fit_moe.py);
   * kl-ladder prints ``[kl-ladder] RUNG x CACHE`` then ``    KL ...`` per
     cell, ``(resumed from saved record)`` for a cell it did not recompute;
     the cell count is --rung x --cache from the step's args (1 under
     --preflight). Resumed cells are excluded from the rate.
   Remaining = per-unit time x units left - time since the last unit line.
2. past runs of the SAME step (same cmd/script and identical args) that
   passed in a non-preflight queue under the queues dir: their median
   seconds, minus this run's elapsed time. Pending steps use this too.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import statistics
import time

_FIT = re.compile(r"^\[(\d+)/(\d+)\]\s+\S+\s+\((\d+(?:\.\d+)?)s\)\s*$", re.M)
_KL_CELL = re.compile(r"^[ \t]+KL[ \t]", re.M)
_KL_RESUMED = re.compile(r"^[ \t]+\(resumed from saved record\)", re.M)


def _ts(iso):
    try:
        return dt.datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


def _count_flag(args, flag):
    return sum(1 for a in args if a == flag or a.startswith(flag + "="))


def _key(st):
    return (st.get("cmd") or "script:" + str(st.get("script")),
            tuple(str(a) for a in st.get("args", [])))


def history(qroot: pathlib.Path, st: dict, exclude: pathlib.Path | None = None) -> list[float]:
    """Seconds of every passed, non-preflight past run of the same step."""
    want, out = _key(st), []
    for q in sorted(pathlib.Path(qroot).glob("*")):
        if exclude is not None and q.resolve() == pathlib.Path(exclude).resolve():
            continue
        try:
            s = json.loads((q / "state.json").read_text())
            qd = json.loads((q / "queue.json").read_text())
        except (OSError, ValueError):
            continue
        if s.get("preflight"):
            continue
        for sd, rec in zip(qd.get("steps", []), s.get("steps", [])):
            if rec.get("status") == "pass" and rec.get("seconds") and _key(sd) == want:
                out.append(float(rec["seconds"]))
    return out


def _from_progress(st, stdout: pathlib.Path, started, now):
    """(remaining s, basis) from the step's own output, or None."""
    try:
        text = stdout.read_text(errors="replace")
        mtime = stdout.stat().st_mtime
    except OSError:
        return None
    since_last = max(0.0, now - mtime)
    cmd = st.get("cmd")
    if cmd == "fit-moe":
        ms = _FIT.findall(text)
        if not ms:
            return None
        i, n, t = int(ms[-1][0]), int(ms[-1][1]), float(ms[-1][2])
        per = t / i
        return (max(0.0, per * (n - i) - since_last),
                f"fit-moe {i}/{n} shards at {per:.0f} s/shard")
    if cmd == "kl-ladder":
        args = [str(a) for a in st.get("args", [])]
        total = 1 if "--preflight" in args else _count_flag(args, "--rung") * _count_flag(args, "--cache")
        done = len(_KL_CELL.findall(text))
        fresh = done - len(_KL_RESUMED.findall(text))
        if not total or fresh <= 0 or started is None:
            return None
        per = max(0.0, mtime - started) / fresh
        return (max(0.0, per * (total - done) - since_last),
                f"kl-ladder {done}/{total} cells at {per:.0f} s/cell")
    return None


def step_eta(st: dict, rec: dict, sdir: pathlib.Path, qroot: pathlib.Path,
             qdir: pathlib.Path | None = None, now: float | None = None) -> dict:
    """{'eta_s': float | None, 'basis': str} for one step."""
    now = time.time() if now is None else now
    status = rec.get("status")
    if status not in ("running", "pending"):
        return {"eta_s": 0.0 if status in ("pass", "skipped") else None, "basis": status or "?"}
    started = _ts(rec.get("started"))
    if status == "running":
        stdout = pathlib.Path(sdir) / "stdout"
        try:          # this attempt's start: retries rename the earlier stdout away
            started = getattr(os.stat(stdout), "st_birthtime", None) or started
        except OSError:
            pass
        got = _from_progress(st, stdout, started, now)
        if got:
            return {"eta_s": round(got[0]), "basis": got[1]}
    past = history(qroot, st, exclude=qdir)
    if past:
        med = statistics.median(past)
        elapsed = (now - started) if (status == "running" and started) else 0.0
        if elapsed > med:
            return {"eta_s": None, "basis": f"over the median of {len(past)} past run(s) "
                                            f"({med:.0f} s) by {elapsed - med:.0f} s"}
        return {"eta_s": round(med - elapsed),
                "basis": f"median of {len(past)} past run(s) of this step ({med:.0f} s)"}
    return {"eta_s": None, "basis": "no progress lines in its output and no past run of this step"}


def queue_eta(qdir: pathlib.Path, qroot: pathlib.Path, now: float | None = None) -> dict:
    """Per-step ETAs plus the queue's: the sum, or None if any open step is unknown."""
    now = time.time() if now is None else now
    qdir = pathlib.Path(qdir)
    s = json.loads((qdir / "state.json").read_text())
    try:
        q = json.loads((qdir / "queue.json").read_text())
    except (OSError, ValueError):
        q = {"steps": [{"name": r["name"]} for r in s["steps"]]}
    steps, total = [], 0.0
    for i, (st, rec) in enumerate(zip(q["steps"], s["steps"])):
        e = step_eta(st, rec, qdir / "steps" / f"{i:02d}-{_slug(st.get('name', ''))}", qroot, qdir, now)
        steps.append(e)
        if rec.get("status") in ("running", "pending"):
            total = None if (total is None or e["eta_s"] is None) else total + e["eta_s"]
    open_ = s.get("status") in ("running", "created")
    return {"steps": steps, "eta_s": round(total) if open_ and total is not None else None,
            "finish": (dt.datetime.fromtimestamp(now + total).astimezone().isoformat(timespec="minutes")
                       if open_ and total is not None else None)}


def _slug(s):  # same as run_queue._slug
    return re.sub(r"[^\w.-]", "_", s)[:40]


def fmt(eta_s) -> str:
    if eta_s is None:
        return "ETA unknown"
    s = int(eta_s)
    return f"ETA {s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"ETA {s // 60}m{s % 60:02d}s"
