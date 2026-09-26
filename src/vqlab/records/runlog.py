"""Run log: one JSONL line when any `vqlab <cmd>` starts, one when it ends.

Every invocation, on every machine that installs vqlab, is recorded with its
full argv, the code it ran (commit + dirty files) and the library versions --
so "which parameters did that fit use?" is a grep, not an archaeology dig
through shell history and EXPERIMENTS.md. The START line is written before
the tool runs, so a run that is killed (watchdog, OOM, closed laptop) still
leaves a record; an unmatched start IS the record that it died.

    ~/.vqlab/runs.jsonl            (override the dir with VQLAB_LOG_DIR)

Cost is two appends and two `git` calls per run. Logging must never break or
slow a run: every failure here is swallowed. Tools that write a build record
(provenance.py) pick up VQLAB_RUN_ID, so an artifact points at its log line.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import time
import uuid

_T0 = {}


def _path() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("VQLAB_LOG_DIR", pathlib.Path.home() / ".vqlab"))
    d.mkdir(parents=True, exist_ok=True)
    return d / "runs.jsonl"


def _append(rec) -> None:
    line = json.dumps(rec, default=str) + "\n"
    # one write() on an O_APPEND fd: concurrent sessions do not interleave
    fd = os.open(_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def start(cmd, argv):
    # Every invocation gets its OWN id. A nested vqlab call (a tool running
    # another tool, or selftest driving the CLI) inherits the caller's id in
    # the environment; reusing it made the child's records look like the
    # parent's. The caller's id is kept as parent_run_id instead.
    parent = os.environ.get("VQLAB_RUN_ID")
    run_id = uuid.uuid4().hex[:16]
    os.environ["VQLAB_RUN_ID"] = run_id          # children + build records
    _T0[run_id] = time.time()
    try:
        import provenance
        rec = {"event": "start", "run_id": run_id, "parent_run_id": parent,
               "time": _now(), "cmd": cmd,
               "argv": list(argv), "cwd": os.getcwd(), "pid": os.getpid(),
               "code": provenance.code_state(), "env": provenance.env_state()}
        rec["code"].pop("repo", None)
        _append(rec)
    except Exception:
        pass
    return run_id


def end(run_id, rc):
    try:
        _append({"event": "end", "run_id": run_id, "time": _now(), "rc": rc,
                 "seconds": round(time.time() - _T0.pop(run_id, time.time()), 1)})
    except Exception:
        pass


def main(argv=None) -> int:
    """`vqlab runs`: the last N invocations, paired start/end."""
    import argparse
    ap = argparse.ArgumentParser(prog="vqlab runs", description="show the run log")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--cmd", help="only this command")
    ap.add_argument("--grep", help="substring of argv")
    ap.add_argument("--json", action="store_true", help="raw start records")
    a = ap.parse_args(argv)
    p = _path()
    if not p.exists():
        print(f"no runs logged yet ({p})")
        return 0
    starts, ends = {}, {}
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (starts if r.get("event") == "start" else ends)[r.get("run_id")] = r
    rows = [s for s in starts.values()
            if (not a.cmd or s.get("cmd") == a.cmd)
            and (not a.grep or a.grep in " ".join(s.get("argv", [])))][-a.n:]
    for s in rows:
        if a.json:
            print(json.dumps({**s, "end": ends.get(s["run_id"])}))
            continue
        e = ends.get(s["run_id"])
        st = (f"rc={e['rc']} {e['seconds']}s" if e else "NO END (killed or running)")
        c = s.get("code", {})
        print(f"{s['time']}  {s['run_id']}  {s['cmd']:14s} {st:24s} "
              f"{str(c.get('commit'))[:8]}{'+dirty' if c.get('dirty') else ''}  "
              f"{' '.join(s.get('argv', []))[:120]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
