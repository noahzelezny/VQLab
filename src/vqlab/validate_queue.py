#!/usr/bin/env python3
"""Overnight validation queue — drain artifacts through the release gates
while nobody is at the keyboard; NEVER publish.

The gates themselves already exist (`check`, `check-release`, `smoke`,
`bundle-accept`, `score`, `kl`, ...). This is only the orchestrator: a
JSONL queue of artifacts, drained sequentially (single GPU, single
writer), each entry running its gate commands via `python -m vqlab.cli`
in a subprocess with output captured to a per-entry log. The morning
summary is one file. Publishing stays a human command — for every entry
that passes, the summary prints the `vqlab publish` line to review, per
docs/PUSH-RUNBOOK.md.

State lives in caches/validate/ (gitignored):
    queue.jsonl          one entry per line: artifact, gates, status
    <name>-<ts>.log      full gate output per run
    MORNING.md           summary of the latest `run` (overwritten)

Usage:
    vqlab validate add --artifact <dir> [--gates check,check-release,bundle-accept]
    vqlab validate list
    vqlab validate run [--timeout-min 90]
    vqlab validate clear-done

A queue entry may instead carry explicit commands (for gates whose args
are not just --artifact, e.g. score):
    vqlab validate add --artifact <dir> --cmd "check --artifact {art}" \
        --cmd "score --model {art} --tasks wikitext"
`{art}` expands to the artifact path. --gates is shorthand for
`--cmd "<gate> --artifact {art}"` per gate.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATE = ROOT / "caches" / "validate"
QUEUE = STATE / "queue.jsonl"
DEFAULT_GATES = "check,check-release,bundle-accept"

# the gates predate the queue and their artifact-arg spellings differ;
# map each to its real surface so --gates shorthand just works.
_GATE_CMD = {
    "check": "check {art}",                       # positional (check_all.py)
    "bundle-accept": "bundle-accept {art}",       # positional (bundle_accept.py)
    "check-release": "check-release --artifact {art}",
    "check-bundle": "check-bundle --artifact {art}",
    "smoke": "smoke {art}",                       # positional (smoke.py)
    "verify": "verify --artifact {art}",
}


def _load():
    if not QUEUE.exists():
        return []
    return [json.loads(ln) for ln in QUEUE.read_text().splitlines() if ln.strip()]


def _save(entries):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(e) + "\n" for e in entries))
    tmp.replace(QUEUE)


def cmd_add(a):
    art = pathlib.Path(a.artifact).resolve()
    if not art.is_dir():
        sys.exit(f"artifact not a directory: {art}")
    if a.cmd:
        cmds = list(a.cmd)
    else:
        cmds = []
        for g in (g.strip() for g in a.gates.split(",")):
            if not g:
                continue
            if g not in _GATE_CMD:
                sys.exit(f"no --gates shorthand for {g!r} (known: "
                         f"{', '.join(_GATE_CMD)}); pass it via --cmd")
            cmds.append(_GATE_CMD[g])
    entries = _load()
    entries.append({
        "artifact": str(art),
        "name": art.name,
        "cmds": cmds,
        "note": a.note or "",
        "status": "pending",
        "added": dt.datetime.now().isoformat(timespec="seconds"),
    })
    _save(entries)
    print(f"queued {art.name}: {len(cmds)} gate command(s)")


def cmd_list(_a):
    entries = _load()
    if not entries:
        print("queue empty")
        return
    for i, e in enumerate(entries):
        print(f"[{i}] {e['status']:8s} {e['name']}  "
              f"({len(e['cmds'])} cmds{', ' + e['note'] if e['note'] else ''})")


def cmd_clear_done(_a):
    entries = _load()
    kept = [e for e in entries if e["status"] == "pending"]
    _save(kept)
    print(f"removed {len(entries) - len(kept)}, kept {len(kept)} pending")


def _run_entry(e, log_path, timeout_s):
    """Run each gate command; stop at the first failure. Returns (ok, ran)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with open(log_path, "a") as log:
        for cmd in e["cmds"]:
            argv = [sys.executable, "-m", "vqlab.cli"] + [
                p.replace("{art}", e["artifact"]) for p in shlex.split(cmd)]
            log.write(f"\n=== {dt.datetime.now().isoformat(timespec='seconds')}"
                      f" $ {' '.join(argv)}\n")
            log.flush()
            try:
                rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout_s, cwd=ROOT).returncode
            except subprocess.TimeoutExpired:
                log.write(f"\n*** TIMEOUT after {timeout_s}s\n")
                return False, cmd
            log.write(f"=== exit {rc}\n")
            if rc != 0:
                return False, cmd
    return True, None


def cmd_run(a):
    entries = _load()
    pending = [e for e in entries if e["status"] == "pending"]
    if not pending:
        print("nothing pending")
        return
    STATE.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    lines = [f"# Validation run {ts}", ""]
    for e in pending:
        log_path = STATE / f"{e['name']}-{ts}.log"
        print(f"validating {e['name']} ... ", end="", flush=True)
        t0 = time.time()
        ok, failed_cmd = _run_entry(e, log_path, a.timeout_min * 60)
        mins = (time.time() - t0) / 60
        e["status"] = "pass" if ok else "FAIL"
        e["ran"] = dt.datetime.now().isoformat(timespec="seconds")
        e["log"] = str(log_path)
        _save(entries)  # persist after each artifact — a crash loses nothing
        print(f"{e['status']} ({mins:.1f} min)")
        lines.append(f"## {e['name']} — **{e['status']}** ({mins:.1f} min)")
        lines.append(f"- log: `{log_path}`")
        if ok:
            lines.append(
                "- ready to review for publish (gated, manual — see "
                "docs/PUSH-RUNBOOK.md):")
            repo = e["name"].replace("--", "/", 1) if "--" in e["name"] \
                else f"TheDrainFlorist/{e['name']}"
            lines.append(
                f"  `python -m vqlab.cli publish --artifact \"{e['artifact']}\""
                f" --repo {repo} --files model.py"
                f" README.md --message \"...\"`")
        else:
            lines.append(f"- failed at: `{failed_cmd}`")
        lines.append("")
    (STATE / "MORNING.md").write_text("\n".join(lines))
    print(f"\nsummary: {STATE / 'MORNING.md'}")


def main():
    ap = argparse.ArgumentParser(prog="vqlab validate")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("add", help="queue an artifact for validation")
    p.add_argument("--artifact", required=True)
    p.add_argument("--gates", default=DEFAULT_GATES)
    p.add_argument("--cmd", action="append",
                   help="explicit vqlab.cli command ({art} expands); "
                        "overrides --gates; repeatable")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("list", help="show the queue")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("run", help="drain all pending entries sequentially")
    p.add_argument("--timeout-min", type=int, default=90,
                   help="per-gate-command timeout (default 90)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("clear-done", help="drop pass/FAIL entries")
    p.set_defaults(fn=cmd_clear_done)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
