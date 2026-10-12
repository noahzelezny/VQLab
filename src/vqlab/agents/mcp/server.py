"""vqlab mcp — the lab's MCP server (stdio JSON-RPC, stdlib only).

Nobody runs the lab by hand any more; agents do. This is the interface they
get. One server per box (it mirrors exo): every tool acts on THIS machine,
and a client that wants the other box talks to the other box's server.

    PYTHONPATH=src python -m vqlab.cli mcp            # serve on stdio
    PYTHONPATH=src python -m vqlab.cli mcp --list     # print the tool table

Design rules (each one paid for):

* ``where_is`` answers deterministically. A model that searches directories
  by hand declared a teacher cache "gone" four times in one night
  (2026-09-15) while it sat in ``Exo Models/``. This tool walks the lab's
  roots and returns the path or NOT_FOUND *with the roots it searched*. It
  cannot editorialize.
* ``run`` launches ALLOWLISTED ``vqlab`` subcommands only, detached (nohup +
  its own process group, so it survives the client), under the box's GPU
  lease (an advisory flock that dies with its holder; the file is
  ``vqlab.config.gpu_lease``, shareable with other GPU tenants), with a
  retry supervisor for the commands that resume from checkpoints. It refuses
  any absolute path outside the lab's roots (artifacts never go on the
  internal disk) and refuses to start while an exo instance is placed.
* ``findings_append`` is the only way the measured record is written from
  here. It allocates the next F-number, requires the pre-registered
  prediction, and takes the verdict from a fixed vocabulary. A falsified
  prediction is recorded as falsified.
* ``publish`` is NOT exposed. Publishing is irreversible and is a human's
  action.

No third-party packages: the exo envs this runs in must not have anything
pip-installed into them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vqlab import config
from vqlab.agents.mcp.common import ToolError
from vqlab.agents.mcp.tools_findings import (VERDICTS, t_findings_append, t_findings_tail,
                                             t_next_f_number, t_release_f_number)
from vqlab.agents.mcp.tools_runs import (RUN_ALLOWLIST, supervise, t_gpu_state, t_list_runs,
                                         t_queue_status, t_run, t_status, t_stop)
from vqlab.agents.mcp.tools_storage import (t_artifact_config, t_disk_free, t_list_artifacts,
                                            t_read_doc, t_where_is)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vqlab"
SERVER_VERSION = "0.1.0"

# --------------------------------------------------------------------------- #
# Tool table (schema is what the model sees — keep descriptions honest)
# --------------------------------------------------------------------------- #

def _schema(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


S = lambda d="": {"type": "string", "description": d}
I = lambda d="": {"type": "integer", "description": d}
B = lambda d="": {"type": "boolean", "description": d}
L = lambda d="": {"type": "array", "items": {"type": "string"}, "description": d}

TOOLS: Dict[str, Dict[str, Any]] = {
    "where_is": {
        "fn": t_where_is, "readonly": True,
        "description": "Find an artifact, fit, teacher cache, corpus or file by (sub)name across the lab's "
                       "storage roots. Deterministic: returns matches, or NOT_FOUND with the roots searched. "
                       "Use this instead of guessing or listing directories.",
        "schema": _schema({"name": S("substring of the directory/file name, ≥3 chars"),
                           "kind": {"type": "string", "enum": ["any", "dir", "file"]},
                           "max_depth": I("directory depth under each root (default 3)")}, ["name"]),
    },
    "list_artifacts": {
        "fn": t_list_artifacts, "readonly": True,
        "description": "Directories carrying a config.json under the lab roots (or under one root), with packed size.",
        "schema": _schema({"root": S("optional: one lab root or sub-directory"), "max_depth": I()}, []),
    },
    "artifact_config": {
        "fn": t_artifact_config, "readonly": True,
        "description": "Read a shipped artifact's config.json — the authoritative record of what shipped "
                       "(authority order: config → FINDINGS-LOG → EXPERIMENTS.md).",
        "schema": _schema({"path": S("artifact dir or config.json path"), "keys": L("optional key subset")}, ["path"]),
    },
    "read_doc": {
        "fn": t_read_doc, "readonly": True,
        "description": "Read a slice of a lab document: docs/*, AGENTS.md, METHODOLOGY.md, REPRODUCING.md, "
                       "docs/FINDINGS.md (the law book) or EXPERIMENTS.md.",
        "schema": _schema({"path": S("repo-relative path"), "start": I("1-based first line"), "lines": I("max 1000")}, ["path"]),
    },
    "run": {
        "fn": t_run, "readonly": False,
        "description": "Launch an allowlisted vqlab subcommand on THIS box, detached, under the GPU lease. "
                       "Returns immediately with a run_id; poll status. Refuses paths outside the lab roots, "
                       "refuses while an exo instance is placed, refuses while the lease is held (deferring is normal). "
                       "publish is not available here.",
        "schema": _schema({"cmd": {"type": "string", "enum": sorted(RUN_ALLOWLIST)},
                           "args": L("arguments exactly as for `vqlab <cmd>`"),
                           "tag": S("short label for the run id"),
                           "retries": I("checkpoint-resume retries (resumable cmds only)"),
                           "force": B("skip the exo-placement check (you know why)")}, ["cmd"]),
    },
    "status": {
        "fn": t_status, "readonly": True,
        "description": "State of a run on this box: status, exit code, attempts, log tail, result path.",
        "schema": _schema({"run_id": S(), "tail": I("log lines (default 40)")}, ["run_id"]),
    },
    "stop": {
        "fn": t_stop, "readonly": False,
        "description": "SIGTERM a run's process group on this box.",
        "schema": _schema({"run_id": S()}, ["run_id"]),
    },
    "list_runs": {
        "fn": t_list_runs, "readonly": True,
        "description": "Recent runs launched on this box.",
        "schema": _schema({"n": I()}, []),
    },
    "gpu_state": {
        "fn": t_gpu_state, "readonly": True,
        "description": "Who holds this box's GPU lease and which exo instances are placed; with "
                       "[boxes.*] in the vqlab config, the same for every other box over ssh "
                       "(lease holder, load, GPU-heavy processes), plus every box's reservation "
                       "(`vqlab reserve`: who has it, until when). Check it before loading a big "
                       "model on a box that might be fitting or is reserved.",
        "schema": _schema({"boxes": B("also report the other configured boxes (default true)")}, []),
    },
    "queue_status": {
        "fn": t_queue_status, "readonly": True,
        "description": "State of `vqlab queue` runs on this box from state.json: per-step status and "
                       "reasons, terminal=true once passed/failed/stopped/deferred/died, and an ETA "
                       "(eta_s, finish, per-step eta_basis) from the running step's own progress lines "
                       "or past runs of the same step -- null means unknown, not zero. Use this (or "
                       "`vqlab queue wait`) to know a queue finished; never grep its log.",
        "schema": _schema({"name": S("substring of the queue dir name"), "n": I("most recent n (default 5)")}, []),
    },
    "disk_free": {
        "fn": t_disk_free, "readonly": True,
        "description": "Free GiB on every configured storage root. Check before any multi-hour writer.",
        "schema": _schema({}, []),
    },
    "next_f_number": {
        "fn": t_next_f_number, "readonly": False,
        "description": "The next free F-number. Pass reserve=true to CLAIM it "
                       "atomically before a long experiment -- a bare read "
                       "races any concurrent session, because the log only "
                       "changes at commit time (two sessions both took 151 on "
                       "2026-09-19). Reservations expire after 6 h and are "
                       "released by findings_append.",
        "schema": _schema({"reserve": B(), "owner": S()}, []),
    },
    "release_f_number": {
        "fn": t_release_f_number, "readonly": False,
        "description": "Give back a reserved F-number whose experiment was abandoned.",
        "schema": _schema({"n": I()}, ["n"]),
    },
    "findings_tail": {
        "fn": t_findings_tail, "readonly": True,
        "description": "The last n finding headlines with their line numbers (read the body with read_doc).",
        "schema": _schema({"n": I()}, []),
    },
    "findings_append": {
        "fn": t_findings_append, "readonly": False,
        "description": "Append the next F-entry to FINDINGS-LOG.md. Every field is required; the pre-registered "
                       "prediction is written verbatim and a falsified prediction stays falsified. "
                       "Verdict ∈ CONFIRMED | FALSIFIED | VOID | NULL | CORRECTS (needs corrects='F<n>').",
        "schema": _schema({"use_reserved": I("an F-number THIS session already claimed "
                                            "with next_f_number(reserve=true); omit to "
                                            "allocate fresh. Without it a reserve-then-"
                                            "append burns a number"),
                           "headline": S(), "artifact": S("artifact name/path the number belongs to"),
                           "instrument": S("scoring path + batching, e.g. 'vqlab score, prose corpus, resident'"),
                           "prediction": S("what was pre-registered before the run"),
                           "measured": S("the number(s)"),
                           "verdict": {"type": "string", "enum": list(VERDICTS)},
                           "body": S("narrative / tables"), "corrects": S("F<n> when verdict=CORRECTS")},
                          ["headline", "artifact", "instrument", "prediction", "measured", "verdict"]),
    },
}


def tool_list() -> List[Dict[str, Any]]:
    return [{"name": n, "description": t["description"], "inputSchema": t["schema"],
             "annotations": {"readOnlyHint": t["readonly"], "destructiveHint": not t["readonly"]}}
            for n, t in TOOLS.items()]


def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Dispatch; returns an MCP tools/call result (never raises for tool errors)."""
    t = TOOLS.get(name)
    if t is None:
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(
            {"error": "UNKNOWN_TOOL", "message": name, "tools": list(TOOLS)})}]}
    try:
        out = t["fn"](**(arguments or {}))
        return {"isError": False, "content": [{"type": "text", "text": json.dumps(out, default=str)}]}
    except ToolError as te:
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(te.as_result(), default=str)}]}
    except TypeError as exc:  # bad/missing arguments
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(
            {"error": "INVALID_ARG", "message": str(exc), "schema": t["schema"]})}]}
    except Exception as exc:  # surface, never crash the server
        return {"isError": True, "content": [{"type": "text", "text": json.dumps(
            {"error": "INTERNAL", "message": f"{type(exc).__name__}: {exc}"})}]}


# --------------------------------------------------------------------------- #
# JSON-RPC over stdio (newline-delimited, per the MCP stdio transport)
# --------------------------------------------------------------------------- #

def handle(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method == "initialize":
        return _ok(mid, {"protocolVersion": PROTOCOL_VERSION,
                         "capabilities": {"tools": {"listChanged": False}},
                         "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION,
                                        "host": config.short_host()},
                         "instructions": ("vqlab lab server for THIS box. Use where_is before claiming "
                                          "anything is missing. Read the law book (read_doc "
                                          "docs/FINDINGS.md) before proposing an experiment. "
                                          "Pre-register a prediction before you run; record it with "
                                          "findings_append. publish is a human action.")})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": tool_list()})
    if method == "tools/call":
        return _ok(mid, call_tool(params.get("name", ""), params.get("arguments")))
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


def _ok(mid: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(inp=None, out=None) -> int:
    inp = inp or sys.stdin.buffer
    out = out or sys.stdout.buffer
    for raw in inp:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            try:
                resp = handle(msg)
            except Exception as exc:
                resp = {"jsonrpc": "2.0", "id": msg.get("id"),
                        "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        if resp is not None:
            out.write((json.dumps(resp, default=str) + "\n").encode("utf-8"))
            out.flush()
    return 0

# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="vqlab mcp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the tool table and exit")
    ap.add_argument("--call", nargs=2, metavar=("TOOL", "JSON_ARGS"), help="call one tool and print the result")
    ap.add_argument("--supervise", metavar="RUN_DIR", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.supervise:
        return supervise(Path(a.supervise))
    if a.list:
        for t in tool_list():
            print(f"{t['name']:16s} {'ro ' if t['annotations']['readOnlyHint'] else 'RW '} {t['description']}")
        return 0
    if a.call:
        print(json.dumps(call_tool(a.call[0], json.loads(a.call[1])), indent=1))
        return 0
    return serve()


if __name__ == "__main__":
    sys.exit(main())
