"""vqlab — one CLI over the standalone pipeline scripts.

The command table is vqlab/commands.py (pure data). The underlying scripts are deliberately standalone (each is a complete,
auditable tool with its own argparse surface, and several run at module
scope). This dispatcher sets sys.argv and executes the chosen script in
its own right, so `vqlab fit-moe --help` shows the script's full surface
and behavior is identical to running the script directly.

Commands are grouped by pipeline stage: `vqlab fit moe` is `vqlab fit-moe`.
The flat names are the canonical keys (COMMANDS, BUILD_OUTPUTS, the run
log, box_quiet, queue steps all speak them) and keep working unchanged; a
namespaced call is resolved to its flat name before anything else runs, so
the two spellings are one code path, not two.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PKG = Path(__file__).parent

# The registry is vqlab/commands.py (pure data, shared with the MCP server
# and the queue). These names are its views, kept for existing callers.
from vqlab import commands as _registry
from vqlab.commands import NAMESPACES, SPELLING, resolve

# Old shape, {flat name: (pre-split filename, help)}: kept for one release.
COMMANDS = {c.name: (c.file, c.help) for c in _registry.COMMANDS.values()}
assert sorted(SPELLING) == sorted(COMMANDS) and len(SPELLING) == sum(
    len(subs) for _, subs in NAMESPACES.values()), "every command in exactly one namespace"


def routed(text):
    """Flat names of the commands a Markdown text routes to: every `code`
    span read as a command line, namespaced (`fit moe ...`) or flat
    (`fit-moe ...`), with or without a leading `vqlab`. The selftest's
    CONTEXT.md check and its test share this, so both spellings count."""
    import re
    out = set()
    for span in re.findall(r"`([^`]+)`", text):
        words = span.split()
        if words and words[0] == "vqlab":
            words = words[1:]
        flat, _ = resolve(words)
        if flat:
            out.add(flat)
    return out


def _namespace_help(ns):
    blurb, subs = NAMESPACES[ns]
    print(f"vqlab {ns} -- {blurb}\n")
    for sub, flat in subs.items():
        alias = "" if flat == sub else f"  [alias: {flat}]"
        print(f"  {sub:22s} {COMMANDS[flat][1]}{alias}")
    print(f"\n`vqlab {ns} <command> --help` shows each command's full surface.")


def _help():
    print("vqlab — size-targeted VQ quantization for MLX\n\nnamespaces (pipeline order):")
    for ns, (blurb, subs) in NAMESPACES.items():
        print(f"  {ns:8s} {blurb} ({len(subs)})")
    for ns, (_, subs) in NAMESPACES.items():
        print(f"\n{ns}:")
        for sub, flat in subs.items():
            alias = "" if flat == sub else f"  [alias: {flat}]"
            print(f"  {ns + ' ' + sub:28s} {COMMANDS[flat][1]}{alias}")
    print("\nEvery flat name is an alias kept for this release (`vqlab fit-moe` is")
    print("`vqlab fit moe`; `vqlab pack` is `vqlab build pack`) and behaves identically.")
    print("`vqlab <namespace>` lists one namespace;")
    print("`vqlab <namespace> <command> --help` shows each command's full surface.")
    print("Read METHODOLOGY.md before publishing any number")
    print("(https://github.com/noahzelezny/VQLab/blob/master/METHODOLOGY.md).")


# Build tools whose output is an ARTIFACT: after a clean exit the CLI writes
# its build record (records/provenance.py) unless the tool already wrote a
# richer one in this run (the fitters do). Candidates are tried in order; a
# FILE output is recorded in the artifact dir that holds it, and only a dir
# with a config.json counts, so a sidecar written to a scratch dir never
# stamps that dir -- nor falls back to stamping --model: mtp-pack once
# recorded two scratch probe heads as builds of the model they read. In-place tools produce an AMENDMENT: the prior record is
# kept in vqlab_provenance.history.jsonl and linked by id.
BUILD_OUTPUTS = {
    "pack": ("--out",), "pack-dense": ("--out",), "build-dense": ("--out",), "slice": ("--out",),
    "stream-convert": ("--out",), "sanitize-stream": ("--out",), "teacher-prep": ("--out",), "mix": ("--out",), "ple-swap": ("--out",), "unpack-dense": ("--out",),
    "pack-ple": ("--artifact",), "graft": ("--artifact",), "graft-extras": ("--artifact",), "splice-ple": ("--artifact",),
    "bundle": ("--artifact",), "rebundle-dense": ("--artifact",),
    "patch-arch": ("--out",), "vision-layout": ("--out", 0),
    "mtp-pack": ("--out",), "mtp-extract": ("--out",),
    "mtp-graft": ("--out",),
}


# Flags under which a build command writes nothing: the automatic build
# record must never fire for them (a --dry-run once wrote a record into the
# teacher's folder; --help exits 0 too). vision-layout's report mode (no
# --fix) is handled by name below.
READ_ONLY_FLAGS = frozenset({"--dry-run", "--plan", "-h", "--help"})


def _parse(argv):
    named, pos, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a.split("=", 1)
                named[k] = v
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                named[a] = argv[i + 1]
                i += 1
            else:
                named[a] = True
        else:
            pos.append(a)
        i += 1
    return named, pos


def _record_build(cmd, rest, script):
    """Write the build record for a build command's artifact. Never raises:
    a record failure is reported, and must not turn a finished build red."""
    import os
    try:
        named, pos = _parse(rest)
        if cmd == "vision-layout" and "--fix" not in named:
            return                                    # report-only mode
        if READ_ONLY_FLAGS.intersection(rest):
            return                                    # nothing was built
        target = None
        for c in BUILD_OUTPUTS[cmd]:
            v = pos[c] if isinstance(c, int) and len(pos) > c else named.get(c)
            if not isinstance(v, str):
                continue
            d = Path(v)
            d = d.parent if d.is_file() else d
            if d.is_dir() and (d / "config.json").exists():
                target = d
                break
        if target is None:
            return
        from vqlab.records import provenance
        old = target / provenance.RECORD
        if old.exists():
            import json
            if json.loads(old.read_text())["tool"].get("run_id") == os.environ.get("VQLAB_RUN_ID"):
                return                                # the tool wrote its own
        inputs = []
        for k, v in list(named.items()) + [(f"arg{i}", v) for i, v in enumerate(pos)]:
            if isinstance(v, str) and v.startswith(("/", ".", "~")) and Path(v).exists() \
                    and Path(v).resolve() != target.resolve() \
                    and Path(v).resolve().parent != target.resolve():
                inputs.append((k.lstrip("-"), Path(v) if Path(v).is_dir() else Path(v).parent))
        provenance.write_build_record(
            target, tool=cmd, script=str(script), argv=[cmd, *rest], inputs=inputs,
            method={"recorded_by": "cli", "note": "this tool's settings are its argv"})
    except Exception as e:
        print(f"vqlab: build record NOT written for {cmd}: {type(e).__name__}: {e}",
              file=sys.stderr)


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _help()
        return 0
    if argv[0] in NAMESPACES and (len(argv) == 1 or (
            argv[1] in ("-h", "--help") and argv[0] not in COMMANDS)):
        _namespace_help(argv[0])
        return 0
    cmd, rest = resolve(argv)
    if cmd is None:
        if argv[0] in NAMESPACES:
            print(f"unknown command: vqlab {argv[0]} {argv[1]}", file=sys.stderr)
            _namespace_help(argv[0])
        else:
            print(f"unknown command: {argv[0]}", file=sys.stderr)
        return 2
    import vqlab
    vqlab.use_knurlogic_arch()  # every command loads Knurlogic's archs (VQLAB_VENDORED_ARCH=0: stock)
    if not any(a in ("-h", "--help") for a in rest):
        from vqlab.bench import box_quiet   # timed commands refuse a busy Mac (rule III)
        box_quiet.guard(cmd)
    module = _registry.COMMANDS[cmd].module
    import importlib.util
    script = importlib.util.find_spec(module).origin
    from vqlab import config as _cfg   # <teachers>/X -> this box's teachers root (a local copy on a box with one)
    rest = [_cfg.expand(a) for a in rest]
    sys.argv = [str(script), *rest]
    from vqlab.records import runlog
    run = runlog.start(cmd, rest)
    rc = 0
    try:
        # By module name, not file path: no file lookup, and relative and
        # absolute imports inside the tool resolve as in any import.
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        raise
    except BaseException as e:
        rc = f"{type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        runlog.end(run, rc)
        if rc == 0 and cmd in BUILD_OUTPUTS:
            _record_build(cmd, rest, script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
