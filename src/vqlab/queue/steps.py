"""Queue steps: the schema, the arguments a step really runs with (its
preflight version included), and the checks made before it may run.

A step names a vqlab command by NAME (`cmd`, resolved through
vqlab.commands) or a repo script by path (`script`); nothing here imports a
tool module. See queue/runner.py's docstring for the queue file format.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import sys

from vqlab import commands as _commands
from vqlab import config as _config
from vqlab.gate import pin as pinmod
from vqlab.util.clock import slug
from vqlab.util.fs import GIB

SCORING = {"score", "kl", "kl-ladder", "kl-pair", "tasks", "decode-timeline",
           "decode-ladder", "active-bytes"}
# Flags whose value is an OUTPUT path: it need not exist yet, its parent must.
OUT_FLAGS = {"--out", "--out-dir", "--output", "-o", "--per-pos-dir",
             "--parts", "--parts-dir", "--out-parts", "--save", "--log", "--json-out", "--cache-out",
             "--kl-per-position", "--keep-dir"}
_PATHLIKE = re.compile(r"""["'](/(?:Volumes|Users|opt|tmp|private)/[^"'\n]+|(?:src|scripts|research|families)/[^"'\s\n]+)["']""")
GPU_FREE = _commands.mcp_gpu_free()
PREFLIGHT_MARGIN = 10 << 30        # never take scratch below this (the step itself is sized separately)


# ------------------------------------------------------------------ schema
def _preflight_problem(st) -> str | None:
    pf = st.get("preflight")
    pfk = [k for k in (pf or {}) if k != "writes_gib"]
    if pf is not None and not (isinstance(pf, dict) and len(pfk) == 1
                               and pfk[0] in ("args", "append", "skip", "full")
                               and ("writes_gib" not in pf or "skip" not in pf)):
        return ("preflight must be one of {args}, {append}, {skip}, {full}; "
                "writes_gib may be added to any of them but skip")
    if pf and (pf.get("append") == [] or pf.get("args") == st.get("args")):
        # a "preflight" identical to the real step is a full-size run
        # into scratch (2026-09-27: two stream-converts left 201 GiB in
        # queue-preflight/ and filled the SSD). Say so explicitly.
        return ("preflight is identical to the real step; give a small "
                "real version, or {\"full\": \"why\"} to run it full-size "
                "on purpose, or {\"skip\": \"why\"}")
    return None


def validate(q) -> list[str]:
    """Problems with a queue file (empty: valid). Normalizes each `cmd` to
    its flat name in place."""
    errs = []
    if not isinstance(q.get("steps"), list) or not q["steps"]:
        return ["queue has no steps"]
    names = set()
    for i, st in enumerate(q["steps"]):
        tag = f"step {i} ({st.get('name', '?')})"
        if not st.get("name"):
            errs.append(f"{tag}: no name")
        elif st["name"] in names:
            errs.append(f"{tag}: duplicate name")
        names.add(st.get("name"))
        if bool(st.get("cmd")) == bool(st.get("script")):
            errs.append(f"{tag}: give exactly one of cmd / script")
        if isinstance(st.get("cmd"), str) and st["cmd"]:
            # `fit moe` and `fit-moe` are one command; the queue (SCORING,
            # GPU_FREE, RESUMABLE, the saved queue.json) speaks the flat name
            flat, extra = _commands.resolve(st["cmd"].split())
            if flat is None or extra:
                errs.append(f"{tag}: unknown command {st['cmd']!r}")
            else:
                st["cmd"] = flat
        if st.get("cmd") == "publish":
            errs.append(f"{tag}: publish is a human action, never queued")
        if st.get("on_fail", "stop") not in ("stop", "continue"):
            errs.append(f"{tag}: on_fail must be stop or continue")
        why = _preflight_problem(st)
        if why:
            errs.append(f"{tag}: {why}")
    return errs


def needs_gpu(st) -> bool:
    return not (st.get("cmd") in GPU_FREE or st.get("gpu") is False)


def retries(st, preflight) -> int:
    if preflight:          # a small run that fails says so at once; a retry hides nothing
        return 0
    n = st.get("retries")
    if n is None:
        n = _commands.DEFAULT_RETRIES if st.get("cmd") in _commands.RESUMABLE else 0
    return n


def expect(st, tree: pathlib.Path, preflight) -> dict:
    """What step_verdict demands of the step's output."""
    exp = dict(st.get("expect") or {"stdout_nonempty": True})
    if preflight:
        # a preflight is a SMALL run: it proves the step runs and says
        # something, not that it produced the full run's lines/files
        exp = {"stdout_nonempty": True,
               **({"allow_traceback": True} if exp.get("allow_traceback") else {})}
    files = [_config.expand(str(f)) for f in exp.get("files", ())]
    exp["files"] = [str(f if pathlib.Path(f).is_absolute() else tree / f)
                    for f in files] if not preflight else []
    return exp


# ------------------------------------------------------------------ paths
_NAMED = re.compile(r"^[\w.+-]+=(?=[/.~]|src/|scripts/)")


def strip_name(v):
    """`prose=/path` -> `/path` (kl-ladder --cache, --rung; kl-pair)."""
    return _NAMED.sub("", v, count=1)


def path_args(args):
    """(inputs, outputs) among a step's args: values that look like paths."""
    ins, outs = [], []
    prev = None
    for a in args:
        if not a.startswith("-"):
            a = strip_name(a)
        if a.startswith("-"):
            prev = a.split("=", 1)[0]
            if "=" in a:
                v = a.split("=", 1)[1]
                (outs if prev in OUT_FLAGS else ins).append(v) if "/" in v else None
            continue
        if "/" in a and not a.startswith(("http:", "https:")):
            (outs if prev in OUT_FLAGS else ins).append(a)
        prev = None
    return ins, outs


def input_bytes(args, tree):
    """Rough size a full-size step writes: the safetensors of the input dirs
    it names (a convert / build writes about what it reads)."""
    n = 0
    for p in path_args(args)[0]:
        pp = pathlib.Path(p) if pathlib.Path(p).is_absolute() else tree / p
        if pp.is_dir():
            n += sum(os.path.getsize(os.path.realpath(f)) for f in pp.glob("*.safetensors"))
    return n


def make_out_parents(args, tree: pathlib.Path):
    """A declared output's parent is created, not demanded: preflight
    redirects outputs into dirs it makes itself, so it can never catch a
    missing parent (2026-09-28: reselect_kl/ did not exist, and the queue
    failed after 72 min of applies)."""
    for o_ in path_args(args)[1]:
        op = pathlib.Path(o_) if pathlib.Path(o_).is_absolute() else tree / o_
        op.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ checks
def _script_literals(st, tree: pathlib.Path, exists) -> list[str]:
    sp = tree / st["script"]
    if not sp.is_file():
        return [f"script not in the pinned tree: {st['script']}"]
    probs = []
    for lit in sorted(set(_PATHLIKE.findall(sp.read_text(errors="replace")))):
        if "{" in lit or "*" in lit:
            continue                       # a template, not a path
        if not exists(lit) and not pathlib.Path(lit).parent.exists():
            probs.append(f"{st['script']} names a path that does not exist: {lit}")
    return probs


def static_check(st, tree: pathlib.Path, args) -> list[str]:
    """Every input path a step names must exist IN THE PINNED TREE (or be
    absolute and exist). The night-4 miss was a DATA path inside a script, so
    a by-path script's string literals are checked too."""
    probs = []
    ins, outs = path_args(args)

    def exists(p):
        pp = pathlib.Path(p)
        return (pp if pp.is_absolute() else tree / pp).exists()

    for p in ins:
        if not exists(p):
            probs.append(f"input path does not exist: {p}")
    for p in outs:
        pp = pathlib.Path(p)
        par = (pp if pp.is_absolute() else tree / pp).parent
        if not par.exists():
            probs.append(f"output's parent dir does not exist: {p}")
    if st.get("script"):
        probs += _script_literals(st, tree, exists)
    return probs


def pin_check(st, args) -> tuple[list[str], list[str]]:
    """(refusals, warnings) for the artifact dirs a scoring step names."""
    if st.get("cmd") not in SCORING:
        return [], []
    ref, warn = [], []
    for a in path_args(args)[0]:
        d = pathlib.Path(a)
        if d.is_dir() and (d / "config.json").exists():
            ok, msg = pinmod.check_pin(d)
            if not ok:
                ref.append(msg)
            elif msg:
                warn.append(msg)
            elif not (d / pinmod.MARKER).exists():
                warn.append(f"NOTE {d}: not a pin (AGENTS.md: pin, then smoke, then measure)")
    return ref, warn


# ------------------------------------------------------------------ preflight


def redirect_outs(args, pfdir: pathlib.Path, outmap: dict):
    """Preflight: send every output-flag value under pfdir (recording
    real -> preflight in outmap), and point any arg naming an EARLIER step's
    real output at that step's preflight output."""
    res, prev = [], None
    for a in args:
        if a.startswith("-"):
            flag, eq, val = a.partition("=")
            if eq and flag in OUT_FLAGS:
                new = str(pfdir / pathlib.Path(val).name)
                outmap[val.rstrip("/")] = new
                a = f"{flag}={new}"
            prev = flag if not eq else None
            res.append(a)
            continue
        if prev in OUT_FLAGS:
            new = str(pfdir / pathlib.Path(a).name)
            outmap[a.rstrip("/")] = new
            a = new
        else:
            name = a[: len(a) - len(strip_name(a))]
            path = a[len(name):]
            for real in sorted(outmap, key=len, reverse=True):
                if path == real or path.startswith(real + "/"):
                    a = name + outmap[real] + path[len(real):]
                    break
        res.append(a)
        prev = None
    return res


def preflight_args(st, args, state, outmap) -> tuple[list[str] | None, dict | None]:
    """(args, None) to run the step's preflight version, or (None, verdict)
    when it must not run: no preflight block, skipped, or too little scratch."""
    pf = st.get("preflight")
    if pf is None:
        return None, {"verdict": "fail", "reasons": [
            "no preflight defined: add preflight.args / preflight.append (a small REAL "
            "run: first cell, one module, a few tokens) or preflight.skip with a reason"]}
    if "skip" in pf:
        return None, {"verdict": "skipped", "reasons": [f"preflight skipped: {pf['skip']}"]}
    if "args" in pf:
        args = [_config.expand(str(a)) for a in pf["args"]]
    elif "append" in pf:
        args = args + [_config.expand(str(a)) for a in pf["append"]]
    pfdir = _config.preflight_dir() / pathlib.Path(state["qdir"]).name / slug(st["name"])
    pfdir.mkdir(parents=True, exist_ok=True)
    args = redirect_outs(args, pfdir, outmap if outmap is not None else {})
    need = preflight_writes(st, args, state.get("tree"), preflight=True)
    free = shutil.disk_usage(pfdir).free
    if free - need < PREFLIGHT_MARGIN:
        return None, {"verdict": "fail", "reasons": [
            f"not enough scratch for this preflight on {pfdir}: {free / GIB:.0f} GiB "
            f"free, full-size step may write ~{need / GIB:.0f} GiB, margin "
            f"{PREFLIGHT_MARGIN / GIB:.0f} GiB"]}
    return args, None


def preflight_writes(st, args, tree, preflight: bool) -> int:
    """Bytes a step writes to scratch: its preflight block's writes_gib when
    stated; else a preflight run that is not full-size writes little and is
    not measured (0); else the full-size step's input size (input_bytes).
    `args` are the args that run: the preflight's for a preflight."""
    pf = st.get("preflight") or {}
    if "writes_gib" in pf:
        return int(float(pf["writes_gib"]) * GIB)
    if preflight and "full" not in pf:
        return 0
    return input_bytes(args, pathlib.Path(tree))


# ------------------------------------------------------------------ launch
def argv(q, st, args, tree):
    """The command line a step runs: a command BY NAME through vqlab.cli,
    or a repo script from the pinned tree."""
    py = st.get("python") or q.get("python") or sys.executable
    if st.get("cmd"):
        return [py, "-m", "vqlab.cli", st["cmd"], *args]
    return [py, str(tree / st["script"]), *args]


def env(state):
    e = dict(os.environ)
    tree = state["tree"]
    src = pathlib.Path(tree) if state.get("installed") else pathlib.Path(tree) / "src"
    e["PYTHONPATH"] = str(src) + (
        os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e.setdefault("PYTHONPYCACHEPREFIX", str(_config.pycache_dir()))
    e.setdefault("VQLAB_FAMILIES_DIR", str(_config.families_dir()))
    e["VQLAB_QUEUE"] = state["qdir"]
    e["VQLAB_QUEUE_COMMIT"] = state["commit"]
    return e
