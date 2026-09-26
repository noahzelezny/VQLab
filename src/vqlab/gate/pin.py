#!/usr/bin/env python3
"""vqlab pin: freeze an artifact for measurement, then smoke the frozen copy.

AGENTS.md: "Measuring? PIN, then SMOKE, then measure." This is that rule as a
tool. A pin is a new directory holding
  * a symlink to every *.safetensors of the source (resolved, so a later
    rebundle or re-pack of the SOURCE cannot reach into the pin),
  * a copy of every other file (config, tokenizer, model.py, ...),
  * optionally the runtime re-baked to a profile (--runtime v1.5|v2), via the
    same tools a human would run (`bundle` for MoE, `rebundle-dense` for dense),
  * vqlab_pin.json (schema vqlab.pin/1) recording what the pin IS and whether
    it has been smoked.

States:
  ok            `vqlab smoke` generated through the pinned runtime.
  failed        the smoke (or the rebundle) failed. Scorers REFUSE this pin.
  deferred-ram  preflight_ram says the pin cannot be resident on this box, so
                no single-box smoke can produce a verdict. Scorers accept it
                with a WARNING; smoke it on a cluster before citing a number.

A pin never deletes: if --out exists the tool refuses (a half-made pin is left
for a human to inspect and remove).

    vqlab pin <artifact> --out <dir> [--runtime v2] [--kind auto|dense|moe]

check_pin(dir) is what scorers call: (allowed, message). A directory without
vqlab_pin.json is not a pin and is allowed unchanged (nothing here breaks
scoring of unpinned artifacts); a pin whose model.py changed since it was
smoked is refused, because the smoke no longer describes it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

SCHEMA = "vqlab.pin/1"
MARKER = "vqlab_pin.json"
SMOKE_LOG = "vqlab_pin_smoke.log"
_SKIP = {"__pycache__", MARKER, SMOKE_LOG, ".DS_Store"}


def _sha(p: pathlib.Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _kind(src: pathlib.Path) -> str:
    """moe if the config names routed experts, else dense."""
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    keys = ("num_experts", "num_local_experts", "n_routed_experts")
    return "moe" if any(tc.get(k) or cfg.get(k) for k in keys) else "dense"


def _cli(*args, env=None, log=None):
    cmd = [sys.executable, "-m", "vqlab.cli", *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if log is not None:
        with open(log, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}\n[rc={p.returncode}]\n")
    return p


def _child_run_id(parent: str | None, cmd: str) -> str | None:
    """The run id `vqlab <cmd>` got when launched from this process."""
    if not parent:
        return None
    log = pathlib.Path(os.environ.get("VQLAB_LOG_DIR",
                                      pathlib.Path.home() / ".vqlab")) / "runs.jsonl"
    found = None
    try:
        for line in open(log):
            r = json.loads(line)
            if (r.get("event") == "start" and r.get("parent_run_id") == parent
                    and r.get("cmd") == cmd):
                found = r.get("run_id")
    except (OSError, ValueError):
        pass
    return found


def pin(src, out, runtime=None, kind="auto", headroom=0.90, max_tokens=8):
    src, out = pathlib.Path(src).resolve(), pathlib.Path(out)
    if not (src / "config.json").exists():
        raise SystemExit(f"FAIL: {src} has no config.json")
    if out.exists():
        raise SystemExit(f"FAIL: {out} already exists. A pin never overwrites; "
                         "inspect it and remove it by hand.")
    out.mkdir(parents=True)
    log = out / SMOKE_LOG
    for f in sorted(src.iterdir()):
        if f.name in _SKIP:
            continue
        if f.suffix == ".safetensors":
            (out / f.name).symlink_to(f.resolve())
        elif f.is_dir():
            shutil.copytree(f, out / f.name, symlinks=False)
        else:
            shutil.copy2(f, out / f.name)

    rec = {"schema": SCHEMA, "source": str(src), "runtime": runtime,
           "kind": _kind(src) if kind == "auto" else kind,
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "pin_run_id": os.environ.get("VQLAB_RUN_ID"),
           "smoke_run_id": None, "state": None, "reason": ""}

    def finish(state, reason=""):
        rec.update(state=state, reason=reason,
                   model_py_sha256=_sha(out / "model.py"))
        (out / MARKER).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"PIN {state.upper()}: {out}" + (f"  ({reason})" if reason else ""))
        return 0 if state in ("ok", "deferred-ram") else 1

    if runtime:
        tool = "bundle" if rec["kind"] == "moe" else "rebundle-dense"
        p = _cli(tool, "--artifact", out, "--runtime", runtime, log=log)
        if p.returncode != 0:
            return finish("failed", f"{tool} --runtime {runtime} rc={p.returncode}; see {SMOKE_LOG}")

    from vqlab._layout import find as _find
    p = subprocess.run([sys.executable, str(_find("preflight_ram.py")), str(out),
                        "--headroom", str(headroom)], capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(p.stdout + p.stderr)
    if p.returncode != 0:
        return finish("deferred-ram", (p.stdout.strip().splitlines() or ["preflight_ram refused"])[-1])

    p = _cli("smoke", out, "--max-tokens", max_tokens, "--headroom", headroom, log=log)
    rec["smoke_run_id"] = _child_run_id(os.environ.get("VQLAB_RUN_ID"), "smoke")
    if p.returncode != 0:
        return finish("failed", f"smoke rc={p.returncode}; see {SMOKE_LOG}")
    return finish("ok")


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab pin", description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--out", help="new directory (must not exist); required unless --check")
    ap.add_argument("--runtime", choices=("v1.5", "v2"),
                    help="re-bake the runtime profile into the pin before smoking")
    ap.add_argument("--kind", choices=("auto", "dense", "moe"), default="auto")
    ap.add_argument("--headroom", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--check", action="store_true",
                    help="only report check_pin() for ARTIFACT (exit 0 allowed / 1 refused)")
    a = ap.parse_args(argv)
    if a.check:
        ok, msg = check_pin(a.artifact)
        print(msg or f"OK {a.artifact}")
        return 0 if ok else 1
    if not a.out:
        ap.error("--out is required")
    return pin(a.artifact, a.out, a.runtime, a.kind, a.headroom, a.max_tokens)


if __name__ == "__main__":
    sys.exit(main())
