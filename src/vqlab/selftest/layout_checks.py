"""Selftest: the package layout (corpus locator, one module object per
name, CONTEXT.md routing, name collisions, `python -m` on a PERMANENT name)."""
from __future__ import annotations

import importlib as _il
import importlib.machinery as _im
import os
import subprocess
import sys

from vqlab import _layout as L
from vqlab.selftest import PY, check, skip


def check_corpus_locator(ctx):
    check("corpus locator: prose / code / lit by name, old spellings too",
          all(L.corpus(n).is_file() for n in ("prose", "code", "lit", "wikitext"))
          and L.corpus(str(L.SRC / "vqlab" / "referee" / "referee_corpus.txt")).is_file())


def check_layout(ctx):
    print("[6b/7] layout")
    # A bare alias that is also a stdlib or installed module name would
    # shadow it (_Alias sits first on sys.meta_path). bench/coverage.py was
    # renamed kernel_coverage.py for exactly this (pytest-cov). Search only
    # OUTSIDE this repo (site-packages, stdlib); find_spec would also report
    # modules we have already imported ourselves.
    repo = str(L.SRC.parent)
    outside = [p for p in sys.path if p and not p.startswith(repo)]
    clash = sorted(n for n in L.PERMANENT if "." not in n and (
                   n in sys.stdlib_module_names
                   or _im.PathFinder.find_spec(n, outside) is not None))
    _check_one_module_object()
    _check_context_routing()
    _nodoc = packages_missing_context(L.PKG)
    check("every package folder has its CONTEXT.md contract", not _nodoc, ", ".join(_nodoc))
    check("no bare alias collides with stdlib / site-packages",
          not clash, ", ".join(clash))
    p = subprocess.run([PY, "-m", "vqlab.pack_artifact", "--help"], capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(L.SRC)})
    check("`python -m vqlab.pack_artifact` (a PERMANENT name) still runs", p.returncode == 0,
          p.stderr.strip().splitlines()[-1] if p.returncode else "")


# Folders that carry .py files as DATA (scored/served, never imported).
DATA_DIRS = {"score/referee", "agents/gui_static/bench"}


def packages_missing_context(pkg):
    """Every top-level package folder, and every nested package its parent's
    CONTEXT.md routes to (`<name>/CONTEXT.md`), carries a CONTEXT.md; a
    nested folder holding .py files is a real package (__init__.py), so an
    implicit namespace folder cannot slip past the check."""
    bad = []
    for d in sorted(pkg.rglob("*")):
        if not d.is_dir() or "__pycache__" in d.parts:
            continue
        rel = d.relative_to(pkg).as_posix()
        if not (d / "__init__.py").exists():
            if any(d.glob("*.py")) and rel not in DATA_DIRS:
                bad.append(f"{rel} (no __init__.py)")
            continue
        parent_ctx = d.parent / "CONTEXT.md"
        routed = parent_ctx.exists() and f"{d.name}/CONTEXT.md" in parent_ctx.read_text()
        if (d.parent == pkg or routed) and not (d / "CONTEXT.md").exists():
            bad.append(rel)
    return bad


def _check_one_module_object():
    from vqlab.runtime import vq_switch as _bare
    import vqlab.runtime.vq_switch as _canon
    import vqlab.vq_switch as _old
    check("bare, dotted and pre-split names give ONE module object",
          _bare is _canon is _old and _canon.__name__ == "vqlab.runtime.vq_switch")
    check("reload by the bare name re-executes the module",
          _il.reload(_bare) is _canon and _canon.__name__ == "vqlab.runtime.vq_switch")


def _check_context_routing():
    from vqlab import commands
    _root_ctx = L.SRC.parent / "CONTEXT.md"
    if _root_ctx.exists():
        _named = commands.routed(_root_ctx.read_text())   # `fit moe` and `fit-moe` both route
        _unrouted = sorted(k for k in commands.COMMANDS if k not in _named)
        check("every CLI command is routed in the root CONTEXT.md (agents navigate by it)",
              not _unrouted, ", ".join(_unrouted))
    else:
        skip("every CLI command is routed in the root CONTEXT.md",
             "an installed package carries no repo root; checked in the checkout")
