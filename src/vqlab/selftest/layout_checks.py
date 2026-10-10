"""Selftest: the package layout (corpus locator, one module object per
name, CONTEXT.md routing, name collisions, pre-split `python -m` names)."""
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
    # A stage module whose bare name is also a stdlib or installed
    # package gets shadowed (or shadows it) on sys.path. bench/coverage.py
    # was renamed kernel_coverage.py for exactly this (pytest-cov).
    # Search only OUTSIDE this repo (site-packages, stdlib); find_spec
    # would also report modules we have already imported ourselves.
    repo = str(L.SRC.parent)
    outside = [p for p in sys.path if p and not p.startswith(repo)]
    clash = sorted(f.stem for d in L.stage_dirs() for f in d.glob("*.py")
                   if f.stem != "__init__" and (
                       f.stem in sys.stdlib_module_names
                       or _im.PathFinder.find_spec(f.stem, outside) is not None))
    _check_one_module_object()
    _check_context_routing()
    _nodoc = sorted(s_ for s_ in L.STAGES if not (L.PKG / s_ / "CONTEXT.md").exists())
    check("every stage folder has its CONTEXT.md contract", not _nodoc, ", ".join(_nodoc))
    check("no stage module name collides with stdlib / site-packages",
          not clash, ", ".join(clash))
    p = subprocess.run([PY, "-m", "vqlab.price", "--help"], capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(L.SRC)})
    check("pre-split `python -m vqlab.<name>` still runs", p.returncode == 0,
          p.stderr.strip().splitlines()[-1] if p.returncode else "")


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
