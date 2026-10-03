"""The architecture files VQLab loads in place of mlx-lm's: Knurlogic's.

mlx-lm resolves a model class with `importlib.import_module(
"mlx_lm.models.<model_type>")`. Stock mlx-lm 0.32 ships no deepseek_v4 and
no qwen4_exp, and its qwen3_5 lacks the edits Knurlogic serves with, so the
families VQLab fits and scores either do not load or load different
arithmetic from what users run. Knurlogic vendors them (engine/families/
<family>/architecture/, with PROVENANCE, pins and parity tests) and exposes
`knurlogic.engine.register(*names)` as public API for exactly this caller.

ONE copy: VQLab depends on knurlogic and never carries its own (a cloned
copy drifted within hours, 2026-10-03). `install()` puts a finder at the
front of sys.meta_path that, on the first import of mlx_lm.models.<name>
for a name Knurlogic vendors, calls register(name) -- lazily, so nothing
loads that a run does not use. A module already in sys.modules is left
alone. VQLAB_VENDORED_ARCH=0 turns it off, to score stock mlx-lm.

The numerics stamp (core/numerics.py) records knurlogic's version and
commit beside mlx / mlx-lm, so moving knurlogic is a stamp change and
scoring refuses caches from the old one.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

PREFIX = "mlx_lm.models."


def enabled() -> bool:
    return os.environ.get("VQLAB_VENDORED_ARCH", "1") != "0"


def _register():
    try:
        from knurlogic.engine import register
        return register
    except ImportError:
        return None


def available() -> list:
    """The architecture names Knurlogic vendors ([] without knurlogic)."""
    reg = _register()
    return list(reg.available()) if reg else []


def source(name: str):
    """The file serving mlx_lm.models.<name>, else None."""
    reg = _register()
    if reg is None or not enabled():
        return None
    src, _ = reg.source_for(name)
    return src


class _Existing(importlib.abc.Loader):
    """Hands back the module knurlogic's register() already built."""

    def create_module(self, spec):
        return sys.modules[spec.name]

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(PREFIX) or not enabled():
            return None
        name = fullname[len(PREFIX):]
        reg = _register()
        if reg is None or name not in reg.available():
            return None
        reg.register(name)                       # its dependencies first, then name
        mod = sys.modules.get(fullname)
        if mod is None:
            return None
        spec = importlib.machinery.ModuleSpec(fullname, _Existing(),
                                              origin=getattr(mod, "__file__", None))
        spec.has_location = spec.origin is not None
        return spec


_FINDER = _Finder()


def install() -> bool:
    """Serve Knurlogic's architectures as mlx_lm.models.<name>. Idempotent;
    False when turned off (VQLAB_VENDORED_ARCH=0) or knurlogic is missing."""
    if not enabled() or _register() is None:
        return False
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    return True


def uninstall() -> None:
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
