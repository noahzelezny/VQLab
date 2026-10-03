"""Architecture files VQLab loads in place of mlx-lm's, cloned from Knurlogic.

mlx-lm resolves a model class with `importlib.import_module(
"mlx_lm.models.<model_type>")`. Stock mlx-lm 0.32.0 ships no deepseek_v4 and
no qwen4_exp, and its qwen3_5 lacks Knurlogic's edits, so the families VQLab
fits and scores either do not load or load different arithmetic from what
Knurlogic serves. These folders are Knurlogic's vendored architectures
(engine/families/<family>/architecture/, knurlogic 3619e00), copied whole:
PROVENANCE.md says where each file came from, pins.json what it was
validated on, THIRD-PARTY.md the upstream (mlx-lm, MIT) license.

`install()` puts a finder at the front of sys.meta_path that answers
`mlx_lm.models.<name>` for these names only, lazily, at first import, so
site-packages is never written and nothing loads that a run does not use.
A module already in sys.modules (Knurlogic registered it in this process)
is left alone. VQLAB_VENDORED_ARCH=0 turns it off, to score stock mlx-lm.

Keep in step with Knurlogic: `vqlab runtime-equiv` compares a slice through
both, and `vqlab doctor` prints each file's sha.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

# name -> path (a file, or a package's __init__.py), and the bases a module
# must load against (a vendored subclass on a stock base mixes arithmetic)
SOURCES = {
    "deepseek_v4": HERE / "deepseek" / "deepseek_v4.py",
    "qwen3_5": HERE / "qwen" / "qwen3_5.py",
    "qwen3_5_moe": HERE / "qwen" / "qwen3_5_moe.py",
    "qwen4_exp": HERE / "qwen" / "qwen4_exp.py",
    "gemma4_text": HERE / "gemma4" / "gemma4_text.py",
    "gemma4": HERE / "gemma4" / "gemma4.py",
    "glm5_next": HERE / "glm5" / "glm5_next" / "__init__.py",
}
DEPENDS_ON = {"qwen3_5_moe": ["qwen3_5"], "gemma4": ["gemma4_text"]}
PREFIX = "mlx_lm.models."


def enabled() -> bool:
    return os.environ.get("VQLAB_VENDORED_ARCH", "1") != "0"


def source(name: str):
    """The vendored file serving mlx_lm.models.<name>, else None."""
    return SOURCES.get(name) if enabled() else None


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(PREFIX):
            return None
        name = fullname[len(PREFIX):]
        src = SOURCES.get(name)
        if src is None or not enabled():
            return None
        for dep in DEPENDS_ON.get(name, []):
            importlib.import_module(PREFIX + dep)
        pkg = src.name == "__init__.py"
        return importlib.util.spec_from_file_location(
            fullname, src, submodule_search_locations=[str(src.parent)] if pkg else None)


_FINDER = _Finder()


def install() -> bool:
    """Serve the vendored architectures as mlx_lm.models.<name>. Idempotent;
    False when turned off by VQLAB_VENDORED_ARCH=0."""
    if not enabled():
        return False
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    return True


def uninstall() -> None:
    if _FINDER in sys.meta_path:
        sys.meta_path.remove(_FINDER)
