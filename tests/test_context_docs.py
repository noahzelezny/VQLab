"""Every CONTEXT.md is an accurate map of its folder.

Agents navigate this package by folder and file name, reading the
CONTEXT.md beside the files. So:

  * every .py file under src/vqlab is named in the CONTEXT.md of its folder
    (files in a sub-folder without its own CONTEXT.md, like agents/mcp/,
    are named there as `mcp/server.py`; top-level modules in the root
    CONTEXT.md as `vqlab/<name>.py`);
  * every `*.py` file a CONTEXT.md names in backticks exists.

`__init__.py` / `__main__.py` are the package itself, documented by the
CONTEXT.md existing. runtime/ is exempt from the first check: every byte
under it, its CONTEXT.md included, is frozen with the shipped runtime
(tests/test_runtime_frozen.py), so its file list changes only with a
deliberate runtime change.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vqlab"
SKIP_STEMS = {"__init__", "__main__"}


def _context_for(path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """(CONTEXT.md that must name `path`, the spelling it must use)."""
    if path.parent == PKG:
        return ROOT / "CONTEXT.md", f"vqlab/{path.name}"
    d = path.parent
    while not (d / "CONTEXT.md").exists():
        d = d.parent
    return d / "CONTEXT.md", path.relative_to(d).as_posix()


def _py_files():
    return sorted(p for p in PKG.rglob("*.py")
                  if "__pycache__" not in p.parts
                  and p.relative_to(PKG).parts[0] != "runtime"
                  and p.stem not in SKIP_STEMS)


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.relative_to(PKG).as_posix())
def test_every_module_is_named_in_its_context(path):
    ctx, spelling = _context_for(path)
    text = ctx.read_text()
    assert re.search(rf"(?<![\w/]){re.escape(spelling)}\b", text), (
        f"{ctx.relative_to(ROOT)} does not name {spelling}")


def _contexts():
    return [ROOT / "CONTEXT.md", *sorted(PKG.rglob("CONTEXT.md"))]


@pytest.mark.parametrize("ctx", _contexts(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_named_file_exists(ctx):
    # `model.py` is the file a bundler writes INTO an artifact, not a module here
    names = set(re.findall(r"`([\w./-]+\.py)`", ctx.read_text())) - {"model.py"}
    bases = (ctx.parent, PKG, PKG.parent, ROOT)
    missing = sorted(n for n in names if not any((b / n).exists() for b in bases))
    assert not missing, f"{ctx.relative_to(ROOT)} names files that do not exist: {missing}"
