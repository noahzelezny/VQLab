"""Where every vqlab module lives, and the two import conventions that span it.

The package is split into stage folders (see CONTEXT.md at the repo root and
the CONTEXT.md inside each folder). The tools inside them are STANDALONE
SCRIPTS by design -- `vqlab <cmd>` runs each one as a file -- and they import
their siblings by bare name (`import vq_pack`, `from families import FAMILY`).
Importing this module makes both conventions work from anywhere:

  * every stage folder goes on sys.path, so bare sibling imports resolve
    across folders exactly as they did when the package was flat;
  * every pre-split dotted name (`vqlab.vq_switch`, `vqlab.geo_build`,
    `vqlab.mtp_head`, ...) is aliased to its new home. This is not only for
    old callers: the SHIPPED runtime text (vq_dense.py's _resolve_kernel,
    dense_shim.py) names `vqlab.vq_switch` / `vqlab.arch_resolve`, and that
    text must stay byte-identical to what published bundles carry, or
    check-bundle reports drift on every artifact.

Locators, for code that reads a sibling as a FILE rather than importing it:

    find("smoke.py")               -> path of any tool, wherever it lives
    runtime_file("vq_switch.py")   -> the repo runtime a bundle is spliced from
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import pathlib
import sys

PKG = pathlib.Path(__file__).resolve().parent
SRC = PKG.parent

# Stage folders, in pipeline order. Their contents are importable by bare
# name. `mtp` is a real package (vqlab.mtp) whose internal module names
# (runtime, sampling, ...) are too generic to expose bare, so it is searched
# by find() but NOT put on sys.path.
STAGES = ("runtime", "core", "plan", "fit", "assemble", "bundle", "gate",
          "score", "bench", "records", "ship", "agents")
SEARCH = STAGES + ("mtp", "score/referee")


def stage_dirs():
    return [PKG / s for s in STAGES]


def find(name: str) -> pathlib.Path:
    """Path of a module FILE by its (pre-split) filename or relative path."""
    for d in (PKG, *(PKG / s for s in SEARCH)):
        p = d / name
        if p.exists():
            return p
    if name.startswith("referee/"):                 # old relative spelling
        return find(name.split("/", 1)[1])
    raise FileNotFoundError(f"vqlab: no module file {name!r} in {SEARCH}")


def runtime_file(name: str) -> pathlib.Path:
    """The repo's copy of a shipped runtime file (vq_switch.py, vq_dense.py,
    vq_pack.py, dense_shim.py, glm5_shim.py). This is the text bundlers
    splice and check-bundle compares against; its flag DEFAULTS are the
    repo's, so a caller that needs a specific profile applies it with
    runtime_profile.apply_profile / resolve_runtime."""
    p = PKG / "runtime" / name
    if not p.exists():
        raise FileNotFoundError(f"vqlab: {name} is not a runtime file ({p})")
    return p


def legacy_path(p):
    """Map a path recorded BEFORE the stage split onto its new home.

    Teacher caches store their corpus path in meta.json (absolute or
    repo-relative) and shell chains pass corpora by path; both still name
    src/vqlab/referee/. Existing paths are returned unchanged."""
    s = str(p)
    if pathlib.Path(s).exists():
        return p
    for old, new in (("src/vqlab/referee/", "src/vqlab/score/referee/"),):
        if old in s:
            cand = s.replace(old, new)
            if pathlib.Path(cand).exists() or not pathlib.Path(s).is_absolute():
                return type(p)(cand) if not isinstance(p, str) else cand
    return p


def _old_names():
    """{old dotted name: new dotted name} for every module a stage folder
    holds, plus the MTP scripts that moved into vqlab.mtp."""
    out = {}
    for s in STAGES + ("mtp",):
        for f in (PKG / s).glob("*.py"):
            if f.stem != "__init__":
                out[f"vqlab.{f.stem}"] = f"vqlab.{s}.{f.stem}"
    return out


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        self.map = None

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("vqlab."):
            return None
        if self.map is None:
            self.map = _old_names()
        new = self.map.get(fullname)
        if new is None or new == fullname:
            return None
        t = importlib.util.find_spec(new)
        return importlib.util.spec_from_loader(
            fullname, self, origin=t.origin if t else f"alias of {new}")

    def create_module(self, spec):
        mod = importlib.import_module(self.map[spec.name])
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass                                        # already executed

    # `python -m vqlab.<old name>` goes through runpy, which asks the loader
    # for code rather than a module. Hand it the real module's code.
    def _target(self, fullname):
        if self.map is None:
            self.map = _old_names()
        return importlib.util.find_spec(self.map[fullname])

    def get_code(self, fullname):
        t = self._target(fullname)
        return t.loader.get_code(t.name)

    def get_source(self, fullname):
        t = self._target(fullname)
        return t.loader.get_source(t.name)

    def is_package(self, fullname):
        return False


def install():
    for d in reversed(stage_dirs()):
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    if str(SRC) not in sys.path:
        sys.path.append(str(SRC))
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.append(_Alias())


install()
