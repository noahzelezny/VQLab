"""Where every vqlab module lives, and the two import conventions that span it.

The package is split into stage folders (see CONTEXT.md at the repo root and
the CONTEXT.md inside each folder). The tools inside them are STANDALONE
SCRIPTS by design -- `vqlab <cmd>` runs each one as a file -- and they import
their siblings by bare name (`import vq_pack`, `from families import FAMILY`).
Importing this module makes both conventions work from anywhere:

  * bare sibling imports (`import vq_pack`) resolve across folders, to the
    SAME module object as the dotted name -- one module, one copy;
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
import importlib.machinery
import importlib.util
import pathlib
import sys

PKG = pathlib.Path(__file__).resolve().parent
SRC = PKG.parent

# Stage folders, in pipeline order. Each holds that stage's tools, importable
# by bare name. `mtp` is NOT a stage: it is the MTP speculative-decoding
# LIBRARY (vqlab.mtp: loop, registry, heads), imported by the MTP tools that
# live in the stage folders like every other tool.
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


CORPORA = {"prose": "referee_corpus.txt",
           "code": "referee_corpus_code_public.txt",
           "lit": "referee_corpus_literary.txt"}
_CORPUS_ALIAS = {"wikitext": "prose", "code-public": "code", "literary": "lit"}


def corpus(name: str = "prose") -> pathlib.Path:
    """A house referee corpus by NAME (prose / code / lit), never by a path
    built from __file__: a scratch script that hardcodes the repo layout
    breaks silently when the layout moves (night 4, 2026-09-26). A path
    argument is passed through legacy_path, so old spellings still resolve."""
    key = _CORPUS_ALIAS.get(name, name)
    if key in CORPORA:
        p = PKG / "score" / "referee" / CORPORA[key]
    else:
        p = pathlib.Path(legacy_path(str(name)))
    if not p.is_file():
        raise FileNotFoundError(f"vqlab: no corpus {name!r} (have {sorted(CORPORA)})")
    return p


def legacy_path(p):
    """Map a path recorded BEFORE the stage split onto its new home.

    Teacher caches store their corpus path in meta.json (absolute or
    repo-relative) and shell chains pass corpora by path; both still name
    src/vqlab/referee/. Existing paths are returned unchanged."""
    s = str(p)
    if pathlib.Path(s).exists():
        return p
    # "vqlab/referee/" (no src/) also covers an INSTALLED package's site-packages path
    for old, new in (("vqlab/referee/", "vqlab/score/referee/"),):
        if old in s:
            cand = s.replace(old, new)
            if pathlib.Path(cand).exists() or not pathlib.Path(s).is_absolute():
                return type(p)(cand) if not isinstance(p, str) else cand
    return p


def _names():
    """{importable name: canonical dotted name} for every stage module.

    Two spellings map onto each canonical module:
      * the BARE name the standalone scripts use (`import vq_pack`), and
      * the pre-split dotted name (`vqlab.vq_switch`), which the SHIPPED
        runtime text itself uses and therefore can never be edited away.
    """
    out = {}
    for s in STAGES:
        for f in (PKG / s).glob("*.py"):
            if f.stem != "__init__":
                canon = f"vqlab.{s}.{f.stem}"
                out[f.stem] = canon
                out[f"vqlab.{f.stem}"] = canon
    # MTP heads lived at vqlab.mtp_head* before the split. Dotted alias only:
    # the vqlab.mtp library's own module names (runtime, loop, ...) are too
    # generic to answer for as bare names.
    for f in (PKG / "mtp").glob("mtp_*.py"):
        out[f"vqlab.{f.stem}"] = f"vqlab.mtp.{f.stem}"
    return out


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """ONE module object per module, whatever name it is imported by.

    Without this, `import vq_switch` and `vqlab.runtime.vq_switch` load the
    file twice: two kernel caches, two sets of env-flag globals, and a test
    that reloads one name silently leaves the other untouched. Sits FIRST on
    sys.meta_path so it wins over the ordinary path search -- including a
    script's own directory, which Python puts on sys.path[0]. It only
    answers for names in _names(), so nothing else is affected, and no stage
    folder has to go on sys.path (which is what made `coverage` shadowable).
    """

    def __init__(self):
        self.map = None
        self._fresh = set()     # ids of modules create_module just imported

    def _canon(self, fullname):
        if self.map is None:
            self.map = _names()
        return self.map.get(fullname)

    def find_spec(self, fullname, path=None, target=None):
        new = self._canon(fullname)
        if new is None or new == fullname:
            return None
        t = importlib.util.find_spec(new)
        return importlib.util.spec_from_loader(
            fullname, self, origin=t.origin if t else f"alias of {new}")

    def create_module(self, spec):
        mod = importlib.import_module(self._canon(spec.name))
        sys.modules[spec.name] = mod
        self._fresh.add(id(mod))
        return mod

    def exec_module(self, module):
        """First import: the canonical import already executed it. RELOAD
        (importlib.reload by the alias name, which is how the flag tests
        re-read env defaults) must genuinely re-execute. Either way the
        module keeps its CANONICAL identity; the import system just stamped
        the alias spec on it."""
        name = module.__spec__.name
        canon = self._canon(name) or name
        # Ask the path finder for the FILE's spec. importlib.util.find_spec
        # would return the module's current __spec__ -- the alias spec the
        # import system just stamped on it -- and recurse back into us.
        parent, _, _ = canon.rpartition(".")
        spec = importlib.machinery.PathFinder.find_spec(
            canon, importlib.import_module(parent).__path__)
        if id(module) in self._fresh:
            self._fresh.discard(id(module))
        else:
            spec.loader.exec_module(module)
        module.__spec__, module.__name__, module.__loader__ = (
            spec, spec.name, spec.loader)

    # `python -m vqlab.<old name>` goes through runpy, which asks the loader
    # for code rather than a module. Hand it the real module's code.
    def _target(self, fullname):
        return importlib.util.find_spec(self._canon(fullname))

    def get_code(self, fullname):
        t = self._target(fullname)
        return t.loader.get_code(t.name)

    def get_source(self, fullname):
        t = self._target(fullname)
        return t.loader.get_source(t.name)

    def is_package(self, fullname):
        return False


def install():
    if str(SRC) not in sys.path:
        sys.path.append(str(SRC))
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())


install()
