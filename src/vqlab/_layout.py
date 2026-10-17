"""Names shipped text depends on, and the locators that find a module as a FILE.

Every module has ONE canonical dotted name: its path under src/vqlab
(`vqlab.runtime.vq_switch`, `vqlab.fit.fit_moe`). PERMANENT lists the only
other spellings, kept forever because shipped text names them:

  * every runtime/*.py stem, as `vqlab.<stem>` AND bare `<stem>`: the
    SHIPPED runtime text (vq_dense.py's _resolve_kernel, dense_shim.py) and
    published model.py files name them, and inside a bundle the files sit
    flat beside model.py. Must stay byte-identical, so the names must stay;
  * `vqlab.mtp_head*`, `vq_397b_codes` and `pack_artifact`: archived queue
    files and ledgers name them.

Whatever name is used, the result is the SAME module object (_Alias): two
copies of vq_switch would mean two kernel caches and two sets of flag
globals.

Locators, for code that reads a module as a FILE rather than importing it:

    find("smoke.py")               -> path of any tool (and vq_397b_codes.py)
    runtime_file("vq_switch.py")   -> the repo runtime a bundle is spliced from
    corpus("prose")                -> a house referee corpus
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

PERMANENT = {
    'arch_resolve': 'vqlab.runtime.arch_resolve',
    'dense_shim': 'vqlab.runtime.dense_shim',
    'gated_delta_chunked': 'vqlab.runtime.gated_delta_chunked',
    'glm5_shim': 'vqlab.runtime.glm5_shim',
    'runtime_profile': 'vqlab.runtime.runtime_profile',
    'vq_dense': 'vqlab.runtime.vq_dense',
    'vq_pack': 'vqlab.runtime.vq_pack',
    'vq_switch': 'vqlab.runtime.vq_switch',
    'vqlab.arch_resolve': 'vqlab.runtime.arch_resolve',
    'vqlab.build.pack_artifact': 'vqlab.build.pack',
    'vqlab.dense_shim': 'vqlab.runtime.dense_shim',
    'vqlab.fit.vq_397b_codes': 'vqlab.fit.fit_moe',
    'vqlab.gated_delta_chunked': 'vqlab.runtime.gated_delta_chunked',
    'vqlab.glm5_shim': 'vqlab.runtime.glm5_shim',
    'vqlab.mtp_head': 'vqlab.mtp.mtp_head',
    'vqlab.mtp_head_ds4': 'vqlab.build.mtp_head_ds4',
    'vqlab.mtp_head_glm5': 'vqlab.mtp.mtp_head_glm5',
    'vqlab.mtp_head_qwen35': 'vqlab.mtp.mtp_head_qwen35',
    'vqlab.pack_artifact': 'vqlab.build.pack',
    'vqlab.runtime_profile': 'vqlab.runtime.runtime_profile',
    'vqlab.vq_397b_codes': 'vqlab.fit.fit_moe',
    'vqlab.vq_dense': 'vqlab.runtime.vq_dense',
    'vqlab.vq_pack': 'vqlab.runtime.vq_pack',
    'vqlab.vq_switch': 'vqlab.runtime.vq_switch',
}

assert not {"queue", "vqlab.queue"} & set(PERMANENT)   # stdlib `queue` stays the stdlib's

# Old FILENAMES the PERMANENT names imply (vq_397b_codes.py -> fit_moe.py,
# pack_artifact.py -> pack.py): archived queue files and ledgers name them.
_FILE_NAMES = {k.rpartition(".")[2]: v.rpartition(".")[2] for k, v in PERMANENT.items()
               if k.rpartition(".")[2] != v.rpartition(".")[2]}


# Folders find() searches, in order: a current filename needs no table.
_FIND_DIRS = ("", "runtime", "core", "core/artifacts", "core/source", "core/box",
              "family", "plan", "fit", "build", "bundle",
              "gate", "score", "bench", "records", "ship", "agents", "mtp",
              "queue", "selftest", "skipzero", "ops", "util", "score/referee")


def find(name: str) -> pathlib.Path:
    """Path of a module FILE by its filename or relative path. The old
    filenames PERMANENT implies (vq_397b_codes.py) map to their module."""
    stem, dot, ext = name.rpartition("/")[2].partition(".")
    new = _FILE_NAMES.get(stem)
    if new and ext == "py":
        name = name[:len(name) - len(stem + dot + ext)] + new + dot + ext
    for d in _FIND_DIRS:
        p = PKG / d / name
        if p.exists():
            return p
    if name.startswith("referee/"):                 # old relative spelling
        return find(name.split("/", 1)[1])
    raise FileNotFoundError(f"vqlab: no module file {name!r} under {PKG}")


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


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """ONE module object per module, whatever name it is imported by.

    Without this, `import vq_switch` and `vqlab.runtime.vq_switch` load the
    file twice: two kernel caches, two sets of env-flag globals, and a test
    that reloads one name silently leaves the other untouched.

    It answers ONLY for the PERMANENT names: `vqlab.*` spellings (our
    namespace) and the bare runtime stems a bundle's runtime text refers to
    (vq_switch, vq_dense, ...). Every other name gets None from it, so it
    cannot shadow another package's module.

    It sits FIRST on sys.meta_path, and that is required, not habit: a
    script directory holding a copy of vq_switch.py (sys.path[0]) would
    otherwise give the runtime a second module object.
    """

    def __init__(self):
        self.map = PERMANENT
        self._fresh = set()     # ids of modules create_module just imported

    def _canon(self, fullname):
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


def install_import_aliases():
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())   # first: see _Alias docstring

install_import_aliases()
