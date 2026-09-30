"""Machine-local storage locations.

vqlab never hardcodes where artifacts live. Each location resolves, in
order, from:

1. its environment variable,
2. the ``[paths]`` table of the config file (``$VQLAB_CONFIG``, default
   ``~/.config/vqlab/config.toml``),
3. a default under ``~/.vqlab``.

Example config::

    [paths]
    scratch   = "/Volumes/Fast/vqlab-scratch"     # builds, pins, run outputs
    models    = "/Volumes/Fast/models"            # served artifacts
    teachers  = "/Volumes/Archive/teachers"       # bf16 teacher checkpoints
    fit_store = ["/Volumes/Archive/vqlab-fits"]   # first entry receives new fits
    roots     = ["/Volumes/Fast", "/Volumes/Archive"]  # where lookups may search
    gpu_lease = "/Volumes/Fast/gpu.lease"         # shared with other GPU tenants

Large outputs belong on the volumes named here: `require_storage` refuses a
write path that is not under one of them.
"""
from __future__ import annotations

import functools
import os
import pathlib
import re
import tomllib

HOME = pathlib.Path.home() / ".vqlab"

_ENV = {
    "scratch": "VQLAB_SCRATCH",
    "models": "VQLAB_MODELS_DIR",
    "teachers": "VQLAB_TEACHERS_DIR",
    "fit_store": "VQLAB_FIT_STORE",
    "roots": "VQLAB_ROOTS",
    "gpu_lease": "VQLAB_GPU_LEASE",
}
_DEFAULT = {
    "scratch": HOME / "scratch",
    "models": HOME / "models",
    "teachers": HOME / "teachers",
    "fit_store": HOME / "fits",
    "gpu_lease": HOME / "gpu.lease",
}


def config_file() -> pathlib.Path:
    return pathlib.Path(os.environ.get("VQLAB_CONFIG")
                        or pathlib.Path.home() / ".config" / "vqlab" / "config.toml")


@functools.lru_cache(maxsize=None)
def _file_paths(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f).get("paths", {})


def _raw(key: str):
    env = os.environ.get(_ENV[key])
    if env:
        return env.split(os.pathsep) if key in ("fit_store", "roots") else env
    return _file_paths(str(config_file())).get(key)


def _one(key: str) -> pathlib.Path:
    v = _raw(key)
    return pathlib.Path(v).expanduser() if v else _DEFAULT[key]


def _many(key: str) -> list[pathlib.Path]:
    v = _raw(key)
    if isinstance(v, str):
        v = [v]
    return [pathlib.Path(p).expanduser() for p in v if p] if v else []


def scratch() -> pathlib.Path:
    """Builds, pins and run outputs."""
    return _one("scratch")


def models() -> pathlib.Path:
    """Served (published or candidate) artifacts."""
    return _one("models")


def teachers() -> pathlib.Path:
    """bf16 teacher checkpoints."""
    return _one("teachers")


def fit_store() -> list[pathlib.Path]:
    """Fit archive roots; the first receives new fits."""
    return _many("fit_store") or [_DEFAULT["fit_store"]]


def gpu_lease() -> pathlib.Path:
    """Advisory lock file every GPU job on this machine takes."""
    return _one("gpu_lease")


def roots() -> list[pathlib.Path]:
    """Where lookups (`where_is`, the MCP path guard) may search."""
    explicit = _many("roots")
    if explicit:
        return explicit
    seen, out = set(), []
    for p in (scratch(), models(), teachers(), *fit_store()):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


_HF_SNAPSHOT = re.compile(r"/models--([^/]+)--([^/]+)/snapshots/([0-9a-f]+)")


def portable(path) -> str:
    """`path` as it should be RECORDED (profiles, build records): relative
    to a configured location (``<teachers>/X``, ``<models>/X``), or
    ``hf://org/name@rev`` for a Hugging Face cache snapshot, so records do
    not carry one machine's disk layout. Anything else stays absolute."""
    p = pathlib.Path(os.path.realpath(path))
    m = _HF_SNAPSHOT.search(str(p))
    if m:
        return f"hf://{m.group(1)}/{m.group(2)}@{m.group(3)[:12]}"
    for label, root in (("teachers", teachers()), ("models", models()),
                        ("scratch", scratch())):
        r = pathlib.Path(os.path.realpath(root))
        if r in p.parents:
            return f"<{label}>/{p.relative_to(r)}"
    return str(p)


def require_storage(path) -> pathlib.Path:
    """Refuse a large-output path outside the configured storage."""
    rp = pathlib.Path(os.path.realpath(path))
    allowed = [pathlib.Path(os.path.realpath(r)) for r in (*roots(), scratch())]
    if not any(rp == a or a in rp.parents for a in allowed):
        raise SystemExit(f"REFUSED: {path} is outside the configured storage "
                         f"({', '.join(map(str, allowed))}); see `vqlab.config`")
    return rp
