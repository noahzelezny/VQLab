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


@functools.lru_cache(maxsize=None)
def _file_table(path: str, table: str) -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f).get(table, {})


def boxes() -> dict:
    """Other machines a queue may run on (`vqlab queue run --on NAME`).

        [boxes.m4]
        ssh       = "user@192.0.2.2"
        repo      = "/Volumes/Shared/vqlab-clone"   # a clone of THIS repo on shared storage
        python    = "/path/to/envs/exo/bin/python"
        config    = "/Volumes/Shared/m4-config.toml" # that box's own [paths]
        queue_dir = "/Volumes/Shared/queues-m4"      # shared, so this box can `queue wait` on it
        teachers  = "/Volumes/Local/teachers"     # optional: that box's LOCAL teacher copies
        hostname  = "box-b"                         # optional: its short hostname, if not NAME
        machine   = "Mac Studio"                    # optional: the name the Knurlogic page lists this box under
                                                    # (vision-smoke --knurlogic NAME needs it)

    `teachers` makes `teachers()` resolve to the box's local copy when
    running ON that box (see `this_box`), so minibase / fit-moe read local
    shards without hand-written paths (a local copy roughly halved a remote
    box's fit read contention over SMB, 2026-10-02). Teacher shards are shared
    by hard link with other tenants: never rewrite one in place.
    """
    return _file_table(str(config_file()), "boxes")


def this_box() -> str | None:
    """The [boxes.NAME] this process runs on: $VQLAB_BOX (`queue run --on`
    sets it on the remote), else the box whose NAME or `hostname` is this
    machine's short hostname, else None (this is the home box)."""
    env = os.environ.get("VQLAB_BOX")
    if env:
        return env
    import socket
    host = socket.gethostname().split(".")[0].lower()
    for name, b in boxes().items():
        if host in (name.lower(), str(b.get("hostname", "")).lower()):
            return name
    return None


def box_teachers(name: str | None = None) -> pathlib.Path | None:
    """The `teachers` override of box NAME (default: this box), or None."""
    b = boxes().get(name if name is not None else (this_box() or ""))
    v = (b or {}).get("teachers")
    return pathlib.Path(v).expanduser() if v else None


def scratch() -> pathlib.Path:
    """Builds, pins and run outputs."""
    return _one("scratch")


def models() -> pathlib.Path:
    """Served (published or candidate) artifacts."""
    return _one("models")


def teachers() -> pathlib.Path:
    """bf16 teacher checkpoints: $VQLAB_TEACHERS_DIR, else this box's
    `[boxes.NAME] teachers` override (a local copy), else [paths]."""
    if not os.environ.get(_ENV["teachers"]):
        local = box_teachers()
        if local:
            return local
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


_PLACEHOLDER = re.compile(r"(^|=)<(teachers|models|scratch)>(?=/|$)")


def expand(arg: str, box: str | None = None) -> str:
    """``<teachers>/X`` (the form `portable` records) -> this box's teachers
    root + /X; likewise ``<models>`` and ``<scratch>``. Also after an ``=``
    (``--src=<teachers>/X``, ``prose=<scratch>/c``). `vqlab <cmd>` applies it
    to every argument, so a queue step naming ``<teachers>/DeepSeek-V4`` reads
    that box's local copy on a box with a `teachers` override and the shared [paths] teachers elsewhere. `box` resolves for
    another box ([boxes.NAME] teachers, else this config's [paths])."""
    def sub(m):
        key = m.group(2)
        if key == "teachers" and box is not None:
            root = box_teachers(box) or _one("teachers")
        else:
            root = {"teachers": teachers, "models": models, "scratch": scratch}[key]()
        return m.group(1) + str(root)
    return _PLACEHOLDER.sub(sub, arg, count=1)


def require_free(path, need_bytes: int, what: str, margin_gib: float = 5.0) -> None:
    """Refuse a writer BEFORE it starts when its output volume cannot hold it.
    `need_bytes` is the writer's own estimate of what it still has to write;
    a run that would die hours in at 0 bytes free is refused in a second.
    VQLAB_SKIP_DISK_CHECK=1 overrides (you know the estimate is wrong)."""
    import shutil
    if os.environ.get("VQLAB_SKIP_DISK_CHECK") == "1":
        return
    p = pathlib.Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free
    need = need_bytes + margin_gib * 2**30
    if need > free:
        raise SystemExit(f"REFUSED ({what}): {path} needs ~{need_bytes / 2**30:.1f} GiB "
                         f"+ {margin_gib:g} GiB margin, only {free / 2**30:.1f} GiB free "
                         f"(VQLAB_SKIP_DISK_CHECK=1 overrides)")


def require_storage(path) -> pathlib.Path:
    """Refuse a large-output path outside the configured storage."""
    rp = pathlib.Path(os.path.realpath(path))
    allowed = [pathlib.Path(os.path.realpath(r)) for r in (*roots(), scratch())]
    if not any(rp == a or a in rp.parents for a in allowed):
        raise SystemExit(f"REFUSED: {path} is outside the configured storage "
                         f"({', '.join(map(str, allowed))}); see `vqlab.config`")
    return rp
