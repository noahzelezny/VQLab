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
    records   = "~/.local/share/vqlab"            # findings log, registry, families, notes

``records`` holds the state vqlab WRITES that is not an artifact: the
findings log, the artifact registry, family profiles, research notes and
queue records, each in a fixed subfolder (see `records()`). It never
defaults into the installed package or a checkout; a checkout user who
wants the old in-repo ``lab/`` sets ``records`` to it.

Large outputs belong on the volumes named here: `require_storage` refuses a
write path that is not under one of them.

Optional ``[card]`` table, read by `vqlab card` (see `card()`): a support /
contact block appended to every generated model card. Unset keys print
nothing; with no table the card has no support section::

    [card]
    sponsor_url = "https://github.com/sponsors/<you>"
    contact     = "you@example.com"          # email or URL
    blurb       = "One paragraph shown above the links."
    contract    = "One line on paid work you take, if any."
"""
from __future__ import annotations

import functools
import os
import pathlib
import re
import tomllib

from vqlab import _layout

HOME = pathlib.Path.home() / ".vqlab"

_ENV = {
    "scratch": "VQLAB_SCRATCH",
    "models": "VQLAB_MODELS_DIR",
    "teachers": "VQLAB_TEACHERS_DIR",
    "fit_store": "VQLAB_FIT_STORE",
    "roots": "VQLAB_ROOTS",
    "gpu_lease": "VQLAB_GPU_LEASE",
    "records": "VQLAB_RECORDS",
}
_DEFAULT = {
    "scratch": HOME / "scratch",
    "models": HOME / "models",
    "teachers": HOME / "teachers",
    "fit_store": HOME / "fits",
    "gpu_lease": HOME / "gpu.lease",
    "records": pathlib.Path(os.environ.get("XDG_DATA_HOME")
                            or pathlib.Path.home() / ".local" / "share") / "vqlab",
}
KEYS = ("scratch", "models", "teachers", "fit_store", "roots", "gpu_lease", "records")
MULTI = ("fit_store", "roots")


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


def card() -> dict:
    """The [card] table (sponsor_url, contact, blurb, contract); {} when unset."""
    return dict(_file_table(str(config_file()), "card"))


def boxes() -> dict:
    """Other machines a queue may run on (`vqlab queue run --on NAME`).

        [boxes.m4]
        ssh       = "user@192.0.2.2"
        repo      = "/Volumes/Shared/vqlab-clone"   # a clone of THIS repo on shared storage
        python    = "/path/to/envs/exo/bin/python"
        config    = "/Volumes/Shared/m4-config.toml" # that box's own [paths]
        queue_dir = "/Volumes/Shared/queues-m4"      # shared, so this box can `queue wait` on it
        teachers  = "/Volumes/Local/teachers"     # optional: that box's LOCAL teacher copies
        models    = "/path/to/models"             # optional: that box's served-artifacts root
                                                    # (check-release --cluster-peer reads it)
        hostname  = "box-b"                         # optional: its short hostname, if not NAME
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


def box_models(peer: str) -> str | None:
    """The `models` root of the [boxes.NAME] whose NAME, `ssh` or `hostname`
    is PEER (as given to --cluster-peer), or None. Returned as the string the
    config holds: it names a path on THAT box, so it is not expanded here."""
    host = peer.split("@")[-1].lower()
    for name, b in boxes().items():
        remote = str(b.get("ssh", "")).lower()
        keys = {name.lower(), remote, remote.split("@")[-1],
                str(b.get("hostname", "")).lower()}
        if peer.lower() in keys or host in keys:
            v = b.get("models")
            return str(v) if v else None
    return None


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


def records() -> pathlib.Path:
    """User-written state that is not an artifact ($VQLAB_RECORDS, [paths]
    records, else ~/.local/share/vqlab or $XDG_DATA_HOME/vqlab)."""
    return _one("records")


def findings_log() -> pathlib.Path:
    """The F-numbered findings log: $VQLAB_FINDINGS_LOG, else
    <records>/FINDINGS-LOG.md."""
    env = os.environ.get("VQLAB_FINDINGS_LOG")
    return pathlib.Path(env).expanduser() if env else records() / "FINDINGS-LOG.md"


def registry_dir() -> pathlib.Path:
    """Artifact registry (artifacts.jsonl, hub.jsonl): <records>/registry."""
    return records() / "registry"


def families_dir() -> pathlib.Path:
    """Family entries and teacher profiles: $VQLAB_FAMILIES_DIR, else
    <records>/families."""
    env = os.environ.get("VQLAB_FAMILIES_DIR")
    return pathlib.Path(env).expanduser() if env else records() / "families"


def checkout_families_dir() -> pathlib.Path:
    """The `families/` data checked in at the root of a source checkout
    (src/vqlab/config.py -> <checkout>/families). A checkout-only read: an
    installed package has no such directory and readers skip a missing one.
    families_dir() is the user's location; registries read both."""
    here = pathlib.Path(__file__).resolve()
    return here.parent.parent.parent / "families"


def research_dir() -> pathlib.Path:
    """Research notes, sweep plans, result tables: <records>/research."""
    return records() / "research"


def queues_dir() -> pathlib.Path:
    """Queue run records: $VQLAB_QUEUE_DIR, else <records>/queues."""
    env = os.environ.get("VQLAB_QUEUE_DIR")
    return pathlib.Path(env).expanduser() if env else records() / "queues"


# Where records lived before they moved under `records` (2026-10): a checkout's
# lab/FINDINGS-LOG.md and registry/*.jsonl, and ~/.vqlab/queues. Read only, to
# warn; nothing here moves or reads the data.
CHECKOUT = _layout.SRC.parent


def _is_checkout() -> bool:
    return (CHECKOUT / "pyproject.toml").is_file()


def legacy_findings_log() -> pathlib.Path | None:
    """A checkout's old lab/FINDINGS-LOG.md, when it exists and is not the
    log this config resolves to."""
    p = CHECKOUT / "lab" / "FINDINGS-LOG.md"
    if not (_is_checkout() and p.is_file()):
        return None
    try:
        if p.resolve() == findings_log().resolve():
            return None
    except OSError:
        pass
    return p


def legacy_records() -> list[dict]:
    """Old-location records that the current resolution no longer reads:
    [{what, path, fix}], only while `records` (and the per-kind override) is
    unset. Empty when nothing was left behind."""
    if source("records") != "default":
        return []
    out = []
    log = legacy_findings_log()
    if log and not os.environ.get("VQLAB_FINDINGS_LOG"):
        out.append({"what": "findings log", "path": str(log),
                    "fix": f"vqlab lab config set records {log.parent}   "
                           f"(or export VQLAB_FINDINGS_LOG={log})"})
    reg = CHECKOUT / "registry"
    if _is_checkout() and any(f.stat().st_size for f in reg.glob("*.jsonl")):
        out.append({"what": "artifact registry", "path": str(reg),
                    "fix": f"vqlab lab config set records <dir>, then copy {reg}/*.jsonl "
                           f"into <dir>/registry/ (default now {registry_dir()})"})
    q = pathlib.Path.home() / ".vqlab" / "queues"
    if not os.environ.get("VQLAB_QUEUE_DIR") and q.is_dir() and any(q.iterdir()):
        out.append({"what": "queue records", "path": str(q),
                    "fix": f"export VQLAB_QUEUE_DIR={q}"})
    return out


def legacy_warnings() -> list[str]:
    return [f"records moved: {r['what']} at {r['path']} is no longer read (records now "
            f"default to {records()}). To keep using it: {r['fix']}"
            for r in legacy_records()]


def source(key: str) -> str:
    """Where `key`'s value comes from: 'env', 'file' or 'default'."""
    if os.environ.get(_ENV[key]):
        return "env"
    return "file" if _file_paths(str(config_file())).get(key) else "default"


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


CHECKOUT_LABEL = "<checkout>"


def checkout_dir() -> pathlib.Path | None:
    """The git checkout this package runs from, or None for a wheel install."""
    repo = _layout.SRC.parent
    return repo if (repo / ".git").exists() and (repo / "src" / "vqlab").is_dir() else None


def private_prefixes() -> list[tuple[str, str]]:
    """[(absolute prefix, placeholder)], longest prefix first: every local
    path that must not leave this machine in a record or a published file,
    and what a record writes instead. Both the configured spelling and its
    realpath are listed (a root may be a symlink)."""
    pairs = []
    co = checkout_dir()
    if co is not None:
        pairs.append((co, CHECKOUT_LABEL))
    pairs += [(teachers(), "<teachers>"), (models(), "<models>"),
              (scratch(), "<scratch>")]
    pairs += [(p, "<fits>") for p in fit_store()]
    pairs += [(p, "<root>") for p in _many("roots")]
    pairs.append((pathlib.Path.home(), "~"))
    out, seen = [], set()
    for p, label in pairs:
        for s in (str(p), os.path.realpath(p)):
            s = s.rstrip("/")
            if s and s != "/" and s not in seen:
                seen.add(s)
                out.append((s, label))
    out.sort(key=lambda x: -len(x[0]))
    return out


def portable_text(s: str, prefixes=None) -> str:
    """Every absolute local path inside `s` (an argv element, a cwd, a
    ``--src=/x`` pair, a message) rewritten to its portable form: Hugging
    Face cache snapshots to ``hf://org/name@rev``, paths under configured
    roots to ``<models>/X`` etc., the checkout to ``<checkout>``, anything
    else under the home directory to ``~/X``. Text with no such path comes
    back unchanged (the same object)."""
    if "/" not in s:
        return s
    s2 = re.sub(r"/[^\s\"']*?/models--([^/\s\"']+)--([^/\s\"']+)/snapshots/([0-9a-f]{12})[0-9a-f]*",
                lambda m: f"hf://{m.group(1)}/{m.group(2)}@{m.group(3)}", s)
    for pre, label in (prefixes if prefixes is not None else private_prefixes()):
        s2 = re.sub(re.escape(pre) + r"(?=/|$|[\s\"',:;)\]])", lambda _m, lab=label: lab, s2)
    return s if s2 == s else s2


def portable_obj(o, prefixes=None):
    """`portable_text` applied to every string in a JSON-shaped value
    (dict keys and values, lists). Numbers, bools and None pass through."""
    prefixes = prefixes if prefixes is not None else private_prefixes()
    if isinstance(o, str):
        return portable_text(o, prefixes)
    if isinstance(o, dict):
        return {portable_obj(k, prefixes): portable_obj(v, prefixes) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [portable_obj(v, prefixes) for v in o]
    if isinstance(o, pathlib.PurePath):
        return portable_text(str(o), prefixes)
    return o


def resolve_recorded(path: str) -> str:
    """A path as a record holds it -> a local path to open: placeholders
    (``<models>/X``, ``<fits>/X``, ``<checkout>/X``, ``~/X``) are expanded on
    this box; an absolute path from an older record is returned as is."""
    if path.startswith("~/") or path == "~":
        return os.path.expanduser(path)
    if path.startswith(CHECKOUT_LABEL):
        co = checkout_dir()
        return str(co) + path[len(CHECKOUT_LABEL):] if co is not None else path
    if path.startswith("<fits>"):
        return str(fit_store()[0]) + path[len("<fits>"):]
    return expand(path)


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
