"""Storage settings: read, validate and write vqlab's locations.

ONE implementation, used by `vqlab lab config` (CLI) and the GUI's Settings
view. Values resolve as `vqlab.config` documents (env var, then the config
file's ``[paths]`` table, then a default); `set_value` writes only the config
file, editing one line of ``[paths]`` and leaving every other line as it was.
Nothing here deletes anything.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from vqlab import config as C

DESCRIPTIONS = {
    "lab": "one folder every location below defaults under",
    "scratch": "builds, pins and run outputs (large)",
    "models": "served or candidate artifacts (large)",
    "teachers": "bf16 teachers, the full-precision models builds start from; the models folder unless kept apart (large)",
    "fit_store": "fit archive; the first entry receives new fits",
    "roots": "where lookups may search (default: the four above)",
    "gpu_lease": "lock FILE every GPU job on this machine takes",
    "records": "findings log, registry, families, research notes, queue records",
}
FILE_KEYS = ("gpu_lease",)       # a file path: its parent dir must be writable


def _reset():
    C._file_paths.cache_clear()
    C._file_table.cache_clear()


def _existing(p: pathlib.Path) -> pathlib.Path | None:
    while not p.exists() and p != p.parent:
        p = p.parent
    return p if p.exists() else None


def free_bytes(path) -> int | None:
    """Free bytes on the volume that holds `path` (or its nearest existing
    ancestor); None when nothing on the way up exists."""
    e = _existing(pathlib.Path(path).expanduser())
    try:
        return shutil.disk_usage(e).free if e else None
    except OSError:
        return None


def source(key: str) -> str:
    env = C._ENV[key]
    if os.environ.get(env):
        return f"${env}"
    if key == "teachers" and C.box_teachers():
        return f"[boxes.{C.this_box()}] teachers"
    if C._file_paths(str(C.config_file())).get(key):
        return str(C.config_file())
    if key == "teachers" and C.teachers() == C.models():
        return "same as models"
    return "default"


def _value(key):
    return {"lab": C.lab, "scratch": C.scratch, "models": C.models, "teachers": C.teachers,
            "fit_store": C.fit_store, "roots": C.roots, "gpu_lease": C.gpu_lease,
            "records": C.records}[key]()


def locations() -> list[dict]:
    """One row per key: key, paths, source, exists, free bytes, description."""
    rows = []
    for k in C.KEYS:
        v = _value(k)
        paths = [pathlib.Path(p) for p in (v if isinstance(v, list) else [v])]
        rows.append({"key": k, "paths": [str(p) for p in paths], "source": source(k),
                     "exists": [p.exists() for p in paths],
                     "free_bytes": [free_bytes(p) for p in paths],
                     "description": DESCRIPTIONS[k]})
    return rows


def _du_kib(path: pathlib.Path, timeout: float) -> int | None:
    try:
        r = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True,
                           timeout=timeout)
        return int(r.stdout.split()[0]) if r.stdout.strip() else None
    except (subprocess.TimeoutExpired, OSError, ValueError, IndexError):
        return None


def campaign_sizes(root=None, timeout: float = 20.0, budget: float = 120.0) -> list[dict]:
    """Top-level folders of scratch with their size (`du -sk`, each capped at
    `timeout` s and all of them at `budget` s; a folder that runs out of time
    reports bytes None rather than hanging)."""
    import time
    root = pathlib.Path(root) if root else C.scratch()
    if not root.is_dir():
        return []
    out, t0 = [], time.monotonic()
    with os.scandir(root) as it:
        entries = sorted((e for e in it if e.is_dir(follow_symlinks=False)
                          and not e.name.startswith(".")), key=lambda e: e.name)
    for e in entries:
        left = budget - (time.monotonic() - t0)
        kib = _du_kib(pathlib.Path(e.path), min(timeout, left)) if left > 0 else None
        out.append({"name": e.name, "path": e.path,
                    "bytes": None if kib is None else kib * 1024,
                    "mtime": e.stat(follow_symlinks=False).st_mtime})
    return out


# what each location holds, in the words the GUI shows
LABELS = {"models": "models", "teachers": "bf16 teachers", "scratch": "work in progress",
          "fit_store": "fit cache", "records": "lab notes"}


def _volume(p: pathlib.Path):
    """(device id, mount point) of the volume holding `p` (or its nearest
    existing ancestor)."""
    e = _existing(p)
    if e is None:
        return None, None
    dev, m = e.stat().st_dev, e.resolve()
    while m != m.parent and m.parent.stat().st_dev == dev:
        m = m.parent
    return dev, str(m)


def usage(timeout: float = 30.0) -> dict:
    """Per volume: total / used / free, which vqlab locations live there and
    how much each holds (`du`, capped at `timeout` s each; None when it runs
    out of time). `system` marks the volume holding the home directory --
    a lab on it competes with the OS for space."""
    home_dev = pathlib.Path.home().stat().st_dev
    vols, seen = {}, set()
    for key, label in LABELS.items():
        v = _value(key)
        for path in (v if isinstance(v, list) else [v]):
            path = pathlib.Path(path)
            if path.resolve() in seen:          # bf16 teachers sharing the models folder
                continue
            seen.add(path.resolve())
            dev, mount = _volume(path)
            if dev is None:
                continue
            if dev not in vols:
                du = shutil.disk_usage(mount)
                vols[dev] = {"mount": mount, "total": du.total, "used": du.used,
                             "free": du.free, "system": dev == home_dev, "locations": []}
            kib = _du_kib(path, timeout) if path.exists() else 0
            vols[dev]["locations"].append({"key": key, "label": label, "path": str(path),
                                           "bytes": None if kib is None else kib * 1024})
    return {"lab": str(C.lab()), "lab_source": source("lab"),
            "configured": C.config_file().exists(), "volumes": list(vols.values())}


def validate(key: str, path: str, create: bool = False) -> dict:
    """Check `path` is usable for `key`: it exists (or, with create, is
    created), it is a directory (the parent dir for a file key), and it is
    writable. -> {ok, key, path, created, free_bytes, error}."""
    if key not in C.KEYS:
        return {"ok": False, "key": key, "path": path,
                "error": f"unknown key {key!r}; one of {', '.join(C.KEYS)}"}
    if not path or not str(path).strip():
        return {"ok": False, "key": key, "path": path, "error": "empty path"}
    p = pathlib.Path(os.path.expanduser(str(path).strip()))
    if not p.is_absolute():
        return {"ok": False, "key": key, "path": str(p), "error": "path must be absolute"}
    d = p.parent if key in FILE_KEYS else p
    res = {"ok": False, "key": key, "path": str(p), "created": False}
    if d.exists() and not d.is_dir():
        return {**res, "error": f"{d} exists and is not a directory"}
    if not d.exists():
        if not create:
            return {**res, "free_bytes": free_bytes(d),
                    "error": f"{d} does not exist (create it, or pass create)"}
        try:
            d.mkdir(parents=True, exist_ok=True)
            res["created"] = True
        except OSError as e:
            return {**res, "error": f"cannot create {d}: {e}"}
    try:
        with tempfile.NamedTemporaryFile(dir=d, prefix=".vqlab-write-test-"):
            pass
    except OSError as e:
        return {**res, "free_bytes": free_bytes(d), "error": f"{d} is not writable: {e}"}
    return {**res, "ok": True, "free_bytes": free_bytes(d), "error": None}


_TABLE = re.compile(r"^\s*\[\[?([^\]]+)\]\]?\s*(#.*)?$")


def _toml_value(key, path):
    s = json.dumps(str(path))
    return f"[{s}]" if key in C.MULTI else s


def write_key(cfg: pathlib.Path, key: str, path) -> None:
    """Set ``[paths] key`` in `cfg`, keeping every other line (comments,
    other tables) unchanged. A multi-line value for the key is replaced."""
    lines = cfg.read_text().splitlines() if cfg.exists() else []
    new = f"{key} = {_toml_value(key, path)}"
    start = next((i for i, x in enumerate(lines)
                  if (m := _TABLE.match(x)) and m.group(1).strip() == "paths"), None)
    if start is None:
        lines += ([""] if lines and lines[-1].strip() else []) + ["[paths]", new]
    else:
        end = next((i for i in range(start + 1, len(lines)) if _TABLE.match(lines[i])),
                   len(lines))
        kre = re.compile(rf"^\s*{re.escape(key)}\s*=")
        hit = next((i for i in range(start + 1, end) if kre.match(lines[i])), None)
        if hit is None:
            ins = end
            while ins > start + 1 and not lines[ins - 1].strip():
                ins -= 1
            lines.insert(ins, new)
        else:
            stop = hit + 1
            val = lines[hit].split("=", 1)[1]
            if val.strip().startswith("[") and "]" not in val.split("#")[0]:
                while stop < end and "]" not in lines[stop - 1].split("#")[0]:
                    stop += 1
            lines[hit:stop] = [new]
    cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.with_name(cfg.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    import tomllib
    tomllib.loads(tmp.read_text())                # never leave an unreadable config
    os.replace(tmp, cfg)
    _reset()


def set_value(key: str, path: str, create: bool = False) -> dict:
    """Validate, then write `key` to the config file. -> validate()'s dict
    plus `config` and, when an env var overrides the file, `shadowed_by`."""
    res = validate(key, path, create=create)
    if not res["ok"]:
        return res
    cfg = C.config_file()
    write_key(cfg, key, res["path"])
    res["config"] = str(cfg)
    env = C._ENV[key]
    if os.environ.get(env):
        res["shadowed_by"] = f"${env}"
    return res


def gib(n) -> str:
    return "?" if n is None else f"{n / 2**30:.1f} GiB"
