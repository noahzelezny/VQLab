"""Filesystem measurements and the lab's small atomic file writes."""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import secrets
import shutil

GIB = 2**30


def dir_bytes(d, suffixes: tuple[str, ...] | None = None, recursive: bool = True) -> int:
    """Bytes of the REGULAR files under `d` (symlinks are another directory's
    bytes and are not counted), optionally only names ending in `suffixes`.

    recursive=True walks the tree (what deleting `d` frees). recursive=False
    sums `d`'s own entries and treats an unreadable `d` as 0 bytes."""
    total = 0
    if recursive:
        for dirpath, _, filenames in os.walk(d):
            for f in filenames:
                if suffixes is not None and not f.endswith(suffixes):
                    continue
                p = os.path.join(dirpath, f)
                if not os.path.islink(p):
                    total += os.lstat(p).st_size
        return total
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file(follow_symlinks=False) and (
                        suffixes is None or e.name.endswith(suffixes)):
                    total += e.stat(follow_symlinks=False).st_size
    except OSError:
        pass
    return total


@contextlib.contextmanager
def atomic_path(path):
    """Yield a temp path beside `path` to write; on success it replaces
    `path` whole (a reader sees the old file or the new one, never a part),
    on failure it is removed. The name is unique, so writers on two boxes
    sharing the folder never share one; not mkstemp, whose 0600 would make a
    shared file unreadable to the other box. Creates the parent."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text_atomic(path, text: str) -> pathlib.Path:
    with atomic_path(path) as tmp:
        with open(tmp, "x") as f:
            f.write(text)
    return pathlib.Path(path)


def copy_atomic(src, dst) -> pathlib.Path:
    """shutil.copyfile, landing whole (atomic_path)."""
    with atomic_path(dst) as tmp:
        shutil.copyfile(src, tmp)
    return pathlib.Path(dst)


def write_json_atomic(path, obj, indent: int | None = 1) -> pathlib.Path:
    return write_text_atomic(path, json.dumps(obj, indent=indent))


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def write_json(path, obj, indent: int | None = None) -> pathlib.Path:
    """json.dump to `path` in place (no temp file: a symlinked `path` is
    written through, as open(path, "w") would)."""
    path = pathlib.Path(path)
    path.write_text(json.dumps(obj, indent=indent))
    return path


def read_jsonl(path, limit: int | None = None, strict: bool = False) -> list:
    """The JSON lines of `path` ([] when it does not exist), only the last
    `limit` when given. A line that is not JSON is skipped, so one torn line
    does not hide the rest; strict=True raises ValueError instead."""
    try:
        lines = [ln for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
    except FileNotFoundError:
        return []
    if limit:
        lines = lines[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            if strict:
                raise
    return out
