"""Filesystem measurements."""
from __future__ import annotations

import os


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
