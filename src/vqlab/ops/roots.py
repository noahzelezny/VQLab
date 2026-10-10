"""The configured storage roots and the path guards every lab operation
shares: is a path under a root, how big is a directory, walk the roots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import vqlab
from vqlab import config
from vqlab.util.fs import dir_bytes

PKG = Path(vqlab.__file__).resolve().parent   # src/vqlab
REPO = PKG.parents[1]
# Where read_doc points a wheel install, which carries no docs/.
DOCS_URL = "https://github.com/noahzelezny/VQLab/blob/master"


def lab_roots() -> List[Path]:
    return config.roots()


def _under(path: Path, roots: List[Path]) -> bool:
    try:
        rp = path.resolve()
    except OSError:
        rp = path
    for r in roots:
        try:
            rp.relative_to(r.resolve() if r.exists() else r)
            return True
        except ValueError:
            continue
    return False


def _dir_bytes(d: Path, suffixes=(".safetensors",)) -> int:
    return dir_bytes(d, suffixes=suffixes, recursive=False)


def _walk(roots: List[Path], max_depth: int):
    for root in roots:
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                yield root, e, depth
                if e.is_dir(follow_symlinks=False) and depth + 1 < max_depth \
                        and e.name != "__pycache__":
                    stack.append((Path(e.path), depth + 1))
