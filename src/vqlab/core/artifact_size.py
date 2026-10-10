"""An artifact's index total_size checked against its shards' tensor bytes.

The library half of `vqlab size` (the command is ship/size.py). It sits in
core so `check-release` (gate) can read the same number the same way.
"""
from __future__ import annotations

import json

GIB = 2 ** 30


def index_total_size(path) -> tuple:
    """(have, want, state) for the index's metadata.total_size against the
    shards' tensor bytes. state is "ok", "missing" or "stale by +x.xx GiB".
    Shared by `size --check-index` and `check-release`, so both gates read
    the same number the same way."""
    import pathlib
    from vqlab.core.artifact import Artifact, tensor_bytes
    art = Artifact.open(path)
    doc = json.loads((pathlib.Path(path) / "model.safetensors.index.json").read_text())
    have = (doc.get("metadata") or {}).get("total_size")
    want = tensor_bytes(art.dir, art.shards)
    state = "ok" if have == want else ("missing" if have is None else
                                       f"stale by {(have - want) / GIB:+.2f} GiB")
    return have, want, state
