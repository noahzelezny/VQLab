"""MCP adapters for the read-only storage tools: where_is, list_artifacts,
artifact_config, read_doc, disk_free. Argument mapping only; the work is in
vqlab.api (where_is, list_artifacts, disk_free) and vqlab.ops (artifact_config
and read_doc, which the api deliberately leaves out: api.artifact_info is the
library's richer artifact read, and read_doc feeds an agent's context, not a
program)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vqlab import api
from vqlab.ops import docs, storage


def t_where_is(name: str, kind: str = "any", max_depth: int = 3) -> Dict[str, Any]:
    """Find a thing by (sub)name across the lab roots. Deterministic."""
    return api.where_is(name=name, kind=kind, max_depth=max_depth)


def t_list_artifacts(root: str = "", max_depth: int = 3) -> Dict[str, Any]:
    """Directories with a config.json under the lab roots (or one root)."""
    return api.list_artifacts(root=root, max_depth=max_depth)


def t_artifact_config(path: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """The shipped config.json -- the record of what actually shipped."""
    return storage.artifact_config(path, keys)


def t_read_doc(path: str, start: int = 1, lines: int = 200) -> Dict[str, Any]:
    """Read a slice of a lab doc (docs/, AGENTS.md, the law book, ...)."""
    return docs.read_doc(path, start=start, lines=lines)


def t_disk_free() -> Dict[str, Any]:
    """Free space on every configured storage root."""
    return api.disk_free()
