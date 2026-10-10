"""Shared pieces of the MCP adapters. The implementations live in the lab
library (vqlab.ops); these names are re-exported so old imports keep working.
`ToolError` IS `vqlab.ops.errors.LabError` (and `vqlab.api.ApiError`)."""

from __future__ import annotations

from vqlab.ops.errors import LabError as ToolError  # noqa: F401
from vqlab.ops.roots import (DOCS_URL, PKG, REPO, _dir_bytes, _under, _walk,  # noqa: F401
                             lab_roots)
from vqlab.ops.runs import _read_meta  # noqa: F401

AGENTS_DIR = PKG / "agents"
