"""LabError: the one refusal type of the lab operations (and of vqlab.api,
which exports it as ApiError, and of the MCP server, as ToolError)."""

from __future__ import annotations

from typing import Any, Dict


class LabError(Exception):
    """A refusal or failure. ``code`` is stable (NOT_ALLOWED, OUTSIDE_ROOTS,
    EXO_PLACED, GPU_BUSY, NOT_FOUND, INVALID_ARG, FAILED, ...); ``extra``
    carries JSON-typed detail (allowlist, roots, lease holder, ...)."""

    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def as_dict(self) -> Dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.extra}

    as_result = as_dict   # the MCP server's spelling
