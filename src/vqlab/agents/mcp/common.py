"""The MCP adapters' error type. `ToolError` IS `vqlab.ops.errors.LabError`
(and `vqlab.api.ApiError`): the server raises and reports the lab's own error."""

from __future__ import annotations

from vqlab.ops.errors import LabError as ToolError

__all__ = ["ToolError"]
