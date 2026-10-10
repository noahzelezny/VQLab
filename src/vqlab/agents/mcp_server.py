#!/usr/bin/env python3
"""vqlab mcp — the lab's MCP server (stdio JSON-RPC, stdlib only).

DEPRECATED location: the server is the package ``vqlab.agents.mcp`` (untangle
step 13). This module only RE-EXPORTS its names so existing imports keep
working. It is a re-export, not the home: ``monkeypatch.setattr`` on a name
here does NOT reach the code that uses it -- patch the defining module
instead: the lab library (``vqlab.ops.runs``, ``.box``, ``.queues``,
``.storage``, ``.docs``, ``.findings``, ``.roots``) for the operations, or
``vqlab.agents.mcp.server`` for the tool table.
"""

from __future__ import annotations

import sys

from vqlab.agents.mcp.common import (AGENTS_DIR, DOCS_URL, PKG, REPO, ToolError,  # noqa: F401
                                     _dir_bytes, _read_meta, _under, _walk, lab_roots)
from vqlab.agents.mcp.server import (B, I, L, PROTOCOL_VERSION, S, SERVER_NAME,  # noqa: F401
                                     SERVER_VERSION, TOOLS, _ok, _schema, call_tool, handle,
                                     main, serve, tool_list)
from vqlab.agents.mcp.tools_findings import (FINDINGS_LOG, VERDICTS, _F_HEAD,  # noqa: F401
                                             _FLOCK_DIR, _FLOCK_TTL, _findings_entries,
                                             _live_reservations, t_findings_append,
                                             t_findings_tail, t_next_f_number,
                                             t_release_f_number)
from vqlab.agents.mcp.tools_runs import (DEFAULT_RETRIES, GPU_FREE, HEAVY,  # noqa: F401
                                         RESUMABLE, RUN_ALLOWLIST, _box_state,
                                         _exo_instances, _lab_state, _lease_holder,
                                         _pid_alive, _queues_dir, _this_host, acquire_lease,
                                         lab_state_path, lease_path, release_lease, runs_dir,
                                         supervise, t_gpu_state, t_list_runs, t_queue_status,
                                         t_run, t_status, t_stop)
from vqlab.agents.mcp.tools_storage import (DOC_ALLOW, t_artifact_config,  # noqa: F401
                                            t_disk_free, t_list_artifacts, t_read_doc,
                                            t_where_is)
from vqlab.core.exo import exo_api  # noqa: F401  (was defined here; kept for callers)
from vqlab.util.fs import dir_bytes  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
