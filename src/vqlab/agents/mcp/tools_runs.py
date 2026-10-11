"""MCP adapters for launching and watching jobs on THIS box: run, status,
stop, list_runs, gpu_state, queue_status. Argument mapping only; the work is
in vqlab.ops (runs, supervisor, box, queues), reached through vqlab.api
where the api has the operation. Two go straight to vqlab.ops:

* ``run`` returns the launcher's own record (run_id, run_dir,
  supervisor_pid, command, note), which is the MCP tool's documented
  output; ``api.run`` wraps the same call in a RunHandle;
* ``queue_status`` belongs with ``vqlab queue wait``; the api does not
  expose queues.

The old module-level names are re-exported for imports: patch the defining
vqlab.ops module, not this one."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vqlab import api
from vqlab.core.box.gpu_lease import acquire_lease, release_lease  # noqa: F401
from vqlab.ops import queues, runs
from vqlab.ops.box import _box_state  # noqa: F401
from vqlab.ops.queues import _queues_dir  # noqa: F401
from vqlab.ops.runs import (DEFAULT_RETRIES, GPU_FREE, HEAVY, RESUMABLE,  # noqa: F401
                            RUN_ALLOWLIST, _exo_instances, _lab_state, _lease_holder,
                            _pid_alive, _this_host, lab_state_path, lease_path, runs_dir)
from vqlab.ops.supervisor import supervise  # noqa: F401


def t_run(cmd: str, args: Optional[List[str]] = None, tag: str = "",
          retries: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
    """Launch an allowlisted vqlab subcommand, detached, under the GPU lease."""
    return runs.launch(cmd, args, tag=tag, retries=retries, force=force)


def t_status(run_id: str, tail: int = 40) -> Dict[str, Any]:
    return api.run_status(run_id=run_id, tail=tail)


def t_stop(run_id: str) -> Dict[str, Any]:
    return api.stop(run_id=run_id)


def t_list_runs(n: int = 20) -> Dict[str, Any]:
    return api.list_runs(n=n)


def t_queue_status(name: Optional[str] = None, n: int = 5) -> Dict[str, Any]:
    """Queue state from state.json (what `vqlab queue wait` blocks on)."""
    return queues.queue_status(name=name, n=n)


def t_gpu_state(boxes: bool = True) -> Dict[str, Any]:
    return api.gpu_state(boxes=boxes)
