# ops/ — the lab operations on THIS box, as a library

**Inputs:** the storage roots, runs dir, GPU lease, findings log and queue
dir from `vqlab.config` (and their `VQLAB_*` env overrides).
**Outputs:** plain JSON-typed dicts; a refusal raises `LabError(code, message,
**extra)` from `errors.py` -- the same class `vqlab.api` exports as
`ApiError` and the MCP server as `ToolError`.

Layer: below `vqlab.api`, above `queue`. Consumers: `vqlab.api` (the stable
surface), and through it every MCP server -- ours (`agents/mcp/`) is a thin
adapter, and a developer's own would be built the same way.

| file | holds |
|---|---|
| `errors.py` | `LabError` |
| `roots.py` | `lab_roots`, `_under` (path guard), `_dir_bytes`, `_walk`, `PKG` / `REPO` / `DOCS_URL` |
| `storage.py` | `where_is`, `list_artifacts`, `artifact_config`, `disk_free` (read-only) |
| `docs.py` | `read_doc`, `DOC_ALLOW` (docs and stage CONTEXT.md files, never source) |
| `runs.py` | `launch`, `status`, `stop`, `list_runs`; `RUN_ALLOWLIST` / `GPU_FREE` / `RESUMABLE` (from `vqlab.commands`), `HEAVY`, `runs_dir`, lab-state |
| `supervisor.py` | `supervise`: the detached process `launch` starts (`python -m vqlab.ops.supervisor --supervise RUN_DIR`) |
| `box.py` | `gpu_state` (lease holder, exo placements, reservations, other boxes over ssh) |
| `queues.py` | `queue_status` (from state.json, never log wording) |
| `findings.py` | `next_f_number` (reserve), `release_f_number`, `findings_tail`, `findings_append` |

## Rules that bite
- `publish` is never launchable (`commands.py` never marks it mcp=True).
- Tests patch THESE modules (`lab.runs._exo_instances`, `lab.findings.FINDINGS_LOG`
  and `_FLOCK_DIR`, ...), never a re-export; always point `FINDINGS_LOG` and
  `_FLOCK_DIR` at a temp dir -- the defaults are the real lab record.
- Nothing here prints or exits; a refusal is a `LabError` with a stable code.
