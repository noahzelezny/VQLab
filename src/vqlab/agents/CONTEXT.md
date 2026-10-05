# agents/ — Agents: the lab over MCP, and queues of runs

## Inputs
JSON-RPC over stdio (`mcp`); a queue file of steps (`queue`).

## Process
`vqlab mcp`: `where_is`, `list_artifacts`, `run` (allowlisted, detached, under the GPU lease), `next_f_number` (reserve=true before a long run), `findings_append`, ...

`vqlab queue run <file>`: a multi-step campaign (a night of scoring) run from a git worktree PINNED at one commit, holding the same GPU lease as `run`, retrying resumable builds, refusing unsmoked pins, and failing a step on a nonzero exit, a traceback, or missing output. `--preflight` runs every step's small real version first; `--detach` survives the shell. `queue status` (and MCP `queue_status`) prints an ETA from the running step's own progress lines (fit-moe `[i/N] shard (Ts)`, kl-ladder cells) or past passed runs of the same step, else "ETA unknown". `--on BOX` first checks over ssh that every path the queue names exists AND reads on that box (an SMB mount can list while reads fail EIO), and refuses with the fix ("remount <volume> on <box>").

`vqlab reserve BOX|here --for WHO --until +3h|HH:MM [--note]`, `--list`, `--release`: intent, not a lock. Stored in `<scratch>/vqlab-reservations.json` (the shared SSD every box mounts; `$VQLAB_RESERVATIONS` overrides; falls back to `~/.vqlab/reservations.json`, local-only, when scratch is unreachable). Expired ones are ignored. MCP `gpu_state` lists every box's; `queue run` / `queue run --on BOX` refuse a box reserved for someone other than `$VQLAB_WHO` (else login) unless `--override-reservation`, which state.json records.

`[boxes.NAME] teachers = "<path>"`: that box's local teacher copies. `config.teachers()` returns it when running ON that box (`$VQLAB_BOX`, which `--on` sets, else hostname), and `vqlab <cmd>` expands a `<teachers>/X` argument through it, so a queue step naming `<teachers>/Model` reads the local copy there.

## Outputs
Detached runs under the MCP runs dir, FINDINGS-LOG entries; queues under `~/.vqlab/queues/<stamp>-<name>/` (state.json, steps/NN-name/{cmd,stdout,stderr,verdict.json}).

## Rules that bite
- Use `where_is` before claiming anything is missing.
- `run` launches through `python -m vqlab.cli`, so every agent job is in the run log.
- A queue step never imports the live checkout: commit mid-run freely. Uncommitted src/ changes are NOT in the queue (it refuses unless --allow-dirty).
- Every queue step needs a `preflight` block; run `--preflight` before any multi-hour queue.
