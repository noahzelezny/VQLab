# agents/ — Agents: the lab over MCP, and queues of runs

## Inputs
JSON-RPC over stdio (`mcp`); a queue file of steps (`queue`).

## Process
`vqlab mcp`: `where_is`, `list_artifacts`, `run` (allowlisted, detached, under the GPU lease), `next_f_number` (reserve=true before a long run), `findings_append`, ...

`vqlab queue run <file>`: a multi-step campaign (a night of scoring) run from a git worktree PINNED at one commit, holding the same GPU lease as `run`, retrying resumable builds, refusing unsmoked pins, and failing a step on a nonzero exit, a traceback, or missing output. `--preflight` runs every step's small real version first; `--detach` survives the shell.

## Outputs
Detached runs under the MCP runs dir, FINDINGS-LOG entries; queues under `~/.vqlab/queues/<stamp>-<name>/` (state.json, steps/NN-name/{cmd,stdout,stderr,verdict.json}).

## Rules that bite
- Use `where_is` before claiming anything is missing.
- `run` launches through `python -m vqlab.cli`, so every agent job is in the run log.
- A queue step never imports the live checkout: commit mid-run freely. Uncommitted src/ changes are NOT in the queue (it refuses unless --allow-dirty).
- Every queue step needs a `preflight` block; run `--preflight` before any multi-hour queue.
