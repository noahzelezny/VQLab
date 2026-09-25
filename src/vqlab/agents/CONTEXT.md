# agents/ — Agents: the lab over MCP

## Inputs
JSON-RPC over stdio.

## Process
`vqlab mcp`: `where_is`, `list_artifacts`, `run` (allowlisted, detached, under the GPU lease), `next_f_number` (reserve=true before a long run), `findings_append`, ...

## Outputs
Detached runs under `caches/`, FINDINGS-LOG entries.

## Rules that bite
- Use `where_is` before claiming anything is missing.
- `run` launches through `python -m vqlab.cli`, so every agent job is in the run log.
