# ship/ — Ship: publish or serve

## Inputs
A gated artifact.

## Process
`vqlab publish` (a HUMAN action, gated on check-release; the MCP server does not expose it), `serve` (OpenAI-compatible, with MTP drafting).

## Outputs
A Hub revision, or a running server.

## Rules that bite
- Generate one token through the shipping runtime before calling anything releasable (III.11).
- Pull the published `model.py` from HF before a rebundle; local copies drift.
