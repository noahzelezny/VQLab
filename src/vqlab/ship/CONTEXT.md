# ship/ — Ship: publish or serve

## Inputs
A gated artifact.

## Process
`vqlab publish` (a HUMAN action, gated on check-release; the MCP server does not expose it), `serve` (OpenAI-compatible, with MTP drafting).

Release text: `card.py` (`vqlab card`: the model card filled from config.json, the KL JSON via `card_tables.render`, `sizes()` and the build record via `records/provenance.records`; stdout unless `--out`), `card_checks.py` (its refusals: a KL fingerprint mismatch -- a later sidecar with no text tensors is allowed and named by `text_match` -- mixed builds, or artifacts changed during the run), `card_tables.py` (`vqlab card-tables`), `size.py` (`vqlab size`: text / +tower / +MTP; `--check-index`/`--fix-index` use `core/artifact_size.index_total_size`). The spelling lint is `gate/spelling.py` (check-release runs it).

## Outputs
A Hub revision, or a running server.

## Rules that bite
- Generate one token through the shipping runtime before calling anything releasable (III.11).
- Pull the published `model.py` from HF before a rebundle; local copies drift.
- Old names `vqlab.records.card`, `vqlab.records.card_tables` and `vqlab.records.size_cmd` (bare `size_cmd`) are deprecated aliases for one release (`_layout.MOVED`/`RENAMED`): same module object. The CLI is unchanged (`card`, `card-tables`, `size`).
- `release_prep.hub_diff` is `core/hub.hub_diff` (re-exported). `mtp_run` takes `add_sampling_args`/`load_all`/`prompt_ids` from `mtp/cli_common.py` (re-exported).
