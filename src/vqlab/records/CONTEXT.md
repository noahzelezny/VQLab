# records/ — Records: nothing happens without one

## Inputs
Any vqlab invocation or build.

## Process
`runlog.py` (every `vqlab <cmd>` -> `~/.vqlab/runs.jsonl`, read with `vqlab runs`), `provenance.py` (the build record in each artifact, read with `vqlab provenance`), `artifact_manifest.py` (tamper stamp for older artifacts), `registry.py` (`vqlab registry`: every artifact from provable facts + Hub drift, into git-tracked `registry/*.jsonl`), `step_verdict.py` (the one rule for whether a queue step failed). `provenance.measured()` is the stamp every scorer writes into its output.

## Outputs
JSONL run log (per user), `vqlab_provenance.json` (per artifact), `registry/artifacts.jsonl` + `registry/hub.jsonl` (in git), a `measured` block in every score.

## Rules that bite
- A build tool writes its record LAST, into its own new output dir.
- The run log never breaks a run: logging errors are swallowed.
- Design and rollout: `docs/PROVENANCE.md`.
