# records/ — Records: nothing happens without one

## Inputs
Any vqlab invocation or build.

## Process
`runlog.py` (every `vqlab <cmd>` -> `~/.vqlab/runs.jsonl`, read with `vqlab runs`), `provenance.py` (the build record in each artifact, read with `vqlab provenance`), `artifact_manifest.py` (tamper stamp for older artifacts).

## Outputs
JSONL run log (per user), `vqlab_provenance.json` (per artifact).

## Rules that bite
- A build tool writes its record LAST, into its own new output dir.
- The run log never breaks a run: logging errors are swallowed.
- Design and rollout: `docs/PROVENANCE.md`.
