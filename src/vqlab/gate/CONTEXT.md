# gate/ — Gate: is it loadable, correct and releasable? Pass/fail only

## Inputs
An artifact directory (most gates need no source model).

## Process
`vqlab check` (all static gates), `check-release`, `check-bundle`, `bundle-accept`, `verify` (outlier gate vs bf16), `smoke` (one token through the SHIPPED runtime), `vision-smoke`, `check-comparator`, `selftest`, `validate` (overnight queue, never publishes).

## Outputs
Exit codes and printed verdicts.

## Rules that bite
- Every gate must FAIL a known-bad input and PASS a known-good one before its pass means anything (III.5).
- A gate that never ran the code reports on the gate, not the artifact. Confirm the path was traversed.
- Measuring? Pin (symlink the shards + ONE model.py), then smoke, then measure.
