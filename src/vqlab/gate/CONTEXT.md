# gate/ — Gate: is it loadable, correct and releasable? Pass/fail only

## Inputs
An artifact directory (most gates need no source model); `spelling` takes release TEXT (cards, the paper draft); its module is `gate/spelling.py` (was `us_spelling.py`; the old name, bare or `vqlab.gate.us_spelling`, is a deprecated alias for one release).

## Process
`vqlab check` (all static gates), `check-release`, `check-bundle`, `bundle-accept`, `verify` (outlier gate vs bf16), `smoke` (one token through the SHIPPED runtime), `vision-smoke`, `check-comparator`, `parity` (family checklist of reference-inference behaviours -- clamps, eps, routing -- each with the test that checks it; exits 1 on an untested required item; no GPU), `selftest`, `validate` (overnight queue, never publishes), `pin` (symlinked copy + smoke + `vqlab_pin.json`; scorers refuse an unsmoked pin), `mtp-smoke-head`, `spelling`. Not CLI: `kernel_accept.py` (device-codebook d4 kernels vs threadgroup + a float32 reference, repo runtime; was `e134_accept.py`) and `verify_dev_codebook.py` (the same agreement through `_fused` on real artifact layers; was `verify_e134_devcb.py`), frozen acceptance checks for the E134 runtime change. Both old names still import. `kernel_reference.py` holds the shared `reference()` dequant-matmul that kernel_accept and bundle-accept score against.

## Outputs
Exit codes and printed verdicts.

## Rules that bite
- Every gate must FAIL a known-bad input and PASS a known-good one before its pass means anything (III.5).
- A gate that never ran the code reports on the gate, not the artifact. Confirm the path was traversed.
- Measuring? Pin (symlink the shards + ONE model.py), then smoke, then measure.
- The Knurlogic client helpers the check-release / vision-smoke arms use are `core/knurlogic_client.py`; gate never imports bench. `pin.py` writes the record whose format and check are `core/pin_record.py`.
