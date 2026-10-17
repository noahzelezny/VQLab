# gate/ — Gate: is it loadable, correct and releasable? Pass/fail only

## Inputs
An artifact directory (most gates need no source model); `spelling` takes release TEXT (cards, the paper draft); its module is `gate/spelling.py`.

## Process
`vqlab check` (all static gates), `check-release`, `check-bundle`, `bundle-accept`, `verify` (outlier gate vs bf16), `smoke` (one token through the SHIPPED runtime), `vision-smoke`, `check-comparator`, `parity` (family checklist of reference-inference behaviours -- clamps, eps, routing -- each with the test that checks it; exits 1 on an untested required item; no GPU), `selftest`, `validate` (overnight queue, never publishes; lives in `queue/validate.py`), `pin` (symlinked copy + smoke + `vqlab_pin.json`; scorers refuse an unsmoked pin), `mtp-smoke-head`, `spelling`. Not CLI: `kernel_accept.py` (device-codebook d4 kernels vs threadgroup + a float32 reference, repo runtime; was `e134_accept`) and `verify_dev_codebook.py` (the same agreement through `_fused` on real artifact layers; was `verify_e134_devcb`), frozen acceptance checks for the E134 runtime change. `kernel_reference.py` holds the shared `reference()` dequant-matmul that kernel_accept and bundle-accept score against.

## Outputs
Exit codes and printed verdicts.

## Rules that bite
- Every gate must FAIL a known-bad input and PASS a known-good one before its pass means anything (III.5).
- A gate that never ran the code reports on the gate, not the artifact. Confirm the path was traversed.
- Measuring? Pin (symlink the shards + ONE model.py), then smoke, then measure.
- The Knurlogic client helpers the check-release / vision-smoke arms use are `core/box/knurlogic_client.py`; gate never imports bench. `pin.py` writes the record whose format and check are `core/artifacts/pin_record.py`.

## Files
- `bundle_accept.py`: `gate bundle-accept`: kernel acceptance against the bundled copy a downloader runs
- `check_all.py`: `gate check`: every release gate that needs no source model
- `check_bundle.py`: `gate check-bundle`: the bundled model.py carries the current repo runtime
- `check_comparator.py`: `gate check-comparator`: a comparator holds the same core tensors as the teacher
- `check_release.py`: `gate check-release`: every file a user needs exists and functions
- `tokenizer_match.py`: check-release's tokenizer gate: pre_tokenizer/normalizer/decoder and a multilingual sample must match the base model's tokenizer.json (card `base_model`, or `--tokenizer-ref`)
- `leak_guard.py`: the publish and check-release refusal for text naming this machine (home dir, storage roots, checkout, LAN IPs); backups never ship
- `kernel_accept.py`: frozen E134 acceptance: device-codebook d4 kernels vs threadgroup + float32 reference
- `kernel_reference.py`: the exact dequant-matmul `reference()` the kernel checks score against
- `mtp_smoke_head.py`: `gate mtp-smoke-head`: a packed draft sidecar loads and forwards; draft step cost
- `parity.py`: `gate parity`: a family's checklist of reference behaviours, tested or not
- `knurlogic_link.py`: `gate knurlogic-link`: one symlink in the Knurlogic models dir so a scratch build loads by name (check-release --knurlogic-smoke); never copies, never removes a real folder
- `pin.py`: `gate pin`: freeze an artifact for measurement, then smoke the frozen copy
- `smoke.py`: `gate smoke`: generate one token through the runtime the artifact ships
- `spelling.py`: `gate spelling`: fail a release on British spellings; --fix rewrites them
- `verify_artifact.py`: `gate verify`: verify a VQ artifact against its bf16 source (outlier gate)
- `verify_dev_codebook.py`: frozen E134 check: device vs threadgroup codebook through `_fused` on real layers
- `vision_smoke.py`: `gate vision-smoke`: put an image through the shipping runtime
