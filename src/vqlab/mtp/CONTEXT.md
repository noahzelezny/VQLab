# mtp/ — MTP speculative decoding (a library, not a stage)

## Inputs
A loaded trunk (mlx-lm) plus a drafting sidecar (`mtp-head-*.safetensors`).

## Process
`vqlab.mtp` is imported, not run: `loop.py` (draft/verify loop,
`mtp_generate`, `mtp_stream_generate`), `registry.py` (one `FamilySpec` per
model family), `caches.py` / `capture.py` / `runtime.py` / `sampling.py`, and
the per-family heads `mtp_head*.py`.

## Outputs
Tokens. The MTP TOOLS live in the stage folders like every other tool:
`plan/mtp_probe*.py` (does the teacher's head draft well?),
`assemble/mtp_extract.py`, `mtp_pack.py`, `mtp_graft.py` (build the sidecar),
`gate/mtp_smoke_head.py`, `bench/mtp_bench.py`, `mtp_accept.py`, and
`ship/mtp_run.py` (`vqlab mtp-generate`).

## Rules that bite
- A new family is a `FamilySpec` in `registry.py` plus a head module. Read that docstring first.
- `mtp-accept` (paired, across prompts) is the reliable acceptance instrument; a single-prompt number is not.
