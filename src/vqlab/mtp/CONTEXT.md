# mtp/ — MTP speculative decoding (a library, not a stage)

## Inputs
A loaded trunk (mlx-lm) plus a drafting sidecar (`mtp-head-*.safetensors`).

## Process
`vqlab.mtp` is imported, not run: `loop.py` (draft/verify loop,
`mtp_generate`, `mtp_stream_generate`), `registry.py` (one `FamilySpec` per
model family), `cli_common.py` (`add_sampling_args`, `load_all`, `prompt_ids`: shared by mtp-generate and mtp-bench), `caches.py` / `capture.py` / `runtime.py` / `sampling.py`, and
the per-family heads `mtp_head*.py`.

## Outputs
Tokens. The MTP TOOLS live in the stage folders like every other tool:
`plan/mtp_probe_qwen4_exp.py` / `plan/mtp_probe_qwen35.py` (does the teacher's head draft well?),
`build/mtp_extract.py`, `build/mtp_pack.py`, `build/mtp_graft.py` (build the sidecar),
`gate/mtp_smoke_head.py`, `bench/mtp_bench.py`, `bench/mtp_accept.py`, and
`ship/mtp_run.py` (`vqlab mtp-generate`).

## Rules that bite
- `load_mtp_head` / `mtp_generate` never install Knurlogic's architectures. For deepseek_v4 / qwen4_exp, call `vqlab.use_knurlogic_arch()` before `mlx_lm.load` (the CLI does it for every command).
- A new family is a `FamilySpec` in `registry.py` plus a head module. Read that docstring first.
- `mtp-accept` (paired, across prompts) is the reliable acceptance instrument; a single-prompt number is not.

## Files
- `bench.py`: measure what MTP drafting is worth on a given model
- `caches.py`: snapshot and rollback of the KV caches for one speculative step
- `capture.py`: per-family capture of the pre-lm_head activation
- `cli_common.py`: CLI plumbing shared by mtp-generate and mtp-bench
- `graft.py`: graft key mapping, the slot check and the sidecar read shared by the heads and the probes
- `loop.py`: the draft/verify loop: `mtp_generate`, `mtp_stream_generate`
- `mtp_head.py`: the qwen4_exp MTP head: build, quantize, save, load
- `mtp_head_glm5.py`: the glm5_next MTP head
- `mtp_head_qwen35.py`: the qwen3_5 / qwen3_5_moe MTP head
- `registry.py`: one `FamilySpec` per model family
- `runtime.py`: trunk loading for the MTP tools, across both runtimes
- `sampling.py`: sampling and exact rejection sampling with residual correction
