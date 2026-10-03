# Changelog

## 0.2.0 (unreleased)

First PyPI release. 0.1.0 (2026-08-26) was the paper snapshot, git only.

### Installing and setup
- `pip install vqlab` works without a checkout: CI runs the full `vqlab
  selftest` from the installed wheel, outside the repo.
- `vqlab config` shows every storage location and where it resolved from;
  `vqlab config init` writes a starter config. Nothing assumes a disk layout.
- Dependencies pinned to the tested ranges (mlx 0.32.x, mlx-lm 0.31-0.32).

### New families and teachers
- DeepSeek-V4-Flash: FP4 experts read natively (mxfp4), a layer-streamed
  scorer validated bitwise against a resident forward, and
  `vqlab teacher-prep` (official release -> exact MLX teacher in one
  command; refuses the HF cache, checks disk, lists every tensor it drops).
- GLM-5.3-Flash and Qwen3.8-Flash-Next profiles and scorers.

### Building artifacts
- `vqlab mix` (per-layer-band sources -> one artifact), `vqlab loo-bands`
  (leave-one-out damage map + its KL queue), `vqlab minibase` (fit one band
  without rewriting the whole base), `vqlab geo-build` (refit named modules,
  keep every other shipped byte), `vqlab reskeleton`.
- vq-skipzero, a shipped format: fully-dead VQ rows dropped on disk and
  served natively by the runtime (397B: -10.8% disk, -11.1% resident, KL
  unchanged). Tools in `vqlab.skipzero`.
- `vqlab.core.artifact`: one shared description of an artifact's layout;
  `mix` and `minibase` run on it, the other writers follow.
- Fixed: `pack` and the bundled loader crashed on a module whose tensors
  straddle a shard boundary; `fit-moe` likewise.
- Disk preflight on every large writer.

### Measuring
- `vqlab kl-ladder` is the release gate (paired, three corpora, |t|>2);
  `vqlab kl-pair` pairs already-scored rungs at zero GPU cost.
- `vqlab decode-timeline`, `prefill-timeline`, `decode-ladder`,
  `active-bytes`, `kernel-truth`, `speed-pair`, `prefill-bench`.

### Runtime (ships inside each artifact)
- Faster decode and prefill kernels across d2/d4/d8 (walk, device-codebook,
  fused gemmseg2), multi-token-prediction drafting, tensor-parallel split
  including skip-zero modules.

### Running long jobs
- `vqlab queue run` (pinned code, GPU lease, preflight, retries),
  `queue wait` (blocks on the state file, never log wording),
  `queue run --on <box>` for a second Mac, and the MCP server for agents
  (`queue_status`, `disk_free`, findings log tools).

### Tests
- Writer contracts: every artifact writer pinned to golden outputs on a
  multi-shard fixture (tests/test_writer_contracts.py).
