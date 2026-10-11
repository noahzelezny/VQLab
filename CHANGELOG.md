# Changelog

## Unreleased

### Removed
- Bare imports of tool modules (`import pack`, `import settings`,
  `import registry`, ...) no longer resolve; use `vqlab.<pkg>.<mod>`.
  `import vqlab` answered for ~150 bare names, so an app's own `settings`
  or `registry` module got vqlab's. Only the runtime stems (`vq_switch`,
  `vq_pack`, `vq_dense`, `dense_shim`, `glm5_shim`, `arch_resolve`,
  `runtime_profile`, `gated_delta_chunked`) stay bare. Old FILE names in
  queue files (`vq_397b_codes.py`, `pack_artifact.py`, ...) still map
  through `_layout.find()`. Deprecated `vqlab.*` spellings now warn at the
  import line, and `importlib.util.find_spec` probes no longer warn.

### Renamed (untangle step 15; old names import for one release)
- `plan/probe_init_sweep.py` is `plan/init_sweep.py`, `plan/mtp_probe.py` is
  `plan/mtp_probe_qwen4_exp.py`, `plan/mtp_probe35.py` is
  `plan/mtp_probe_qwen35.py`; `fit/mtp_head_ds4.py` moved to
  `build/mtp_head_ds4.py`. Command names (`probe-init`, `mtp-probe`,
  `mtp-probe35`, `mtp-head-ds4`) are unchanged.
- **`vqlab.fit.fit_additive` changed meaning.** It was the additive-VQ
  library (`fit_additive()`, k-means helpers); that library is now
  `vqlab.fit.additive_codebooks`. The name `fit_additive` now belongs to the
  `fit-additive` command module (formerly `vqlab.fit.additive_vq`, which
  still aliases it). Code that did `from vqlab.fit.fit_additive import
  fit_additive` must import from `vqlab.fit.additive_codebooks`.
- Generic `install()` functions say what they install:
  `_layout.install_import_aliases`, `ple_stream.install_ple_stream`,
  `sz_resident.install_resident`, `serve.install_serving` (`install` kept as
  an alias). `runtime/glm5_shim.install` is shipped text and keeps its name.
  `sz_resident_build.plan` is `plan_resident` (`plan` kept as an alias).

## 0.1.0 (2026-10-10)

First release. VQLab fits size-targeted, data-free vector-quantization
codebooks for large language models on Apple Silicon (MLX), assembles and
bundles the quantized artifacts with their Metal runtime, and scores them
against their bf16 teachers (paired KL on three corpora, perplexity, task
benchmarks) before release. One `vqlab` CLI covers plan, fit, build, bundle,
gate, score, bench and ship; `vqlab mcp` exposes the lab to agents.

(A `v0.1.0` git tag from 2026-08-26 marked the paper
snapshot; it was never a release, and moves to this commit when this ships.)

### Since the first draft of these notes (2026-10-03)
- **CLI namespaces**: `vqlab plan | fit | build | bundle | gate | score |
  bench | ship | lab <command>`; every flat name (`vqlab fit-moe`) still
  works. `fit/vq_397b_codes.py` is now `fit_moe.py`, `assemble/pack_artifact.py`
  is `pack.py` (old names import).
- **Own environment and the architectures it needs**: `vqlab doctor`
  (interpreter, mlx builds, arch shas, storage, token). VQLab depends on
  [Knurlogic](https://github.com/noahzelezny/Knurlogic) for the
  architecture files stock mlx-lm 0.32 lacks or differs on (deepseek_v4,
  qwen4_exp, qwen3_5/_moe, gemma4, glm5_next): `import vqlab` serves them as
  `mlx_lm.models.<name>` through Knurlogic's own `register()`, so scoring
  loads exactly the arithmetic Knurlogic serves, from one copy.
  `VQLAB_VENDORED_ARCH=0` scores stock mlx-lm.
- **Numerics stamps**: every teacher cache, score and per-position array
  records mlx, mlx-lm, the arch file's sha and the scorer variant; scoring
  refuses a cache from another build (`--allow-build-mismatch` records the
  override), `kl-pair` verifies a pairing from sidecars, `card-tables`
  prints "Measured with" and flags MIXED builds. `vqlab runtime-equiv`
  runs one slice through two interpreters. DeepSeek's reference
  shared-expert clamp is the default scorer variant.
- **Family parity**: plugins declare their reference inference code and a
  checklist of its behaviours; `vqlab parity` runs each item's test (a
  skipped test counts as untested). `vqlab act-stats` counts how often each
  clamp fires.
- **Writers refuse a no-op** and exit 1 (pack, pack-dense, pack-ple,
  splice-ple, ple-swap, mix, geo-build, reselect, harvest-parts,
  vision-layout --fix, size --fix-index, publish when the Hub already
  holds the bytes); `--help` and `--dry-run` never write a build record.
  One tensor classifier (`core.artifact.tensor_class`) behind every
  tower / MTP / text decision.
- **Release gates**: `check-release` fails a wrong or missing index
  `total_size` and requires an image smoke for artifacts with a tower
  (`vision-smoke --knurlogic` drives DeepSeek Vision-Exp). `vqlab card`
  writes the whole model card from config + KL JSON + build record and
  refuses numbers measured on other bytes.
- **Disk and machines**: `fit-moe --pack` (unpacked K>256 codes never hit
  disk), `vqlab scratch reclaimable` (prints rm lines, never deletes),
  `vqlab slice` (a small real teacher for preflights), `vqlab reserve`
  (who has a box until when; queues refuse a box reserved for someone
  else), queue ETAs from the step's own timings, `queue run --on` reads a
  byte of every path on the remote first.
- **Fixed**: the selftest's packer step had never packed (it ran pack-dense
  on a raw fit); mtp-pack no longer records a scratch head as a build of
  the model it read; graft-extras says to pass the official release when
  given an MLX conversion.

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
