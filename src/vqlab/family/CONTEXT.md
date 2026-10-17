# family/ — The family registry, family-aware loading and stamping, and the per-family plugins

## Inputs
A family name or a checkpoint's `model_type`; the checked-in `families/<family>/entry.json` data (source checkout only) and the user's `config.families_dir()`.

## Process
- `registry.py` is the single family registry: `FAMILY` (MoE/verify source layouts and projection maps), `DENSE_FAMILIES` / `DENSE_REMAP`, `teacher_caches()` (registered KL teacher caches), `scorer_variant(model_type)`.
- `model_loader.py`: which library (mlx-lm or mlx-vlm) loads a family, and `load_for_family()`, the one place models are loaded by family. checks: `pytest tests/test_model_loader.py`.
- `version_stamp.py`: where versions are stamped. `build()` records the numerics build a number was computed under (mlx, mlx-lm, knurlogic, arch file sha, scorer variant); `check_cache` / `pairing_problems` refuse a mismatch. `python -m vqlab.family.version_stamp --model <dir>`. The stamp's pure field logic (`COMPARED`, `diff`, `describe`) is `core/artifacts/stamp_fields.py`, re-exported here.
- `arch_equivalence.json`: arch files whose bytes differ but whose output was measured bit-identical on a real model. `check_cache` / `pairing_problems` accept a stamp pair whose ONLY difference is an arch_sha256 pair listed there for that model_type (status `arch-equivalent`, a printed note); any other difference still refuses. An entry needs a bit-exact comparison on a real artifact.
- `__init__.py` defines `FamilyPlugin` and discovers one plugin per module (`deepseek_v4.py`, `qwen4_exp.py`, `qwen3_5.py`, `gemma4.py`, `glm5_next.py`): fit layout, streamed scorer, tokenizer hook, reference parity items, `fused` (projections the runtime fuses; read by stream-convert's skeleton aliasing) and `keep_fp` (module suffixes kept full precision, the router by default; read through `keep_fp(name)` by stream-convert's `body_spec` and the qwen3_5 and qwen4_exp MTP heads).

## Outputs
`vqlab.family.registry.FAMILY` and friends, read by fit/, build/, plan/, gate/, score/, bench/; `vqlab.family.plugins()`.

## Rules that bite
- A new family is ONE entry: a plugin module here, or a `families/<name>/entry.json` data file. Don't fork a copy of the table into a tool. A code entry always wins over a data entry of the same name.
- family sits above core and records: neither imports it. A record or card that only READS a stamp uses `core.artifacts.stamp_fields`.
- Renaming or moving code must not change a stamped value (tests/test_numerics_build.py::test_build_blob_names_no_module).
- Depth and geometry laws are family-local. Never inherit an allocation across families.

## Files
- `registry.py`: the model-family registry: the single source of truth for every family
- `model_loader.py`: which library (mlx_lm / mlx_vlm) loads a family's models
- `version_stamp.py`: the numerics build a number was computed under, and the check that two match
- `deepseek_v4.py`: DeepSeek-V4-Flash as one plugin (fit layout, streamed scorer, tokenizer hook)
- `gemma4.py`: gemma4: the reference behaviours VQLab checks
- `glm5_next.py`: glm5_next (GLM-5.3-Flash): the reference behaviours VQLab checks
- `qwen3_5.py`: qwen3_5 / qwen3_5_moe: the reference behaviours VQLab checks
- `qwen4_exp.py`: qwen4_exp (Flash-Next): the reference behaviours VQLab checks
