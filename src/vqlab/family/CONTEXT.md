# family/ — The family registry, family-aware loading and stamping, and the per-family plugins

## Inputs
A family name or a checkpoint's `model_type`; the checked-in `families/<family>/entry.json` data (source checkout only) and the user's `config.families_dir()`.

## Process
- `registry.py` is the single family registry: `FAMILY` (MoE/verify source layouts and projection maps), `DENSE_FAMILIES` / `DENSE_REMAP`, `teacher_caches()` (registered KL teacher caches), `scorer_variant(model_type)`.
- `model_loader.py`: which library (mlx-lm or mlx-vlm) loads a family, and `load_for_family()`, the one place models are loaded by family. `python -m vqlab.family.model_loader --selftest`.
- `version_stamp.py`: where versions are stamped. `build()` records the numerics build a number was computed under (mlx, mlx-lm, knurlogic, arch file sha, scorer variant); `check_cache` / `pairing_problems` refuse a mismatch. `python -m vqlab.family.version_stamp --model <dir>`. The stamp's pure field logic (`COMPARED`, `diff`, `describe`) is `core/stamp_fields.py`, re-exported here.
- `__init__.py` defines `FamilyPlugin` and discovers one plugin per module (`deepseek_v4.py`, `qwen4_exp.py`, `qwen3_5.py`, `gemma4.py`, `glm5_next.py`): fit layout, streamed scorer, tokenizer hook, reference parity items.

## Outputs
`vqlab.family.registry.FAMILY` and friends, read by fit/, assemble/, plan/, gate/, score/, bench/; `vqlab.family.plugins()`.

## Rules that bite
- A new family is ONE entry: a plugin module here, or a `families/<name>/entry.json` data file. Don't fork a copy of the table into a tool. A code entry always wins over a data entry of the same name.
- family sits above core and records: neither imports it. A record or card that only READS a stamp uses `core.stamp_fields`.
- Renaming or moving code must not change a stamped value (tests/test_version_stamp_identity.py).
- `vqlab.core.families`, `vqlab.core.numerics`, `vqlab.core.version_stamp` and `vqlab.core.runtime_load` are pre-2026-10-10 names, aliased for one release.
- Depth and geometry laws are family-local. Never inherit an allocation across families.
