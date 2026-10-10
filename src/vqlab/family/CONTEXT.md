# family/ — The family registry and the per-family plugins

## Inputs
A family name or a checkpoint's `model_type`; the checked-in `families/<family>/entry.json` data (source checkout only) and the user's `config.families_dir()`.

## Process
`registry.py` is the single family registry: `FAMILY` (MoE/verify source layouts and projection maps), `DENSE_FAMILIES` / `DENSE_REMAP`, `teacher_caches()` (registered KL teacher caches), `scorer_variant(model_type)` (stamped by core/version_stamp.py). `__init__.py` defines `FamilyPlugin` and discovers one plugin per module (`deepseek_v4.py`, `qwen4_exp.py`, `qwen3_5.py`, `gemma4.py`, `glm5_next.py`): fit layout, streamed scorer, tokenizer hook, reference parity items.

## Outputs
`vqlab.family.registry.FAMILY` and friends, read by fit/, assemble/, plan/, gate/, score/; `vqlab.family.plugins()`.

## Rules that bite
- A new family is ONE entry: a plugin module here, or a `families/<name>/entry.json` data file. Don't fork a copy of the table into a tool. A code entry always wins over a data entry of the same name.
- family sits above core: core never imports it (core reaches the registry only through `vqlab._family_registry()`).
- `vqlab.core.families` is the pre-2026-10-10 name of registry.py, aliased for one release.
- Depth and geometry laws are family-local. Never inherit an allocation across families.
