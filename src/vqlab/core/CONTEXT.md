# core/ — Shared libraries used by every stage

## Inputs
Model-family names, source checkpoints.

## Process
`expert_src.py` (every reader of MoE source weights), `runtime_load.py` (which library loads a family), `arch_finder.py` (the Knurlogic meta-path finder behind `vqlab.use_knurlogic_arch()`; `install_arch_finder()`), `mem_budget.py`, `ple_stream.py`, `fitstore.py` (the fit store's layout, index, census and content-based teacher naming; `vqlab fits` is its CLI), `version_stamp.py` (where versions are stamped: the numerics build a number was computed under -- mlx, mlx-lm, knurlogic, arch file sha, scorer variant -- and the checks that refuse a mismatched cache or pairing; `python -m vqlab.core.version_stamp --model <dir>`).

The family registry (FAMILY, teacher_caches, scorer_variant) is NOT here: it is `family/registry.py`.

## Outputs
Imported as `vqlab.core.<module>`.

## Rules that bite
- core imports NOTHING from `vqlab.family` (layer rule, tests/test_layering.py). A core module that needs a family lookup asks the package root at call time: `vqlab._family_registry()`.
- Old names `vqlab.core.families`, `vqlab.core.numerics` (and bare `families` / `numerics`) and `vqlab.family.arch` are aliases for one release (`_layout.ALIASES`): same module object.
- The version stamp's fields are compared across caches and runs. Renaming or moving code must not change a stamped value (tests/test_version_stamp_identity.py).
- Depth and geometry laws are family-local. Never inherit an allocation across families.
