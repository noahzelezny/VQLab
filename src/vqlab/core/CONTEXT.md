# core/ — Shared libraries used by every stage

## Inputs
Source checkpoints and their family entries (passed in by the caller).

## Process
`expert_src.py` (every reader of MoE source weights; its loader checks are `tests/test_expert_src_loader.py`), `arch_finder.py` (the Knurlogic meta-path finder behind `vqlab.use_knurlogic_arch()`; `install_arch_finder()`), `stamp_fields.py` (what a numerics build stamp holds and how two are compared and printed: `COMPARED`, `diff`, `describe`), `mem_budget.py`, `ple_stream.py`, `fitstore.py` (the fit store's layout, index, census and content-based teacher naming; `vqlab fits` is its CLI).

Anything that needs the family table is NOT here, it lives in `family/`: the registry (`family/registry.py`), which library loads a family (`family/model_loader.py`), and building a version stamp (`family/version_stamp.py`).

## Outputs
Imported as `vqlab.core.<module>`.

## Rules that bite
- core imports NOTHING from `vqlab.family` (layer rule, tests/test_layering.py), directly or through an indirection. A core module that needs a family lookup is in the wrong package: move it up.
- Old names `vqlab.core.families`, `vqlab.core.numerics`, `vqlab.core.version_stamp`, `vqlab.core.runtime_load` (and bare `families` / `numerics` / `version_stamp` / `runtime_load`) and `vqlab.family.arch` are aliases for one release (`_layout.ALIASES`): same module object.
- Depth and geometry laws are family-local. Never inherit an allocation across families.
