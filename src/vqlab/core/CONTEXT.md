# core/ — Shared libraries used by every stage

## Inputs
Source checkpoints and their family entries (passed in by the caller).

## Process
Three concept groups, each a sub-package with its own CONTEXT.md:

- `artifacts/` — an artifact on disk: its layout (`artifacts/artifact.py`), size check (`artifacts/artifact_size.py`; the `vqlab size` command is `ship/size.py`, and this half sits below gate so `check-release` reads the same number), MoE code packing (`artifacts/codepack.py`, shared by `vqlab pack` and `fit-moe --pack`), the pin record every scorer reads (`artifacts/pin_record.py`), the numerics stamp fields (`artifacts/stamp_fields.py`) and Hub comparisons (`artifacts/hub.py`).
- `source/` — reading teachers and fits: MoE source weights (`source/expert_src.py`), the fit store (`source/fit_store.py`; `vqlab fits` is its CLI), the streamed-probe eval budget (`source/mem_budget.py`), qwen4_exp's streamed PLE (`source/ple_stream.py`) and act-stats clamp counting (`source/clamps.py`).
- `box/` — this machine and its services: the GPU lease (`box/gpu_lease.py`), the exo API (`box/exo.py`), the Knurlogic client (`box/knurlogic_client.py`) and the Knurlogic arch finder behind `vqlab.use_knurlogic_arch()` (`box/arch_finder.py`).

These are the pieces several stages share, so none imports a peer or a higher stage.

Anything that needs the family table is NOT here, it lives in `family/`: the registry (`family/registry.py`), which library loads a family (`family/model_loader.py`), and building a version stamp (`family/version_stamp.py`).

## Outputs
Imported as `vqlab.core.<group>.<module>` (e.g. `vqlab.core.box.gpu_lease`).

## Rules that bite
- core (all three groups) is one layer, L1: it imports NOTHING from `vqlab.family` (layer rule, tests/test_layering.py), directly or through an indirection. A core module that needs a family lookup is in the wrong package: move it up. Imports between the groups are intra-core and legal.
- Depth and geometry laws are family-local. Never inherit an allocation across families.

## Files
- `artifacts/`: an artifact on disk (see `artifacts/CONTEXT.md`)
- `source/`: reading teachers and fits (see `source/CONTEXT.md`)
- `box/`: this machine and its services (see `box/CONTEXT.md`)
