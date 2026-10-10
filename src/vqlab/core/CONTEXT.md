# core/ — Shared libraries used by every stage

## Inputs
Source checkpoints and their family entries (passed in by the caller).

## Process
`artifact_size.py` (`index_total_size`: the index's metadata.total_size against the shards' tensor bytes; the `vqlab size` command is `ship/size.py`, and this half sits below gate so `check-release` reads the same number), `expert_src.py` (every reader of MoE source weights; its loader checks are `tests/test_expert_src_loader.py`), `arch_finder.py` (the Knurlogic meta-path finder behind `vqlab.use_knurlogic_arch()`; `install_arch_finder()`), `stamp_fields.py` (what a numerics build stamp holds and how two are compared and printed: `COMPARED`, `diff`, `describe`), `mem_budget.py`, `ple_stream.py` (`install_ple_stream()`), `fitstore.py` (the fit store's layout, index, census and content-based teacher naming; `vqlab fits` is its CLI), and the pieces several stages share so none imports a peer or a higher stage: `clamps.py` (`ClampCounter`, `corpus_ids` for act-stats; family/deepseek_v4's hook uses them), `gpu_lease.py` (`lease_path`, `lease_holder`; box_quiet and the MCP server), `knurlogic_client.py` (`post`, `entry`, `ready`, `drafting`, `rank_logs`, `BUNDLED`/`VENDORED`; speed-pair-knurlogic and the check-release / vision-smoke Knurlogic arms), `codepack.py` (the MoE code packing rule shared by `vqlab pack` and `fit-moe --pack`), `pin_record.py` (the `vqlab_pin.json` format and `check_pin`, read by every scorer; `vqlab pin` writes it), `hub.py` (`remote_files`, `hub_diff` for release-prep; the registry's Hub table shares its fetch).

Anything that needs the family table is NOT here, it lives in `family/`: the registry (`family/registry.py`), which library loads a family (`family/model_loader.py`), and building a version stamp (`family/version_stamp.py`).

## Outputs
Imported as `vqlab.core.<module>`.

## Rules that bite
- core imports NOTHING from `vqlab.family` (layer rule, tests/test_layering.py), directly or through an indirection. A core module that needs a family lookup is in the wrong package: move it up.
- Old names `vqlab.core.families`, `vqlab.core.numerics`, `vqlab.core.version_stamp`, `vqlab.core.runtime_load` (and bare `families` / `numerics` / `version_stamp` / `runtime_load`) `vqlab.family.arch` and `vqlab.assemble.codepack` are aliases for one release (`_layout.DEPRECATED`): same module object.
- Depth and geometry laws are family-local. Never inherit an allocation across families.

## Files
- `arch_finder.py`: the Knurlogic architecture meta-path finder behind `vqlab.use_knurlogic_arch()`
- `artifact.py`: how an artifact directory is laid out: index, shards, config (`write_index`, ...)
- `artifact_size.py`: an index's total_size checked against its shards' tensor bytes
- `clamps.py`: clamp counting for act-stats: `ClampCounter`, house-corpus ids
- `codepack.py`: the one packing decision for MoE expert codes (`vqlab pack` and `fit-moe --pack`)
- `exo.py`: the exo cluster's HTTP API as the GPU gate reads it: is an instance placed?
- `expert_src.py`: every reader of MoE source weights: the expert-stack loader, all layouts
- `fitstore.py`: the fit store: layout, index, census, content-based teacher naming
- `gpu_lease.py`: the GPU lease file every GPU tenant flocks in turn
- `hub.py`: Hugging Face Hub comparisons: what a local artifact holds vs its repo
- `knurlogic_client.py`: Knurlogic HTTP/job client helpers (bench and gate arms)
- `mem_budget.py`: per-block eager-eval budget for the streamed probes
- `pin_record.py`: the `vqlab_pin.json` format and `check_pin`, read by every scorer
- `ple_stream.py`: make qwen4_exp's PLE n-gram tables stream instead of going resident
- `stamp_fields.py`: what a numerics build stamp holds; compare and print two
