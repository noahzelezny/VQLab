# core/source/ — reading teachers and fits

Everything that reads the inputs a fit or a probe starts from: MoE source
weights in every layout, the fit store, the per-block eval budget for
streamed reads, qwen4_exp's streamed PLE tables, and the house-corpus clamp
counting act-stats runs over a teacher.

## Rules that bite
- Part of the core layer (L1): imports nothing from `vqlab.family` or above.
- `expert_src.py`'s loader checks are `tests/test_expert_src_loader.py`.

## Files
- `expert_src.py`: every reader of MoE source weights: the expert-stack loader, all layouts
- `fit_store.py`: the fit store: layout, index, census, content-based teacher naming (`vqlab fits` is its CLI)
- `teacher_profiles.py`: reading teacher profiles (`read_profile`, `profiles_in`), shared by the fit store and family/registry.py
- `mem_budget.py`: per-block eager-eval budget for the streamed probes
- `ple_stream.py`: make qwen4_exp's PLE n-gram tables stream instead of going resident (`install_ple_stream()`)
- `clamps.py`: clamp counting for act-stats: `ClampCounter`, house-corpus ids (family/deepseek_v4's hook uses them)
