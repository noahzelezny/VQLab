# core/artifacts/ — an artifact on disk

What an artifact directory holds and how it is read: its layout, its size
check, the MoE code packing, the pin record every scorer reads, the numerics
stamp fields, and comparisons against its Hugging Face repo.

## Rules that bite
- Part of the core layer (L1): imports nothing from `vqlab.family` or above.
- Machine-specific values (storage roots) come from `vqlab.config`, never from code.
- Old names `vqlab.core.<module>` (and `vqlab.assemble.codepack` for `codepack`)
  resolve here for one release (`_layout.DEPRECATED`).

## Files
- `artifact.py`: the one place that knows how an artifact directory is laid out
- `artifact_size.py`: an artifact's index total_size checked against its shards' tensor bytes (the library half of `vqlab size`; the command is `ship/size.py`)
- `codepack.py`: the one packing decision for MoE expert codes, shared by `vqlab pack` and `fit-moe --pack`
- `pin_record.py`: the pin record (`vqlab_pin.json`) and the check every scorer runs on it
- `stamp_fields.py`: what a numerics build stamp holds, and how two are compared and printed
- `hub.py`: Hugging Face Hub comparisons: what a local artifact holds vs its repo
