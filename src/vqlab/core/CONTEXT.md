# core/ — Shared libraries used by every stage

## Inputs
Model-family names, source checkpoints.

## Process
`families.py` (the single family registry), `expert_src.py` (every reader of MoE source weights), `runtime_load.py` (which library loads a family), `mem_budget.py`, `ple_stream.py`, `fitstore.py` (the fit store's layout, index, census and content-based teacher naming; `vqlab fits` is its CLI).

## Outputs
Imported by bare name from any stage.

## Rules that bite
- A new family is ONE entry in `families.py`. Don't fork a copy into a tool.
- Depth and geometry laws are family-local. Never inherit an allocation across families.
