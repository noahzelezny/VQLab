# bundle/ — Bundle: put the runtime inside the artifact

## Inputs
An assembled artifact plus `runtime/`.

## Process
`vqlab bundle` (MoE `model.py`), `rebundle-dense`, `patch-arch` (loader fix, runtime untouched), `vision-layout`.

## Outputs
`<artifact>/model.py` = the runtime text + loader shim, at a named profile (v1.5 / v2).

## Rules that bite
- A rebundle must PRESERVE the flag defaults the artifact shipped with (`runtime_profile.preserve_shipped`).
- Announce a fleet-wide rewrite first, then gate ONE artifact per family (`smoke` + `vision-smoke`) before touching the next (F154).
- The release baseline is the HF revision, not the local `Exo Models/` copy.
