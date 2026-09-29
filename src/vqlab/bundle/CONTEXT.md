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
- `bundle` on an `sz-pack` output serves it RESIDENT through the runtime switch: carries vq_skipzero modules' geometry over, sets `vq_skipzero.loader = "runtime"`, drops the experimental hooks. `vqlab pin <sz-pack> --out <new> --runtime <profile>` is the one-step build. It refuses to write through a symlinked model.py/config.json.
