# plan/ — Plan: price a build and decide where the bits go, BEFORE fitting

## Inputs
Byte budget, a shipped rung, and a teacher (for leverage probes).

## Process
`vqlab family-profile --teacher <dir>` FIRST for any teacher: headers only, seconds, writes `families/<family>/teachers/<teacher>/profile.json` (legal geometries, exact bytes, GiB per bit). Then:
`vqlab price`, `layer-leverage`, `alloc-sweep`, `probe-init`; hard guards `preflight-ram` / `preflight-disk`.

## Outputs
A recipe or geomap for `fit/`, and printed cost/value curves.

## Rules that bite
- Rank layers by the JUMP in `traj_rel`, not by `local_rel`. Isolation probes are anti-signal (F95).
- Escape the cheapest width broadly before enriching narrowly (FINDINGS I.3).
- Price a rung before fitting it, and stamp every size pre- or post-vision-graft.
