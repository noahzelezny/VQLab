# plan/ — Plan: price a build and decide where the bits go, BEFORE fitting

## Inputs
Byte budget, a shipped rung, and a teacher (for leverage probes).

## Process
New teacher? `vqlab onboard --teacher <dir>` sequences the whole characterisation pass and resumes. Otherwise:
`vqlab family-profile --teacher <dir>` FIRST for any teacher: headers only, seconds, writes `families/<family>/teachers/<teacher>/profile.json` (legal geometries, exact bytes, GiB per bit). Then:
`vqlab price`, `layer-leverage`, `alloc-sweep`, `probe-init`; hard guards `preflight-ram` / `preflight-disk`.

- `vqlab zero-groups <artifact|--fits>` prices a skip-zero-groups format: counts vq_scales==0 and |s|<6.2e-5 groups and the code GiB they hold (397B rungs ~10% of text bytes, concentrated in L0-L4).

## Outputs
A recipe or geomap for `fit/`, and printed cost/value curves.

## Rules that bite
- Rank layers by the JUMP in `traj_rel`, not by `local_rel`. Isolation probes are anti-signal (F95).
- Escape the cheapest width broadly before enriching narrowly (FINDINGS I.3).
- Price a rung before fitting it, and stamp every size pre- or post-vision-graft.
