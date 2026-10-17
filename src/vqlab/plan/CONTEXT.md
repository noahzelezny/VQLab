# plan/ — Plan: price a build and decide where the bits go, BEFORE fitting

## Inputs
Byte budget, a shipped rung, and a teacher (for leverage probes).

## Process
New teacher? `vqlab onboard --teacher <dir>` sequences the whole characterisation pass and resumes (its code is `agents/onboard.py`: it launches GPU steps through the MCP server, so it lives in agents). Otherwise:
`vqlab family-profile --teacher <dir>` FIRST for any teacher: headers only, seconds, writes `families/<family>/teachers/<teacher>/profile.json` (legal geometries, exact bytes, GiB per bit). Then:
`vqlab price`, `recipe compile` (recipe -> queue file), `layer-leverage`, `alloc-sweep`, `probe-init` (`init_sweep.py`); MTP draft probes `mtp-probe` (`mtp_probe_qwen4_exp.py`) and `mtp-probe35` (`mtp_probe_qwen35.py`); hard guards `preflight-ram` / `preflight-disk`.

- `vqlab zero-groups <artifact|--fits>` prices a skip-zero-groups format: counts vq_scales==0 and |s|<6.2e-5 groups and the code GiB they hold (397B rungs ~10% of text bytes, concentrated in L0-L4).

## Outputs
A recipe or geomap for `fit/`, and printed cost/value curves.

## Rules that bite
- Rank layers by the JUMP in `traj_rel`, not by `local_rel`. Isolation probes are anti-signal (F95).
- Escape the cheapest width broadly before enriching narrowly (FINDINGS I.3).
- Price a rung before fitting it, and stamp every size pre- or post-vision-graft.

## Files
- `alloc_sweep.py`: `plan alloc-sweep`: the per-layer allocation frontier for a rung
- `family_profile.py`: `plan family-profile`: size up a teacher from its headers, before any fit
- `init_sweep.py`: `plan probe-init`: on what fraction of tensors does ++ seeding sell the tail?
- `layer_leverage.py`: `plan layer-leverage`: which layers' quantization damage matters most
- `mtp_probe_qwen35.py`: `plan mtp-probe35`: MTP acceptance probe for qwen3_5 / qwen3_5_moe
- `mtp_probe_qwen4_exp.py`: `plan mtp-probe`: MTP acceptance probe for qwen4_exp
- `preflight_disk.py`: `plan preflight-disk`: refuse a build whose output volume lacks the space
- `preflight_ram.py`: `plan preflight-ram`: refuse a resident load of a model bigger than RAM
- `price.py`: `plan price`: size-target a VQ build before fitting it
- `recipe_queue.py`: `plan recipe compile`: a build recipe -> a `vqlab queue` file, qwen3_5 MoE, qwen3_8_dense and affine chains from steps that have each run, proven by their preflight, with the MTP head packed as the published sidecars were (wiring from `families/<f>/teachers/<t>/mtp.json`, else probed); the GUI's Recipe writes through it
- `zero_groups.py`: `plan zero-groups`: code bytes that encode nothing, sized from scales alone
