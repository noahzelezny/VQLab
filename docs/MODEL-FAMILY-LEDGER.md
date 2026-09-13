# Model-family ledger — what exists, per family

One reference table: families as rows; rungs, geometry, gates, bench
rows, fit assets, and pending work as columns. Every geometry cell below
was read from the artifact's config on 2026-09-13 (not from memory);
bench cells name only rows that exist on disk in a known harness.
Update this file when a rung ships or a fit lands — it goes stale the
day someone doesn't.

Version vocabulary: **arc6 runtime** = the 2026-09-09 fleet republish
(all 20 Hub repos, gate-smoked). **v2 mixed-geometry** = a rung whose
expert geometry was set by the measured ladder process rather than
uniform/hand-set. Per Noah: released-as-v2 so far = GLM (no v1 existed)
and the 397B small rung.

| family | released rungs (bpw) | expert geometry (config-verified) | mixed-geo status | ppl rows on hand | task-bench rows on hand | fit assets on disk | pending |
|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | 2.7 / 3.1 / 3.6 | 2.7: K512×102+K2048×24 · 3.1: K512/K2048/K8192 three-tier · 3.6: K8192×111+K16384×15 (all d4) | **v2 mixed (ladder-derived)** | ladder-era ppl tables in quantlab/EXPERIMENTS.md (E8–E13b; affine-study artifacts — VQ-rung ppl lives on the HF cards, unaudited here) | none found on disk — HF cards unaudited | glm53_vq_fit d4 K512(+seedB)/K2048/K8192/K16384-partial, d8k16384 (SSD) | none open |
| **Qwen3.5-397B** | 2.2 / 2.4 / 2.6 / 3.1 | 2.2: d8-K16384×151 + d4-K256×20 + d4-K2048×9 · 2.4/2.6/3.1: FLAT (d4 K256/K512/K2048) | **2.2 = v2 mixed**; larger rungs flat | championship-ladder ppl in quantlab/EXPERIMENTS.md (E24–E29, 60k-char wikitext, MlxRing harness) | hellaswag/piqa/winogrande JSONs: AgenicAI/quantlab/results_tasks/ (VQ-2.2/2.4/3.1 + spicyneuron comparators) | codebooks in-artifact; HDD demote_fit-d4/d8; bf16 on SSD | flagship VQ swap still Noah's call (older thread) |
| **Qwen3.6-35B** | 3.4 / 3.8 / 4.6 / 5.4 | 3.4: flat d4-K2048 · 3.8: flat d4-K8192 · 4.6: d4-K2048×30 + d2-K512×90 · 5.4: flat d2-K1024 | 4.6 mixed (provenance unaudited); rest flat | base + reselect arms, wikitext/B/C (scratch) | hellaswag/piqa/winogrande ×2 arms (bench_venv, direct) | codebooks in-artifact ONLY; bf16 teacher on HDD | alternation+bf16 refit candidate (F81, gate-safe, minor); geometry ladder never run on this family |
| **Qwen3.8-27B** (dense) | 3.9 / 4.5 / 4.8 | vq_linear schema: 4.5 = flat d2-K256×192 (others unaudited) | flat | none on disk | none on disk | unaudited | schema differs (vq_linear/vq_embed) — audit before touching |
| **Qwen3.8-Flash-Next** | 2.1 / 3.2 / 4.4 / 5.5 | 2.1: d2-K256×6(front)+d8-K16384×138 · 3.2: d2×18+d4-K2048×126 · 4.4: d2-K1024×18+d2-K256×126 · 5.5: flat d2-K1024 | hand-set front-protection mixes; **NOT ladder-derived** | base/R0/R1/bands/promo ×3 corpora (scratch) | **one-harness table: 8bit ref + base + R1** (streamed, results_bench3) | qwen4exp_vq_fit d2k256/d2k1024/d8k16384/full + hot6/quiet + PLE fits (SSD); bf16 on HDD | **art_flash_r1 = ship candidate** (F85/F86, gate green, awaiting Noah); other rungs' knees unmeasured |
| **gemma-4-26b** | 6.2 | flat d2-K2048×90 | flat | GEMMA4_PPL_ANOMALY.md (caveats apply) | struct-sweep rows: quantlab/results_crush/ | unaudited | none open |
| **gemma-4-e4b PLE** | 1 repo | vq_ple (no expert vq_modules) | n/a | none on disk | none on disk | qwen4exp_ple_fit* family (SSD) | none open |

## Depth-law card (why mixes can't be copied across families)

| family | measured law | evidence |
|---|---|---|
| GLM-5.3 | front lobe = ballast; value in tail; attention protected everywhere | E10/E12/E13b (affine ladders) |
| 397B | tail-graded, knee ~tail30 | E24–E29 |
| Flash-Next | **true U**: front (0-1) and late (32+) expensive, trough 12-19 cheap — and only one K-step deep | F82/F83/F84/F86 (end-to-end VQ probes) |
| 35B / 27B / gemma | **unmeasured** | — |

## Codebook portability

Codebooks are per-module tensors inside each artifact — mix-and-match
within a family = copying module tensors (the diff-builder's whole
mechanism, `scratchpad/flash_geo_build.py`, GEOMAP-driven, set
pack_bits). Nothing is unrecoverable since the bf16 teachers were
archived (`/Volumes/Thunderbay HDD/Teacher Models/`, 35B + Flash-Next;
397B bf16 on the SSD): any missing geometry refits at ~1-5 min/module.

## Bench-row comparability (F87)

qwen4_exp loglikelihoods shift 0.1-0.7 nats with batch composition IN
THE UPSTREAM FORWARD. Flash rows are comparable only within one harness
+ one batching; the streamed harness (score_tasks_q4exp.py, b256) is
the canonical one and is the only way to score the 178 GB 8bit ref.
