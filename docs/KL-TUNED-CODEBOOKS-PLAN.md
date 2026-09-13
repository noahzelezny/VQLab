# KL-Tuned Codebooks + VQ-KV Cache — v2 campaign plan (2026-09-11)

One investment, two products, one paper.
STATUS 2026-09-13: Part 1 code re-selection CLOSED — VERDICT NO-SHIP (F78:
ppl damage invariant to teacher and calibration; the objective itself).
Phase-1 KL table tuning (through-the-model loss) is the one unexplored
road. Part 2 (VQ-KV) untouched. Paper gains a strong measured negative.
Everything below rides the existing v2-release train and its PPL/KL
referee; no new gates invented.

## Thesis

Every shipped VQ artifact lives under the *MSE-VQ* cap: k-means/Lloyd
optimizes weight-space MSE, which (a) is a proxy known to invert against
real quality (findings archive, F65/F66 caveat discipline), and (b) treats
all error directions equally while the model does not. Re-optimizing ONLY
the codebook tables (~2 MB/model) against teacher KL moves the fixed point.
Same bits, same codes, same kernels, same runtime — better tables in the
same slots.

## Part 1 — KL-tuned weight codebooks

- **Trainables**: codebook entries (optionally group scales). Codes FROZEN
  in phase 1. Model frozen. ~2-4 MB of parameters.
- **Loss**: KL(teacher || student) over a calibration corpus; teacher =
  the bf16/affine-8bit reference (existing teacher-cache machinery — note
  the standing task: caches need regeneration on the current schema).
- **Optimizer**: plain Adam on tables; forward uses the existing runtime
  (codebook is an mx array — differentiable through the gather).
- **Phase 2 (optional, AQLM-style)**: greedy/beam re-selection of codes
  under the SAME KL-informed objective, alternate with table tuning.
  Gate phase 2 on phase 1 showing a real win first.
- **Acceptance**: per-rung ppl/KL referee (the v2 release gate, unchanged);
  bit-exact runtime (tables are data, not code). Ship as v2 artifacts.
- **Compute**: M4 128GB, LoRA-scale. Days of wall clock across the lineup,
  not weeks.
- **Risk**: overfitting the calibration corpus — hold-out referee corpus
  disjoint from tuning corpus, standard.

## Part 2 — VQ-KV cache sidecar

- **Shape**: calibration-fit KV codebooks shipped as a per-repo sidecar
  (like the MTP heads); runtime assignment kernel encodes V (and maybe K)
  at decode time.
- **The make-or-break number, measure FIRST**: online assignment cost —
  K dot products per subvector per token per layer. Probe on 35B before
  designing anything.
- **Starting split**: VQ on V (smooth), scalar on K (outlier-prone) —
  matches the scalar-KV literature's asymmetry findings.
- **Quality gate**: attention-output error + end-to-end ppl at long
  context; compare against KIVI-style 4/2-bit scalar baselines.
- **Audience**: 64 GB M5/M6 agent swarms — KV residency IS the agent count.
- **Synergy**: Part 1's tuning harness IS Part 2's calibration machinery.

## Paper skeleton (working title: "Better Tables in the Same Slots")

1. MSE-VQ cap: proxy-inversion evidence + F65/F66 homogeneity (why
   allocation and specialization DON'T help — measured negatives are the
   motivation).
2. KL-tuned tables: method, per-rung quality deltas at identical bpw.
3. Code re-selection (if phase 2 pays).
4. VQ-KV sidecar: design, assignment cost, agent-count-per-GiB curves.
5. The serving story: F64 batch curves, residency math.

## Order of operations

1. Regenerate teacher caches (current schema, multi-sample) — standing
   prerequisite, unblocks everything.
2. Part 1 phase 1 on ONE rung (35B-3.4, fastest referee loop). Decision
   point: KL delta vs referee noise.
3. KV assignment-cost probe (cheap, parallel with 2).
4. Scale winners across the lineup on the v2 train.

---

## RESULTS SO FAR (2026-09-12) — validated, with caveats

Two whole-model runs done (F71/F72 in FINDINGS-LOG):

| model | geometry | mean KL | top-1 | worst-pos KL | benchmarks |
|---|---|---|---|---|---|
| 35B-3.4 | d4-K2048 | −2.2% | +0.51 | **−15.6%** (better) | net +1.2 |
| Flash-2.1 | d8-K16384 | −1.7% | +0.22 | **+8.7%** (worse) | not run |

The mechanism transfers across scale/arch/geometry (not a d4 special), but
the d8/2.1bpw win is smaller and NOT tail-clean. Recipe is real and free;
per-rung quality referee is mandatory, not optional.

## OPEN THREADS (priority order)

1. ~~d8 tail regression~~ **RESOLVED (F73): there is no tail regression.**
   Per-position KL vectors show F72's +8.7% max-KL was ONE position; tail
   quantiles improved (P99.5 −7.8%, P99.9 −8.4%) and the decile
   decomposition matches the 35B exactly (worst-decile positions repaired,
   small tax on the bulk). The d8 blocker on scaling is lifted. Residual
   upside question: does full-Gram ICM beat block-diagonal? (icm_probe —
   the banked Grams are full IN×IN, so no recapture needed.)
2. **bf16 teacher for BOTH models** — hygiene now, not diagnosis (F73).
   Current teacher is affine-8bit (also the fit target). NO bf16 exists on
   disk for either model: 35B bf16 ≈ 70 GB HF download, Flash bf16 ≈
   320 GB — Noah's call whether either is worth it before the ppl sweep
   (which compares artifacts, not teachers, so it does not depend on this).
3. **Format-matched d2 re-selection** — layers 0-1 (d2/K256, UNPACKED codes)
   were skipped. They need unpacked-uint output, not bit-packed. Small
   module count; do it so whole-model claims are honest.
4. **PPL sweep, per rung** — the release-gate number (Noah's policy). Only
   after 1–3.
5. **Then**: iterate re-selection (alternate re-select ↔ light table tune),
   and the VQ-KV sidecar (Part 2, untouched).

## TEACHER RETENTION POLICY (Noah, 2026-09-12)

bf16 teacher weights are ARCHIVED on the HDD, not deleted after use —
supersedes the "rebuildable: redownload" delete-by-default posture in
quantlab/ARTIFACTS.md for teacher-class artifacts. Home:
`/Volumes/Thunderbay HDD/Teacher Models/`. They are read start-to-finish
once per streaming-teacher run, so HDD bandwidth is fine; they are also
the source for any future fit. In flight 2026-09-12: 35B bf16 (72 GB) and
Flash-Next bf16 (360 GB) re-downloading there (scratchpad/
teacher_downloads.log).

## REUSABLE ASSETS (scratchpad/, all working)
- overnight_reselect.py — whole-model re-select + KL referee (MODEL_ART,
  ON_PREFIX, AFF env-driven; standalone stage R is flash_reselect_offline.py,
  per-module checkpointed, memory-capped)
- flash_teacher_stream2.py — arch-aware layer-streaming teacher (qwen4_exp)
- flash_referee.py — swap codes + KL, skips format-mismatched modules
- bench_venv/ — pinned lm_eval 0.4.12 + mlx-lm 0.31.3 for card-protocol tasks
- assemble_reselect.py — writes a re-selected artifact to disk

## FLASH v2 GEOMETRY CAMPAIGN (opened 2026-09-13, Noah's direction)

Focus the v2 campaign on Flash-Next (the loved model). Basis: the shipped
2.1bpw mix is HAND-SET (6 modules d2/K256 layers 0-1 + 138 uniform
d8/K16384, confirmed from config) — never knee-measured. The proven
GLM/397B process (E10 depth law, E12/E13b shape ladders, E24-E29 tail
ladder) says expert bits belong in the TAIL and front protection is
ballast — but E13b's law is family-dependent, so Flash must be measured.

Process: iso-byte candidates as DIFFS against shipped bytes (unchanged
modules keep their exact shipped tensors; only geometry-changed modules
refit from the bf16 teacher with the F80/F81 alternation recipe). Gate
per rung: four-corpus ppl + generation smoke; benchmarks on the winner.

Rungs, in order:
  R0 ballast test — front d2->d8-K16384 (6 modules, artifact SHRINKS
     ~175 MB). Flat ppl => front protection is ballast, bytes available.
  R1 tail ladder — promote tail-N layers' experts d8-K16384 -> d4-K4096
     (+1.25 b/w), paid by early-mid K16384 -> K4096 (-0.25 b/w);
     ladder N per E25 until the knee.
  R2 combine R0+R1 winners.
