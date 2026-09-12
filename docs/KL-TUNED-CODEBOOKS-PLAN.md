# KL-Tuned Codebooks + VQ-KV Cache — v2 campaign plan (2026-09-11)

One investment, two products, one paper. Status: SPEC — nothing started.
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

1. **d8 tail regression — diagnose before scaling.** Flash worst-position KL
   got worse. Hypotheses, unseparated: (a) block-diagonal Gram ignores
   cross-subvector correlation the d8 SIMD kernel couples — try a
   block-diagonal-over-2-subvectors Gram, or full-Gram re-selection on one
   layer as an upper bound; (b) affine-8bit teacher sits far from the 2.1bpw
   student — regenerate a bf16 teacher (streaming path already works,
   scratchpad/flash_teacher_stream2.py) and re-referee. Cheap, decisive.
2. **bf16 teacher for BOTH models** — current teacher is affine-8bit (also
   the fit target). Campaign-grade referee needs the true bf16 teacher;
   35B streaming teacher is quick, Flash streaming proven (~2 min).
3. **Format-matched d2 re-selection** — layers 0-1 (d2/K256, UNPACKED codes)
   were skipped. They need unpacked-uint output, not bit-packed. Small
   module count; do it so whole-model claims are honest.
4. **PPL sweep, per rung** — the release-gate number (Noah's policy). Only
   after 1–3.
5. **Then**: iterate re-selection (alternate re-select ↔ light table tune),
   and the VQ-KV sidecar (Part 2, untouched).

## REUSABLE ASSETS (scratchpad/, all working)
- overnight_reselect.py — whole-model re-select + KL referee (MODEL_ART,
  ON_PREFIX, AFF env-driven; standalone stage R is flash_reselect_offline.py,
  per-module checkpointed, memory-capped)
- flash_teacher_stream2.py — arch-aware layer-streaming teacher (qwen4_exp)
- flash_referee.py — swap codes + KL, skips format-mismatched modules
- bench_venv/ — pinned lm_eval 0.4.12 + mlx-lm 0.31.3 for card-protocol tasks
- assemble_reselect.py — writes a re-selected artifact to disk
