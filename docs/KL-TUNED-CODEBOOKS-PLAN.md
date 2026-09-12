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
