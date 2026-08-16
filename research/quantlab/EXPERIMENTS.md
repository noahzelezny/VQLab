# Quantization experiments — Qwen3.5 family on Apple Silicon

Noah Zelezny + Claude, 2026-08-08/09. Hardware: M3 Ultra 96GB + M4 Max 128GB (Thunderbolt).
Toolchain: mlx-optiq 0.4.18 (pinned + patched, see README.md), mlx-lm/mlx.
Goal: choose the method for a vision-preserving ~2.6bpw quant of Qwen3.5-397B-A17B.

> **STATE OF RECORD (2026-08-13, supersedes everything below where they
> conflict — this is a chronological lab log and several loud mid-log claims
> were later voided by instrument fixes):**
> - **Ship artifacts: TWO, one per duty cycle (E29).** All 6-bit structure,
>   4-bit qkv/z, bf16 routers; they differ only in how deep the 2-bit expert
>   region reaches.
>   **DAILY `struct6-tail3x3`** (122.3 GiB, 3-bit last 3 layers) — PPL
>   **3.1557**, beats spicyneuron 2.6bit (3.1830, blind) at its own size;
>   sized to leave unified-memory headroom for daytime workers.
>   **OVERNIGHT `struct6-tail30`** (142.5 GiB, 3-bit last 30) — PPL
>   **2.3982**, 24% better for +20 GiB, within 1.6% of spicyneuron's 3.5bit
>   (2.3614) while 23 GiB smaller AND sighted. The `run_overnight.py` author.
>   Recipe story: E24 (spicymirror) → E25 (tail ladder) → E29 (shape + knee).
>   **The ladder is DONE — the knee is at ~tail30** (tail33 bought 0.002 PPL
>   for 2.3 GiB). More depth is not a lever; see E29 for what is.
> - **Instrument of record: `referee/score_streaming.py`** (single-box,
>   raw wikitext, 8192-token prefix). Every 2-node cluster score before E23
>   is VOID (E18c non-determinism + E23 >8k-token corruption + chat-wrapping);
>   that includes **E17's "t2.1 beats spicyneuron by 30%" — later measured
>   FALSE on the fixed instrument** (E23: spicyneuron 3.183 < t2.1 3.999).
> - **DWQ at 397B/2-bit: CLOSED, falsified** (E27 attention OOD collapse,
>   E28 experts-only OOD collapse, E20 block-wise hurts; the sane patch
>   gained nothing). E16's 35B DWQ win is real but does not transfer.
> - Voided entries are tagged ⛔ inline. Read those tags before citing.
> - **Every same-quantizer lever is measured dead (08-14):** depth knee
>   (E29), DWQ (E20/E27/E28), allocation priors (E14/E30), fused rotation
>   (E31 autopsy + E32 falsification — weights are FLAT, no outlier groups;
>   rotation Gaussianizes and hurts), group size (E33 — real at 35B, ~nil
>   past the knee at 397B; payoff ∝ remaining 2-bit loss). Tooling:
>   `rotate_fuse.py`, `probe_rotation_divergence.py`, `convert_35b_struct.py`.
>   Open axes per Noah (08-14): GPTQ-style compensated rounding (in
>   progress, 35B rung first), then vector/lattice quant incl. kernel work
>   (accepted); hot/cold expert split REJECTED ("different flavor of
>   compromise").

## Headline findings

1. **Per-layer sensitivity calibration fails on MoE, mechanistically.** The probe
   (quantize ONE layer, measure output KL) rates attention as insensitive because it
   measures layers in isolation; errors across simultaneously-degraded attention layers
   compound. Falsified end-to-end (E7): attention-only@2bit → PPL 46.4 (18.3G artifact);
   experts-only@2bit → PPL 10.4 (10.9G). The probe's "safest" cut is the most damaging.
2. **The community static recipe (attention/router high, experts low) is the right
   prior for MoE** — it is itself calibration, distilled from many end-to-end
   experiments on the true objective. Nothing we measured beat it (E5, E6).
3. **Calibrated allocation DOES win on dense** at matched budget in the steep zone
   (E4: 19.8 vs 29.1-29.7 at ~2.7bpw on 9B) — the failure is MoE-specific.
4. **MoE tolerates extreme quant gracefully; dense does not.** Dense 9B: 2.72→2.30bpw
   quadruples PPL. MoE 35B: all-experts-2bit costs ~25% PPL over tuned statics.
5. **optiq's sweep resume is deterministic** (E3): kill + overnight gap + resume
   reproduced PPL to 3 decimals. Checkpoint machinery is trustworthy for multi-day runs.

## Tool bugs found (mlx-optiq 0.4.18, upstream-worthy)

- **Expert param-count budget bug:** batched-expert sweep entries record ONE expert's
  params while the bit choice applies to all N (35B checkpoint sums ~2B params).
  Artifacts ship ~60% over budget while reporting on-target achieved_bpw.
  Fix here: `OPTIQ_EXPERT_PARAM_MULT=<num_experts>` (patches/).
- **Attention allocation inversion on MoE:** flat isolation-KL for attention lets greedy
  crush it to minimum bits — inverted from every shipped mixed quant.
  Mitigation here: `OPTIQ_ATTN_FLOOR_BITS=4` (patches/). Root fix needs grouped probes.
- **Reference-mode hygiene:** `Reference: auto` silently downgrades bf16 → uniform_4bit
  when RAM is tight (M3 96GB for the 35B). A 4bit reference scores 4bit as zero error by
  construction — usable for relative 2-vs-3 ranking, INVALID for absolute curves or for
  mixing with bf16-referenced entries. All reported sweep-driven results here used bf16
  references (9B sweeps on M3, 35B sweep on M4); the only uniform_4bit-referenced data
  (E9's fill-in + an aborted 4-point partial) fed forced allocations that ignore scores,
  and the files are renamed QUARANTINE-mixed-ref-* — never fit curves to them. v2 sweeps:
  M4, bf16 reference, explicitly — do not trust auto.
- **The bf16 reference path silently drops every routed expert** (found 08-10, mid
  E10 sweep). `_exact_with_bf16_reference` selects targets via
  `_quantizable_linears()`, which filters `isinstance(module, nn.Linear)` and
  `w.ndim == 2` — `SwitchLinear` is neither a Linear subclass nor 2-D, so all
  `mlp.switch_mlp.*` modules vanish. The other two paths do NOT: the static path's
  inline filter explicitly admits `SwitchLinear` with `ndim == 3`, and the
  uniform_4bit path's `_quantized_layers()` admits `QuantizedSwitchLinear`.
  Hence 391 targets under bf16 vs 511 under uniform_4bit on the 35B — the 120-target
  gap is exactly 40 layers × {gate,up,down}_proj of routed experts.
  **This is a reference-mode difference, not a version difference** (both 0.4.18;
  an earlier note here blamed 0.4.18-vs-0.4.19 granularity — that was wrong).
  Consequence: under bf16 the allocator never sees ~90% of the model's parameters,
  so `total_params` and `_compute_allocation_bpw` are computed over the non-expert
  remainder and the resulting achieved_bpw is meaningless — a far larger version of
  the expert param-count budget bug above. **bf16-referenced sweeps are valid for
  CURVES on attention / routers / shared experts / lm_head, and invalid for
  producing an artifact.**
- **Benchmark scale is dependency-sensitive:** identical artifacts scored PPL ~1.3-1.8 on
  fresh-dep installs vs ~8-46 on the pinned harness. ~1.x on wikitext is physically
  implausible → fresh-dep harness is broken. NEVER compare PPL across harnesses; pin the
  full dependency set, not just the tool. (Bit us twice: E5 M4 table, E7 first pass.)

## Experiment log

- **E1 (08-08):** Vision audit. spicyneuron 2.6/3.5bit 397B = text-only (mlx-lm convert
  strips vision); mlx-community 4bit (mlx-vlm convert) keeps it. Root cause of the
  "blind daily driver". [research 1934a079]
- **E2 (08-08):** exo vision pipeline debugging — 4 stacked bugs fixed (card
  capabilities, Scout probe cache, sidecar-blind loader glob, unpatched 2nd node).
  OptiQ-9B passes blank/shape/color image tests through exo. [f828e303]
- **E3 (08-08/09):** 9B PPL baseline + resume test. 8bit 8.70 / OptiQ~6bpw 8.95 /
  flat-4bit 9.24. Resume deterministic. [e47cd33e, c91de858]
- **E4 (08-09):** Matched-budget dense shootout: calibrated 2.72bpw 19.80 beats static
  2.77-2.78bpw 29.1-29.7. Static targeting unreliable at low bpw. [c91de858]
- **E5 (08-09):** 35B MoE ladder: calibrated ships 18.3G claiming 2.6bpw and still loses
  to 11.3G static. (Absolute PPLs from this M4 run later invalidated by the harness bug;
  size + ordering conclusions confirmed on the trusted harness.) [b293ff81]
- **E6 (08-09):** Hybrid (attn floor + budget fix): honest 2.61bpw allocation, but at
  13.2G / PPL 8.75 sits above the static frontier (t2.6: 12.9G / 8.30). Fixed
  calibration still loses on MoE. [9fbc2733]
- **E7 (08-09):** Falsification: forced attn-only-2bit (46.37, 18.3G) vs
  experts-only-2bit (10.37, 10.9G), trusted harness. Isolation assumption dead.
- **E8 (08-09):** Complete 35B ladder on the trusted harness (single scale, final):

  | artifact | size | PPL |
  |---|---|---|
  | static t2.0 | 10.4G | 46.22 |
  | experts-only@2bit (E7-B) | 10.9G | 10.37 |
  | static t2.2 | 11.3G | 9.14 |
  | static t2.4 | 12.1G | 8.65 |
  | static t2.6 | 12.9G | 8.30 |
  | hybrid (E6) | 13.2G | 8.75 |
  | M4 calibrated-orig (E5) | 18.3G | 11.96 |
  | attention-only@2bit (E7-A) | 18.3G | 46.37 |

  Confirms E5 quantitatively: original calibrated loses to an 11.3G static by 2.8 PPL
  while carrying 7G more bits (its partial 3-bit attention explains why it sits at 12,
  between healthy statics and the pure-2bit-attention disaster at 46).

  **THE CLIFF IS ATTENTION, NOT TOTAL BITS:** static t2.0 (46.2) lands at the same PPL
  as attention-only-2bit (46.4) — at target 2.0 the static recipe is forced to drop
  attention to minimum bits, and that alone reproduces the catastrophe. Meanwhile ALL
  experts at 2-bit costs only ~1.2 PPL over the t2.2 static. On this family the entire
  low-bpw quality game is "how few bits can everything EXCEPT attention take."

- **E9 (08-09):** Operator-designed allocation (attn=6, experts=2, other=4) via forced
  mode: 11.3G / 10.19. Beats the crude crush (10.9G / 10.37) — attention enrichment helps —
  but loses to static t2.2 at IDENTICAL size (11.3G / 9.14). Attention protection
  saturates around 4 bits (the cliff is below ~3); the remaining frontier is grading
  WITHIN the expert mass (which experts deserve 3s), i.e. per-expert differentiation —
  untested, requires per-expert signal (routing frequency or per-expert probes).
  Incidental: an accidental ~4bpw static-flavored artifact scored 7.614 at 18.6G (best
  absolute PPL of the study; extends the frontier above t2.6).

- **E10 (08-10):** Complete 511-target × 5-bit sweep, true bf16 reference, M4 (10.5h,
  die 64-70C on a USB fan, zero throttle events). First fully-honest fine-grained
  dataset: every attention projection, router, shared expert, routed-expert stack and
  lm_head, at {2,3,4,5,6} bits. Required patching the bf16 target filter (see bug above).

  **THE DEPTH FINDING — sensitivity is monotone-decreasing with depth, and the static
  recipe's U-shape is wrong.** Mean 2-bit KL by layer band:

  | layers | 0-3 | 4-7 | 8-11 | 12-15 | 16-19 | 20-23 | 24-27 | 28-31 | 32-35 | 36-39 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 2b KL | 3.58e-2 | 3.33e-2 | 2.96e-2 | 2.28e-2 | 1.45e-2 | 1.08e-2 | 7.10e-3 | 5.59e-3 | 4.57e-3 | 5.27e-3 |

  Correlation against `_structural_priority`'s U-shaped prior (protect first AND last
  block equally at 0.90): **r = +0.185** — essentially no predictive power. Correlation
  against plain depth-descending: **r = +0.951**. Layer 39 measures 3.9× cheaper than
  layer 0 while the recipe assigns both the same top tier. Layers 30-38 are
  over-protected; layers 10-20 are under-protected. There IS a small final-layer bump
  (layer 39 = 9.4e-3 vs layers 37-38 ≈ 3.4e-3), so the correct prior is
  monotone-decreasing WITH a modest tail bump — not a symmetric U.

  Routed experts follow the same law and span the widest range: layers 0-7 mean
  3.20e-2 vs layers 32-39 mean 3.30e-3, a **9.7× spread within one component family**.
  "Experts are robust cargo" is true only in the back half of the network.

  Family means at 2-bit (n, total params): lm_head 1.66e-1 (1, 0.51B) · router 2.44e-2
  (40, 0.02B) · self_attn 2.41e-2 (40, 0.27B) · linear_attn 1.97e-2 (150, 1.01B) ·
  routed experts 1.59e-2 (120, 32.2B) · shared expert 1.22e-2 (120, 0.13B) ·
  shared_expert_gate 9.23e-3 (40, 0.0001B). Per-billion-params, the tiny components
  dominate: shared_expert_gate 4509 KL/Bparam, router 46.6, shared expert 11.6,
  self_attn 3.5, routed experts 0.059. **Noah's shared-expert hypothesis: not
  supported in isolation** — shared experts rank below attention, though their
  KL-per-param is 200× the routed experts', so they are cheap to protect and worth a
  floor on that basis alone.

  Allocation produced (first honest MoE param accounting we have gotten from optiq):
  87@2 / 27@3 / 19@4 / 28@5 / 350@6, achieved 2.61 bpw, estimated 10.6 GB — but the
  artifact is **14 GB on disk, still ~32% over the estimate**. Budget accounting is
  closer than the 18.3 GB era but not yet trustworthy; do not treat achieved_bpw as
  ground truth.

  **UNVALIDATED — this is isolation KL again.** E7 is the standing proof that isolation
  ranking can invert under simultaneous degradation. Depth is arguably the case where
  isolation is most trustworthy (an early layer's error genuinely propagates through 39
  downstream blocks, and the probe captures exactly that), but the claim needs the same
  end-to-end falsification attention got: build two matched-size artifacts, one with the
  stock U-shaped depth prior and one with the measured monotone prior, and benchmark.
  Until that runs, the depth law is a strong hypothesis, not a result.

- **E11 (08-10):** Depth-law falsification — **REFUTED.** Two matched builds
  (`--method static --target-bpw 2.6 --candidate-bits 2,3,4`, same bf16 snapshot,
  trusted harness; control re-benched 8.298 vs E8's 8.30 before trusting anything):

  | arm | depth prior | size | bpw | PPL |
  |---|---|---|---|---|
  | A | stock U-shape | 13.90G | 2.601 | **8.298** |
  | B | measured monotone (`OPTIQ_DEPTH_PRIOR=monotone`, exp(-2.732x) + 3.0x tail bump) | 13.92G | 2.601 | **8.770** |

  Bit maps confirm the intervention: B moved expert 4-bit budget from layers 32-39
  (16→3 stacks) to layers 0-7 (15→24) at identical bpw — and lost by 0.47 PPL.
  End-to-end, protecting LATE layers beats protecting early layers, inverting the
  isolation-KL ranking (E10) which scored late layers 4-10x cheaper.

  **Isolation KL has now inverted under end-to-end test on two independent axes**
  (E7: attention-vs-experts; E11: early-vs-late). Proposed mechanism: probing one
  degraded early layer lets the intact downstream stack re-normalize the error, so
  isolation underestimates early-layer recoverability; under simultaneous
  degradation, late-layer noise lands raw on the logits. Treat ALL isolation curves
  as hypothesis generators, never allocation inputs.

  Cost asymmetry worth recording: the E10 sweep took 10.5h; this falsification took
  4 minutes (static builds on the M3 are ~2.5 min — it is the sweeps that are
  expensive, not the tests). No isolation-derived claim should ever ship unfalsified.
  E10 survivors: component-level facts (lm_head 20x outlier; router/shared-expert
  KL-per-param) — structural, not depth, claims.

- **E12 (08-10):** Depth-shape ladder — completes the E11 axis. Four matched builds
  (2.601 bpw, ~13.9G, same recipe/harness; `OPTIQ_DEPTH_PRIOR=flat|reverse` added
  alongside `monotone`):

  | shape | PPL |
  |---|---|
  | reverse (late-only, mirror of monotone) | **8.277** |
  | U-shape (stock) | 8.298 |
  | monotone (early-only) | 8.770 |
  | flat (no structure) | 8.772 |

  **All depth value lives in the tail.** Front-loading = flat exactly (8.770 vs
  8.772 — early protection bought nothing), while both tail-protecting shapes
  cluster at the top. Reverse edges the U by 0.021 (a tie at this resolution) with
  the early lobe deleted → the U's front protection is ballast. Bit maps verified:
  reverse gave all 24 late expert stacks 4-bit and crushed the front.

  Mechanism now evidenced, not hand-waved: early quant noise is absorbed by the
  downstream stack; late noise lands raw on the logits. Isolation KL measured
  exactly backwards along depth (anti-signal, not noise) because probing one early
  layer against an intact network is the condition where downstream laundering
  hides the damage best.

  White-paper-grade summary: "isolation sensitivity is anti-correlated with
  end-to-end importance along depth on this MoE family; the folklore U-shape
  carries dead weight in its early lobe; a simple back-loaded prior matches it" —
  four matched artifacts + control, ~20 min of falsification compute.
  For the 397B: keep the battle-tested U recipe (reverse's edge is within noise),
  floor lm_head, and cite E12 in the write-up.

- **E13 (08-10):** Steepness ladder — **degenerate at fixed budget.** k=1.35 and k=5.5
  reverse-curves produced allocations bit-identical to k=2.73 (0 of 511 layers differ);
  k=11 killed unbuilt. Mechanism: the static allocator consumes priority as a RANKING
  (greedy tier-fill to budget), so any monotone depth curve yields the same layer order,
  and the constant component offsets (+0.30 attn, +0.40 router) preserve the cross-family
  boundary. Steepness changes numbers, never ranks. Duplicate artifacts deleted.
  Incidental: same artifact benched twice scored 8.277/8.277 and 46.219/46.22 —
  harness repeatability ±0.001.

- **E13b (08-10):** Where shape CAN matter — budget ladder, reverse vs stock (E8 refs):

  | target | stock U | reverse | diff map (stock→reverse) |
  |---|---|---|---|
  | t2.6 | 8.298 | 8.277 | (E12) |
  | t2.4 | 8.65 | **8.474** | 14 tensors, ALL experts: early 4→2, late 2→4 |
  | t2.2 | **9.14** | 9.332 | 16 tensors, ALL linear_attn: early 4→2, mid 2→4 |
  | t2.0 | 46.22 | 46.219 | 0 tensors — identical floor allocation, not a comparison |

  **THE DEPTH LAW IS FAMILY-DEPENDENT.** Expert bits belong in the tail (t2.4: moving
  them wins 0.18). Attention cannot be crushed ANYWHERE — early included (t2.2: the
  reverse prior 2-bits layers 1-9's linear_attn and pays 0.19; a mild dose of the E7
  cliff). The E12 U-vs-reverse tie was two family effects cancelling through one shared
  knob. t2.0 confirms the cliff row is allocation-degenerate: at that budget every
  shape collapses to the same minimum-bits floor.

  → E14 (running): hybrid prior — reverse depth for experts only, stock U for the rest
  (`OPTIQ_DEPTH_PRIOR=reverse_experts`). Predicted to win at every budget if the
  decomposition is right.

- **E14 (08-10):** Hybrid prior — **the decomposition validated in the most literal
  way possible.** At each budget the hybrid built the BIT-IDENTICAL artifact to that
  budget's winning parent: = reverse at t2.4 (8.474) and t2.6 (8.277), = stock at
  t2.2 (9.140). The allocation boundary cuts through exactly one family per budget
  (experts at t2.4/2.6, attention at t2.2), and the hybrid applies each family's
  measured law, so it selects the winner by construction:

  | target | stock U | reverse | hybrid |
  |---|---|---|---|
  | t2.6 | 8.298 | 8.277 | **8.277** (=reverse) |
  | t2.4 | 8.65 | 8.474 | **8.474** (=reverse) |
  | t2.2 | 9.14 | 9.332 | **9.14** (=stock) |

  No third effect hiding anywhere: two family laws (expert bits belong in the tail;
  attention cannot be crushed at any depth) fully explain the ladder. Four
  bit-identical rebenches tonight (8.277 / 8.474 / 9.140 / 46.219) put harness
  repeatability at ±0.001. `reverse_experts` is strictly ≥ stock at every budget
  measured — the recommended shape going forward. Patch lives in the AgenicAI .venv
  optimizer AND must be mirrored to ~/Documents/AgenicAI/quantlab per README before any production build.

- **E15 (08-11):** First 397B production builds (t2.6 + t2.4, reverse_experts,
  streaming, M3) — and the scale-up asteroid field: **13 takes, 7 real defects, one
  environmental gremlin**, none visible at 35B scale. Every fix verified
  bit-identical to the naive math before relaunch. The ladder, in discovery order:

  1. **Giant single quantize kernel** (a 512-expert stack is 2-4B params; one
     `mx.quantize` Metal buffer exceeds the GPU watchdog) → chunk along experts.
  2. **Chunking without flushing** — optiq's own `quantize_best` chunk loop never
     `mx.eval`'d per chunk, so the lazy graph re-fused everything into one buffer
     → per-chunk eval.
  3. **Whole-model RAM accumulation** — `convert_llm_to_mlx` holds the full
     quantized model before saving; 127G output on a 96G box → `Killed: 9` at
     ~34 min → route to the per-block streaming convert (`OPTIQ_FORCE_STREAMING`).
  4. **Streaming path lacked the lowbit range search** (silent methodology fork:
     2-bit layers would ship with collapsed min/max ranges) → wrap in
     `use_search_encoder`, and make its >max_bits fallback chunk-safe too.
  5. **Unit enumeration trusted a delegating `.layers` property** — Qwen3.5-VLM's
     wrapper returns the inner text model's list, so "61 units" = the WHOLE
     language model as unit 0 PLUS the 60 blocks again → generic children-scan
     descent (verified against the live model, not just fixtures).
  6. **2-D giants unchunked** — lm_head/embed at 397B are 618M params each; the
     ndim==3-only shims let them run as single kernels → extend to ndim>=2.
  7. **THE ROOT: lazy-load ops bind to the stream active at op-creation.** A
     GPU-ambient `load(lazy=True)` makes every weight's mmap/cast op a GPU op
     that stalls on SSD page-in whenever the disk is busy (e.g. right after a
     4GB shard flush) → watchdog. This was the "layer 7 curse": deterministic
     collision of the flush schedule with the load graph, moving with page-cache
     warmth. Fix: `load()` under `mx.stream(mx.cpu)`; compute hops to GPU
     explicitly on CPU-resident chunks.

  Environmental: repeated watchdog kills + exo zombie runners (4 stuck
  RunnerLoading from failed morning placements) progressively degraded the Metal
  driver until even known-good code failed in seconds — mid-debug, the test rig
  itself was lying. Detection: re-run a previously-passing repro when failures
  accelerate; cure: restart exo (clears runners), GPU recovers without reboot.

  Result: t2.6-revexp = 146.9G language model (2.60 weights-bpw + ~0.5 affine
  scales/biases overhead = ~3.1 effective) + 0.91G bf16 vision sidecar (333 keys)
  + honest 128@2/1@3/636@4 bit map. ~25 min per arm once correct. Streaming
  build peak RAM ≈ one block. For comparison: spicyneuron 2.6bit = 121G blind.

- **E16 (08-11):** DWQ-on-MoE dry run — **possible AND helpful.** mlx_lm dwq,
  35B bf16 teacher (65G) + optiq mixed revexp-t2.4 student (12G) on the M4:
  peak 110.9G of 128G, 44 tok/s, ~90 min for 256 steps x 1024 tok. Val
  KL-to-teacher 0.108 → 0.072 (-33%). Trusted-harness PPL: **8.474 → 8.345**
  at identical 12.1G — recovers 73% of the t2.4→t2.6 quality gap for free;
  0.047 behind the 12.9G champion while 0.8G smaller. A real overnight corpus
  (10-50x) plausibly crosses 8.298 → smaller-AND-better demonstrated at 35B.
  Mechanical notes: mlx dwq accepts a pre-quantized optiq student directly
  (--quantized-model); three env fights on the M4 (anaconda MPICH aborts MLX's
  MPI probe → patch init sites to ring backend + guard all_sum, or better
  MLX_MPI_LIBNAME=<nonexistent> disables MPI probing globally; pip install
  datasets). DWQ output drops the vision sidecar + processor configs —
  re-attach from the pre-DWQ artifact before serving.
  397B path: teacher = cluster-served sighted 4bit via exo + cached top-k
  teacher logits (one serving pass, ~100s MB on disk), student trains on M4.

- **E17 (08-11): ⛔ VOIDED by E23 — every number in this entry came from a
  broken instrument.** The scoring path corrupted past ~8k tokens (kernel bug,
  E23) and was chat-wrapped (E18c); on the fixed instrument spicyneuron BEATS
  t2.1 (3.183 vs 3.999, E23), the exact opposite of this entry's headline.
  The "2.1 floor" and the internal t2.x ranking are also void (E25: expert
  tail-promotion returns keep accelerating; the ladder has no knee at 2.1).
  What survives: the t2.0/t1.8 byte-identical allocator-regime observation
  (a diff of bit maps, no scoring involved), the echo_score bug fix below,
  and the ops notes. Kept verbatim for the record: 397B championship ladder
  on the cluster — **t2.1-revexp beats the community daily driver, and the
  size floor is a cliff at 2.1 bpw.** *(← this claim is the voided one)*

  First: a real bug in the scoring path, found because every 397B score came back
  absurd (179k / 206k PPL) while the 35B smoke test was fine. `_echo_score` called
  `set_pipeline_prefill(model, is_prefill=True)`, and that flag gates OFF
  `PipelineLastLayer`'s cross-rank `all_gather` (auto_parallel.py:191) — the very
  gather the function's own comment cites as its SPMD-safety argument. Correct for
  ordinary prefill (only the last rank's logits feed decode), fatal for scoring:
  non-final ranks return a pre-send partial tensor as "logits". **Invisible at
  world_size=1** — which is exactly why the 35B smoke test passed and every
  multi-node score was garbage. Fix: don't set the flag; eat one all_gather per
  chunk. First post-fix score was sane and the harness re-ran bit-identical
  (t2.1: total_nll 32788.7168 twice, PPL 9.1056).

  The ladder (same 60k-char wikitext slice, 14,844 tokens, 2-node MlxRing):

  | model | shard | size | PPL |
  |---|---|---|---|
  | **t2.1-revexp** | Tensor | 124G | **9.106** (bit-identical rerun) |
  | spicyneuron 2.6bit (community daily driver) | Tensor | 121G | 13.026 |
  | t2.4-revexp | Tensor | 138G | 18.948 (bit-identical rerun) |
  | t2.6-revexp | Tensor | 147G | 23.980 (Pipeline: 24.065) |
  | t1.8-revexp | Tensor | 122G | **535.268** |

  **Sharding mode is score-neutral.** t2.6 scored 23.980 tensor vs 24.065 pipeline —
  0.35%, float accumulation order, nowhere near a rank change. Cross-mode comparisons
  in this table are safe. (Within a mode the harness is exact: t2.1 and t2.4 each
  re-ran bit-identical to the last decimal, so the smaller-is-better inversion up the
  ladder is real signal, not measurement noise.)

  **t2.1 wins at size parity** (+3G over spicyneuron, 30% better PPL, and sighted
  where spicyneuron is blind) — the multi-day reverse_experts thesis cashing out.

  **WHY the cliff is vertical — a regime boundary at 2.1 (probed 08-11, t2.0):**
  t2.0 was built to bisect the 2.1↔1.8 gap. It came out **byte-identical to t1.8**
  (same 765-tensor bit map; shards 1/16/32 sha256-identical) — so it was never
  scored; it *is* t1.8, PPL 535. The bit maps show what actually happens:

  | build | 2-bit | 3-bit | 4-bit |
  |---|---|---|---|
  | **t2.1** | 211 | 1 | **553** |
  | t2.0 | 625 | 140 | **0** |
  | t1.8 | 625 | 140 | **0** |

  The allocator does not degrade smoothly across budget — it **flips regimes**.
  At 2.1 it keeps 553 tensors at 4 bits; at any lower target it keeps NONE, dumping
  everything into 2/3-bit. That discontinuity is the cliff: t2.1 is the last budget
  where 4-bit survives at all. Consistent with E13's rank-based-allocator finding
  (the informative knob is budget, not curve shape) — here the budget knob turns out
  to have exactly one usable detent for this model, and t2.1 sits on it.
  Corollary: sub-2.1 targets are wasted compute (they all rebuild the same collapsed
  artifact). The lever for going smaller is `--candidate-bits` (allow 1-bit, or a
  finer ladder so 4-bit degrades gradually) or DWQ — never `--target-bpw`.

  **The floor is a cliff, not a slope.** t1.8 is 2G smaller than t2.1 and 59x worse,
  with visibly destroyed generation (`'\n "θ\n</think>\n**!\n.\n'`). Mechanism: with
  `--candidate-bits 2,3,4` the allocator has already pinned essentially every routed
  expert at 2 bits by t2.1, so a lower target buys almost no bytes — it can only
  start stripping the structures the recipe exists to protect (attention floor,
  router, lm_head, tail experts), i.e. all of E11-E14's damage for none of the
  savings. **t2.1 is the recipe's floor and its optimum simultaneously.** Going
  smaller needs a different lever (1-bit candidates, or DWQ recovery), not a lower
  `--target-bpw`.

  Ops notes: node ids change on EVERY exo restart (match `nodeIdentities`
  friendlyName, never hardcode); custom cards vanish on restart (re-POST
  /models/add, snake_case `model_id`); previews need ~30s settle after node join
  and report `memory_delta_by_node` — check it for lopsided splits before placing.
  Do NOT run a quantize build on the M3 while placing a 397B on the cluster: they
  fight for the same RAM and the placement retry-loops. Evicting an instance can
  orphan a runner subprocess holding ~75G — kill it or the next placement OOMs
  (abort trap 6). SMB load ≈ 11 min per 397B vs ~4 min to rsync it local first.

- **E18 (08-11, overnight — verdict in E20: block-wise DWQ HURTS both
  objectives; its baseline t2.1 figures are also E23-void):** Block-wise DWQ on the t2.1
  champion — built because stock `mlx_lm dwq` CANNOT run here: it holds the
  full student + fp32 trainables + Adam = 260.6 GiB (the 122.9 GiB student
  alone exceeds the M4; `--pipeline` only splits the teacher). Harness:
  `~/Documents/AgenicAI/quantlab/dwq_blockwise.py` — BRECQ-style sequential per-block distill
  (student block trained so `f(student_stream) -> teacher_output`, streams
  advance through the trained block, so drift is corrected then carried).
  Peak stays ~71 GiB flat for the 397B (one bf16 teacher block + one student
  block + streams + full Adam on that block's scales/biases; dead blocks
  dereferenced). Per-block patch files = resumable checkpoints; `merge`
  averages same-init runs (model-soup regime, drifts ~1%); `assemble`
  rewrites shards + re-attaches the vision sidecar/processor configs (E16).
  Validated on 35B end-to-end: patches NaN-free, assembly byte-faithful
  (52 patched / 1705 untouched identical), assembled model generates
  identically to baseline. (A `!!!`-output scare was the E15 wedged-Metal
  transient — rerun on a quiet box was clean. Check the rig before the code.)

  Measured facts worth keeping:
  * **Training-mode linear-attn is the bottleneck**: `GatedDeltaNet` uses
    `use_kernel=not self.training` — the differentiable path is an unfused
    recurrence, launch-bound (GPU 33-69%). Linear blocks ~573s vs full-attn
    ~80s at 512x1 on a quiet M3 — a 7x per-block gap, 45:15 mix.
  * **Batch does NOT fix it — it inverts.** M4 probe, same block/data:
    bs4 56s/69.1G, bs8 175s/92.8G, bs16 387s/154G (paging). The MoE
    backward's working set blows past residency long before launch overhead
    is amortized. bs4 is the optimum on both boxes; a "obvious" bs8 switch
    would have been a 3x slowdown. Probe before retuning.
  * **The M4 Max BEATS the M3 Ultra per-block on this launch-bound workload**
    (56s vs ~90s-equivalent) — Noah called it from GPU non-saturation alone.
  * Everything-else-closed matters: killing Scout's backend halved the M3's
    per-block time (1043s -> 573s).
  * 512 fresh samples x 1 epoch == 256 x 2 epochs in wall clock (same 128
    steps) but every step sees new data — strictly better use of a night.

  Overnight config: M3 seed 123 + M4 seed 456 (teacher on a local T7 copy),
  both 512x1x bs4 -> `397b-t2.1-dwq-patches-{m3,m4}` -> 1:1 merge ≈ 1024
  effective samples -> assemble -> referee vs 9.106.

  **RESULT (08-12 morning): apparent NEGATIVE — but ⚠️ RETRACTED AS UNPROVEN
  the same afternoon (see E18c). Every number below is a SINGLE measurement
  from the multi-node harness later shown to swing 6x on identical reruns, so
  "DWQ destroyed the model" is NOT established — it may even have helped.**
  Referee, same frozen corpus:

  | model | PPL |
  |---|---|
  | t2.1 baseline (untouched champion) | **9.106** |
  | + block-wise DWQ, merged (M3∪M4, ~1024 samples) | 37.381 |
  | + block-wise DWQ, M4 solo (coherent chain, 60/60) | 46.787 |

  Both runs were healthy by every local metric (M4: 60/60 trained, mean
  nmse -14.6%, zero reverts; M3: 54/60, -6.7%). The A/B decides the cause:
  merged BEATS solo, so the merge isn't the poison — averaging *diluted* a
  harm that each coherent chain carries in full. The harm is the method:
  1. **Per-block hidden-MSE optimizes norm-heavy directions, not
     logit-relevant ones.** nmse was already ~1e-4; squeezing it rearranges
     exactly the tail components that end-to-end KL would have protected.
     This is E11's isolation-vs-end-to-end anti-correlation showing up in
     OPTIMIZATION, not just allocation. Local metric up, model down.
  2. **The error-correcting chain bakes calibration-set statistics into the
     weights.** Each block was trained to map ITS stream back onto the
     teacher trajectory — the learned "correction" is the drift pattern of
     tulu chat data specifically, and misfires on wikitext. (E16's
     full-model DWQ improved the same eval from the same chat data — but it
     optimized KL on FINAL LOGITS, which generalizes.)

  Standing conclusions (the METHOD survives even though the verdict does not):
  **local losses are hypothesis generators, never ship criteria — referee
  before believing any patch** (this run "looked"
  perfect until the one end-to-end number); per-block distill needs a
  logit-aware objective (final-logits KL with truncated backprop, or
  teacher-input/no-correction ablation) before it's worth another night.
  The champion remains **t2.1 at 9.106, un-DWQ'd**. Failed artifacts kept
  for analysis pending Noah: `…-t2.1-DWQ`, `…-t2.1-DWQ-solo` (124G each on
  Thunderbay), patch dirs (23G x2 + merged). Queued next: the 209G 4-bit
  asymptote score (kit: `~/Documents/AgenicAI/quantlab/referee/`).

- **E18b (08-12) — RESOLVED: exo is NOT buggy. Our numbers were CHAT-WRAPPED
  perplexity, which is a different metric from the raw-text PPL a model card
  implies — and the two RANK QUANTS DIFFERENTLY.**

  Found while building the 35B analog curve, which (unlike the 397B ladder)
  contained a model whose ranking we already knew: our 2.4-bit "beat" the
  community 8-bit AND bf16 through exo — impossible. Scoring the same files
  locally with a second, independent implementation
  (`~/Documents/AgenicAI/quantlab/referee/score_local.py`, ~60 lines, same chunked-NLL math)
  reproduced exo exactly ONCE the chat template was applied:

  | 35B model | local RAW | local CHAT | via exo |
  |---|---|---|---|
  | mlx-community 8bit | **4.865** | 11.012 | 11.060 |
  | ours: RevExp-t2.4 | 5.555 | **9.265** | 9.214 |

  Local-chat matches exo within 0.5% ⇒ exo's scoring math was always right.
  The cause: `echo_score` is reachable ONLY via /v1/chat/completions, so the
  corpus is always wrapped in a user turn. That is not a ~9-token wrapper tax
  (arithmetic kills that theory outright: the NLL gap would need ~1356
  nats/token on 9 tokens, vs a ~11.9 ceiling) — it reframes ALL 14,835 corpus
  tokens as chat content and changes every prediction. **Raw and chat-wrapped
  PPL are different measurements, and the ranking genuinely inverts between
  them.** Both orderings are real and reproducible; neither is an error.

  Consequences: (1) every exo number in this file is chat-wrapped and is NOT
  comparable to anyone's published wikitext PPL — including E17's 9.106 /
  13.026. The championship comparison stays internally valid (same treatment
  both sides) but **cannot be published as "perplexity" without re-measuring
  raw.** (2) Fix landed: `utils_mlx.py apply_chat_template` now honors a
  `<|RAW_SCORE|>` prompt sentinel that bypasses the template, so cluster-held
  models too big for one box can be scored on the standard metric
  (`score_via_exo.py --raw`; patch must exist on EVERY node).

  Two false alarms worth remembering, both mine, both from skipping the
  loader: *calibration contamination* (referee corpus is wikitext-2 **test**,
  optiq calibrates on **validation** — disjoint) and *different base
  checkpoints* (the +1.0 norm gap vs raw bf16 is mlx_lm's OWN `sanitize()`
  convention for MTP-carrying checkpoints, `models/qwen3_5.py:307-330`;
  t2.1 and spicyneuron differ from bf16 IDENTICALLY ⇒ same base).
  **Diff models through the loader, never the raw safetensors bytes.**

  METHOD, the generalizable one: **an eval harness is an instrument and needs
  its own calibration.** ~10 recorded numbers and repeated bit-identical
  reruns proved only reproducibility, never validity; a cheap second
  implementation caught it in one afternoon. And **always anchor a ladder with
  a model whose ranking you already know** — the 397B ladder had no such
  anchor, which is why this hid for days.

- **E18c (08-12) — ⚠️ MULTI-NODE SCORING IS NON-DETERMINISTIC. EVERY 397B PPL
  IN THIS FILE IS UNRELIABLE, IN BOTH DIRECTIONS. NOTHING IS PUBLISHABLE.**

  Three runs of the IDENTICAL command on the IDENTICAL model (t2.1, raw,
  2-node Tensor, 14,835 tokens every time): **4.824 / 3.500 / 22.077.** Not
  numerical noise — a 6x spread. So neither E17's crown (t2.1 9.106 beating
  spicyneuron 13.026) nor its apparent raw-metric reversal (spicyneuron 2.979
  beating t2.1 4.824) is established. Both rested on single measurements from
  an unstable instrument.

  Localization so far:
  * **35B single-node raw is EXACT** — 5.5577 three times, and it matches an
    independent local mlx_lm implementation (5.5548, 0.05%). So the corpus,
    the chunked-NLL math, the `<|RAW_SCORE|>` patch and single-node execution
    are all sound.
  * The instability appears only **multi-node**.
  * Ruled out: `kv_bits` (static per-model table, not memory-derived).
  * Prime suspect, and it is the same function we already had to fix once for
    this exact path: `auto_parallel.py:191`
    `output = mx.distributed.all_gather(output, group)[-output.shape[0]:]`.
    Taking a fixed trailing slice of a cross-rank gather is right for pipeline
    decode (only the last rank's logits matter) but is not obviously right for
    a chunked scoring pass under **Tensor** sharding, where each rank holds a
    shard of the final projection. Whichever rank's response the API reads
    then decides the number — which is exactly the observed symptom.
  * Note the asymmetry to explain: E17's CHAT-wrapped reruns WERE bit-identical
    (total_nll 32788.7168 twice). Same-instance repeat scoring was launched to
    separate within-instance stability from across-placement variation; if
    within-instance is stable, the variance is in placement/rank assignment.

  UNTIL FIXED: score only single-node (≤96G models), or fix the multi-node
  gather. The 397B cannot be honestly benchmarked on this cluster right now.

  RULE, earned the hard way twice today: **a number is not a result until it
  reproduces — and reproducing within one placement is not enough; it must
  reproduce across fresh placements, on the metric you intend to claim.**

- **E20 (08-12) — BLOCK-WISE DWQ DOESN'T WORK, EITHER OBJECTIVE. Settled at 35B
  on a trustworthy instrument.** The 397B verdicts (E18) came from the broken
  multi-node harness, so the whole question was re-run at 35B, where scoring is
  exact and every number below re-ran BIT-IDENTICAL:

  | 35B artifact | raw PPL | vs baseline |
  |---|---|---|
  | RevExp-t2.4 (untouched) | **5.5548** | — |
  | + block-wise hidden-MSE DWQ | 5.6538 | **+1.8% WORSE** |
  | + block-wise logit-lens KL DWQ | 5.7652 | **+3.8% WORSE** |

  Identical setup both arms (512 samples x 512 tok x 1 epoch, bs4, seed 123,
  40/40 blocks trained, zero reverts, mean local loss down 6.6% / 5.7%).

  **This kills the E18 post-mortem's central hypothesis.** We blamed hidden-MSE
  for optimizing norm-heavy rather than logit-relevant directions and predicted
  a logit-aware objective would fix it (E19). The logit lens is *worse*. So the
  fault is not the objective — it is something structural in sequential
  per-block distillation as implemented here: most likely the error-correcting
  chain (each block trained to map ITS drifted input back onto the teacher
  trajectory bakes in calibration-set statistics that don't generalize), or the
  premise that per-block optima compose into a better whole model.

  Do NOT scale this up. More samples is the wrong lever when the sign is
  negative — E19's 397B run was paused at 25/60 blocks on this evidence.
  If block-wise distillation is revisited, the thing to test FIRST is the
  chain: feed each block the TEACHER's activations instead of the student's
  (no error correction, no drift accumulation) and see if the sign flips.
  Cheap at 35B: ~45 min to train, ~1 min to score, deterministic.

  Method note: both arms looked equally healthy locally (6.6% / 5.7% loss
  reduction, no reverts) and both were worse end-to-end. **A per-block loss
  going down carries no information about model quality** — that is now
  measured twice, not argued.

- **E21 (08-12) — A DETERMINISTIC, exo-FREE MULTI-NODE SCORER. The 397B is
  measurable again.** `~/Documents/AgenicAI/quantlab/referee/score_pipeline.py` + `mlx.launch`.
  Validated on the 35B, whose correct answer we know:

  | harness | PPL | reruns |
  |---|---|---|
  | local mlx_lm, 1 node | 5.5548 | bit-identical |
  | mlx.launch, world_size=4 | 5.5547 | — |
  | **mlx.launch, world_size=2** | **5.5522** | **3x bit-identical, PASS** |

  vs exo's multi-node 4.824 / 3.500 / 22.077 on ONE unchanged model. The 0.05%
  residual is float accumulation order across shards, not instability.

  Sharding-mode nuance (Noah flagged it, and he is right): Qwen3.5-MoE exposes
  no `model.pipeline`, so **mlx_lm's** `pipeline_load` raises and we pass the
  group in the tensor slot — `sharded_load(repo, None, group, True)`. That is a
  LOADER limit, not a model limit: **exo pipelines the same model fine** via its
  own PipelineFirst/LastLayer wrappers, which is how E17 has a Pipeline score
  for t2.6 (24.065) at all.

  **exo's scoring math was never wrong and tensor parallelism is not to blame:**
  mlx_lm's own tensor sharding (the model's `shard()`, its own all_reduce) is
  exact. What differs is that no serving layer sits in between deciding which
  rank's response the caller reads.

  Four traps, each of which silently produced a WRONG-BUT-PLAUSIBLE run:
  1. **`--repeat` is eaten by mlx.launch.** argparse prefix-matches it onto
     `--repeat-hosts`, so `--repeat 3` duplicates every host (2 hosts x 3 =
     world_size 6) and your script never sees the flag. Named `--trials` here.
     **Always print world_size and check it equals your host count.**
  2. **mlx.launch ssh's to EVERY rank including localhost** — a box that cannot
     ssh to itself over the cluster interface fails with "exited with code 255"
     and a red-herring traceback about `cat None`.
  3. The script must be **executable with a venv shebang on every host**
     (it is exec'd directly, not passed to an interpreter), and args go
     **without** a `--` separator.
  4. Cleanup like `ps aux | grep score_pipeline | xargs kill -9` **matches its
     own command line** and kills the shell mid-script, orphaning ranks that
     then join the next ring. Keep cleanup and launch in separate commands.

  Architecture conclusion: **exo is the serving runtime, mlx_lm is the
  measuring instrument, and they must not be the same thing.** Today's whole
  mess came from using the serving stack as a scientific instrument. exo keeps
  its job (vision sidecar loading, placement, the API Scout talks to — none of
  which mlx_vlm can do multi-node: its only `mx.distributed` references are in
  trainers). Also measured: exo's idle overhead is 0.27 GB, so there is no
  RAM tax worth optimizing there.

- **E22 (08-12) — THE INSTABILITY IS 397B-SCALE-SPECIFIC, NOT A STACK BUG.
  E21's "exo was the culprit" is RETRACTED. No 397B perplexity has ever been
  validly measured in this project.** The control matrix, all cells same
  script/hosts/corpus/day:

  | | 35B (6G/rank) | 397B (62G/rank) |
  |---|---|---|
  | mlx 2-node | **bit-identical x5** (5.5522, incl. a rig-honesty rerun AFTER all failures) | t2.1: 77.9→4810; spicyneuron: 61.8→2912 — both FAIL |

  Both 397B artifacts fail the SAME way (plausible run 1, catastrophic run 2),
  on mlx_lm's stack with no exo involved — so exo's serving layer is acquitted,
  our artifact is acquitted (spicyneuron fails identically), and the rig is
  honest (the 35B repro passes bit-identical in the same degraded-day state).
  The only surviving variable is PER-RANK FOOTPRINT.

  Mechanism hypothesis (testable, one run): the M3's 62G shard NEVER became
  resident during either 397B run (rank RSS stayed ~5G; the M4's sat at 69.5G)
  — the M3 is the pressured box in both failures. File-backed mmap pages being
  evicted and re-faulted under an active GPU computation would produce exactly
  this: garbage instead of an error, worsening as pressure accumulates, and
  physically impossible at 35B where the shard fits trivially.
  **PREDICTION: strip the M3 (Claude closed, nothing resident) and the 397B
  scores stabilize.** That is the first task for a fresh session.

  Status of every 397B claim: E17's crown, its raw-metric reversal, E18's
  397B DWQ numbers — ALL unmeasured, in both directions. What stands: the 35B
  DWQ verdict (hurts, both objectives, E20), the vision differentiator, and a
  validated 35B-scale instrument with a determinism gate built in.

- **E23 (08-12) — ROOT CAUSE LOCALIZED: 397B tensor-sharded inference is
  bit-deterministic BELOW context 8192 and corrupts process state the moment
  context crosses it. First valid 397B PPL ever measured.** The per-chunk
  trace (`score_pipeline.py --per-chunk`) told the whole story in one run:
  chunks 0–7 (context <8192) reproduce BIT-IDENTICALLY across fresh
  launches (every digit, twice); chunk@8192 jumps nll 1571→8000+ and never
  recovers; and a SECOND trial in the same process is garbage from chunk 0
  even with fresh caches — the overrun poisons process state (weights or
  heap), which is why trial 2 always "exploded." E22b's memory-pressure
  hypothesis and the intermediate "doubled all_reduce" read (trial2 nll was
  exactly 2× trial1 — mix arithmetic of half-healthy + half-garbage, killed
  by the trace) are both dead. All historical "non-determinism" was
  identical healthy prefix + varying garbage tail. Falsified along the way:
  wired limit (no effect), 256-token repro (bit-identical ×4 — too short to
  trigger). Nothing in the config says 8192 (max_position 262144);
  suspicion points at a long-context kernel path (SDPA at ≥8K with 1 KV
  head/rank, or GatedDeltaNet state) writing out of bounds — upstream mlx
  repro still owed. 35B never crosses the buggy path at the same context
  lengths (14835 fine), so it is size/shape-, not length-only-, dependent.

  **Refinement (same day): the trigger is artifact-dependent.** spicyneuron's
  trigger is strictly CONTEXT length (`--reset-ctx 8192` → full corpus,
  3 trials, bit-identical ×3, PPL 3.0714, 44k cumulative tokens clean). t2.1's
  is roughly CUMULATIVE tokens processed (~8k): with resets at 4096 it still
  corrupted at chunk@8192 on a near-fresh cache, and capped trials 2–3 were
  poisoned start to finish. Same code, different quant mix → different
  trigger; whatever the kernel bug is, buffer shapes/sizes gate it.

  **The instrument that works for BOTH: score the first 8192 tokens, one
  trial per fresh launch (`--max-tokens 8192`).** Every model tested is
  bit-deterministic there across launches (every digit, 2–3 launches each).
  THE FIRST VALID 397B CHAMPIONSHIP (raw wikitext NLL, first 8192 tokens of
  the frozen referee corpus, growing context, step 1024):

  | model | nll/token | raw PPL (prefix-8192) | size |
  |---|---|---|---|
  | spicyneuron 2.6bit | 1.157816 | **3.1830** (bit-identical ×3 launches) | 120G |
  | t2.1 RevExp | 1.386002 | **3.9988** (bit-identical ×2 launches) | 124G |

  spicyneuron wins raw text by a real margin (+19.7% nll for t2.1) at 4G
  smaller. t2.1's differentiator remains what it always was: VISION (t2.1
  sighted 3/3, spicyneuron blind — E17b, still solid). E17's internal crown
  (t2.1 > t2.4 > t2.6) was measured on the broken instrument and is
  unverified; re-run t2.4/t2.6 on prefix-8192 if the ranking ever matters.

- **E24 (08-12) — SPICYMIRROR: their recipe, our weights, plus eyes. 3.1775,
  new champion.** Autopsy of spicyneuron's config: default 2-bit experts,
  8-bit structure (full attn, linear_attn out_proj, shared experts, embed,
  lm_head), 4-bit linear_attn in_proj_qkv/z — and routers/gates/in_proj_a/b
  NOT QUANTIZED (bf16). t2.1's loss decomposed: qkv crushed to 2-bit in
  37/45 layers + ~3.2G burned on two 4-bit expert layers. Current mlx_lm's
  `mixed_2_6` recipe is NOT what built spicy (llama.cpp-style, 3.162bpw,
  157G — killed mid-build). Build of record:
  `~/Documents/AgenicAI/quantlab/convert_spicy_mirror.py` — quantizes EXACTLY the tensors whose
  `.scales` appear in spicy's weight index (557=557 verified), bits from
  their config overrides, CPU device (GPU convert dies:
  kIOGPUCommandBufferCallbackErrorSubmissionsIgnored; E15 gotcha). v1 lesson
  (2-bit routers → PPL 34.85): ROUTERS ARE NEVER QUANTIZED in a good MoE
  recipe. Result 2.613 bpw, 121G LM + 0.91G vision sidecar (t2.1's, re-attached):
  **prefix-8192 raw PPL 3.1775, bit-identical ×2 launches — beats
  spicyneuron's 3.1830 by a hair, same size, sighted.**
  `TheDrainFlorist--Qwen3.5-397B-A17B-spicymirror` on Thunderbay + M4.
  Ladder next (each ~20min build + ~3min score): (a) structure 8→6-bit +
  freed ~1.3G into 3-bit tail experts (same-size-better); (b) structure
  4-bit floor w/ sane qkv (smaller, risky); (c) E16-style end-to-end DWQ on
  the winner (overnight). OPS: model copies M3→M4 go via the M4's SMB mount
  with 4 parallel streams (`find -type f | xargs -P4 cp`) — ~1GB/s vs
  275MB/s single-stream rsync/ssh; NFS was retired 08-03 (SMB won).

- **E25 (08-12) — THE EXPERT-TAIL LADDER. 2-bit experts are the binding
  constraint; tail placement beats spread; E17's "2.1 floor" is dead.**
  All prefix-8192 raw, bit-identical ×2 launches, built by
  `~/Documents/AgenicAI/quantlab/convert_variant.py` (spicy-mirror base, dials:
  --structure-bits / --tail-expert-bits / --tail-layers / --promote-every;
  size predicted from source shapes BEFORE building, matched-size guard):

  | variant | struct | experts@3bit | PPL | GiB(q) |
  |---|---|---|---|---|
  | struct6-tail10 | 6 | last 10 | 3.0157 | 126.4 |
  | spread10 (every 6th) | 6 | 10 spread | 3.0490 | 126.4 |
  | struct6-tail6 | 6 | last 6 | 3.1130 | 123.4 |
  | struct6-tail3x3 | 6 | last 3 | **3.1580** | 121.2 |
  | struct6-tail3 (=2 layers) | 6 | last 2 | 3.1712 | 121.0 |
  | spicymirror (E24) | 8 | none | 3.1775 | 120.3 |
  | spicyneuron | 8 | none | 3.1830 | 120.3 |
  | struct6 | 6 | none | 3.1841 | 118.9 |
  | struct4-tail4 | 4 | last 4 | 3.3182 | 120.6 |
  | struct4 | 4 | none | 3.371* | 117.6 |

  (*struct4 = single launch; its confirm run died on a GPU timeout from
  overlapping scoring with two parallel builds — don't do that.)
  FINDINGS: (1) marginal PPL/layer of tail promotion ACCELERATES through 10
  layers (−0.0065→−0.0243/layer) — no knee found ≤10; 2-bit experts are
  starved, so E17's "experts pinned at 2-bit / t2.1 is the floor" was an
  artifact of the broken instrument + the 2,3,4 candidate ladder's own
  allocator. (2) POSITION IS REAL at matched size: tail10 3.0157 vs
  spread10 3.0490 — E12's late-layer law survives on a valid instrument.
  (3) Structure bits: 8→6 costs +0.0066 (cheap to demote), 6→4 costs
  ~+0.19 (never); routers/gates stay bf16 (2-bit routers = +11 PPL, E24).
  **STATIC RECIPE OF RECORD @~121G: struct6-tail3x3** (6-bit structure,
  4-bit qkv/z, 3-bit last-3-layer experts, 2-bit rest, bf16 routers) =
  3.1580, beats spicyneuron at its own size, sighted. Every further GB ≈
  −0.015..0.024 PPL via deeper tail — buyer's choice. NEXT: E16-style
  end-to-end DWQ on the winner via cached teacher top-k logits (751G
  teacher can't be resident; stream once, cache, train against cache).

- **E26 (08-12, night) — 397B DWQ MADE RUNNABLE: cached-prefix solo-box
  formulation, after distributed training died three measured deaths.**
  Goal: E16-style sequence-level DWQ on struct6-tail3x3 (tail-6 trainable).
  What failed, with cause measured each time: (1) PIPELINE-parallel training
  impossible — `mx.distributed.send` has NO vjp (loud error; the shim's
  forward is fine and validates identically to single-node); (2) tensor-
  parallel training passes gradient-reach on both ranks but the M4 rank dies
  in Metal watchdog (GPU Timeout) on the first optimizer step at 397B —
  micro-batching AND MLX_MAX_OPS_PER_BUFFER=20 both failed to save it;
  (3) probes: a single tail layer bwd = 0.6s, the ENTIRE tail-6+norm+head
  bwd solo on the M4 = 3.3s, NO watchdog — the kill needs the 60-layer
  sharded context, not the training math. INSIGHT: the 54-layer prefix is
  FROZEN, so its activations are training-constant — cache them once and
  the distributed problem evaporates. Pipeline:
  `dwq_cache_teacher_stock.py` (stock-format top-1024 targets by STREAMING
  the 751G teacher, ~2h; MUST use the STUDENT tokenizer — t2.4's differs
  from teacher and misaligns every batch; 397B pair verified identical) →
  `dwq_cache_student_prefix.py` (layer-54 acts, ~25 min) →
  `dwq_train_tail_solo.py` on M4 (14.8G resident, 3 epochs overnight,
  initial valid KL 1.0545 = distributed run's 1.0556 ✓). 35B smoke of the
  tensor path: valid KL 2.2392→1.3257 in 20 steps (tail-6) — DWQ moves
  HARD when it runs. Assembly: paste tail_patch.safetensors scales/biases
  into a copy of the champion, referee prefix-8192 ×2 vs 3.1580.
  Traps for the file: mlx.launch --env for remote env vars; barrier before
  first distributed op (rank skew inside recv = watchdog death); E23 test
  first (champion is context-triggered: reset-8192 full corpus ×3
  bit-identical 3.0579 — cumulative-safe for training).

- **E26b (08-13, overnight ops) — three real causes behind "training is
  slow/stuck", none of them the method:** (1) **The M4 ran on battery** —
  the TB5 cable swap disrupted its power path; it slept mid-run (evicting
  the model; steps 9s→90s), then at 2% charge macOS throttles the GPU ~10×
  EVEN ON AC, and the weak charge source loses to training draw (2%→5%→2%).
  Laptop trainers: `caffeinate -is` ALWAYS, and check `pmset -g batt`
  FIRST when a run degrades. (2) **`ps` rss lies about unified memory** —
  trainer showed 0.2-15G rss while `sample <pid>` showed 78.8G real /
  119G peak. Variable-length batches make every new seq length allocate a
  fresh ~25G transient set and the MLX buffer cache retains ALL of them:
  fine on 128G, death-by-compression on 96G. Fix: `mx.set_cache_limit(8G)`
  (footprint 78.8G→50G). Measure footprint with `sample`, never ps.
  (3) **The M3 Ultra is ~6× slower than the M4 Max on this workload**
  (~60s/step vs 9s) — E18's launch-bound GDN-backward finding, now
  confirmed at the full-step level. Trainer of record for tail-DWQ: the
  M4, on wall power. Also: TB bridge (10.0.0.x) went down with the cable
  event — mDNS hostnames still work for ssh; referee can ring over LAN.
  All three data waves (6144 samples) cached and banked on Thunderbay.

- **E27 (08-13) — DWQ ROUND 1 VERDICT: FALSIFIED for wikitext PPL at 397B
  (tulu-corpus, tail-6). Two mechanisms, both measured:**
  (1) **Trained ATTENTION scales destroy OOD behavior**: the epoch-1 patch
  (valid KL 0.0253→0.0205, on-distribution tail-KL 0.0201 = healthy) scores
  **PPL 201,892 deterministically** — per-block probe shows the prefix
  bit-clean through layer 53, then layer 55 (full-attn, largest trained
  deltas: k_proj relmax 7.6%) inflates activations 1.71×, compounding to
  5.7× by 59 → uniform logits. On-distribution fine, wikitext catastrophic:
  2-bit models are calibration-manifold-fragile and mis-trained attention
  walks OOD inputs off it. (2) **Expert-scale training doesn't transfer**:
  attention-dropped patch (72 mlp/expert tensors) = 3.1619 vs champion
  3.1557 — sane but no gain; tulu-KL improvements do not move wikitext.
  ALSO BUILT: `referee/score_streaming.py` — single-box streaming referee
  (champion 3.1557 vs 2-node 3.1580, the known 0.07% decomposition
  residual; ~70s/score) after 2-node scoring of the ep1 artifact broke —
  NOTE the 202k was REAL (both instruments agree), not E23. Diagnosis
  ladder that cracked it: assembly verified (cand==patch), patch NaN-free,
  round-trip byte-clean, rig-honesty (champion re-scored 3.1580 exact),
  cross-copy md5s, then per-block activation diff — the last one named
  layer 55. E16's 35B DWQ win does NOT replicate at 397B tail-only/tulu.
  IF REVISITED: freeze attention scales in the trainer (unfreeze predicate
  excludes *_proj in attn), train experts only, and use a general-text
  corpus; expectations modest. The static champion stands.

- **E28 (08-13) — DWQ ROUND 2 (experts-only, attention frozen): ALSO
  FALSIFIED — the fragility is the 2-bit expert scales themselves, not
  attention.** Ran the fair test E27 called for: `--freeze-attention`
  (only mlp/expert scales trainable, 1210M params) + OOD canary gate
  (wikitext-2 VALIDATION slice through the frozen prefix, disjoint from
  the referee corpus). Epoch-1 verdict: tulu valid KL 1.0545 → **0.0230**
  (aced on-distribution — memorization-speed descent, 1.05→0.10 by step
  100) while canary NLL 1.0655 → **10.7312** (10× OOD collapse). Gate
  fired, patch refused. Together with E27's attention-dropped patch
  (sane weights, 3.1619, no gain), both branches are now closed: trained
  expert scales at 2-bit either break OOD or don't help wikitext.
  KL-training a 2-bit MoE's quantization scales on a chat corpus
  overfits the calibration manifold regardless of which sublayer is
  trained. "Train earlier/middle expert layers" would hit the same wall
  (same 2-bit expert scales — one more idea removed). **DWQ at
  397B/2-bit is closed. struct6-tail3x3 (3.1580 2-node / 3.1557
  streaming, 122G, sighted) ships as-is.** If EVER revisited:
  general-text calibration corpus + canary-gated early stopping inside
  epoch 1 (damage complete by step ~250), but E27's no-gain result caps
  expectations near zero.

- **E29 (08-13) — THE SHAPE LAW, THE KNEE, AND A SECOND SHIPPED ARTIFACT.
  Breadth beats richness; tail promotion dies at the half-way line; our
  3.0-bit lands within 1.6% of a community 3.5-bit that is 23 GiB bigger.**
  Motivated by a role split Noah named: the 122 GiB champion is the DAILY
  driver (must leave unified-memory headroom for caption workers, sidecars,
  Chroma), but the overnight runner owns the box, so quality is worth bytes
  up to a 145 GiB ceiling (DeepSeek V4-Flash, already swapped nightly, is
  145 GiB — so this is proven-feasible territory, not a new risk).

  **(1) MATCHED-BYTE SHAPE SWEEP — three builds at IDENTICAL 141.42 GiB,
  differing only in WHERE the bits sit** (new `--expert-schedule` dial in
  `convert_variant.py`, since the tail dial can only promote off a fixed
  2-bit base):

  | shape | layers left at 2-bit | PPL |
  |---|---|---|
  | **flat 3-bit ×30 (tail30)** | 30 (L0-29) | **2.3982** |
  | ramp 3-bit ×10 + 4-bit ×10 | 40 (L0-39) | 2.5042 |
  | spike 4-bit ×15 | 45 (L0-44) | 2.7224 |

  PPL tracks HOW DEEP THE 2-BIT REGION REACHES and nothing else — 0.32 PPL
  spanned by allocation shape alone at constant bytes. **Rule: escape 2-bit
  broadly before enriching narrowly.** And it is not a preference but an
  optimum: a 4-bit layer costs exactly TWO 3-bit promotions, so under a byte
  budget with 2/3/4 candidate widths, maximizing non-2-bit layers means never
  buying 4-bit at all. The flat shape is provably the best available at its
  size. (Prediction was pre-registered before C scored — C worst, ~2.6-2.7;
  it landed 2.7224. The mechanism predicted the number, not just the order.)
  Note this REFINES rather than contradicts E12/E25's position law: tail10
  beat spread10 when the COUNT was fixed and only position varied; here the
  count/depth trade is the variable and count wins.

  **(2) THE KNEE IS AT ~TAIL30.** tail33 (144.8 GiB, three more promoted
  layers) = **2.3961** — 0.002 PPL for 2.3 GiB, against the ~0.031 PER LAYER
  the ladder paid up to tail30. Tail promotion stops paying the instant it
  crosses into the front half of the network — E12's late-layer law arriving
  exactly where it should. E25's "no knee found ≤10" is now bounded: the knee
  is at ~30. tail33 was built, scored, deleted. **Noah's 145 GiB ceiling was
  never the binding constraint; the physics stopped paying first.**

  **(3) ANCHOR — and the honest claim.** spicyneuron 3.5bit (165.6 GiB,
  3.53 bpw, BLIND) = **2.3614** on the same instrument. So:

  | model | GiB | bpw | PPL | eyes |
  |---|---|---|---|---|
  | spicyneuron 3.5bit | 165.6 | 3.53 | 2.3614 | no |
  | **ours tail30** | **142.5** | **3.02** | **2.3982** | yes |
  | ours tail3x3 (daily) | 122.3 | 2.59 | 3.1557 | yes |
  | spicyneuron 2.6bit | 120.6 | 2.57 | 3.1830 | no |

  The anchor did double duty: it VALIDATED tail30's number (a larger,
  higher-bit model from an independent quantizer landing at 2.36 puts 2.3982
  in the right regime — a broken build or lying instrument would have shown
  their 3.5bit near 3.0), and it set the corpus's achievable floor. **We do
  NOT beat their 3.5bit** — we are 1.6% behind at 23 GiB less and sighted.
  Publish the EFFICIENCY claim, not a win. And the remaining 1.6% is NOT
  reachable by more depth (see the knee) — it needs a different axis.
  Every number here reproduced bit-identically ×2 (tail30 total_nll
  7165.8109 twice; tail33 7158.4977 twice).

  **SHIPPED: two artifacts, one per duty cycle.** Daily = struct6-tail3x3
  (122.3 GiB). Overnight = struct6-tail30 (142.5 GiB, 24% lower PPL),
  now the `run_overnight.py` author default, placed Tensor/MlxJaccl.
  Both carded as BUILTINs on both nodes.

  **NEXT AXIS — falsify before building.** The layer axis is closed and
  dialed; per-EXPERT precision is untouched. Blocker is the format: all 512
  experts of a layer live in ONE batched `[512, 4096, 16]` tensor with one
  bit-width. The tractable version is a hot/cold SPLIT (two batched tensors,
  two SwitchLinear calls, no new kernel) — but it rests on routing being
  skewed, and MoE load-balancing loss exists precisely to flatten it.
  `probe_routing_skew.py` measures the real distribution in one forward pass
  (wraps `SwitchGLU.__call__`, whose `indices` ARE the selections). Gini
  <0.10 = uniform, idea dies for the price of a coffee; >0.25 = real mass to
  allocate. RUN THIS BEFORE WRITING ANY KERNEL OR SPLIT LOGIC. If it pays, it
  should lift the DAILY driver too, not just the overnight one — the 122 GiB
  build has far more 2-bit layers for a hot/cold split to work on.

- **E30 (08-13) — ROUTING SKEW IS HUGE BUT DOMAIN-BOUND. General-purpose
  per-expert allocation is DEAD; domain-specialized allocation is ALIVE.**
  `probe_routing_skew.py` (wraps `SwitchGLU.__call__`; its `indices` ARE the
  selections — no model surgery), 2048 tokens, 512 experts/layer.

  **(1) Skew is enormous, and load balancing did NOT flatten it.** Mean Gini
  **0.635**; the top 10% of experts take **48.9%** of routing (uniform = 10%),
  top 25% take ~84%, and ~130 of 512 experts per layer never fire at all.
  Holds on all three corpora (gini 0.60 code, 0.63 wikitext, 0.69 prose).

  **(2) BUT THE HOT SET MOVES WITH THE INPUT — and static bits cannot track a
  moving target.** Overlap of each layer's top-10% expert set:

  | pair | overlap | note |
  |---|---|---|
  | wikitext-A ∩ wikitext-B | **54.5%** | CONTROL: same domain = the noise floor |
  | code ∩ prose | **49.6%** | indistinguishable from the floor |
  | wikitext ∩ prose | 14.6% | |
  | wikitext ∩ code | **9.2%** | chance is 10% — literally random |

  The same-domain control is what makes this readable: 54.5% vs a 10% chance
  baseline proves the probe measures something real, so wikitext-vs-code at
  9.2% is a genuine disjointness, not sampling noise. **Run the control before
  believing any overlap number** — at 2048 tokens each expert sees only ~32
  routes, and without the floor this table is uninterpretable.

  **(3) THE USABLE FINDING: stability is a property of the domain FAMILY.**
  code∩prose ≈ the same-domain floor means technical text of any shape routes
  to the same experts. So a hot/cold split is illegitimate for a
  general-purpose model and legitimate for a duty-cycle-specialized one — and
  we ALREADY ship per duty cycle (E29). The overnight author's diet is code +
  engineering prose, exactly the family where the hot set holds still.
  Economics if it works: promoting the hot 25% costs a QUARTER of a full-layer
  promotion for ~84% of routing mass — ~4x the coverage per byte, enough to
  hot-promote all 60 layers for half of what tail30 spent on 30.
  Mechanism is a hot/cold SPLIT (two batched tensors, two SwitchLinear calls,
  model-code change only — no new kernel); per-expert widths remain
  inexpressible because 512 experts share one `[512, 4096, 16]` tensor.

  **BLOCKER BEFORE BUILDING: our referee is wikitext-only.** A code-specialized
  quant scores WORSE there while being better at the job, so measuring it on
  the current instrument would repeat E17's broken-instrument mistake in a new
  costume. Build a code-domain referee corpus FIRST, validate it against a
  known-good artifact, and only then allocate. Per-corpus routing JSON (counts
  + top-25% ids per layer) in `/tmp/skew_{wikitext,code,prose,wiki_b}.json` —
  regenerate rather than trust, they are snapshots.

## Decision

**SUPERSEDED 08-13 — the current decision is the STATE OF RECORD block at the
top of this file: ship `struct6-tail3x3` (built by `convert_variant.py`
mirroring spicyneuron's weight-index map, NOT by optiq target-bpw), scored by
the streaming referee. The paragraphs below are the 08-11 decision, kept for
the record; their optiq-t2.1 recipe and PPL figures were voided by E23-E25.**

**397B-A17B production quant = static mixed recipe at `--target-bpw 2.1`
(`--candidate-bits 2,3,4`) with the `reverse_experts` depth prior (E14: strictly ≥
the community U at every measured budget; expert bits tail-weighted,
attention/router/lm_head floored per the stock component offsets) + bf16 vision
tower (mlx-vlm-style packaging), from the verified bf16 (751G, Thunderbay SSD).**
Hours-scale. Calibration earns no place on MoE until probes measure layer GROUPS —
and isolation curves are hypothesis generators only (E7/E11/E12: two proven
end-to-end inversions).

**⛔ Superseded 08-11 (E17) — itself later VOIDED (E23, broken instrument; see
STATE OF RECORD): the budget is 2.1, not 2.6.** Measured on the cluster,
t2.1 (124G, PPL 9.106) beats t2.4 (138G, 18.948) and t2.6 (147G, 24.065) outright —
smaller AND better, so the earlier "~2.6bpw" figure was never the optimum, just the
first budget built at scale. 2.1 is also the FLOOR: t1.8 collapses to 535 PPL with
destroyed generation (2-bit candidate floor binding — see E17). Build command of
record:

```bash
OPTIQ_FORCE_STREAMING=1 OPTIQ_DEPTH_PRIOR=reverse_experts optiq convert "$BF16_SRC" \
  --method static --target-bpw 2.1 --candidate-bits 2,3,4 -o "$OUT/397b-t2.1-revexp"
```

## E31 (08-14) — Rotation autopsy: the 2026-06 failure was a BROKEN BUILD, not "rotation doesn't work here"

Step 0 of the fused-rotation arc: before spending a build, establish whether the
June rotorquant salad indicts rotation or indicts that artifact. Verdict: **that
artifact**. Rotation is NOT falsified on this rig — it was never tested here.

Local artifacts (`majentik--Qwen3.5-397B-A17B-RotorQuant-MLX-2bit`,
`…-TurboQuant-…`) are deleted; evidence is the June diagnosis
([[project_exo_model_cards]], instrument-first, 2026-06-17) plus the repos'
`config.json` re-fetched from HF 08-14.

Evidence:
1. **The failed build is UNIFORM 2-bit.** Both 397B repos: `bits: 2,
   group_size: 64, mode: affine`, **no per-module overrides** — embeddings,
   q/k/v/o, router, experts and lm_head ALL at 2 bits. The working comparator
   (spicyneuron 2.6bit) is 8-bit embed / 4-bit attn / 2-bit experts+router.
2. **Our own lab already proves that recipe is lethal, rotation or not.** E7:
   attention-at-2-bit → PPL 46.4 (salad) while experts-at-2-bit → 10.4. The June
   build's allocation alone predicts its output; rotation is not needed to
   explain it.
3. **Confound is total, and the June note's "CONFIRMS the 2-bit router" line is
   wrong.** RotorQuant and TurboQuant are the same vendor and the same
   uniform-2-bit pipeline — n=1 pipeline, not two independent methods. And the
   router hypothesis was refuted the same day by spicyneuron 2.6bit generating
   coherently with a 2-bit router+experts.
4. **No online-rotation story to blame.** The configs carry no rotation /
   hadamard / custom `quantization_method` key and no custom loader or kernel
   code; the June stock-loader probe found **0 missing params**, and mlx
   `gather_qmm` was verified faithful at 2-bit. The rotation was baked into the
   weights (fused) and stock MLX decoded it correctly — the artifact loaded
   right and was allocated wrong.
5. **Rotated weights decode fine on this stack**: the same vendor's dense
   `Qwen3.5-27B-RotorQuant-2bit` (also uniform 2-bit) is coherent.

Consequence for this arc: fused rotation is unexplored territory here, and the
prudent design is rotation **on top of our own struct6 allocation** (6-bit
structure, 4-bit qkv/z, bf16 routers) — never uniform 2-bit, which is the one
variable known to produce this failure. Nothing found argues for touching online
activation rotation.

## E32 (08-14) — Fused rotation FALSIFIED for weight-only 2-bit under group-64 affine

The ladder (0.8B exactness → 35B exactness → 35B quant A/B) ran to a clean
negative. Tooling: `rotate_fuse.py` (residual-stream R1 rotation, norm-fold,
HF zero-centered-norm aware, mtp dropped), `convert_35b_struct.py` (397B
struct6 recipe at 35B scale: experts 2b / structure 6b / qkv,z 4b / routers
bf16), `probe_rotation_divergence.py` (per-layer H-equivalence probe).

Exactness (rotation without quantization must be a no-op):
  0.8B  16.8462 → 16.8558 (+0.06%, bf16 re-round noise)   PASS ×2
  35B   6.1906 → 6.1957 (+0.08%) plain H                  PASS ×2
  35B   6.1906 → 6.2233 (+0.5%)  randomized D·H seed 42   PASS
The transform is provably correct — what follows is a quantization
interaction, not a bug.

Quantized A/B at IDENTICAL recipe + size (11G, bit-for-bit same histogram),
all numbers reproduced bit-identically:

  | 35B struct6 artifact | wikitext | code   |
  |---|---|---|
  | baseline             | 7.4285   | 3.1653 |
  | plain Hadamard       | 7.9208 (+6.6%) | 3.2826 (+3.7%) |
  | randomized D·H       | 7.9687 (+7.6%) | 3.3416 (+5.6%) |

Rotation HURTS, both corpora, both family members — randomized slightly
worse than plain. Mechanism: MLX affine quant keeps a scale per 64-channel
INPUT group, which is already a crude outlier isolator — only groups that
contain an outlier pay for it. Rotation smears every outlier across all
groups, so every group inherits a wider range and a coarser 2-bit grid.
The literature's rotation wins come from activation quant (not done here)
and from quantizers without group-level adaptivity (QuIP#'s lattice etc.).
Weight-only + small-group affine is exactly the regime where incoherence
processing has nothing to sell.

Mechanism CORRECTION (same day, Noah's "rotate only the clean groups?"
question forced the measurement): the outlier premise is FALSE for these
weights. Per-(row, 64ch-group) max ratios sampled across expert / linear-attn
/ full-attn matrices: p99.9 = 1.8-2.7x, dirty fraction at a 4x threshold
~0.00%. Qwen3.5 weights are already flat at group-64 granularity — there
were never outlier groups to fix. So rotation's harm is not "smearing
outliers into clean groups" (first-draft story above): it is Gaussianization
— each rotated entry becomes a sum of 2048 light-tailed weights, and a
Gaussian's tails fit a 4-level affine grid worse than the flatter native
distribution. Incoherence processing solves spiky weights; these aren't.
(Selective rotation is also structurally unavailable — one global R shared
by every reader/writer — but the premise fails before the constraint binds.)

Wrong-turns log (both cost a build, neither voids the result):
- HF checkpoints store norms ZERO-CENTERED (w-1); mlx_lm sanitize shifts
  +1 at load, keyed on mtp/conv1d presence. First 35B rotation folded raw
  g → PPL 71M. rotate_fuse now detects and compensates; output is emitted
  fully sanitized.
- GatherMM has no CPU path for bf16 — MoE probes must compute on GPU
  (params may still load on the cpu stream), like score_streaming does.

Consequence for the HF debut: the 3bit-class gap (1.6% wikitext) will NOT
close via rotation. With tail depth at its knee (E29), DWQ falsified
(E20/E27/E28), allocation domain-bound (E30) and rotation falsified (this),
the fused-weights lever drawer is EMPTY on this quantizer. What's left is
changing the QUANTIZER, not the weights: mode="mxfp4"-class formats, smaller
group sizes for the 2-bit region (paying bytes), or vector/lattice quant
(needs kernel work). E31's autopsy conclusion stands unchanged: the June
salad was allocation, not rotation — but rotation now has its own honest
verdict on top: correctly applied, it still loses here.

## E33 (08-14) — Group-size 32 on the 2-bit expert region: the first lever in four that points the right way

Follow-on from E32's flatness finding: if per-group ranges are already
uniform, 2-bit is starving for GRID RESOLUTION, not outlier protection.
Test: struct6 recipe with experts 2-bit at gs32 instead of gs64 (fp16
scale+bias per 32 → +1 bit/weight overhead; effective 2.5→3.0bpw there).

  | 35B struct6 | size | eff bpw | wikitext | code |
  |---|---|---|---|---|
  | experts 2b gs64 | 11G | 2.746 | 7.4285 | 3.1653 |
  | experts 2b gs32 | 13G | 3.211 | 7.3028 (-1.7%) | 3.1378 (-0.87%) |

Reproduced ×2 bit-identically. Both corpora improve — consistent with the
flatness mechanism.

397B result (built same day, tail30+gs32-on-2bit-region, 152.7 GiB,
vision grafted, all numbers reproduced bit-identically ×2 for the
candidate; baselines single-run today but match the E29/E30 records):

  | artifact | GiB | wikitext | code |
  |---|---|---|---|
  | spicyneuron 3.5bit | 165.6 | 2.3614 | 2.6005 |
  | tail30 (ship) | 142.5 | 2.3982 | 2.5928 |
  | tail30+gs32 | 152.7 | 2.3903 (-0.33%) | 2.5980 (+0.20%) |

**The 35B -1.7% did NOT transfer.** At 35B every expert was 2-bit, so the
2-bit region was the whole loss; at 397B tail30 already promoted the half
that mattered and the knee (E29) says the residual 2-bit region barely
binds. gs32 sharpens a region that is no longer the constraint: -0.33%
wikitext, -0.2% on code, +10 GiB. NOT SHIPPED. The lever is real but its
payoff is proportional to how much 2-bit loss remains — near the knee,
almost none. E29's closing line stands verbatim: the remaining 1.6% needs
a different axis (per-expert precision / hot-cold split, gated on the
routing-skew probe).

## E34 (08-14) — GPTQ error-compensated rounding: v1 (fp-input) FALSIFIED, and it named the mechanism

The zero-byte lever: keep MLX's exact group-64 affine format, change only
WHICH grid level each weight rounds to, minimizing OUTPUT error (Hessian
from calibration activations) instead of weight error. Toolchain has none
of this — built in `gptq_solver.py` (+ `gptq_experts_35b.py` capture/solve,
`assemble_gptq_35b.py` swap-in).

Pre-flight validation (all passed before spending the run):
- **Pack layout reverse-engineered + verified**: 2-bit indices are 16 per
  uint32, little-endian within the word; hand-packed indices survive
  `mx.dequantize` bit-exactly. GPTQ output is format-identical to RTN.
- **Solver**: textbook GPTQ (lazy block updates, `Hinv = chol(inv(H))`
  upper). Synthetic anisotropic test showed the right signature —
  weight relerr UP 0.45→0.80, OUTPUT relerr DOWN 0.451→0.357.
- **Grid honesty**: scales/biases snap to bf16 INSIDE the solver, before
  rounding decisions, so it optimizes against the grid the artifact
  actually decodes with.
- **Calibration hygiene**: 200k tokens, wikitext-TRAIN + scout/exo code,
  provably disjoint from both referee corpora (which are wikitext-TEST and
  a fixed mlx_lm/exo/scout-JS file set, all excluded by name).
- 10,240 experts solved (40 layers x 256), per-expert Hessians from real
  routing, **zero starved experts** anywhere.

Result — v1, 35B struct6, identical 11G/format, RTN rebuild reproduced its
own prior numbers bit-identically (16427.6274 / 9439.236):

  | 35B struct6 | wikitext | code |
  |---|---|---|
  | RTN | **7.4285** | **3.1653** |
  | GPTQ v1 (fp inputs) | 8.5649 (+15.3%) | 3.3560 (+6.0%) |

**Not a broken build** (checked first, per E31): the artifact dequantizes to
exactly the solver's intent (maxdiff 0.0005 = bf16 rounding), grids match
RTN (scale ratio 1.0), weight relerr moved 0.486→0.767 as designed.

**Mechanism, measured:**
- GPTQ HALVES per-expert output error on calibration inputs: 0.315→0.133.
  It fully delivers on its own objective.
- Routing is NOT the confound: 95.2% expert-selection overlap bf16 vs
  quantized.
- So each layer is individually 2x more accurate, yet the 40-layer stack is
  15% worse. That is the fp-input simplification: compensation is fit to
  full-precision inputs, but at inference each layer receives QUANTIZED
  upstream activations. GPTQ buys accuracy by moving weights far from
  original; when the input distribution shifts, the cancellation lapses and
  only the large weight error remains — compounded over 40 layers.

**Pipeline perf — 4x, and a lesson in not guessing.** v2's first run sat at
150s/layer. I blamed worker count (6→18: NO change), then BLAS threading
(pinning `VECLIB_MAXIMUM_THREADS=1` on the whole process made the PARENT
single-threaded — actively worse), then the Hessian build (benchmarked: 9s
of 150s). Sampling parent-vs-worker CPU showed ~100s/layer of the parent
alone at exactly 100% = one core. The actual culprit, found by benchmarking
the candidates directly instead of reasoning about them:
`np.savez_compressed` = **119s/layer** to save 0.5 GiB on a 3.3 TiB-free
drive. Plain `np.savez` = **0.3s**. 150s/layer → 39-44s.
**Use `np.savez`, never `savez_compressed`, for these checkpoints** — at
397B scale (60 layers, 8x payload) that one word is a ~2 HOUR pure-waste
tax. Also worth threading the Hessian build (2.4x on its 9s) for the 397B.
Three wrong guesses, one 30-second benchmark: instrument the candidates,
don't reason about them.

### E34b — v2 (sequential propagation): ALSO LOSES. Pattern promoted, n=4.

v2 = reference GPTQ's fix for exactly the mechanism v1's measurement
indicted: activations captured from the QUANTIZED (RTN struct6) model,
weights solved from bf16. Same solver, same corpus, same format.

  | 35B struct6 (11G, identical format) | wikitext | code |
  |---|---|---|
  | RTN | **7.4285** | **3.1653** |
  | GPTQ v1 (fp inputs) | 8.5649 (+15.3%) | 3.3560 (+6.0%) |
  | GPTQ v2 (quantized inputs) | 8.3184 (+12.0%) | 3.3756 (+6.6%) |

Reproduced bit-identically ×2. Sequential propagation recovered only 2.9%
of a 15% deficit on wikitext and nothing on code — it is directionally
right and nowhere near sufficient. **RTN beats GPTQ at 2-bit here.**

**FINDING (n=4): activation-FITTED quantization fails at 2-bit on this MoE
family** — E27 (DWQ attn), E28 (DWQ experts), E34a, E34b — while pure
weight-space methods (static allocation, tail depth) hold. Mechanism is
consistent across all four: these methods buy accuracy on the calibration
distribution by moving weights FAR from original, and at 2 bits (4 levels)
there is no representational headroom to encode the correction — the
compensation is clipped/rounded away and only the damage survives.

Two MoE-specific aggravators worth naming, both measured:
1. **Hessian data starvation is structural.** With 8-of-256 routing, each
   expert sees ~3% of tokens: 6,272 tokens for a 2048-dim Hessian =
   **3.1x oversampling**. A DENSE model of the same width on the same
   corpus gets **98x**. That looked like the escape hatch — it isn't; see
   E34c/E34d below, which closed it for the price of 15 minutes instead of
   a 2-hour run.
2. Routing is NOT a confound (95.2% overlap) — ruled out in E34a.

### E34c/E34d — the data escape hatch, opened and closed by measurement

`probe_hessian_convergence.py` (E34c): build H from two DISJOINT halves of
the calibration set, compare the GPTQ decisions. Result: the halves disagree
on **25.7%** of all weights while GPTQ only moves **19.6%** off RTN — the
disagreement EXCEEDS the entire signal. Verdict printed:
"NOISE-DOMINATED — more data is the lever." **That verdict was WRONG**, and
it was wrong because it was a single point.

`probe_hessian_scaling.py` (E34d): same test at 256/512/1024/2048
tokens-per-subset. BOTH columns grow with N, and the ratio is FLAT:

  | tok/subset | disagreement | signal (vs RTN) | ratio |
  |---|---|---|---|
  | 256 | 19.06% | 14.66% | 1.30 |
  | 512 | 25.38% | 19.73% | 1.29 |
  | 1024 | 32.66% | 25.63% | 1.27 |
  | 2048 | 34.47% | 26.74% | 1.29 |

At small N the Hessian is near-singular, damping dominates, GPTQ ~= RTN —
little signal AND little noise. As N grows the Hessian conditions up and the
solver moves more weights off RTN; a flat ratio means **every extra decision
it earns is one an independent data sample would have made differently.**
No SNR improvement across an 8x range. More data is NOT the lever.

**Sharpened root cause: at 2 bits GPTQ's sequential update is a CHAOTIC
MAP.** Each rounding decision feeds the next; with 4 levels a hair's
difference in the Hessian flips an early decision that cascades through
every column after it. Better Hessians don't damp the cascade — they make
the solver more confident about which noise to follow. This is why n=4
holds and why it is not fixable with calibration data.

Residual uncertainty, stated honestly: tested 256-2048 tok/subset against a
~4,700-tok/expert pool; a 32x corpus is untested and the ratio could break
there. Flat over 8x is decent evidence, not proof.
Method note: measure the RATIO, never extrapolate either column — the
fitted slope is POSITIVE, so log-log extrapolation prints impossible >100%
disagreements (it did, once; removed from the script deliberately).

## E35 M0 (08-14) — VECTOR QUANTIZATION WORKS. The first lever to beat RTN.

After E31-E34 killed every scalar-format lever against the same wall (2-bit
affine = 4 rigid levels, no headroom for any correction), VQ removes the
wall instead of fighting it. `vq_fit.py`: k-means product quantization over
d=4 weight subvectors, per-(row,64) fp16 scale, codebooks fit in PURE
WEIGHT SPACE (no Hessian, no activations — E34 put those at 0-for-4).

Scored as a QUALITY PROXY (VQ reconstruction written as bf16, so the
existing referee scores it with no kernel and no exo change; bytes are
analytic). All artifacts share IDENTICAL 6-bit structure — only the expert
treatment differs. Every number reproduced bit-identically x2.

  | expert format | bpw | wikitext | vs RTN | %gap | code | vs RTN | %gap |
  |---|---|---|---|---|---|---|---|
  | RTN affine 2-bit | 2.50 | 7.4285 | — | 0% | 3.1653 | — | 0% |
  | **VQ d4 K256** | **2.25** | **7.1807** | **-3.34%** | 23% | **3.0881** | **-2.44%** | 35% |
  | **VQ d4 K1024** | **2.75** | **6.5818** | **-11.40%** | **77%** | **2.9752** | **-6.01%** | **86%** |
  | bf16 experts (ceiling) | 16.00 | 6.3310 | -14.77% | 100% | 2.9446 | -6.97% | 100% |

**K256 spends EXACTLY the same 2 bits/weight as scalar RTN** (8 bits per
4-weight group either way) and still wins — 256 LEARNED joint patterns beat
256 rigid grid combinations. It is also 10% SMALLER because affine needs
scale AND bias per group while VQ needs only scale (centroids are arbitrary
4-vectors; asymmetry is free).

**The curve is STEEP: +0.5 bpw (K256->K1024) buys 54 points of the gap.**
Codebook size is the binding constraint at d=4 — so K=4096 (3.0 bpw) and
larger-d lattices are the direction, and the 3-bit tail has a VQ analogue.

397B projection (state as projection, not result): the debut needs -1.6% on
wikitext (tail30 2.3982 vs spicyneuron 2.3614 @ 165.6 GiB). Under tail30
only ~half the expert mass is 2-bit, so expect roughly half the 35B effect
by coverage alone — plus E33's warning that tail promotion already absorbed
the worst 2-bit loss (gs32 diluted 5x). Even a 5x dilution of -11.40% is
-2.3%, which clears the bar; K1024 on the 2-bit region costs ~+5.6 GiB
(~148 GiB, still 17 GiB UNDER their comparator). This is the strongest ship
case the arc has produced — but it is arithmetic, not a measurement, until
M3 runs.

Fit cost is GPU-native (k-means distance = GEMM): 35B in 302s at K256,
~4x that at K1024 (assignment scales with K). 397B ~1h / ~4h respectively —
a different world from GPTQ's CPU-bound per-expert solves.

**What M0 did NOT do** (see VECTORQUANT_PLAN.md): the scored artifacts store
experts as bf16 (67G). No codes are saved, no small artifact exists, and
nothing can RUN one. M1 (Metal LUT-matmul kernel via `mx.fast.metal_kernel`,
present in mlx 0.32.0 — no fork needed) is the real remaining risk; decode-
at-load is NOT a fallback for the 397B (bf16 runtime footprint ~800 GiB).

## E35 M0b (08-14/15) — VQ AT 397B: the frontier moves. Both size classes won.

Fused one-pass fit+assemble (`vq_397b_fused.py`) over the shipped artifacts;
2-bit expert region replaced by a VQ reconstruction stored bf16 (QUALITY
PROXY — real bytes are analytic, no kernel yet). Structure/tail/routers
byte-identical. Every number below reproduced bit-identically x2 on the
single-box referee. Instrument validated: tail3x3 re-measured 3.1557,
matching its months-old record exactly.

  | artifact | GiB* | wikitext | code | eyes |
  |---|---|---|---|---|
  | **C  tail3x3+VQ K256**  | **111** | **2.8197** | **2.6504** | yes |
  | spicyneuron 2.6bit      | 120.6 | 3.1843 | 2.6667 | no |
  | ours tail3x3 (RTN)      | 122.3 | 3.1557 | 2.6542 | yes |
  | **A  tail3x3+VQ K1024** | **134** | **2.4328** | **2.6042** | yes |
  | ours tail30 (RTN)       | 142.5 | 2.3982 | 2.5928 | yes |
  | **B  tail30+VQ K1024**  | **148** | **2.3579** | **2.5961** | yes |
  | spicyneuron 3.5bit      | 165.6 | 2.3614 | 2.6005 | no |
  *GiB for VQ rows is the ANALYTIC size of the real format, not the proxy.

**Both classes won outright, on BOTH corpora:**
- 2.6bit class: C beats spicyneuron 2.6bit by 11.45% wikitext / 0.61% code
  at ~10 GiB LESS.
- 3bit class: B beats spicyneuron 3.5bit by 0.15% wikitext / 0.17% code at
  ~17 GiB less. (E29's "publish the efficiency claim, not a win" is now
  SUPERSEDED — it is a win.)
- A is the efficiency standout: matches their 3.5bit on code (2.6042 vs
  2.6005) at 31.6 GiB less.

**Dilution law confirmed quantitatively.** 35B gave -11.4% wikitext. At
397B: tail3x3 (57/60 layers still 2-bit) kept nearly all of it (-10.65% vs
its own baseline); tail30 (only 30/60) got -1.68%. VQ's payoff is
proportional to REMAINING 2-bit mass — the same law E33 found for group
size, now with a second data point at 7x dilution.

**Domain asymmetry, stated because it constrains the claim.** At K256 the
win is 11% wikitext but only 0.6% code; at K1024 code improves ~2.3%.
Codebook size matters MORE for code than for prose. Never publish the
wikitext number alone.

**K-curve on tail3x3 is steep and not flat at the top:** K256 2.8197 ->
K1024 2.4328 (23 GiB). Probe relerr keeps falling to K2048 (0.2311 ->
0.1947), so K2048/K4096 are unexplored upside. K512 (2.50 bpw = exactly
RTN's budget, ~122 GiB) is in flight as run D.

**Still proxy-only.** No codes are stored, no kernel exists, nothing can RUN
these. M1 (Metal LUT-matmul) remains the gating work for shipping.

## Open directions (v2 drawer)

- Grouped-perturbation calibration (probe families jointly; compounding becomes
  measurable). Then marginal-curve fitting / water-filling allocation (4-5 sweep
  points per layer, leave-one-out validated) becomes meaningful.
- Routing-frequency-weighted expert precision from real workload traces (demote cold
  experts, don't delete — graceful degradation).
- DWQ distillation polish pass after the static quant (bf16 teacher already on disk).
- Fused-rotation (QuaRot/SpinQuant-style) weight-only quant — composes with allocation;
  requires runtime validation on exo first (fused = plain weights, online ops = DOA).

## Reproduction

All artifacts + logs: `/Volumes/Thunderbay SSD/Exo Models/optiq-ab*` and `.{ab,hybrid,falsify}*.log`.
Sweep checkpoint (35B, 391 layers × {2,3,4}): `optiq-ab-35b/sensitivity_checkpoint.json`.
Env-gated patches + verify procedure: `~/Documents/AgenicAI/quantlab/README.md`.
Research-store chain: 1934a079 → f828e303 → e47cd33e → c91de858 → b293ff81 → 9fbc2733 → b56200c2 (E17) → a6f6f613 (E17b) → (this doc).

## Rig gotcha (08-14 evening) — a 39 MB/s USB link makes mx.load return ZEROS

**CORRECTED — the first version of this entry blamed exFAT mmap. Wrong.**

Symptoms on the M4's T7 (bf16 397B copy): `mx.load` served all-zero tensors
(VQ relerr exactly 1.0000), Metal command-buffer timeouts on the same reads,
and 18 MB/s file copies. Three false diagnoses in order: (1) exo's dev mlx
build — refuted, a pinned 0.32.0 venv failed identically; (2) corrupt copy —
refuted, RAW read() of the exact tensor byte ranges is 99.9% nonzero; (3)
exFAT mmap — refuted, an APFS disk image ON the T7 reproduced the zeros with
byte-perfect data inside it.

ROOT CAUSE: the drive was enumerating at **USB 2.0 speed — 39 MB/s measured**
(should be ~1000 MB/s), after being moved to a different port. An 8.6 GB
mmap'd tensor then needs ~220 s to fault in, which exceeds BOTH the Metal
watchdog and the device's own 30 s read timeout (`ioreg`: Read Time Out
Duration=30000) — so the GPU gets aborts or zero pages. One degraded link
explains all three symptoms.

Lessons: (a) `mx.load` mmaps, so GPU reads inherit the DEVICE's latency —
a slow link corrupts silently rather than erroring; (b) when data reads fine
via `read()` but wrong via mmap, suspect the transport, not the filesystem;
(c) always check link speed before blaming software — `ioreg -rc
IOBlockStorageDevice` + a raw read-rate measurement takes 30 seconds and
would have skipped all three wrong turns.

## M1 (08-15) — the VQ runtime EXISTS. Kernel correct, fast enough, artifact runs.

One day on the idle M4 while the M3 ground the E/F/G fit queue. All code in
vq_switch.py / patch_mlx_lm.py / vq_35b_codes.py / m1*_*.py; plan+results in
M1_KERNEL_PLAN.md. Highlights:

- **M1a** `mx.fast.metal_kernel` fused LUT-matmul == fp64 numpy decode to
  ~2e-7 (fp32 acc) on synthetic d4/K128 + d8/K16384 AND real 397B L0 codes.
- **M1b** decode shapes: best kernel (threadgroup cb+x, uchar4 code loads)
  = 0.66-0.88x gather_qmm. Bar (>=0.5x) met. Gotchas: the honest baseline
  is the mlx_lm sorted call shape (flat unsorted gather_qmm degrades ~80x
  and flatters rivals); K<=256 must ship uint8 codes.
- **M1c** prefill: decode-to-dense + PADDED batched GEMM **beats gather_qmm
  1.21-1.28x** at 8192 tok. Row-batched gather_mm (65k M=1 matvecs) is the
  trap: 0.43x. The feared simdgroup tile kernel was never needed.
- **M1d** (single box) `rotlab-35B-vqK256codes` = **10.1 GiB** artifact
  (62G bf16 source), coherent generation at **85 tok/s / 11 GB peak** on
  the M4. Loader hook walks attributes (tree_unflatten breaks on layers.0);
  fitter strips `.weight` so the MODULE path carries codes.
- **M1e** (35B) referee, bit-identical x2: codes 7.0378/3.0755 vs twin
  7.0313/3.0750 — +0.09%/+0.02%, the fp16-vs-bf16 arithmetic-path spread,
  not a bug. E35 quality claims unmoved.

Remaining to close M1: exo two-node integration, then the 397B codes
fitters (vq_397b_fused --emit-codes analogue) -> real 111 GiB C artifact.

## M2 (08-15) — THE REAL C ARTIFACT + the stored-vs-analytic bpw trap

`rotlab--397B-tail3x3-vqK256codes` = **110.8 GiB**, 27 shards, 171 expert
tensors, mean relerr 0.3156, fitted in 2h09 on the M4 (`vq_397b_codes.py`).
Self-contained: `model_file: model.py` + `vq_modules` geometry in config, so
STOCK mlx_lm loads it — the M4's mlx_lm was reverted to pristine before the
referee run specifically to prove that path at 397B scale.

**Referee (M4, stock mlx_lm, reproduced bit-identically x2): wikitext 2.7655**
— 1.9% BETTER than the bf16 proxy's 2.8197. Not noise and not a bug: the
shipping artifact stores scales in **fp16 (10 mantissa bits)** where the
proxy used **bf16 (7 bits)**, so the real thing is more accurate than its own
preview. (Same direction as the 35B refit: 7.1807 -> 7.0313.) vs spicyneuron
2.6bit 3.1843 @ 120.6 GiB = **13.2% better in 9.8 GiB less**.

**THE TRAP — analytic bpw assumes bit-packing we never implemented.**
`BPW = log2(K)/d + 16/group` is what the fitters print, but codes are stored
in WHOLE BYTES (uint8 for K<=256, uint16 above). Padding is invisible in the
formula:

  | run | geom | analytic | STORED | total |
  |---|---|---|---|---|
  | C | d4 K256   | 2.25 | **2.25** | 110.8 GiB (packs exactly) |
  | F | d4 K128   | 2.00 | **2.25** | 110.8 GiB (7 bits in 8) |
  | G | d8 K16384 | 2.00 | **2.25** | 110.8 GiB (14 bits in 16) |
  | E | d4 K2048  | 3.00 | **4.25** | ~181 GiB (11 bits in 16) |

Consequences:
- **F is STRICTLY DOMINATED by C** — identical size, half the codebook,
  worse quality. Disarmed 08-15 before it auto-started. It had no reason to
  exist as configured.
- **G is the SAME SIZE as C**, not smaller. Its case is quality only
  (relerr 0.3099 vs C's 0.3156, ~2%) for 38h + a kernel that does not exist
  (256 KB codebook cannot live in Apple's 32 KB threadgroup memory).
- **E as a real artifact would be 181 GiB**, bigger than B (148) which
  already beats spicyneuron 3.5bit. E's value is the axis question only.
- Byte-aligned geometries are quantised: d4/K<=256 = 2.25 bpw (110.8 GiB),
  d8/K<=256 = 1.25 bpw (~68 GiB), d8/K<=65536 = 2.25 GiB. **Nothing lands
  between 68 and 110.8 GiB without bit-packing** — that is the real lever for
  a sub-100 GiB artifact, and it is the same work for F and G.
- Free upgrade if G ever runs: at d=8 a uint16 holds 16 bits, so **K=65536
  is the same file size as K=16384** with a 4x bigger codebook — but 4x the
  assign cost (~169h, rejected).

NEXT: measure whether 110.8 GiB actually GENERATES on a 128 GB box
(`m2_fits_in_128.py`) — the referee STREAMS, so it proves quality, not
residency. If C fits, the accessibility claim is already won and F/G revert
to pure quality experiments.

## M2b (08-15) — RESIDENCY on a 128 GB box, and the lazy-graph prefill bug

**It fits.** `rotlab--397B-tail3x3-vqK256codes` (110.8 GiB) loads on a 128 GB
M4 Max in ~60s, **110.8 GiB resident, swap +0**. MLX warns 113.4 GB needed vs
a 120 GB recommended working set — real but not fatal.

**The ceiling was OUR BUG, not the model size.** First ladder (chunk=128,
pre-fix) showed prefill growing **3.35 MB/token** where this architecture's KV
cache is only **0.059 MB/token** — a 57x gap. Root cause found by arithmetic,
not guesswork: `_prefill` built the entire chunk loop as ONE LAZY MLX GRAPH,
so every chunk's decoded dense expert weights (2.0 GiB each for gate_up)
stayed alive until the final `concatenate`. Four chunks = 8 GiB live on a box
with ~9 GiB headroom.

FIX (commit 67a0473): `mx.eval()` each chunk's output inside the loop so each
`w` frees before the next decode; `_DECODE_CHUNK` auto-sized from free
headroom (env `SCOUT_VQ_DECODE_CHUNK`) instead of hardcoded 128.

  | context | peak (before) | tok/s (before) | peak (AFTER) | tok/s (AFTER) |
  |---|---|---|---|---|
  | 463    | 118.0 GiB | 2.2 | **112.4 GiB** | **2.9** |
  | 1,871  | 122.6 GiB | 1.2 (empty out) | (running) | |
  | 7,503  | 122.7 GiB | 0.4 | | |

**Lesson worth keeping: on a lazily-evaluated framework, a loop that only
evaluates at the end holds EVERY iteration's transients simultaneously.** The
peak-memory symptom looks exactly like "the model is too big for the machine"
and would have sent us chasing a smaller artifact for nothing.
Instrument-first ([[feedback_measure_the_metric_that_binds]]): the KV-cache
arithmetic is what proved the model size was innocent.

### M2b VERDICT — 397B runs on ONE 128 GB Mac at 30k context

Post-fix ladder (`m2_fits_in_128.py`, M4 Max 128 GB, stock mlx_lm, artifact's
own embedded model.py):

  | context | peak GiB | swap |
  |---|---|---|
  | 463 | 112.4 | -48 MB |
  | 1,871 | 115.1 | -48 MB |
  | 7,503 | 115.5 | -64 MB |
  | 15,016 | 115.8 | -72 MB |
  | **30,031** | **117.7** | **-128 MB** |

**A 65x increase in context cost 5.3 GiB** and swap FELL at every rung. Peak
stays under the 120 GiB recommended working set throughout. Pre-fix the same
model pinned at 122.6 GiB by 1,871 tokens with swap rising and empty output.

CLAIM (defensible): *Qwen3.5-397B-A17B, 110.8 GiB, runs on a single 128 GB
Apple Silicon Mac at 30k context* — with quality beating spicyneuron's 2.6bit
on BOTH corpora at 9.8 GiB less. As far as we can tell this is a first.
DO NOT quote the tok/s column above: it divides generated tokens by wall time
INCLUDING prefill, so it measures prefill at long prompts. Use
`m2_speed_split.py` (separate prefill / decode) for any published speed.

### M2c — honest throughput (m2_speed_split.py, chunked prefill, real corpus)

  | context | prefill tok/s | DECODE tok/s | peak GiB |
  |---|---|---|---|
  | 512 | 48.2 | 22.2 | 112.4 |
  | 2,048 | 51.5 | 21.9 | 112.8 |
  | 8,192 | 40.7 | 18.8 | 113.0 |
  | 14,000 | 38.8 | 20.1 | 113.1 |

Decode is FLAT at ~19-22 tok/s through 14k context, peak ~113 GiB. The
"2.9 -> 0.4 tok/s collapse" in m2_fits_in_128's table was measurement
artifact (generated/wall-time-incl-prefill). Second measurement trap the
same day: a MONOLITHIC prefill forward pass OOMs Metal at 8k on the nearly-
full box (kIOGPUCommandBufferCallbackErrorOutOfMemory) while the chunked
path mlx_lm actually uses runs 30k fine — benchmark the path users take.

**FULL PUBLISHABLE CLAIM for C:** Qwen3.5-397B-A17B, 110.8 GiB, single
128 GB Apple Silicon Mac: 30k context, ~20 tok/s decode, ~40-50 tok/s
prefill, wikitext 2.7655 / code 2.6383 (beats spicyneuron 2.6bit on both
at 9.8 GiB less), stock mlx-lm, zero patches.

## E36 (08-15) — d8 probe: the win is PER-TENSOR, not per-format (mixed geometry)

Same-methodology ladder on real 397B L0 tensors (M4, m1a_emit_codes,
fp16-codebook relerr), d4 K256 as the C anchor:

  | geometry | bpw(packed) | down_proj | gate_up |
  |---|---|---|---|
  | d4 K256 (C) | 2.25 | 0.1930 | 0.4161 |
  | d8 K256    | 1.25 | 0.3413 | 0.6437 |
  | d8 K1024   | 1.50 | 0.2335 | 0.5939 |
  | d8 K4096   | 1.75 | **0.1794** | 0.5185 |

- **68 GiB (d8 K256, byte-aligned, no packing) is DEAD**: +77%/+55% error.
- **down_proj prefers d8**: K4096 BEATS the C anchor at 22% fewer bits.
- **gate_up never recovers at d8** — even K4096 sits far below its d4
  anchor. The morning sweep's "d8 ~2%/bit better" held only for down_proj
  (1/3 of expert mass). Uniform-d8 G spends its 38h applying the wrong
  geometry to 2/3 of the weights.
- **MIXED GEOMETRY is already format-legal**: vq_modules carries dim/K per
  module; the runtime builds each module independently. Frontier candidates:
    gate/up d4K256 + down d8K4096  -> ~103.6 GiB, quality >= C everywhere
    gate/up d4K1024 + down d8K4096 -> ~118 GiB, ~A-grade gate/up at C size
  Both need 12-bit packing + the L2-resident d8 kernel (K4096 codebook =
  64 KB > 32 KB threadgroup). d8 K1024 (16 KB) fits threadgroup if a
  cheaper mixed point is wanted (down 0.2335, slightly below C).

## E (08-15 overnight) — d4 K2048 on tail3x3: the 3-bit-class number, and the domain asymmetry AGAIN

`zzvq-tail3x3-K2048` — bf16 quality proxy, 698.6 GiB, fit 36,451s on the M3
(171 tensors, **mean relerr 0.1952** vs C's 0.3156, and below the 35B K1024
reference of 0.222). Referee: stock mlx_lm, single-box, **all four runs
bit-identical** (wikitext nll 6919.536 x2, code 7828.8672 x2).

  | corpus | E (K2048) | C proxy | margin |
  |---|---|---|---|
  | wikitext | **2.3272** | 2.8197 | 17.5% |
  | code | **2.6004** | 2.6504 | **1.9%** |

**The asymmetry is the finding, not the headline.** Relerr improved 38% and
wikitext followed (17.5%), but code moved 1.9% — the same shape as K256's
11%-vs-0.6%. Codebook size buys wikitext far more than it buys code on this
family. A publish that quoted wikitext alone would misrepresent E by ~9x.

Where E actually stands, against the outside comparator (spicyneuron 3.5bit,
165.6 GiB, 2.3614 / 2.6005): E **wins wikitext by 1.4% and ties code**
(2.6004 vs 2.6005) at **142.8 GiB packed** — a 3-bit-class win at 22.8 GiB
less. Against our own B proxy (~148 GiB) it is a wash: E +1.3% wikitext,
B +0.17% code.

**E does NOT displace C.** Paying 32 GiB over C — and leaving single-128 GB
territory — to buy 1.9% on code is the wrong trade for the accessibility
goal. C stays the daily driver; E is the heavy-hitter/cluster artifact.

**E is a PROXY — three things stand between it and shipping:**
1. a real codes fit at K2048 (does not exist yet; K>256 ⇒ uint16 codes),
2. bit-packing (11-bit fields) — unpacked E is **196.3 GiB**, not 142.8,
3. vision, if it is to be the differentiator (see below).

## Vision (08-15) — mlx_lm DISCARDS it for this arch; "with vision" is an mlx-vlm project

Checked while scoping a vision graft. Facts, measured:

- Source `Qwen--Qwen3.5-397B-A17B-bf16` carries **333 vision tensors in 2
  shards = 0.85 GiB bf16**. Cheap to copy.
- Shipped C carries **0** of them — and so does the spicymirror comparator
  (0 / 2212). Nobody in this class ships vision.
- Cause, in stock `mlx_lm/models/qwen3_5_moe.py` `Model.sanitize()`:
      if key.startswith("vision_tower") or key.startswith("model.visual"):
          continue
  The loader drops them by construction. `mlx_lm` is text-only for this
  architecture; vision lives in **mlx-vlm**, a separate package with its own
  loader and arch registry.

**Consequence:** grafting the tensors puts bytes in the file that `mlx_lm`
silently discards — the capability would not exist. Shipping working vision
means supporting mlx-vlm, including a `model_file`-equivalent VQ shim for
its model class. That is an unscoped investigation, NOT a copy job.

**Retraction:** C's config declaring `vision_config` with no vision weights
was called a publication-blocking defect earlier in the session. It is not —
it is the normal shape of every mlx_lm conversion of this model, comparator
included. The error was inferring a loader failure without reading
`sanitize()`.

### The exo backdoor — vision ALREADY WORKS there, and the graft is NOT inert

Second correction, same session: "grafting the tensors is inert" is true only
for `mlx_lm`. **exo never uses mlx_lm's loader for vision.** It carries its
own path, `exo/src/exo/worker/engines/mlx/vision.py`:

- `model_cards.py` builds a `VisionCardConfig` for ANY model whose config has
  both `vision_config` and `image_token_id`;
- `weights_repo` is set to the model's OWN id, and `vision.py` resolves it
  via `build_model_path(...)` — i.e. exo reads the vision tower out of the
  SAME directory as the language model.

Confirmed by Noah 08-15: **the 397B has been made to see, through exo only** —
never through mlx_vlm, which we have never patched. So the capability is real
and the mechanism is exo's, not ours.

**Consequence for shipping — vision splits three ways, by consumer:**

  | consumer | what it takes | patches |
  |---|---|---|
  | **exo** | graft the 333 tensors (0.85 GiB) into the artifact | **none** |
  | `mlx_lm` | nothing works — `sanitize()` drops the keys | text-only, period |
  | `mlx-vlm` | subclass its `Model` + a `model_file` hook upstream | PR (~20 lines) |

mlx-vlm feasibility, checked: it already ships `mlx_vlm/models/qwen3_5_moe/`,
and its `language.py` imports `SwitchGLU` **from `mlx_lm.models.switch_layers`**
— the exact class `VQSwitchLinear` replaces, so the VQ runtime transplants
without a rewrite. The only true blocker is that `get_model_and_args()`
resolves classes purely by `model_type` via `importlib`, with NO `model_file`
hook (that mechanism is mlx_lm-only). Prove any of this at 35B first — that
artifact is 10 GiB and surfaces loader surprises in minutes, not hours.

**Standing gotcha, now measured:** all fifteen local 397B artifacts carry
**0 vision tensors** — C, the new mixed build, tail30, tail3x3, and both
spicyneuron comparators. Four of them (tail30, tail3x3, C, mixed) declare
`vision_config` WITHOUT the weights, which is precisely the "serves BLIND
while looking healthy" shape the debut plan warns about. Whatever carried
vision in the working exo run is not in the current artifact set, so a ship
that promises vision MUST graft explicitly and be verified on an image.

## E37 (08-15) — mixed geometry FALSIFIED: E36's d8 win was a LAYER-0 ARTIFACT

Built the E36 frontier candidate for real: `rotlab--397B-tail3x3-vqMixed-d4K256-d8K4096`,
gate/up d4 K256 + down d8 K4096, 171 tensors, 110.8 GiB, 3h34m on the M4
(`vq_397b_codes.py --geom`). It loads under STOCK mlx_lm, scores deterministically,
and **loses on both corpora**:

  | corpus | mixed | C (bar) | delta |
  |---|---|---|---|
  | wikitext | **3.0819** | 2.7655 | **+11.4% worse** |
  | code | **2.6820** | 2.6383 | **+1.66% worse** |

Both x2 bit-identical (total_nll equal to 4 decimals, DETERMINISM spread 0.00%),
M3 referee, stock unpatched mlx_lm. Mean fit relerr 0.3474 vs C's 0.3156.

**ROOT CAUSE — the probe was taken at the one layer where the sign is positive.**
E36 fitted layer 0 only. Same methodology re-run at layer 40 inverts the result:

  | down_proj | d4 K256 | d8 K4096 | d8 vs d4 |
  |---|---|---|---|
  | layer 0 (E36's probe) | 0.1930 | **0.1794** | -7.0% (d8 better) |
  | layer 40 | **0.3117** | 0.4142 | **+32.9% (d8 WORSE)** |

The full fit log shows why: d8 down_proj relerr climbs monotonically 0.1793 (L0)
-> 0.4148 (L56) and plateaus. d4 climbs too (it is a property of the TENSORS, not
the geometry — d4 K512 goes 0.1566 -> 0.2645), but **d8 degrades far harder**:
2.31x from L0 to plateau vs d4's 1.69x. Layer 0 is anomalously easy to VQ, and
d8's extra dimensions only pay off while the subvector distribution is that benign.

**THE LESSON, and it is bigger than this experiment: a single-layer probe does not
generalize across depth.** E36's whole premise ("down_proj PREFERS d=8, beats the
anchor at 22% fewer bits") was one number from one tensor. Any future geometry
decision must be probed at BOTH a shallow and a deep layer before a 3.5h fit is
spent on it — the two-point probe costs ~4 minutes (24s at K256, 192s at K4096).

Second, quieter finding: **the domain asymmetry reversed direction here.** Every
prior entry warned that wikitext-only publishing could hide a code regression;
this time code (+1.66%) nearly absorbed a loss that wikitext (+11.4%) reports
loudly. Scoring both corpora is what makes either number trustworthy.

Consequences:
- **Mixed geometry is CLOSED as a quality lever.** Not the mechanism — the
  mechanism works perfectly (see below) — the d8-for-down_proj bet itself.
- **The frontier plan's step 2 (12-bit packing -> ~103.7 GiB) is dead as written**:
  it existed to shrink d8 K4096 down_proj, which is quality-negative. Packing a
  worse artifact smaller is not the trade we want.
- **C remains champion**, unchallenged: 2.7655 / 2.6383 at 110.8 GiB.
- Artifact DELETED after scoring (111 GiB, confirmed to do nothing for us).

**What survives and is worth keeping — the runtime, which is now strictly more
capable than C needs:**
- d8 Metal kernels landed and are FAST (M1 ladder green, see M1_KERNEL_PLAN):
  L2-resident 64 KB codebook holds at **0.86-1.39x gather_qmm** decode — better
  than the shipped d4 path (0.84-0.93x) — and 1.19x prefill.
- **Mixed geometry is proven end-to-end**: 114 d4K256 + 57 d8K4096 modules in one
  checkpoint, per-module dispatch, loading and scoring under stock mlx_lm with
  zero patches. The MACHINERY for any future per-tensor geometry decision exists
  and is referee-validated. Only this particular geometry choice was wrong.
- `vq_397b_codes.py --geom` plumbs per-projection geometry through fit + config.

## E38 (08-15) — F (d4 K128) revisited: PASSES the depth test, but the magnitude is the problem

E37's fix applied prospectively. Before spending another 3.5h fit, ran the
two-point probe (L0 + L40, both projections) on the one geometry that packing
could still help. **Packing is a no-op for C** — d4 K256 codes are exactly 8
bits in a uint8, nothing to reclaim — so the sub-100 GiB question rests
entirely on a sub-byte geometry. Measured byte split of C:

  | component | GiB | share |
  |---|---|---|
  | VQ codes | 85.5 | 77.2% |
  | VQ scales (fp16) | 10.7 | 9.6% |
  | codebooks | 0.0003 | ~0% |
  | structure (tail3x3, attn, embeddings) | 14.6 | 13.2% |

d4 K128 packed to 7 bits reclaims 1/8 of the codes: **110.8 -> 100.1 GiB**.
That also un-dominates F: M2 killed it as "same size, worse quality"; packed
it is 10.7 GiB SMALLER, so it becomes a real trade instead of a strict loss.

Probe (m1a_emit_codes, fp16-codebook relerr, identical methodology):

  | tensor | d4 K256 (C) | d4 K128 (F) | F/C |
  |---|---|---|---|
  | L0 down | 0.1930 | 0.2566 | 1.330 |
  | L0 gate_up | 0.4161 | 0.4500 | 1.081 |
  | L40 down | 0.3117 | 0.3686 | **1.182** |
  | L40 gate_up | 0.3119 | 0.3691 | **1.183** |

**F passes the test d8 failed**: the penalty does not widen with depth, it
CONVERGES to a steady ~1.18x on both projections. Note also how tightly the
two projections converge at L40 (0.3117/0.3119 and 0.3686/0.3691) — layer 0
is the outlier for every geometry we have measured, which is E37's lesson
restated from a second direction.

**But steady is not the same as affordable.** C's measured mean over all 171
tensors is 0.3156; at a flat 1.18x, F projects to ~0.372 — WORSE than the
E37 mixed artifact's 0.3474, and that one lost wikitext by 11.4%. The single
calibration point we own says +10.1% relative relerr cost +11.4% wikitext PPL.
F is asking for +18%.

That extrapolation is one point and the error DISTRIBUTION differs (E37
concentrated its damage in down_proj at depth; F spreads uniformly), so F is
not dead on arithmetic — but it does not justify a 3.5h 397B fit on those
odds. Calibrating at 35B instead (minutes, not hours): fit K128 against the
existing `rotlab-35B-vqK256codes` (7.0378 wiki / 3.0755 code) to turn
"relerr says probably bad" into a real PPL delta for the K256->K128 halving.
`vq_35b_codes.py --out-proxy` made optional for this (the twin costs ~6x the
codes artifact in disk and the M1e runtime-vs-twin check is long closed).

## Probe (08-15) — K-scaling of codes-fit time: E is a ~4 h fit, not 6-10 h

`probe_k_fit_time.py` (M3, free GPU; layer 40 down_proj, real bf16 tensor,
the fitter's kmeans + assign loops copied shape-for-shape):

  | K | kmeans | assign (64 experts) | full tensor |
  |---|---|---|---|
  | 256 | 0.9s | 1.0s | 8.1s |
  | 2048 | 6.7s | 7.4s | 59.2s |

**K2048 costs 7.3x K256 in compute** — sub-linear in K (8x nominal), the GEMM
amortising. Extrapolated compute for all 171 tensors: **K256 ~0.3 h,
K2048 ~2.0 h.**

Calibrate against the real C fit (K256, 2h09 on the M4): compute is only
~0.3 h of that, so **~1.8 h was I/O** (390 GB source read + artifact write).
I/O is K-independent, so:

    E codes fit  ~=  2.0 h compute + ~1.8 h I/O  ~=  **4 h**

Independently corroborated by the chip's mixed fit (down=d8K4096, heavier
compute than K2048): **3h34m measured**. Same ballpark, from a different
geometry — the model holds.

**Corrects the 6-10 h estimate given earlier from arithmetic alone.** The
error was treating the whole of C's 2h09 as K-scaling when 85% of it is I/O
that does not scale with K at all. Cost of the probe: ~2 minutes.

### E38b — the 35B calibration: F is DEAD, and with it the sub-100 GiB route

Fitted `rotlab-35B-vqK128codes` from the same source and script as the
existing K256 artifact, regenerated BOTH artifacts' `model.py` from the
current runtime so the only variable is codebook size, and scored both on
both corpora x2 (M4, all runs bit-identical, DETERMINISM spread 0.00%):

  | 35B | wikitext | code |
  |---|---|---|
  | K256 | 7.0378 | 3.0755 |
  | K128 | 7.7555 | 3.2408 |
  | **penalty** | **+10.20%** | **+5.37%** |

(K256 reproduced its historical M1e score 7.0378 / 3.0755 EXACTLY through the
new runtime — incidental proof that the d8 kernel work caused zero regression
on the d4 path. The K128 artifact also weighed **10.1 GiB, identical to
K256** — the stored-bpw trap again, at 35B.)

Projected onto the 397B artifacts:

  | artifact | size | wikitext | code |
  |---|---|---|---|
  | C (shipped) | 110.8 GiB | **2.7655** | **2.6383** |
  | F (packed, projected) | 100.1 GiB | 3.0475 | 2.7801 |
  | spicyneuron 2.6bit | 120.6 GiB | 3.1843 | 2.6667 |

**F beats spicyneuron on wikitext (-4.30%) and LOSES on code (+4.25%).** That
forfeits the only claim worth publishing — C beats the community quant on
BOTH corpora — in exchange for 10.7 GiB. And the verdict is robust to
transfer error: F would need a code penalty under **1.08%** to win there,
5x smaller than measured; even halving the observed penalty still loses code.

**VERDICT: the sub-100 GiB route is closed.** Not because bit-packing fails —
it works and would deliver the 10.7 GiB — but because the only geometry that
can USE packing (sub-byte codes = K128) costs ~10% wikitext to get there.
C at 110.8 GiB remains the artifact, unchallenged on both corpora.

Cost of reaching this: ~25 minutes (two-point probe + 35B calibration) against
the 3.5h fit + referee it replaced. This is what E37's lesson is worth in
practice — the cheap experiment answered the expensive question.
