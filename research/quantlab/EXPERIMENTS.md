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

> **SUPERSEDED same night (see "F REAL" below).** Noah reweighed the
> trade-off: losing code by ~1-4% at 20.5 GiB smaller is still an
> ACCESSIBILITY win ("someone can run it who couldn't before"), so F was
> fit, refereed and packed on 08-16 — real numbers 3.1706 / 2.6988 at
> 100.1 GiB. Two notes on this section with hindsight: (1) its VERDICT
> conflated "F loses code" (true, -1.20% measured) with "route closed"
> (a value judgment that was Noah's to make, and he made it the other
> way); (2) its projected wikitext 3.0475 was still 4% optimistic vs the
> real 3.1706 — even the calibrated transfer missed. The 35B measurement
> itself stands and remains the best K256->K128 calibration point.

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

## Probe (08-15) — K128 shallow AND deep: depth-STABLE, uniformly ~+18% relerr

The E37-mandated two-point-plus probe, applied to the last sub-100 candidate
(d4 K128 packed = 100.1 GiB). Layers 0 / 40 / 56, both projections, K128 vs
the C geometry K256 on the same tensors (`probe_k_fit_time.py`, M3):

  | layer, proj | K128 | K256 | penalty |
  |---|---|---|---|
  | L0 down | 0.2523 | 0.1876 | +34% |
  | L0 gate_up | 0.4534 | 0.4306 | +5% |
  | L40 down | 0.3689 | 0.3118 | +18% |
  | L40 gate_up | 0.3692 | 0.3118 | +18% |
  | L56 down | 0.3674 | 0.3136 | +17% |
  | L56 gate_up | 0.3736 | 0.3186 | +17% |

**No d8-style depth pathology**: the penalty is ~17-18% at every deep layer
and BOTH projections — flat, not climbing. (L0 is again the outlier in both
directions, confirming E37's "layer 0 is unrepresentative".) K128 is a
legitimate geometry, just uniformly coarser.

Quality projection (calibrate on E: -38% relerr bought code -1.9%, wiki
-17.5%): +18% relerr on C's 0.3156 → ~0.373, projecting code ~+0.9% and
wikitext ~+8% vs C. That puts K128 at roughly **spicy-2.6bit parity on code**
(C's margin there is only 1.07%) while still clearly ahead on wikitext, at
**20 GiB less than their 120.6**. A real but marginal claim: "ties on code,
wins wikitext, 100 GiB". Whether that is worth a fit + referee is a
judgment call, not a science question — the fit is cheap (~2.5 h incl. I/O)
but only AFTER bit-packing exists, since 7-bit codes don't pack until then.

## Packing VALIDATED (08-15/16) — 7-bit codes, real model, bit-identical

Bit-packing is built and proven end-to-end. `vq_pack.py` (format + packer),
packed Metal kernels in `vq_switch.py`, `pack_artifact.py` (converter),
and the `add_model_file.py` shim now carry `pack_bits`/`in_features`.

**35B K128 (7-bit), M4, stock mlx_lm via the bundled model.py:**

  | | wikitext | code | size |
  |---|---|---|---|
  | unpacked | 7.7654 | 3.2463 | 10.1 GiB |
  | **packed** | **7.7654** | **3.2463** | **9.2 GiB (0.908x)** |

nll agrees to 4 decimals on BOTH corpora (16790.9817 / 9646.1924), each
reproduced x2. This is an EQUALITY gate, not a tolerance: packing changes
representation, never values, so any drift would mean a decode bug.

Why 0.908x and not 0.875x: only CODES shrink (8->7 bits). Codebooks, scales
and the entire non-expert region are untouched. The 397B's expert share is
larger, which is why F gains more there (110.8 -> 100.1 GiB).

Also validated at 8 bits on the 35B K256 artifact (M3): 15985.0024 / 7.0378
identical packed vs unpacked — byte-neutral in size, but it exercises the
same shim/loader path.

**A kernel ceiling found while gating this, unrelated to packing:**
`_SRC_FUSED` caches the codebook as float4 (16 B/entry), so K2048 needs
32 KB before x is cached and Metal REFUSES to load the kernel (36864 >
32768). **E could not have decoded in ANY format.** `_SRC_FUSED_D4_BIGK`
caches codebook+x as half4 — value-identical (both are fp16 on disk),
24 KB total. Caught the night E was being fit; without it the artifact
would have failed at first token.

Two silent bugs the equality gate caught, both of which produce plausible
wrong numbers rather than errors:
1. **`#if BITS > 0` does not work in MLX kernels.** Template params are C++
   template parameters, NOT preprocessor defines — the preprocessor saw BITS
   undefined, took the unpacked branch, and read packed words as raw codes.
2. Dispatch routed packed tensors to the unpacked big-K kernel.

Operational note: `vq_35b_codes.py` does NOT embed model.py — that is a
separate `add_model_file.py` step. Omitting it yields "Received 360
parameters not in model" (120 modules x 3 tensors), because no
VQSwitchLinear is ever constructed.

## E REAL ARTIFACT (08-16) — the 3bit class is won. Both classes now held.

`rotlab--397B-tail3x3-vqK2048codes` — real codes, **196.3 GiB unpacked /
142.9 GiB packed**, fit 15568s (4h19m) on the M3, 171 tensors, mean relerr
**0.1936** (proxy 0.1952). Referee: stock mlx_lm, single-box streaming, both
corpora, bit-identical x2.

  | | E real (142.9 packed) | spicy 3.5bit (165.6) | C (110.8) |
  |---|---|---|---|
  | wikitext | **2.3519** | 2.3614 | 2.7655 |
  | code | **2.5987** | 2.6005 | 2.6383 |

**E beats spicyneuron 3.5bit on both corpora at 22.7 GiB less.** With C
holding the 2.6bit class, BOTH debut classes are won with real runnable
artifacts.

**Publish it honestly: wikitext +0.40%, code +0.07%.** The code figure is a
TIE, not a win — the claim is "matches on code, wins wikitext, at 23 GiB
less". The size advantage is the headline; a 0.07% margin is exactly what
the both-corpora rule exists to stop us overselling.

**Proxy fidelity is NOT one-directional** — worth knowing before trusting any
future proxy:
  | | wikitext | code |
  |---|---|---|
  | C: proxy -> real | 2.8197 -> 2.7655 (-1.9%) | 2.6504 -> 2.6383 (-0.5%) |
  | E: proxy -> real | 2.3272 -> 2.3519 (**+1.1%**) | 2.6004 -> 2.5987 (-0.1%) |
C's real artifact beat its proxy on both; E's real artifact came in WORSE on
wikitext and better on code. So "real beats proxy (fp16 vs bf16 scales)" was
an over-generalisation from a single data point. Treat proxies as ±1-2%
indicators per corpus, never as a decimal-accurate prediction. E's relerr
improved (0.1936 vs 0.1952) while its wikitext ppl got worse — relerr and
perplexity do not move in lockstep at this resolution.

**The half4 kernel fix is validated at 397B scale**: every K2048 module
decodes through `_SRC_FUSED_D4_BIGK`, and it produced coherent, reproducible
numbers. Without it this artifact could not have run at all.

## F REAL (08-16) — the 100 GiB point exists, and BOTH ppl projections were wrong

`rotlab--397B-tail3x3-vqK128codes` — **110.8 GiB unpacked / 100.1 packed**,
fit in only **1577s (26 min)**, mean relerr **0.3698**. Referee x2 both
corpora, bit-identical.

  | | F (100.1 packed) | spicy 2.6bit (120.6) | C (110.8) |
  |---|---|---|---|
  | wikitext | **3.1706** | 3.1843 | 2.7655 |
  | code | 2.6988 | **2.6667** | 2.6383 |

F wins wikitext by 0.43%, **LOSES code by 1.20%**, at **20.5 GiB less** than
its comparator. That is the honest claim, and it is still a real win on
Noah's terms (08-15): "slightly worse at -20GB means someone can run it who
couldn't before."

**Two forecasting failures worth carrying:**
1. **Both ppl projections missed, mine worse.** Predicted wikitext: mine
   ~2.843 (relerr-calibrated), the other session's 3.0475. Actual **3.1706**
   — 11% and 4% optimistic respectively. Yet the RELERR prediction was
   nearly exact (+17-18% probed, +17.2% actual, 0.3156 -> 0.3698). So the
   relerr model transfers; **relerr -> perplexity does not.** Same lesson E
   taught in the opposite direction the same night. Do not publish or plan
   around a projected ppl: fit it and referee it.
2. **F's fit ETA was 6x off** — estimated ~2.5 h, took 26 min. K128 compute
   is ~16x cheaper than K2048 and it writes 86 GiB less, but that does not
   cover the gap: the I/O share in the earlier calibration was overstated
   for this shape. Fit cost is not a simple compute + fixed-I/O model.

## THE LINEUP (08-16) — three real artifacts, all measured, all reproduced x2

  | | packed GiB | wikitext | code | vs its comparator |
  |---|---|---|---|---|
  | **F** | 100.1 | 3.1706 | 2.6988 | +0.43% / -1.20% vs 2.6bit @120.6 |
  | **C** | 110.8 | 2.7655 | 2.6383 | +13.2% / +1.07% vs 2.6bit @120.6 |
  | **E** | 142.9 | 2.3519 | 2.5987 | +0.40% / +0.07% vs 3.5bit @165.6 |

C wins its class outright on both corpora. E takes the 3bit class (code is a
TIE at 0.07% — publish it as "matches on code"). F trades 1.2% of code for
20.5 GiB and is the accessibility artifact. Every number: stock mlx_lm,
single-box streaming referee, bit-identical reruns.

## PACKED AT 397B SCALE (08-16) — both artifacts, both corpora, all exact

Ran `pack_artifact.py` on the real E and F artifacts and refereed the packed
copies x2 on both corpora. Every score is IDENTICAL to the unpacked twin:

  | artifact | unpacked | packed | ratio | wikitext | code |
  |---|---|---|---|---|---|
  | E (11-bit) | 196.3 GiB | **142.8** | 0.728x | 2.3519 = 2.3519 | 2.5987 = 2.5987 |
  | F (7-bit) | 110.8 GiB | **100.1** | 0.904x | 3.1706 = 3.1706 | 2.6988 = 2.6988 |

nll agrees to 4 decimals in all 8 packed runs (E 7006.0744 / 7823.3298,
F 9452.9411 / 8133.0694). Packing is now validated at **7, 8 and 11 bits**,
on both the 35B and the full 397B — the sizes are real, not computed, and
quality is provably untouched.

Predicted vs actual packed size: E 142.9 predicted / 142.8 actual, F 100.1 /
100.1. The shape-derived size table in `vq_pack.py` is trustworthy.

**SHIPPABLE STATE — three artifacts, stock mlx_lm, zero patches:**
  - `rotlab--397B-tail3x3-vqK128codes-packed`  100.1 GiB
  - `rotlab--397B-tail3x3-vqK256codes`         110.8 GiB (C, byte-aligned)
  - `rotlab--397B-tail3x3-vqK2048codes-packed` 142.8 GiB
Remaining before publish: vision graft (exo backdoor works; mlx-vlm needs a
model_file PR), model cards for F and E, and a decode-speed measurement of
the packed path on a quiet box (the packed-vs-unpacked timings collected so
far were all taken under contention and are NOT comparable).

## Vision graft SHIPPED + mlx-vlm PR drafted (08-16)

`graft_vision.py`: copies the 333 `model.visual.*` tensors (0.85 GiB bf16)
from the source model into a `model-vision-graft.safetensors` shard and
registers them in the artifact index. Run against ALL THREE artifacts —
C, E-packed, F-packed — each now vision-complete on disk, configs already
carried `vision_config` + `image_token_id`.

Regression proof: C re-refereed AFTER the graft under mlx_lm — **2.7655,
nll 8332.9789, exact match**. The graft is invisible to mlx_lm (sanitize
drops visual keys), works today under exo (VisionCardConfig reads the tower
from the artifact's own dir), and awaits the model_file hook under mlx-vlm.

**mlx-vlm PR branch ready** (scratchpad clone, branch `model-file-loading`,
19-line diff to `load_model` in `mlx_vlm/utils.py`): mirrors mlx_lm's
`model_file` mechanism verbatim — config.json names a Python file inside the
checkpoint, `importlib.util.spec_from_file_location` loads it as the arch
module, registry path untouched for every existing model. Hook mechanism
verified against a synthetic checkpoint. NOT yet pushed/opened — Noah's
GitHub, Noah's call.

## FINAL TABLE (08-16) — the release lineup, everything measured

  | | F | C | E |
  |---|---|---|---|
  | geometry | d4 K128, 7-bit packed | d4 K256, byte-aligned | d4 K2048, 11-bit packed |
  | **size** | **100.1 GiB** | **110.8 GiB** | **142.8 GiB** |
  | wikitext ppl | 3.1706 | 2.7655 | 2.3519 |
  | code ppl | 2.6988 | 2.6383 | 2.5987 |
  | vs comparator | spicy 2.6bit @120.6: wiki **-0.43%**, code +1.20% | spicy 2.6bit @120.6: wiki **-13.2%**, code **-1.07%** | spicy 3.5bit @165.6: wiki **-0.40%**, code -0.07% (tie) |
  | decode tok/s (M4 Max 128GB) | **20.4-20.9** | 19-22 | n/a single-box |
  | prefill tok/s | 42-48 | 40-50 | n/a single-box |
  | peak GiB @8k ctx | **103.0** | ~115.5 | — |
  | load time | 116 s | ~60 s | — |
  | vision tower | grafted | grafted | grafted |
  | runs on | 128 GB Mac, roomy | 128 GB Mac, tight | ≥192 GB or cluster |

Notes for the record:
- **F is NOT faster than C, and should not be.** Decode is bound by
  bytes-read-PER-TOKEN, and this is an A17B MoE: both read the same routed
  experts, F's codes at 7/8 width = ~8% best-case traffic cut, absorbed by
  K-independent costs. The 20 GB buys RESIDENCY (peak 103 vs ~115.5), not
  tok/s. Size -> headroom; active-bytes -> speed.
- E has no single-box speed number because no box we own holds 142.8 GiB;
  its speed story is exo cluster serving, a different measurement.
- All ppl: stock mlx_lm, single-box streaming referee, bit-identical x2,
  identical packed-vs-unpacked verified per artifact.

## mlx-vlm model_file PR OPENED (08-16)

https://github.com/Blaizzy/mlx-vlm/pull/1926 — branch `model-file-loading`
on the noahzelezny fork, authored noahzelezny@thedrainflorist.com. 20-line
hook in `load_model` + `tests/test_model_file_loading.py` (3 tests: end-to-
end synthetic checkpoint, missing-file error, registry-path-untouched).
Contract note discovered writing the test: a `model_file` module's
ModelConfig should NOT expose `text_config`/`vision_config` attributes
unless it also supplies TextConfig/VisionConfig classes —
`update_module_configs` keys off those attrs. Our future VLM model.py must
handle this (our config DOES carry vision_config).

## exo 2-node verification (08-16) — and the sharding bug it caught

Renamed to the community bpw convention (overall bits/weight = total bytes /
total params, the exl2 style; size-in-name is not a convention anyone uses):
`Qwen3.5-397B-A17B-VQ-2.2bpw` / `-2.4bpw` / `-3.1bpw`.

**THE BUG THAT WOULD HAVE SHIPPED.** E stalled on the ring at 59/60 layers,
M4 at 66 GiB vs M3 at 16.5 GiB, no traceback, runners eventually vanishing.
Noah's control experiment — spicyneuron 2.6bit places fine on the same ring —
proved exo healthy and pointed at our artifact. Cause, found by READING
`auto_parallel._sharded_to_all` rather than by another 10-minute placement:
it shards `codes` IN PLACE along the last axis AFTER module construction.
The unpacked path survived because `input_dims` was already a property over
live shapes; **the packed path cached `in_features`**, so post-shard it
described a tensor twice the size it held. Single-box never shards — which is
exactly why F-packed ran all morning on the M4 and this hid until the cluster.
Fix: derive dims from current tensors, never cache (`vq_switch.py`), then
REGENERATE every artifact's bundled `model.py` (they embed the runtime).

**2.2bpw VERIFIED SERVING 2-node tensor-sharded** — both runners Ready,
coherent structured reasoning, `finish_reason: stop`, correct answers.

Test-design trap worth keeping: the first two runs looked like garbage
("Definition", then "."). Not corruption — **max_tokens truncation on a
thinking model**, which the model card warns about and the test author (me)
ignored. Budget 3000+ tokens for a verification prompt. `/no_think` did NOT
suppress reasoning on this model; token budget is the only real lever.

### All three VERIFIED on the 2-node ring (08-16)

  | artifact | size | format | placement | generation |
  |---|---|---|---|---|
  | VQ-2.2bpw | 100.1 GiB | 7-bit packed | (post-reset) | coherent, finish=stop |
  | VQ-2.4bpw | 110.8 GiB | unpacked | 80s | coherent, finish=stop |
  | VQ-3.1bpw | 142.8 GiB | 11-bit packed | 336s | coherent, finish=stop |

3.1bpw — the artifact that stalled at 59/60 this morning — now places and
serves, and answered "what is vector quantization?" correctly in two
sentences (1189 tokens, 5282 chars of reasoning). The cached-in_features fix
is confirmed on the hardest case.

**Sequencing gotcha, learned the hard way:** DELETE-then-place wedges exo.
Deleting an instance leaves runners in RunnerShuttingDown indefinitely, and
the next placement spawns runners that die instantly, leaving ghost
"RunnerLoading" entries whose layer counts are FICTION — the state API
reports progress for processes that do not exist. Always verify against
`ps` on both nodes, never the API alone. `exo_verify_artifact.sh` now does a
full `exo-reset.sh --restart` between artifacts (~60s) instead.

Env gotcha for any HF automation: Noah's `HF_HOME` is
`/Volumes/Thunderbay SSD/Mlx_Models` (set in ~/.zshrc), and the auth token
lives there. Non-interactive shells don't source .zshrc, so scripts must set
HF_HOME explicitly or run unauthenticated — which would fail a 100+ GiB
upload at the very end. `upload_to_hf.sh` sets it and pre-flights whoami.

### STOCK mlx_lm proven on the CLUSTER too (08-16)

Noah's rule — "only keep deviations if they're necessary" — so we tested the
strongest claim instead of assuming it. Restored PRISTINE `mlx_lm` on BOTH
exo nodes (`utils.py` from the `.orig-vq` backup, `models/vq_switch.py`
deleted) and re-placed VQ-3.1bpw 2-node tensor-sharded.

**Result: SERVING after 623s, output IDENTICAL to the patched run** — same
text, same 1189 tokens, same 5282 reasoning chars. exo calls
`mlx_lm.utils.load_model`, which honors `model_file` natively, so the
artifact's bundled `model.py` supplies the VQ runtime. The patch was only
ever a convenience for the pre-packaging 35B artifact — and that one now
carries `model.py` too, so **`patch_mlx_lm.py` is retired; nothing needs it.**

Standing deviation list is now exactly ONE item: the codebook-replicate line
in exo's own `auto_parallel.py` (codebooks are shared LUTs and must not be
sliced). Nothing else, anywhere, for any consumer.

Also measured: **VQ-3.1bpw decodes ~17.4 tok/s** on the ring (M3 Ultra 96 GB
+ M4 Max 128 GB over TB5/RDMA) — installed in its card, replacing the earlier
byte-arithmetic estimate. Memory splits proportional to node capacity
(M3 45.6 GiB / M4 80.6 GiB), not evenly.

## TASK BENCHMARKS (08-16) — HellaSwag/PIQA/WinoGrande, all five models, ONE harness

Motivation: the HF cards show perplexity only; spicyneuron's card shows task
accuracies. Their card states NO harness, version, shots, or limit — and
their reported stderrs imply n≈500 (0.598±0.022 ⇒ n=497; all three tasks
land on ~500). We had already measured their 2.6bit at wikitext 3.1843
where their card says 3.852 — same weights, 21% apart, the gap IS the
pipeline. So: never quote their numbers next to ours; run their artifacts
on our own instrument instead. This entry does exactly that.

**Instrument:** `score_tasks_streaming.py` — an lm-eval `LM` subclass whose
`loglikelihood()` streams one transformer block at a time (the
score_streaming.py trick) while batching ALL task sequences through each
block before freeing it. One pass over the model's bytes per sweep, flat
~15 GB, so the 165.6 GiB comparator scores on one box. lm-eval builds the
prompts and computes the metrics; we only supply logprobs. Reuse was
load-bearing: WinoGrande varies the CONTEXT per option (shared
continuation) — easy to invert in a hand-rolled scorer.

**Validation before any number was trusted** (referee/README rule):
0.8B debug model — batched path vs referee: total_nll 23135.2109 BOTH.
VQ-2.2bpw real artifact — ppl 3.1706, the published number (total_nll
differs at 3e-8 relative: float summation order, per-token concat vs
per-1024-chunk; the claim is "identical to 4 decimals", not bit-identical).
0.8B end-to-end sanity: hellaswag 0.49 / piqa 0.71 / winogrande 0.57 —
inside the published band for that size.

**Settings (state these on any card):** lm-eval 0.4.12, mlx_lm 0.31.3,
0-shot, limit 1000 per task, seeded selection (identical items for every
model), acc_norm for hellaswag/piqa, acc for winogrande. 10-shot was
measured (request construction only) at 4.62M tokens vs 289,598 — 16x the
compute — and skipped; shot count only matters vs OUTSIDE numbers, which
we are not comparing against.

  | model | GiB | hellaswag | piqa | winogrande |
  |---|---|---|---|---|
  | VQ-2.2bpw | 100.1 | 0.8610 | 0.8410 | 0.7870 |
  | VQ-2.4bpw | 110.8 | 0.8830 | 0.8440 | 0.7840 |
  | spicy 2.6bit | 120.6 | 0.8800 | 0.8410 | 0.7710 |
  | VQ-3.1bpw | 142.8 | 0.9030 | 0.8400 | 0.7800 |
  | spicy 3.5bit | 165.6 | 0.9040 | 0.8460 | 0.7670 |

**Paired analysis** (`analyze_task_bench.py`: every model saw the SAME 1000
items, so item difficulty cancels; McNemar exact on discordant pairs +
paired bootstrap 95% CI — the independent ±0.011-0.015 stderr is the wrong
bar for deltas):

- **HellaSwag is the one task that discriminates, and it reproduces the
  perplexity ordering exactly:** 2.2 < 2.4 ≈ spicy2.6 < 3.1 ≈ spicy3.5.
  Every rung is significant (2.2→2.4 p=0.0013; 2.4→3.1 p=0.0017;
  spicy2.6→spicy3.5 p=0.0001). Discordant counts are lopsided the same
  way every time (e.g. 2.2 vs 3.1: 8 wins / 50 losses).
- **The size-class matchups are statistical TIES on all three tasks:**
  VQ-2.4bpw vs spicy 2.6bit p=0.76/0.77/0.29 at **9.8 GiB smaller**;
  VQ-3.1bpw vs spicy 3.5bit p=1.00/0.33/0.25 at **22.8 GiB smaller**.
  That is the publishable claim: same task accuracy, meaningfully smaller.
- **PIQA and WinoGrande resolve nothing** at n=1000 — every pair ties.
  They are "nothing broke" checks here, not rankings. WinoGrande shows a
  consistent but non-significant drift in OUR favor (all four VQ-vs-spicy
  deltas positive, up to +0.020 p=0.12); n=1000 cannot promote it and we
  do not claim it.
- **spicyneuron's published task numbers are confirmed incomparable:**
  their 2.6bit on OUR harness: hellaswag 0.8800 (their card: 0.598 — 28
  points; our RAW acc is 0.743, so it is not an acc/acc_norm confusion),
  piqa 0.8410 (0.802), winogrande 0.7710 (0.718). Second independent
  confirmation of the pipeline gap after the ppl one. Their artifacts are
  GOOD — their published numbers just measure something else.

**Surprise, logged as tracker VQ-PF1:** identical 289,598-token workload
runs ~9.6 s/block on spicy (MLX affine gather_qmm) vs ~90 s/block on our
VQ artifacts — ~9x on prefill-shaped work, corroborated by the cards'
prefill figures (489 vs 42-48 tok/s, ~10x, measured completely
differently). Decode is at parity (bandwidth-bound; 20.4-20.9 tok/s F).
PUZZLE: vq_switch.py's own microbench header says chunked decode+GEMM is
1.21-1.28x gather_qmm — contradicts end-to-end; diagnose before fixing
(bucket-loop re-decode amplification in MY scorer is candidate #1 and
would partially exonerate the kernel). Constraint recorded: any fix must
keep the on-disk layout frozen so it ships as a few-KB model.py update,
not a 100+ GiB re-upload.

> **RESOLVED same night — see the VQ-PF1 entry below. The VQ kernel is NOT
> the problem: it is at PARITY with gather_qmm at this workload's real
> shape (61.5 ms both).** The 9x is padded-GEMM waste in `_prefill`, which
> pads every expert in a decode chunk to that chunk's MAX row count against
> a router skewed 8.7x — 5.80x surplus FLOPs at the shipped
> _DECODE_CHUNK=128. My candidate #1 above was wrong; the header's
> 1.21-1.28x came from a bench whose synthetic `rng.integers` router pads
> only 1.20x. Do not cite the 9x as a property of vector quantization —
> it is a chunking-policy bug with the layout untouched, and chunk=8 alone
> already takes a real block 1135 -> 690 ms.

**Not measured:** full task sets (13.1k items; ~6x the compute — the
paired design at n=1000 already separated what is separable), any
few-shot setting, generative tasks (the scorer is loglikelihood-only by
construction). Timings: ours 1.28-1.50 h/model, spicy 10-11 min/model,
sweep total ~4.6 h. Raw per-item records in results_tasks/*.samples.json
(local, untracked); summaries committed.

## VQ PREFILL DIAGNOSIS (08-16) — the 9x is padded-GEMM waste, not the kernel

Follow-up to the task-bench surprise above (tracker VQ-PF1). Diagnose-only:
instrument, name the cause, quantify the recoverable headroom. No fix is
applied in this entry.

**The puzzle.** vq_switch.py's header claims the chunked decode+GEMM prefill
path benches 1.21-1.28x gather_qmm; the 08-16 sweep measured ~90 s/block for
our VQ artifacts vs ~9.6 s/block for spicy 2.6bit on an identical
289,598-token workload. Both numbers are real. The regime that separates them
is **router skew**, and the microbench never saw it.

**Workload shape** (what the scorer actually asks of the MoE): 8000 sequences
/ 289,598 tokens, `--batch-seqs 256` ⇒ 32 padded buckets of ~9,216 tokens.
E=512 experts, top_k=10 ⇒ N=92,160 (token,expert) pairs per bucket per
projection, mean 180 rows/expert. 3 projections x 32 buckets x 60 blocks.

**Probe 1 — the expert kernel is exonerated** (`probe_vq_prefill_regime.py`,
real layer-3 gate_proj tensors out of the shipped 2.4bpw artifact, uniform
router, vs an affine 2-bit gather_qmm baseline of identical shape):

  | tokens/call | rows/expert | gather_qmm | vq _prefill | ratio | decode share |
  |---|---|---|---|---|---|
  | 1024 | 20 | 7.5 ms | 22.8 ms | 0.33x | 45% |
  | 4096 | 80 | 28.5 ms | 36.7 ms | 0.78x | 28% |
  | **9050** | **177** | **61.5 ms** | **61.5 ms** | **1.00x** | 17% |
  | 18100 | 354 | 122.4 ms | 101.8 ms | 1.20x | 10% |
  | 72400 | 1414 | 484.6 ms | 302.8 ms | 1.60x | 3% |

  At the scorer's own shape the kernel is at **parity**, and the header's
  1.21-1.28x reproduces exactly where m1c_prefill_bench measured it (high
  rows/expert). Decode-materialization is NOT the story: decoding all 512
  experts costs 10.2 ms, 17% of the call, and is chunk-size-sensitive only
  from 10.2 ms (chunk 128) to 41.3 ms (chunk 4) — worst case 1.7x, not 9x.

**Probe 2 — the gap is real and this instrument reproduces it**
(`probe_block_prefill.py`, one REAL transformer block, bucket [256,36]):
spicy 2.6bit **295 ms/bucket ⇒ 9.4 s/block** (the sweep measured 9.6 — the
instrument is faithful); VQ-2.4bpw **1135 ms/bucket ⇒ 36.3 s/block**.

**Probe 3 — where the block's time goes** (`probe_block_breakdown.py`, stage
timing inside the real `VQSwitchLinear.__call__`, 3 calls/block, N=92,160):

  | stage | ms | % of block |
  |---|---|---|
  | `_prefill` (3 calls) | 783.9 | 72.6% |
  | `broadcast_to(...).reshape(N, IN)` | 143.4 | 13.3% |
  | output reshape | 55.0 | 5.1% |
  | astype fp16 | 27.7 | 2.6% |
  | idx -> host sync, argsort | 0.3 | 0.0% |
  | rest of block (attn, norms, shared expert) | 68.8 | 6.4% |

  `_prefill` is **261 ms/call here vs 61 ms/call in probe 1** at the same
  token count. The only difference between the two is the expert histogram.
  (The bucket-loop re-decode of hypothesis #1 is real — 32 buckets do decode
  each layer's experts 32x — but it amplifies BOTH models' weight traffic
  equally and is only 17% of our call, so it does not explain the ratio.)

**THE CAUSE** (`probe_routing_pad.py`, real router indices captured off a
real block): `_prefill` pads every expert in a decode chunk to that chunk's
**maximum** token count and runs one batched GEMM over `[ne, cap, IN]`. Real
MoE routing is heavily skewed — **max 1574 rows/expert against a mean of
180, 8.7x** — so at the shipped `_DECODE_CHUNK=128` the per-chunk caps are
[773, 1574, 1197, 633] and the GEMM does **5.80x the necessary FLOPs**. A
uniform rng router over the same E, N and chunk pads only **1.20x**. Timed
side by side on identical data: `_prefill` **238 ms under the real router vs
76 ms under uniform**. That 3.1x is the missing factor, and it is invisible
to any benchmark whose router is `rng.integers` — which is exactly what
m1c_prefill_bench.py uses. **The kernel was never the problem; the padding
policy is.** The remainder of the gap is the ~226 ms/block (21%) spent
materializing the `[N, IN]` fp16 row matrix that `gather_qmm` never needs,
because it gathers rows inside the matmul.

**Recoverable headroom, on-disk layout untouched** (`probe_pad_headroom.py`,
real router histogram; "id" = shipped policy, chunks experts by expert id;
"count" = chunk experts by SIMILAR token count, a pure host-side reordering
of which experts share a GEMM — codes, codebooks and scales are not read
differently, let alone repacked):

  | chunk | pad (id) | time (id) | pad (count) | time (count) |
  |---|---|---|---|---|
  | 8 | 2.59x | 146.9 ms | **1.08x** | 84.0 ms |
  | 16 | 3.36x | 150.5 ms | 1.19x | **75.9 ms** |
  | 32 | 4.08x | 198.2 ms | 1.41x | 98.0 ms |
  | 64 | 4.81x | 236.6 ms | 1.88x | 108.5 ms |
  | **128 (shipped)** | **5.92x** | **263.1 ms** | 2.84x | 124.7 ms |
  | 256 | 8.01x | 491.4 ms | 4.83x | 189.5 ms |
  | 512 | 8.91x | 505.2 ms | 8.91x | 446.7 ms |

  Count-sorted chunking at chunk=16 lands `_prefill` at **75.9 ms — 3.5x
  faster than shipped, and 1.24x gather_qmm's 61.5 ms**, i.e. it *recovers
  the header's own claim* on the real router. Confirmed at block level with
  the existing env knob alone (no code change): `SCOUT_VQ_DECODE_CHUNK=8`
  takes the real block from 1135 to **690 ms/bucket (22.1 s/block, 1.65x)**.

**Verdict.** The fused VQ-GEMM tile kernel proposed as the candidate fix is
NOT indicated: the existing kernel already matches gather_qmm at this shape
once it stops multiplying padding. Two layout-free changes, in cost order:
(1) chunk experts by token count rather than expert id inside `_prefill`;
(2) lower the `_DECODE_CHUNK` default, which is currently auto-sized purely
for memory headroom (128 on a 96 GB M3 Ultra) with no regard for the pad
tax it buys. Both are a few lines inside `_prefill` and ship as the few-KB
bundled `model.py` update the HARD CONSTRAINT requires. Deliberately not
done here: neither is applied, and neither is trustworthy until
`score_tasks_streaming.py --selftest` reproduces the published ppl
(2.4bpw 2.7655) to 4 decimals through the changed path.

**Caveats.** All timings are layer 3 of the 2.4bpw artifact on a 96 GB M3
Ultra, with a synthetic uniform-length bucket [256,36]; the real sweep's
length-sorted buckets have a long tail, which is why the reproduced ratio
here is 3.9x (36.3 vs 9.4 s/block) against 9x in the sweep — longer buckets
raise N per call and, with a skewed router, the cap along with it. Routing
skew was measured on ONE block with random hidden states; a real-token
histogram could differ in magnitude, not in kind (probe_routing_skew.py has
the corpus-driven version if the number needs hardening).

## VQ-PF1 FIX APPLIED (08-16) — count-sorted chunking, 1.83x prefill, ppl to the digit

Acts on the diagnosis above. ONE change in `_prefill`: chunk experts by
SIMILAR ROW COUNT instead of by expert id
(`touched[np.argsort(counts[touched])]`), plus the row-order restore that
requires — reordering experts reorders output rows, and `_prefill`'s
contract is to return rows in its INPUT order (`__call__` layers its own
`inv` on top). Tracked via `row_ids`, undone with one `[N, OUT]` gather.

**Correctness, established BEFORE any timing claim:**
- Synthetic, skewed routers (Dirichlet α=0.1/0.5/3.0, skew 3.2-7.3x):
  new(count, chunk16) vs shipped(id, chunk128) **bit-identical, maxdiff
  0.000e+00** in all three.
- Real artifacts, `--selftest` through the changed path, all three exact:
  2.2bpw **3.1706** (nll 9452.9414) · 2.4bpw **2.7655** (8332.9785) ·
  3.1bpw **2.3519** (7006.0742).

**Speed** (2.2bpw selftest, old vs new runtime back-to-back, same box):
**167.6 s -> 91.4 s, 1.83x.** 2.4bpw 97.1 s, 3.1bpw 110.7 s (no clean
pre-fix baseline was taken for those two — only 2.2bpw is a controlled
comparison).

**THE TRAP, and why the default chunk did NOT change.** The diagnosis
recommended lowering `_DECODE_CHUNK` as fix #2. It IS faster — chunk=16
took the same selftest to 76.3 s. It also **changes the output**: ppl
3.1754, nll 9465.2217, against the published 3.1706 / 9452.9414. Not a
bug and not the count-sort — count-sorted chunking at the shipped 128
reproduces the published nll to the digit (9452.9414). `ne` is the
batched-GEMM batch dimension; a different `ne` selects a different Metal
GEMM tiling and fp16 accumulates in a different order. So the reordering
ships and the default does not: **exact reproducibility of every published
number beat ~20% more prefill.** `SCOUT_VQ_DECODE_CHUNK` still exposes the
trade for anyone who wants it, now documented as breaking bit-exactness.

**Process note worth keeping.** The first "validation" of this fix was
worthless and looked fine: ppl 3.1706, unchanged. The artifacts carry a
BUNDLED `model.py` (`config.json → model_file`), so `mlx_lm` was loading
the OLD runtime and my edited `vq_switch.py` never executed — the matching
ppl was matching because NOTHING HAD CHANGED. Three timing runs were also
spent on the old code. Editing `vq_switch.py` does not change an artifact;
the bundled copy must be re-spliced (`model.py` = `vq_switch.py` + the
1798-char loader shim). Check `grep -c` for your own change in the
artifact's `model.py` before believing any number.

**Shipping shape:** artifact `model.py` is 29,719 bytes; codes, codebooks,
scales and `config.json` are untouched, so this is a single small file per
repo — HF is content-addressed, so existing downloaders fetch only it, not
100+ GiB. The HARD CONSTRAINT held.

## CHUNK KNEE: THERE IS NO KNEE (08-17) — and last night's 1.20x was noise

Follow-up to the VQ-PF1 fix entry, which left one claim provisional at n=1:
chunk=16 measured 76.3 s / ppl 3.1754 against the shipped 128's 91.4 s /
3.1706, and the open question was whether that +0.0048 ppl was a systematic
cost buying real speed (a knee worth taking) or float noise.

Swept chunk 4/8/16/32/64/128, 2.2bpw selftest, QUIET box (the HF upload and
the 790 MB Claude-Code ingest that contended with every earlier timing had
both finished):

  | chunk | ppl | total_nll | seconds |
  |---|---|---|---|
  | 4 | 3.1741 | 9461.9189 | 106.1 |
  | **8** | **3.1696** | 9450.3555 | 111.9 |
  | 16 | 3.1754 | 9465.2217 | 103.5 |
  | 32 | 3.1706 | 9452.9414 | 105.5 |
  | 64 | 3.1706 | 9452.9414 | 98.3 |
  | 128 (shipped) | 3.1706 | 9452.9414 | 103.5 |

**BOTH halves of the provisional claim fail.**

1. **The ppl shift is float ordering, not damage — proven by the sign.**
   It is NOT monotone in chunk: chunk=8 lands at 3.1696, *below* the
   published 3.1706, while chunk=16 sits above it. Systematic accumulation
   error would make every smaller chunk worse; instead they scatter ±0.005
   on both sides. Chunks 32/64/128 return nll 9452.9414 IDENTICALLY, so a
   different Metal GEMM tiling is only selected below ne=32. Each value is
   itself deterministic — chunk=16 reproduced 3.1754 / 9465.2217 exactly,
   matching last night to the digit (n=2).

2. **Smaller chunks are NOT faster.** 98-112 s across the entire range with
   no trend; chunk=64 fastest, chunk=8 SLOWEST. Last night's 76.3 s was
   measured while the HF upload saturated the SSD — contention noise that
   happened to land low, and I reported the 1.20x from it. On a quiet box
   the speed difference between chunk 8 and chunk 128 is ~8% and points the
   WRONG way.

**Verdict: no knee, no trade, nothing to tune.** Shipping count-sorted
chunking at the default 128 was right, for a better reason than the one we
had: not "exactness is worth 20% of prefill" but "there was never 20% of
prefill on offer." The 1.83x from count-sorting itself stands (167.6 s
pre-fix vs ~100 s here). `SCOUT_VQ_DECODE_CHUNK` stays as an escape hatch
for memory pressure — its ORIGINAL purpose, per the transient-size table —
not as a speed knob.

**Method note for the next person timing anything here.** Three separate
wrong numbers in two days came from timing on a contended box: this 76.3 s,
the "9x" prefill gap (partly), and 3.1bpw looking FASTER than 2.2bpw in the
08-16 task sweep (1.28 h vs 1.43 h) purely because the upload finished
between them. Perplexity is contention-immune and can be measured any time;
WALL TIME CANNOT. Check `ps` for the embedder, the ingest, and smbd before
believing a duration.

## CORRECTION + SHIP (08-17) — the knee is REAL, ~1.4x, and free. Default 128 -> 32.

This reverses the "no knee" entry above. That conclusion was measured with
the WRONG INSTRUMENT and it cost us a 1.4x we already had.

**The instrument error.** The knee sweep timed `score_tasks_streaming.py
--selftest`, which re-reads the entire model from disk on every pass (~63 s
of a ~100 s run for 100 GiB at ~1.7 GB/s). That run is DISK-bound, so a
compute-side change is buried: a 1.4x on the ~40 s compute half moves the
total by ~10%, inside the run-to-run noise. Measuring a RESIDENT block
instead (`probe_block_prefill.py`, weights materialized before the timed
region) shows it immediately.

**Resident measurements, M4, [256,36] bucket, steady-state ms/bucket:**

  | chunk | 2.2bpw | 2.4bpw | 3.1bpw |
  |---|---|---|---|
  | 16 | 677.2 | 651.9 | 663.6 |
  | **32** | **716.1** | **683.0** | **691.9** |
  | 64 | 791.4 | 756.3 | 775.7 |
  | 128 (old default) | 984.1 | 943.2 | 947.3 |

  Reproduced (run 2, quiet box, --buckets 4): 2.2bpw 701.8/1037.6,
  2.4bpw 953.8/1426.2, 3.1bpw 1018.5/1490.5.

  **128 -> 32 = 1.37x, 1.38x, 1.37x (run 1); 1.48x, 1.50x, 1.46x (run 2).**
  Claim conservatively: **~1.4x, n=2, all three artifacts.**

**READ THE RATIOS, NOT THE ABSOLUTES.** 2.4bpw measured 683 ms in run 1 and
954 ms in run 2 — 40% apart on identical work, because something else was
running. The A/B survives only because both arms sit inside the same run.
Never quote these ms as throughput.

**Why 32 and not 16** (16 is marginally faster still): 32 is the SMALLEST
chunk that reproduces published perplexity EXACTLY. Validated on all three
at the new default — 2.2bpw nll 9452.9414 / 3.1706, 2.4bpw 8332.9785 /
2.7655, 3.1bpw 7006.0742 / 2.3519, every one an exact match. Below 32 the
Metal GEMM tiling changes and fp16 sums in a different order: chunk 16 gives
9465.2217, chunk 8 gives 9450.3555 — one above the published value and one
BELOW it. A lower ppl from float reordering cannot be banked; it is the same
model measured differently. Take the knee, keep exactness.

**The gap vs the affine comparator, resident, VQ vs spicyneuron 2.6bit:**

  | tokens/call | rows/expert | VQ @128 | VQ @32 | spicy | gap @128 | gap @32 |
  |---|---|---|---|---|---|---|
  | 9,216 | 180 | 972.8 | 679.3 | 461.1 | 2.10x | **1.47x** |
  | 16,384 | 320 | 2365.4 | 1773.4 | 1427.1 | 1.61x | **1.24x** |
  | 65,536 | 1,280 | 10030.0 | 7718.4 | 6473.5 | 1.57x | **1.19x** |

  The gap NARROWS with prompt length and does not reverse — the isolated
  expert kernel wins at high rows/expert but the whole block does not,
  because the ~21% spent materializing the [N, IN] fp16 row matrix (which
  gather_qmm never needs) does not go away. At long-prompt shapes VQ is
  within **1.19x** of affine. The earlier "4-5x" was arithmetic, never
  measured; the measured end-to-end task-bench gap was 3.3x (0.73 h vs
  0.22 h) at chunk 128 and is not yet re-measured at 32.

**Remaining lever, unbuilt:** fuse the row gather into the matmul so the
[N, IN] matrix is never materialized (~21% of block time). Only worth it if
someone needs the last 1.19x.

**Shipped:** default cap 128 -> 32 in `_default_decode_chunk`, bundled
model.py re-spliced in all three artifacts (Thunderbay AND the M4 local
copies, which are now independent files), pushed to HF. Weights, codebooks,
scales and config.json untouched — every published ppl and task number
stands.

## FUSED-THRESHOLD (08-17) — real at block level, a WASH end-to-end. Not shipped.

The lead: mlx-lm's default 512-token prefill step gives N = 512*10 = 5,120
(token,expert) pairs — just OVER VQ_FUSED_MAX_N=4096, so the default takes
the chunked padded-GEMM path at ~10 rows/expert, our worst regime. Would the
fused kernel be faster there? Raising the threshold is a model.py-only
change: invisible to users, no env vars.

**Block level (M3, resident single block, n=3 each, control included):**

  | bucket | N | padded | fused | verdict |
  |---|---|---|---|---|
  | [1,256] | 2,560 | 35.5 | 35.1 | control: both fused, identical ✓ |
  | [1,512] | 5,120 | 106.0-108.2 | 67.1-68.0 | **fused 1.58x** |
  | [1,1024] | 10,240 | 129.6-131.7 | 131.9 | tie — crossover |
  | [1,2048] | 20,480 | 192.4-196.5 | 268.9 | padded 1.4x |

  Crossover ~N=10k, so 8192 would be the right threshold. The N=2,560
  control (same path under both settings) matching to 1% is what makes the
  probe trustworthy.

**PPL gate (score_local, step 512, so the fused path is actually exercised —
--selftest would be a FALSE PASS, its single 82k-pair call never crosses the
threshold):** both paths deterministic, padded 1.096101, fused 1.095358
(6.8e-4 relative, fused LOWER). Plausible mechanism: the fused kernel
accumulates in fp32 (`float acc`, fma) where the padded path runs an fp16
GEMM. Published numbers unaffected — selftest with the raised threshold
still returns 9452.9414 / 3.1706 exactly (its calls sit far above any
threshold considered).

**End-to-end (M4, m2_speed_split, ctx 8192, step 512, five runs both
orders):** 4096 → 73.7, 56.9 tok/s; 8192 → 53.6, 62.0, 58.2. **A wash** —
the instrument's ±15% swamps the effect. The 1.58x is real for the expert
projection but the projection is only part of block time, and prefill
includes attention and everything else. Order-reversal killed the
throttling confound (8192 won the reversed round).

**Decision: NOT SHIPPED.** The only reason to move the threshold was speed;
without a measurable end-to-end gain we would be changing arithmetic
(slightly different logits for prompts whose chunks land in 4096 < N <=
8192) in exchange for nothing a user can perceive. Contrast chunk-32, which
shipped because it was BOTH measurable end-to-end (1.42-1.50x) AND exact.
Kept as a lead: if the fused-row-gather work ever removes the ~21% row
materialization, the balance shifts and this threshold deserves a re-test.

Method note: this took three instruments to resolve — resident block probe
(found it), chunked scorer (gated it), end-to-end throughput (rejected it).
Any one alone would have given the wrong answer or shipped a no-op.

---

## E39 (08-17) — GEMMA-4 ARRIVES, AND ppl DIES WITH IT

Goal: replace the gemma-4-e4b sidecar with gemma-4-26b-a4b at the same
~8.4G, keeping audio. Detail: **GEMMA4_PPL_ANOMALY.md**, **CRUSH_RESULTS.md**,
**LADDER_GEMMA.md**.

**RAW LIKELIHOOD IS INVALID ON gemma-4-it — MODEL PROPERTY, NOT A PORT BUG.**
The family assigns absurd probabilities to external text (plain English ppl
~100, Austen ~700) while its own greedy output scores 1.42 and it generates
fluently. Falsified as an mlx bug by an INDEPENDENT referee: HF transformers
5.5.4, fp32, on unquantized bf16, reproduces it (115.6 / 695.1 vs mlx's
96.6 / 729.2). mlx_lm and mlx_vlm agree to the decimal — they share lineage
and referee nothing, which is why the third implementation was needed.
Consequence: **no wikitext ppl, no literary ppl, no raw-continuation MC
(hellaswag/litbench-as-shipped) may be cited for gemma-4, absolutely or
cross-family.** Qwen is unaffected.

**REPLACEMENT INSTRUMENT: `kl_damage.py`** — KL to the model's own bf16
output. Sharpening is common-mode between teacher and student so it cancels
exactly, and no notion of "correct" text is needed. Cache format is
byte-compatible with `dwq_cache_teacher.py`. Top-k truncation is *helped*
here: a collapsed distribution is concentrated, so k=64 holds 96.5%.

**KL VALIDATED AGAINST ppl ON QWEN** (where both work): across 9 rungs KL
and ppl rank identically, monotonically, 1.00x -> 3.15x. Dense conversion:
KL <50 mnats free, ~200 = +10% ppl, >1000 broken. **These DO NOT transfer to
MoE** — gemma-4-26b at 8-BIT reads 441 mnats / 79.95% agreement where Qwen
q4 reads 45.8 / 89.82%. Routing is discrete: perturb anything upstream and a
different 8-of-128 experts fire. Measure gemma against its own 8-bit
reference, not zero.

**gemma-4-26b-a4b IS A HYBRID, NOT A PURE MoE.** Every layer carries a dense
`mlp.{gate,up,down}_proj` ALONGSIDE `experts.switch_glu.*`; `v_proj` exists
in only 25 of 30 layers (`attention_k_eq_v`). Split: experts 90.5%,
attention 4.4%, embed 2.9%, dense mlp 2.1%. Non-expert at FULL 8-bit costs
only 2.54G, leaving ~2.05 bpw for experts inside 8.4G.

**AFFINE LADDER (12 rungs, `convert_gemma_struct.py`, all KL-scored):**

  | rung | size | top-1 agree |
  |---|---|---|
  | struct8-e8 | 25G | 79.95% (ceiling) |
  | uniform-q8 | 25G | 79.33% |
  | struct6-e8 | 24G | 45.20% |
  | struct6-e3 | 11G | 38.01% |
  | struct8-e2 | 9.1G | 34.90% |
  | uniform-q3 | 10G | 4.46% |

  Converter VERIFIED: struct8-e8 reproduces uniform-q8 and edges it (bf16
  router), so the `switch_glu`/`router.proj` re-targeting is correct against
  real tensors. Structured beats uniform at every small size. **E8's
  attention cliff reproduced on a new family** — struct6-e8 collapses to
  45.2% purely from qkv at 4-bit, already-known and re-learned the slow way;
  use `--structure-bits 8 --qkv-bits 8`, it is nearly free here.
  **Affine tops out at 9.1G / 34.9% against an 80% ceiling. That gap is the
  VQ-shaped hole.** VQ NOT YET RUN; blocked on the `down_proj` packing
  decision (moe_intermediate 704 -> NSUB 176, 176 % 32 != 0).

**mlx FACTS.** gemma-4 cannot be layer-streamed (DecoderLayer returns a
tuple, threads PLE + shared KV, alternates sliding-window masks a hand-rolled
loop gets silently wrong) — `score_tasks_streaming.py --direct` added, and
VALIDATED to the digit against the streamed path on Qwen, 2x faster.
`tok.encode()` does not prepend BOS and gemma degenerates without it — fixed.
E-series quants ship dead shared-KV k/v tensors mlx_lm won't build (e2b 140 =
20x7, e4b 126 = 18x7); provably dead (mlx_lm and mlx_vlm both give ppl 96.62
with and without), so `--allow-unmatched` is safe there and refused by
default elsewhere. e2b/e4b are DENSE — "E" is effective params via per-layer
embeddings + KV sharing, NOT expert routing — so only 26b-a4b is a VQ target.

## E40 (08-17) — QWEN3.8-27B: q4 IS FREE, HAND-MIXED LOSES TO UNIFORM

Dense 27.78B, bf16 55.6G. Both instruments valid, measured together.

  | rung | size | ppl | vs bf16 | KL | agree |
  |---|---|---|---|---|---|
  | q4 | 14G | 5.2055 | **0.996x** | 45.8 | 89.82% |
  | m3-a4 | 13G | 5.4377 | 1.041x | 106.4 | 85.36% |
  | q3 | 11G | 5.8323 | 1.116x | 187.8 | 79.48% |
  | m2-a4 | 11G | 7.0976 | 1.358x | 504.3 | 69.67% |
  | q2 | 5.7G | 16.4349 | 3.146x | 1426.9 | 46.07% |

**q4 at 14G is free** (4x compression, 0.996x). **Hand-designed static mixed
allocation LOSES to uniform at matched budget** — m2-a4 1.358x vs q3 1.116x
at 11G, and m2-a6 at 12G is still worse than q3 at 11G. Cause: mlp is 61.6%
of a dense model, so 2-bit mlp IS the dense cliff (headline 4) and attention
protection cannot buy it back. Mirror image of MoE, where experts are ~90%
and tolerate 2-bit.

**This does NOT falsify E4** — these are STATIC hand allocations, not optiq's
KL-calibrated ones. OptiQ sweep IN PROGRESS (~10.6h sensitivity on the M3
Ultra, ~0.5 components/min x 326). Bar to clear: **beat 1.116x at 11G**.
Note the bf16-reference SwitchLinear blind spot does NOT fire on a dense
model — only conv1d is invisible, at 2.0M params (0.01%). `optiq_realloc.py`
re-allocates from an existing `sensitivity_checkpoint.json` so extra
target-bpw / attention-floor combinations are free (the floor lives in
optimizer.py, not sensitivity.py).

Model also carries an MTP head (0.425B, speculative decoding) and a vision
tower (0.46B) — droppable bytes if unused.

## E41 (08-17) — METHODOLOGY: the agreement metric has a FLOOR, and the two metrics are complementary

Not a gemma result — this changes how every agreement number in this repo
should be read, on any family.

Measure two artifacts INDEPENDENTLY established as equivalent against each
other, and whatever that reads is the instrument's floor:

  | comparison | KL (mnats) | top-1 agree |
  |---|---|---|
  | struct8-e8 vs ITSELF (control) | 0.000 | 100.00% |
  | struct8-e8 vs uniform-q8 | 397 | 82.32% |
  | struct8-e8 vs bf16 | 441 | 79.95% |
  | VQ K256 d4 vs bf16 | 3363 | 42.65% |

**Two near-lossless artifacts disagree on 17.7% of positions.** The self
comparison returns exactly 100.00%/0.000, so this is not instrument error.

1. **Agreement SATURATES near the top.** It cannot separate "equivalent"
   from "8-bit" (82.32 vs 79.95). So a near-lossless rung reading ~80% is
   the EXPECTED result, not a quality ceiling to climb toward — an earlier
   draft of CRUSH_RESULTS.md made exactly that mistake.
2. **Agreement DISCRIMINATES lower down.** The affine ladder separates four
   rungs cleanly across 27-45 (27.29 / 34.90 / 38.01 / 42.73). A saturated
   instrument cannot do that, which is what licenses spending compute on
   improvements in that region. NOTE: a low floor does NOT by itself imply
   real damage — the floor locates only the TOP of the scale. The ladder's
   resolution is the argument, not the floor value.
3. **KL behaves oppositely** — still separating 397 from 441 where agreement
   has given up. So QUOTE KL AS MULTIPLES OF ITS NOISE FLOOR (~400 mnats
   here; VQ at 3363 is ~8.5x it) and keep agreement as the intuitive
   companion. Neither is trustworthy alone across the full range.
4. **The floor is a property of the MEASUREMENT SETUP**, not a constant.
   Re-measure it whenever the corpus or cache changes, or you are comparing
   across two different sticks.

MECHANISM, from the 397B session's per-item data (which existed because
`results_tasks/*.samples.json` are tracked — a new metric over old per-item
outputs, no GPU): quant disagreement is SYMMETRIC (on disagreeing items
neither model matches gold more often — 29v31, 16v18) and concentrated on
NEAR-TIES (top-2 margin 1.2-2.4 nats on disagreements vs 16-18 on
agreements, up to 14.7x). Quants reshuffle uncertain decisions and leave the
confident mass alone.

CAVEAT: that evidence is 4-choice argmax on tasks; ours is top-1 over a
~250k vocab on free text. Comparable PHENOMENA, not comparable NUMBERS.

## E42 (08-18) — THE SIDECAR ANSWER, and OptiQ falsified on dense

Full tables in CRUSH_RESULTS.md.

**GEMMA SWAP: VIABLE.** Generative chat-native litbench (the only valid
cross-model form — see below), 104 items, position-debiased:

  26b bf16 (48G)          84.62%   <- the premise: bigger MoE IS better
  e4b bf16 (19G) CYCLIC   82.69%   <- the incumbent, our own harness
  VQ-K256-d4 (8.4G) CYC   79.81%   <- 44% of the size, within noise

A 26B MoE at the CURRENT SIDECAR'S EXACT SIZE (8.4 GiB) scores within this
bench's resolution of e4b at bf16. Audio still graftable.

**INSTRUMENT WARNING, THE THIRD TONIGHT.** Single-token chat scoring put the
26b at 37.5% — BELOW its own 8-bit quant. Cause: it opens a <|channel>thought
block and reasons through every option, so reading letter logprobs at the
first position measures willingness-to-answer-immediately, not comprehension.
e4b answers directly and was unaffected, which is exactly why the artefact
looked like a capability gap. --generative fixes it (37.5% -> 100% on a
6-item smoke; e4b unchanged at 78.85%, the control that proves the fix does
not simply inflate).

**OPTIQ ON QWEN3.8-27B: E4 DOES NOT REPLICATE.** Real calibrated sweep, 497
components, ~8h:

  uniform q3      11G  1.116x  <- STILL THE BEST
  optiq-b30       13G  1.179x
  optiq-b30-af6   13G  1.621x  (attention floor 6)

Both lose to plain uniform while being 2G larger. The floor makes it worse:
forcing 6-bit attention leaves no budget elsewhere so 100% of mlp goes to
2-bit, reproducing the m2-a6 shape that already failed. Unfloored calibration
assigned 2- and 3-bit to some ATTENTION layers, so flat isolation-KL on
attention is NOT MoE-specific — but flooring is not the cure on dense either,
because the bytes must come from the mlp bulk (61.6% of params). Three
independent mixed-precision attempts all lose to uniform: use uniform here.

**TOOLING.** optiq resumes from sensitivity_checkpoint.json when
candidate_bits match (core/sensitivity.py:870) — the af6 variant cost 3
MINUTES instead of ~10h by copying the checkpoint into the new output dir
first. Sensitivity is a property of model+calibration, not of the budget.

## E43 (08-18) — K=2048: CODEBOOK CAPACITY WAS THE BINDING CONSTRAINT

**Question.** K256 VQ left gemma at 42.65% agreement vs an 80% 8-bit
ceiling, and Qwen3.6 at 79.50% vs a 96% bar. Untried levers ranked in E36/
E37's wake: d-geometry, tail schedules, larger K. Which one binds?

**Method.** Change ONLY K: 256 -> 2048, d=4 held, same struct-e2 bases,
same bf16 value sources, same fitter. gemma on M4 (2653s, needed
--expert-chunk 16 after K=8192 died to a Metal command-buffer timeout),
Qwen3.6 on M3 (3097s). Score on the same KL caches; gemma additionally on
litbench generative+cyclic, the instrument its community-4bit incumbent was
measured with.

**Result.** relerr 0.31 -> 0.187 on BOTH families, and it converted on both:

- gemma: 3363 -> 1856 mnats, 42.65 -> 56.56% agreement (13.7G unpacked).
  litbench 86.54% vs bf16's 84.62% — AT the bf16 ceiling (not "above":
  n=104, SE ~3.4pp), vs community 4bit 79.81% at 15G.
- Qwen3.6: 79.50 -> 87.33% agreement; packs 17.6 -> 13.0 GiB (0.734x,
  120/120 tensors) and the packed artifact scores IDENTICALLY (85.535
  mnats to three decimals). Beats community 4bit (85.61% @ 19G) at 68% of
  its size. Still 9 points short of the 8bit bar (96.18%).

**Answer: K.** Codebook capacity, not geometry or scheduling, was the
K=256 bottleneck. Cost profile makes it strictly first-try: fit wall-clock
~same, size cost only at storage (8 -> 11 packed bits/code, 2.25 -> ~3.1
bpw effective on a fully packable family).

**Falsified along the way.** (a) "litbench is saturated at 79.81" — no,
K2048 moved it +6.73; the tie at K256 was two models at the same quality,
not an instrument ceiling. (b) "Qwen3.6 punishes compression too hard for
VQ to matter" — no, same lever, same conversion. (c) My packed-size quote
for gemma (~2.75 bpw) — down_proj (NSUB=176, 1/3 of code elements) can't
block-pack at 32 codes/block, so gemma's effective width is 12.7 bits ->
~3.42 bpw. Qwen (NSUB=128) packs fully.

**Open.** gemma K2048 packed size (running); whether 87.33% Qwen justifies
a publish given it misses the stated 8bit goal; 16-code block layout to
reclaim gemma's unpacked third; K=4096/8192 needs either a fitter that
survives Metal timeouts (smaller chunks?) or a different box.

## E44 (08-18, evening) — THE INSTRUMENT WAS BROKEN THREE WAYS, AND ONE FIT WAS TOO

**Context.** Built winrate_bench.py to settle the litbench/KL disagreement
on gemma (E43): blind paired win-rate on literary prose, judged by
Qwen3.8-27B q4, every pair judged in both orders. First run returned 60/60
"inconsistent" — zero decisive pairs. Diagnosis found FOUR distinct
failures, three in the instrument and one latent in the fitter.

**1. Generation never left the thinking channel.** 420 max_tokens; the 26b
plans/drafts/self-critiques for 3000+ tokens on a writing task. All 180
"generations" were raw reasoning — the tokenizer's channel markers
(<|channel> id 100 opens, <channel|> id 101 closes) confirmed no close ever
appeared. First diagnosis wrongly blamed the marker string itself; the
marker was correct, the budget was the bug.
FIX: apply_chat_template(..., enable_thinking=False) — gemma's template
pre-closes an EMPTY thinking block and prose starts immediately (verified:
first token is story text). Fair across models, kills the reasoning-length
confound, and 700 tokens suffices.

**2. The judge parser read our own prompt back.** Qwen3.8 is a reasoner; it
restates the instruction ('answer "1", "2", or "tie"') while thinking. A
first-match \b(1|2|tie)\b regex matched that restatement: 120/120
judgments returned "1".
FIX: require an explicit trailing "VERDICT: x" line and take the LAST
match. Lesson generalizes: NEVER first-match-parse a reasoner's output —
litbench hit the identical trap (E39) and solved it the same way.

**3. (What went right.) The dual-order guard converted garbage into an
obvious null.** Judging each pair in both orders and requiring agreement
meant a constant-"1" judge scored 0 decisive rather than a plausible 30-30.
A single-order design would have produced a believable, WRONG win-rate that
was one commit away from a model card.

**4. tail30 fit collapse — kmeans is UNSEEDED, fits are non-deterministic.**
tail30 (a strict superset of tail20's d2 coverage) scored WORSE than tail20
(160 vs 51 mnats, 83.79 vs 89.77%). Impossible if fits were healthy. Log
audit: L26 down_proj relerr 1.0000 (weight destroyed), four more tensors
0.12-0.26 — the SAME layers fitted at ~0.032 in tail20. kmeans centroid
init is mx.random with no fixed seed; a bad draw (or a transient Metal
fault — this was the M4, which also produced the K8192 command-buffer
timeout) collapses a tensor and the fitter shipped it silently.
FIXES, two layers:
  - fitter now refits (fresh init) any tensor over --relerr-abort 0.35 and
    ABORTS the run if still failing (vq_397b_codes.py).
  - verify_artifact.py decodes every tensor FROM THE ARTIFACT (packed or
    unpacked) and measures relerr against the bf16 source independently —
    catches fit collapse, packing faults, write faults, stale resume
    shards. Smoke-verified: packed tail20's decoded relerrs match the fit
    log to 4 decimals. RUN THIS BEFORE ANY HF UPLOAD.
Audit of all shipping artifacts (qwen K2048/K4096/tail20, gemma K2048):
clean, worst 0.215 = L00, legitimately hard. tail30 shard-2 refit queued
with the gate armed.

**Meta-lesson.** Every failure tonight was SILENT-PLAUSIBLE: reasoning that
looked like generations, a parse that looked like a verdict, a fit that
looked done. The pattern that caught each one was a REDUNDANT CHECK THAT
CANNOT SHARE THE FAILURE (order-swapped judging; a superset artifact
scoring worse than its subset; decode-from-artifact verification).
Instruments need controls before results are believed — same lesson as the
agreement-floor test (E41), which is why the bench has a bf16-vs-bf16
control in its design.

## E45 (08-18, evening) — QWEN3.6 TAIL LADDER: NEAR-PARITY AT 18.1G, THE K LADDER IS SPENT, AND relerr STOPPED PREDICTING

**Goal (Noah).** A second Qwen artifact: 8-bit quality (96.18% agree /
0.999x ppl @ 35G) at the smallest size — "8bit parity someone can run on a
32GB machine is a perfect accessibility artifact."

**Rungs built this evening** (all packed, all verified vs the same KL cache
and referee ppl):

| rung | packed | ppl | vs bf16 | KL (mnats) | agree |
|---|---|---|---|---|---|
| mlx-community 8bit | 35G | — | 0.999x | — | 96.18% |
| **vq-tail20-d2k2048** | **18.1G** | 4.7541 | **1.007x** | 50.8 | 89.77% |
| vq-K4096-d4 | 13.9G | 4.8100 | 1.019x | 68.5 | 87.88% |
| vq-K2048-d4 | 13.0G | 4.8584 | 1.029x | 85.5 | 87.33% |
| mlx-community 4bit | 19G | — | 1.041x | — | 85.61% |

**FINDING 1 — the K ladder is exhausted.** 2048->4096 halved nothing:
+0.55 agree for +0.9G, after 256->2048 bought +7.83. Do not chase K=8192.

**FINDING 1b (added 08-19) — AND THE COST IS SUPERLINEAR, which is
independently decisive.** The quality argument above says "not worth it";
the cost argument says "not reachable". bpw = log2(K)/d, so each +0.25 bpw
at fixed d DOUBLES K, while k-means assignment costs O(n*K*d). Extrapolated
from a measured gemma d4-K8192 fit (4255 s):

| d4 target | K | fit time |
|---|---|---|
| 3.50 bpw | 8,192 | 1.2 h (measured) |
| 4.25 bpw | 65,536 | 9.5 h |
| 5.75 bpw | 4,194,304 | 605 h |

So the two curves close from both ends: quality flattens (this entry) while
cost explodes. Note this is the ASSIGNMENT search, not memory — E50's
scatter-add removed the memory wall and the ladder still stops. Only
approximate NN or hierarchical k-means would move it.

Credit: the 397B session, who pointed out that E45 was making only the
quality half of the argument.

**FINDING 2 — relerr stopped predicting exactly there.** Fit relerr stayed
log-linear right through K=4096 (0.313 -> 0.187 -> 0.158, ~16%/doubling)
while agreement flatlined. relerr is a trustworthy proxy only until it
isn't; it gave no warning. Score early, never extrapolate more than one
rung ahead of a scored point (the "K=4096 ~90%" prediction was wrong).

**FINDING 3 — the tail law transfers to Qwen3.6 and shows up on PPL.**
tail20 (L20-39 at d2k2048, body d4k2048): tail relerr 0.032 vs body 0.187
(~6x), and ppl lands at 1.007x vs 8-bit's 0.999x — 0.8 points of ppl from
parity at 52% of 8-bit's size. Also beats the 19G community 4bit while
smaller. This is the accessibility artifact.

**FINDING 4 — the two instruments diverge, and the split is informative.**
tail20 is 1.007x on ppl but 89.77% on agreement (8bit: 96.18%). For MoE
this is the expected signature: routing flips swap one plausible token for
another — ppl absorbs it, top-1 agreement punishes it. Same shape as the
gemma litbench/KL disagreement (E43/E44). Quote BOTH numbers; neither alone
is the truth.

**FINDING 5 — a near-lossless tail moved agreement only +2.44.** So the
BODY (L0-19 at d4) is the binding constraint on agreement now. Extending
d2 to the body (~23G) extrapolates to ~92%, still short of 96 — full
agreement-parity is NOT reachable under ~32GB with this fitter. Parity on
ppl effectively already happened at 18.1G.

**Bookkeeping.** tail30 collapsed in fit (E44), shard-2 refit queued with
the sanity gate armed; expect it at ~20.7G packed between tail20 and d2-body
if healthy. gemma sighted artifacts measured: K256 9.43G, K2048 12.53G
(vision graft +1.07G, text path KL-identical). Blind win-rate verdicts
(winrate_bench, E44's fixed instrument) still running; results go in
CRUSH_RESULTS when they land.

## E46 (08-18, late) — d=2 GEOMETRY: HEADLINE RETRACTED, SEE THE BRACKET AT THE END

**RETRACTED HEADLINE (corrected same night, see BRACKET below).** This entry
originally claimed "at roughly matched bytes, halving d beats raising K." It
does not. The comparison it rested on — d2-K256 at 4.00 bpw vs d4-K2048 at
2.75 bpw — is a 45% BIT RAISE, not matched bytes, and the error flattered the
new result. Caught by the 397B session. The matched-bpw bracket at the end of
this entry is the corrected finding. What survives: d=2 keeps CLIMBING where
the d4 K-ladder is known to flatten, and it is far cheaper to fit.

gemma-4-26b-a4b expert fits (same base, same source, same fitter):

| geometry | code bpw | mean relerr | note |
|---|---|---|---|
| d4 K256  | 2.00 | 0.3136 | the 9.43G small artifact |
| d4 K2048 | 2.75 | 0.1877 | the 12.53G quality artifact |
| d2 K256  | 4.00 | 0.0873 | 2.15x better than d4K2048 |
| d2 K512  | 4.50 | **0.0589** | best gemma fit achieved |

Fit COST also fell: d2K256 took 313s vs d4K2048's 2653s (smaller codebook
dominates the k-means cost), so the better geometry is also 8x cheaper to
search.

**WHY WE MISSED IT FOR SO LONG — no d=2 kernel existed.** vq_switch.py
shipped fused decode kernels for d=4 (float4) and d=8 (2x half4) only.
`_fused` is never passed in_features, so it ASSUMES d=4 and derives
NSUB=IN/4. A d=2 artifact therefore read across codebook entries and emitted
pure `<pad>` — while `_prefill` (which DOES receive in_features) decoded it
correctly. Net effect: d=2 scored fine on every teacher-forced instrument
(KL 949.960 / 68.27%, verify_artifact PASS at relerr 0.087) and generated
nothing. Forcing SCOUT_VQ_FUSED_MAX_N=0 proved the diagnosis: correct prose
at 10.8 tok/s vs 48.7 for d=4.
FIXED (commit 4b2d016): d=2 fused kernel + EXPLICIT dim dispatch that raises
on unsupported (dim, pack_bits) instead of falling through to another dim's
kernel. Verified independently: prose correct with no env var, KL bit-identical
at 949.960/68.27%, d=4 unregressed at 47.2 tok/s, and d=2 now runs at
**51.0 tok/s — FASTER than d=4** (K<=256 means uint8 codes and a codebook
small enough to sit in threadgroup memory).

**METHODOLOGICAL RULE — QWEN IS SCORED ON PPL, FULL STOP.** Twice tonight a
Qwen call was made on top-1 agreement and twice it was wrong:
  - K=4096 predicted ~90% from relerr, landed 87.88% (+0.55 over K2048).
  - tail30 was dismissed as "+0.53 agreement for +2.6G, a poor trade" — then
    ppl showed 4.7210 vs bf16's 4.7215, i.e. **1.000x, PARITY**, where
    tail20 sits at 1.007x. tail30 is the accessibility artifact, not tail20.
MoE routing is discrete, so any perturbation flips which experts fire and
agreement punishes that categorically while ppl absorbs it. Qwen's ppl is
VALID (unlike gemma's) — use it, and treat agreement as a secondary signal.
Corollary: gemma has no ppl, so gemma decisions need the BLIND WIN-RATE, not
KL alone (E44/winrate_bench). Testing is mandatory for gemma, not optional.

**relerr is a proxy that expires.** It stayed log-linear across the whole K
ladder (0.313 -> 0.187 -> 0.158, ~16%/doubling) while agreement flatlined at
K=4096. Never extrapolate more than one rung past a SCORED point.

**Open at time of writing.** Qwen flat d2-K256 fitting (~18.8G projected, no
packing needed since uint8) — if it matches tail30's 1.000x parity it wins on
size and simplicity, since tail20/tail30 are mixed-geometry and need packing.
gemma d2-K512 packing/scoring (~16G sighted projected). Both d2 gemmas still
need blind win-rate judging before any quality claim.

## E47 (08-18/19, overnight) — M4 COMPUTE IS INTERMITTENTLY WRONG, AND THE VENV HOLDS THE LIVE RUNTIME

Two infrastructure findings that invalidate results silently. Both were
caught by verify_artifact.py, neither by any fit log.

### 0. TWO DIFFERENT FAILURE MODES — do not conflate them (397B session)

  LOUD: kIOGPUCommandBufferCallbackErrorTimeout. It RAISES, the run dies,
  nothing reaches disk. Retry and move on. Seen on both boxes under load.
  SILENT: wrong compute that passes the fitter's own relerr gate and LANDS
  IN THE ARTIFACT. Only this one needs a standing verification gate.
Conflating them makes verification look more burdensome than it is.

### 1. M4 (nozzlebook-pro, M4 Max 128GB) returns wrong compute intermittently

Controlled A/B — SAME artifact file, SAME deterministic verification, hashes
of the per-tensor relerr list:

    M3  run A  9ca617fa...   run B  9ca617fa...     <- identical
    M4  run A  eafc930d...   run B  9ca617fa...     <- DIFFERENT

M4's run B agrees with M3, so 9ca617 is the truth and run A was simply
wrong. A follow-up 5 runs on M4 all returned the correct hash (including 5
taken DURING a concurrent fit), so the fault is INTERMITTENT, roughly 1 in 7
observed here — not load-triggered in any way we could pin down.

Corrupted artifacts M4 produced tonight, all passing the fitter's own gate:
  - tail30 (qwen): 5 tensors collapsed, one at relerr 1.0000, while the SAME
    layers fitted at 0.032 in tail20. Scored 160 mnats/83.79%; after refit on
    M3, 46.8 mnats/90.30% — the difference between "not worth shipping" and
    the PARITY artifact.
  - gemma d2-K512: fit log reported worst 0.0611; the WRITTEN artifact held 4
    tensors at 0.54-0.99. Refit on M3: clean, 744 mnats / 72.72%.
  - qwen d2-K64: 3 tensors >0.5 (L11 gate_proj 0.988, L38 down_proj 0.957).
  - Also the only box to hit kIOGPUCommandBufferCallbackErrorTimeout (K=8192,
    and again at K=32 where codebook size cannot be the cause).
M3 (M3 Ultra) produced ZERO corrupted artifacts across the same night.

**POLICY: every M4-fitted artifact is verified ON M3 before any number from
it is believed.** Fit logs structurally cannot see this — the fitter reports
what it COMPUTED, not what reached disk.

**USE `--outlier 3.0`, NOT `--threshold`.** An absolute bar is
geometry-specific: healthy d4-K128 on the 397B lands ~0.46, so 0.35 would
fail every tensor and the gate would be ignored within a day. Corruption is
an OUTLIER AGAINST THE ARTIFACT'S OWN PEERS — tail30's dead tensor read
1.0000 beside peers at 0.032. `--outlier MULT` flags anything above MULT x
the artifact's own median and needs no per-geometry tuning. Validated both
ways: catches all 3 corrupt tensors in qwen d2-K64 (5.7-5.9x median) and
passes a clean artifact with no false alarm. Credit: 397B session.

### 2. There are TWO copies of vq_switch.py and the LIVE one is in the venv

The artifact shim does `importlib.import_module(f"mlx_lm.models.{model_type}")`,
so the VQ runtime resolves to
`venv/lib/python3.12/site-packages/mlx_lm/models/vq_switch.py` — NOT the copy
add_model_file.py inlines into each artifact's model.py (dead code for these
loads). A patched repo vq_switch.py therefore has NO effect until copied into
the venv.

Cost: ~1h chasing a "broken d=2 uint16 decode" that was a stale venv copy
still holding the old fall-through dispatch. Compounding it, the real
artifact WAS also corrupt (finding 1), so two independent faults produced one
symptom and each masked the other. The decode kernels are in fact D-generic:
verified numerically for d2-uint16 K=512 against a numpy reference, max rel
diff 2.5e-4.

**POLICY: patching vq_switch.py means `cp vq_switch.py $VENV/lib/python*/
site-packages/mlx_lm/models/`, on every box.**

### BRACKET (the corrected, matched-bpw finding)

Same base/source/fitter, same KL cache, gemma-4-26b-a4b:

| geometry | code bpw | agreement |
|---|---|---|
| d4 K256  | 2.25 | 42.65% |
| d2 K32   | 2.50 | 48.84% |
| **d4 K2048** | **2.75** | **56.56%** |
| **d2 K64**   | **3.00** | **57.68%** |
| d2 K256  | 4.00 | 68.27% |
| d2 K512  | 4.75 | 72.72% |

**AT MATCHED BYTES, d=4 WITH A BIG CODEBOOK WINS.** Interpolating the d2 line
to 2.75 bpw gives ~53.3%, BELOW d4-K2048's 56.56%. d2-K64 beats the d4 anchor
by only +1.12 points while spending 9% more bits. Returns per bit in this
regime actually favour d4 (27.8 pts/bpw for K256->K2048 vs 17.7 for
K32->K64).

**WHAT IS STILL TRUE, and why d=2 is not dead:**
1. d=2 keeps climbing to 68.27% (4.00 bpw) and 72.72% (4.75 bpw), while the
   d4 K-ladder is KNOWN TO FLATTEN — qwen K2048->K4096 bought +0.55 points
   for +0.9G (E45). Nobody has a d4 comparator above K2048 on gemma, so the
   high-bpw regime is untested, not won.
2. Fit COST is genuinely 8x lower and is independent of the bits question:
   d2-K256 313s vs d4-K2048 2653s (smaller codebook dominates k-means).
3. K<=256 at d=2 needs NO packing (uint8, byte-aligned) and decodes FASTER
   than d4 (51.0 vs 47.2 tok/s), sidestepping gemma's NSUB=176 stranded
   third entirely.

**PRE-REGISTERED PREDICTION, RESOLVED.** The 397B session pre-registered,
BEFORE d2-K64 existed, that if d2 sat on the d4 line it would land at 63.52%
(d4 slope 27.82 pts/bpw from the K256->K2048 interval), with <61% meaning
"d2 is worse per bit" and >66% meaning "d2 genuinely steeper".
    d2-K64 actual: 57.68%  ->  5.84 points BELOW the d4 line.
And the gap WIDENS with bpw (-0.77 at 2.50, -5.84 at 3.00). On gemma, at
matched bytes, d=2 is WORSE per bit than d=4 with a large codebook.

Qualifier that cuts both ways: the d4 line is EXTRAPOLATED past K2048, and
E45 showed the d4 K-ladder flattens (qwen K2048->K4096 = +0.55 for +0.9G).
So 63.52% probably overstates real d4 up there. Hence:

**THE TEST THAT SETTLES IT** (running): gemma d4-K8192 at 3.50 bpw. The d2
line interpolates to ~63% there. If d4-K8192 lands below that, d=2 scales
further and the whole lineup is worth re-walking; if it matches or beats it,
d=2 is only a fit-cost and packing convenience, not a quality lever.

**METHOD NOTE.** The original error is the one to remember: comparing two
points that differ in TWO variables (geometry AND bits) and attributing the
whole difference to the one being advocated. Bracket the anchor next time —
it cost 6 minutes of fitting to answer.
## E48 (08-19) — A BOUNDED METRIC CANNOT BE LINEARLY EXTRAPOLATED: the agreement ladder is a chord of a saturating curve

**Origin.** The 397B session pre-registered d2-K64 = 63.52% from a d4 slope
of 27.82 agreement-points/bpw fitted over K256->K2048 (42.65 -> 56.56%).
Measured: 57.68%. They then retracted their own reasoning before the result
could be used, and the retraction is the finding.

**THE ERROR, STATED GENERALLY.** Top-1 agreement is bounded above by a
MEASUREMENT FLOOR, not by 100%. E41 put that floor near ~82% for this
family: even a near-lossless 8-bit gemma quant only agrees with bf16 79.95%
of the time, because MoE routing is discrete and any perturbation flips which
8-of-128 experts fire. So the agreement-vs-bpw curve SATURATES toward ~80,
and any slope fitted in the 40-60% band is a CHORD of that curve, not its
tangent. Extrapolating the chord upward systematically OVERSTATES the
ceiling-ward arm.

How badly: extending the same 27.82 pts/bpw line to 3.50 bpw predicts 77.4%
— within 5 points of the floor itself, i.e. the linear model claims K8192
would be nearly indistinguishable from bf16. Nobody believes that, and that
implausibility was visible BEFORE the fit was run.

**RULE.** Put the metric's FLOOR next to every agreement-vs-size table, so
saturation is visible at a glance rather than discovered by extrapolating
through it. Fit slopes only over the interval they were measured on, and
never predict a rung above ~60% agreement from a slope fitted below it.

**This is the same shape as E45's "relerr is a proxy that expires", one
level up.** E45: the FIT PROXY stopped tracking quality (relerr stayed
log-linear while agreement flatlined at K=4096). E48: the SCORING METRIC
stops tracking damage linearly as it approaches its own floor. Both are
"this number has a domain of validity"; between them they cover the fit side
and the score side of the same ladder.

**PRE-REGISTERED, PENDING (gemma d4-K8192 @ 3.50 bpw, fitting on M3).**
Interpolating d4 between K2048 (2.75, 56.56%) and K8192 (3.50, X), the d4
line passes exactly through d2-K64's 57.68% at 3.00 bpw when X = 59.92%.
  X > ~59.9%  -> d4 still wins at 3.00 bpw; d=2 is the worse lever
  X < ~59.9%  -> the d4 line bent UNDER d2; d=2 is competitive after all
  X ~ 59.9%   -> the two geometries are indistinguishable per bit here
Recorded BEFORE the number exists, by both sessions.

## E49 (08-19) — QWEN: A CHEAPER TAIL STRICTLY DOMINATES A RICHER ONE

**Setup.** tail30 (layers 10-39 at d2, body 0-9 at d4-K2048) reached ppl
PARITY at 20.7G with a d2-K2048 tail (5.75 bpw). Hypothesis: the tail is
OVER-fed — drop the tail codebook and the artifact gets smaller without
losing parity.

**Result — it got smaller AND better, on every axis:**

| rung | packed | ppl vs bf16 | KL | agree |
|---|---|---|---|---|
| mlx-community 8bit | 35G | 0.999x | — | 96.18% |
| **tail30 d2k512 tail** | **17.9G** | **0.991x** | 44.6 | **90.75%** |
| tail30 d2k2048 tail | 20.7G | 1.000x | 46.8 | 90.30% |
| tail20 d2k2048 tail | 18.1G | 1.007x | 50.8 | 89.77% |
| mlx-community 4bit | 19G | 1.041x | — | 85.61% |

Strict dominance: -2.8 GiB, better ppl, better agreement. **Spending FEWER
bits on the tail beat spending more.**

**READ CAREFULLY — WHAT CHANGED IS THE ALLOCATION, NOT THE TOTAL.** Both
rungs pay for 30 tail layers; the cheaper one simply stops over-serving them
and the bytes it frees are not spent anywhere. So this is not "less is more"
in general — it says the d2-K2048 tail was PAST ITS OWN KNEE, i.e. the tail
saturates in codebook size just as the body's K-ladder does (E45). The
lesson generalizes as: when a schedule and a geometry are tuned separately,
the schedule's optimum moves once the geometry improves. tail30's depth was
chosen when the tail was d4; nobody re-tuned it after d2 arrived.

**0.991x is BELOW bf16 and that is not a bug.** Mild quantization slightly
reducing referee ppl has been seen repeatedly in this lab (Qwen3.8 q4 at
0.996x, E40; mlx-community 8bit at 0.999x). Treat anything in 0.99-1.00x as
"at parity", not as "beats the teacher" — the corpus is finite and the
effect is within its noise.

**Next rung queued:** tail30 with a d2-K256 tail (4.25 bpw). If cheaper wins
again the knee is lower still, and K<=256 needs no packing for those tensors.

### 3. SCOPED NEGATIVE RESULT — no M4 corruption reached the Hub (397B session, 08-19)

All three PUBLISHED 397B artifacts verified: decoded from the artifact files
(not fit logs) against the 751G bf16 source, on the M3, MLX pinned to CPU so
the GPU was never involved. 171 tensors each, 513 total, ZERO corruption.

| artifact | mean relerr | worst | worst/mean | outlier(3x) |
|---|---|---|---|---|
| 2.2bpw d4-K128 | 0.3698 | 0.4655 | 1.26x | clean |
| 2.4bpw d4-K256 | 0.3156 | 0.4332 | 1.37x | clean |
| 3.1bpw d4-K2048 | 0.1936 | 0.2963 | 1.53x | clean |

Two independent cross-checks make this believable rather than merely a PASS:
  - the 2.4bpw VERIFIED mean (0.3156) equals the FIT-TIME mean recorded in
    EXPERIMENTS.md to four decimals, from a decode path sharing no
    reconstruction code with the fitter;
  - E36 measured this same model at d4-K256 layer 0 as down 0.1930 /
    gate_up 0.4161 months ago with different code; the 2.4bpw L00 gate/up
    land at 0.4163/0.4189.
Worst tensors are at L00/L01 in every artifact — the known layer-0-is-hard
structure (E37), not damage.

Scope it honestly: whatever the ~1-in-7 M4 error rate applies to, it did not
hit these fits. The fits behind the release were either M3 or lucky, and we
should NOT retro-fit a story about which.

**AND THIS IS THE CONCRETE CASE FOR `--outlier` OVER `--threshold`.**
`--threshold 0.35` would have FAILED ALL THREE of these clean, shipped
artifacts — the 2.2bpw alone has 100+ healthy tensors above 0.35. `--outlier
3.0` passes all three by an enormous margin (worst case 1.26-1.53x mean
against a 3x bar). A fixed bar would have produced three false alarms on a
verified-good release, and a gate that cries wolf gets widened until it
catches nothing. Median-relative is the design that survives.

## E50 (08-19) — THE BRACKET, WITH THE bpw ARITHMETIC FIXED: d4 wins per bit, but SATURATES; d2 keeps going

**FIRST, MY OWN ERROR.** E46's bracket used WRONG bpw values. Packed bpw is
`log2(K)/d + 16/64`, so d4-K2048 is **3.00** bpw (I wrote 2.75) and d2-K32 is
**2.75** (I wrote 2.50). Every "matched-bpw" statement in E46, and the
397B session's pre-registered 59.92% boundary which was computed FROM MY
NUMBERS, inherited the error. The conclusions survive; the arithmetic did
not. Corrected ladder, all points MEASURED on the same base/source/fitter/
KL cache (gemma-4-26b-a4b):

| geometry | packed bpw | agreement |
|---|---|---|
| d4-K256 | 2.25 | 42.65% |
| d4-K512 | 2.50 | 45.04% |
| d2-K32 | 2.75 | 48.84% |
| d4-K2048 | 3.00 | 56.56% |
| d2-K64 | 3.25 | 57.68% |
| d4-K8192 | 3.50 | 61.32% |
| d2-K256 | 4.25 | 68.27% |
| d2-K512 | 4.75 | 72.72% |
| d2-K1024 | 5.25 | 75.90% |
| **d2-K2048** | **5.75** | **77.89%** |
| (8-bit ceiling) | — | 79.95% |

**AT MATCHED BPW, d4 WINS — BY 1-2 POINTS, NOT 6.** Interpolating d4 between
MEASURED neighbours (no extrapolation this time):
    2.75 bpw: d2-K32 48.84% vs d4 50.80%  -> d4 +1.96
    3.25 bpw: d2-K64 57.68% vs d4 58.94%  -> d4 +1.26
E46's "-5.84" came from extrapolating a d4 chord fitted at 42-56%, exactly
the saturation error E48 describes. The real d4 slope COLLAPSES with bpw:
27.8 pts/bpw over K256->K2048, then **9.5** over K2048->K8192.

**BUT d4 CANNOT REACH THE INTERESTING REGION AT ALL.** bpw = log2(K)/d, so at
d=4 every +0.25 bpw costs a K DOUBLING. Reaching d2-K2048's 5.75 bpw would
need K = 2^22 — infeasible in memory, in fit time, and in codebook storage.
d4 tops out around K=8192 / 3.50 bpw / 61.32%. d=2 reaches 5.75 bpw with
K=2048 and lands at **77.89%, within 2.06 points of the 8-bit ceiling.**

**THE HONEST SYNTHESIS.** d=2 is NOT a better use of a bit. It is the only
geometry that can SPEND enough bits to approach the ceiling, because K grows
exponentially in (bpw x d). Where both are feasible, prefer d4. Where you
want quality above ~3.5 bpw, d4 is not an option and d2 is. That is a
different and more useful claim than either "d beats K" (E46, retracted) or
"d2 is a bit-buying illusion" (the reading after the first bracket).

**METHOD.** Three wrong conclusions in one night, all from the same family of
error: comparing points that differ in two variables, extrapolating a chord
of a saturating curve, and finally plain arithmetic in the x-axis itself.
The fix that actually worked was measuring the neighbours instead of
interpolating from far away — d4-K512 and d4-K8192 cost ~1.3h combined and
turned a 6-point claim into a 1-2 point one.

## E51 (08-19) — BLIND JUDGING: the d2 gemmas are INDISTINGUISHABLE from bf16, and the judge reproduces the KL ordering

**The question this closes.** Since E43 two valid gemma instruments disagreed:
litbench said VQ-K2048 was at the bf16 ceiling while KL said it diverged 4.2x
more than 8-bit. E44 built the blind win-rate to break the tie, and E44's
first run judged bf16 vs the two d4 artifacts. This is the same instrument
run against the three d2 rungs built overnight.

**Protocol.** 60 literary continuations per artifact, greedy, generated
through each model's own chat template with thinking disabled. Pairs
anonymized A/B with the assignment RANDOMIZED per pair and the key withheld
in a separate file the judge was never pointed at. Judge: claude-sonnet-5,
told explicitly that ties are legitimate. Decoded afterwards; exact
two-sided sign test on decisive pairs.

| artifact | KL agree | bf16 | quant | tie | p | verdict |
|---|---|---|---|---|---|---|
| d4-K256 (E44) | 42.65% | 34 | 12 | — | 0.0016 | bf16 SIGNIFICANTLY better |
| d4-K2048 (E44) | 56.56% | 36 | 20 | 4 | 0.044 | bf16 SIGNIFICANTLY better |
| d2-K512 | 72.72% | 23 | 16 | 21 | 0.34 | indistinguishable |
| d2-K1024 | 75.90% | 16 | 25 | 19 | 0.21 | indistinguishable |
| **d2-K2048** | **77.89%** | **11** | **23** | **26** | 0.058 | indistinguishable |

**FINDING 1 — the artifacts we were shipping yesterday were MEASURABLY WORSE
than bf16; the d2 rungs are not.** Both d4 artifacts lose significantly. No
d2 rung does. gemma-quality at 18.74G sighted is statistically
indistinguishable from 48G bf16 on blind literary judging.

**FINDING 2 — THE JUDGE REPRODUCES THE KL ORDERING, and this is the real
result.** bf16's share of DECISIVE pairs falls monotonically down the KL
ranking: 74% -> 64% -> 59% -> 39% -> 32%, across five artifacts and two
geometries. An independent, blind, human-style instrument recovered the
ordering our automated metric predicted. That is the corroboration litbench
could not supply because it had saturated (E43). It also rehabilitates KL for
gemma: it over-reports the MAGNITUDE of MoE damage, but it RANKS correctly.

**FINDING 3 — ties rise with quality:** 4 -> 21 -> 19 -> 26. On d2-K2048 the
judge cannot separate the texts in 43% of pairs.

**DO NOT READ THIS AS "THE QUANT BEATS bf16".** The judge has a strong
positional lean toward B (raw splits 33-6, 24-17, 24-10) and bf16 sat in A
for 34/60, so the bias pushes TOWARD the candidate. The defensible ceiling on
this data is "indistinguishable", not "better", and d2-K2048's p=0.058 points
the wrong way to be claimed as a win.

**WHY THE ORDERING SURVIVES THE BIAS.** All three pair sets were built with
the same seed over the same sorted ids, so the A/B assignment is IDENTICAL
across them (verified). The positional confound is therefore the SAME
constant in all three comparisons — it contaminates each absolute number but
CANCELS when the artifacts are compared with each other. Design note for
reuse: keep the pair assignment fixed across candidates for exactly this
reason; it converts an instrument bias into a common-mode term.

**Instrument caveats that remain.** One judge, one pass, no dual-order
replication (E44's dual-order local judge was too weak to rank these at all —
47/60 order-inconsistent). n=60 with up to 26 ties leaves as few as 34
decisive pairs, so confidence intervals are wide. A second judge from a
different family, or dual-order replication, would tighten this; neither
changes the ordering, which is the load-bearing part.

## E52 (08-19) — THE FUSED ROW-GATHER LEVER DOES NOT EXIST: MLX ALREADY FUSES IT

**Retires the standing "remaining lever, unbuilt" from the VQ-PF1 entry.**
That note said ~21% of block time goes to materializing the `[N, IN]` fp16
row matrix `gather_qmm` never needs, and that fusing the gather into the
matmul was worth it "if someone needs the last 1.19x". Measured properly, the
recoverable amount is **zero**.

**Two instruments, and the second one overturns the first.**

1. Timing the `_prefill` inner loop with `mx.eval` BETWEEN the gather and the
   matmul reproduces the historic figure almost exactly — across all three VQ
   modules of a real block, real router histogram, chunk 32, n=3:

   | stage | ms | share of VQ `_prefill` |
   |---|---|---|
   | decode | 17.87 | 33.7% |
   | **gather** | **12.88** | **24.2%** |
   | matmul | 22.32 | 42.1% |

   Against a whole-block time of 107.2 ms (n=3, same shape), that is **12.0%
   of block** — so even taken at face value the old "21% of block" was stale,
   roughly halved by count-sort + chunk-32 shrinking `cap` (xp is
   `[ne, cap, IN]`, so its size scales with the padding those fixes removed).

2. But the real `_prefill` does NOT evaluate them separately — it builds one
   graph per chunk and calls `mx.eval` once, at the end. Timing gather+matmul
   as ONE eval, n=3:

   | | run 1 | run 2 | run 3 |
   |---|---|---|---|
   | separate (gather + matmul) | 2.02 | 1.68 | 1.58 |
   | together, one eval | 1.23 | 1.41 | 1.30 |
   | **left for a hand-written kernel** | **-0.29** | **+0.06** | **+0.04** |

   MLX already absorbs 16-39% of the separate sum. What remains is 0.04 ms on
   a 1.3 ms operation — indistinguishable from zero, and negative once.

**THE LESSON, and it is the general one: an instrument that serializes what
production fuses will invent work that does not exist.** The `mx.eval` I
inserted to attribute time between two stages is the same `mx.eval` that
destroys the fusion being measured. Any lazy-graph framework has this hazard,
and the tell is that the "cost" belongs to an intermediate the real code never
materializes on its own. Before optimizing a stage, time the stage boundary
you actually ship — not the one you inserted to see inside.

**Decision: NOT BUILT, and the lead is closed rather than deferred.** A fused
gather+LUT-matmul Metal kernel is substantial work, changes summation order
(the chunk-32 result showed fp16 order moves published ppl), and buys ~0%.
Contrast count-sort and chunk-32, which shipped because they were measurable
end-to-end AND bit-exact.

Corollary for the runtime as it stands: with the gather effectively free, VQ
`_prefill` is decode 34% / matmul 42% / gather ~0, so the only remaining VQ
tax over affine is the weight decode itself. That is the fused-kernel path
(`_fused`), already measured and rejected as a wash end-to-end at the default
prefill step.

## E53 (08-19) — NO DEPTH GRADIENT IN FIT ERROR FOR VQ CODEBOOK: UNIFORM K IS DEFENSIBLE ON THE FIT SIDE (see the AMENDMENT — relerr is now known to INVERT against ppl)

**The question.** E25 established on THIS model that concentrating affine bit
promotions in the last 10 layers beat spreading them at matched size (tail10
3.0157 vs spread10 3.0490). The published VQ lineup is FLAT — one K across
all 171 modules — so the obvious lever was: carry E25's tail law over to VQ,
body at K128 + tail at K2048, landing ~2.3bpw against the 2.4bpw's 2.00. The
question this settles is whether the fit side supports it.

**It does not.** `probe_k_fit_time.py`, gate_up_proj, K128 vs K2048, bf16
source, 64 experts assigned:

  | layer | K128 | K2048 | reduction |
  |---|---|---|---|
  | 0 | 0.4427 | 0.2978 | 32.7% |
  | 3 | 0.4038 | 0.2409 | 40.3% |
  | 14 | 0.3698 | 0.2011 | 45.6% |
  | 28 | 0.3686 | 0.1888 | 48.8% |
  | 40 | 0.3689 | 0.1879 | 49.1% |
  | 42 | 0.3683 | 0.1867 | 49.3% |
  | 56 | 0.3696 | 0.1941 | 47.5% |

1. **The body is flat.** From L14 on, K128 relerr spans 0.3683-0.3698 across
   42 layers — a **0.4% spread**. K2048 spans 7.7%. There is no depth trend
   to exploit.
2. **The deepest layer is not the neediest.** L56 (0.1941) is 2.8% WORSE than
   L28 (0.1888) at K2048 — non-monotone, so "spend late" has no fit-side
   justification.
3. **What looked like a gradient was a SHALLOW ANOMALY.** An n=2 probe (L3 vs
   L40) read 40.3% vs 49.1% and looked like a clean depth trend. Five points
   show it is L0-L3 being anomalous while everything else is uniform. n=2
   cannot distinguish an endpoint anomaly from a gradient — this is E37's
   two-point rule biting from the other side: two points are enough to
   FALSIFY a claimed trend, never enough to ESTABLISH one.
4. **The two projections have OPPOSITE depth profiles, and they cancel.**
   down_proj gets worse with depth (L3 0.3084 -> L40 0.3692); gate_up gets
   better then flattens (L3 0.4038 -> L40 0.3689 -> flat). E37 recorded the
   down_proj half ("layer 0 is anomalously easy to VQ"); the gate_up half runs
   the other way. Any schedule tilted toward either end helps one projection
   at the other's expense.
5. **The hard tensors are not codebook-limited.** L0 gate_up gains only 32.7%
   from a **16x** larger codebook, vs ~49% in the body. Shallow gate_up is
   limited by something other than codebook capacity (subvector geometry or
   intrinsic outliers), so a "head" schedule is not indicated either. This is
   consistent with the shipped artifacts, whose worst tensors are L00/L01 in
   all three (verify_artifact, 513 tensors).

**CONCLUSION: uniform K is close to optimal for VQ on this model, and the flat
lineup is a defensible design rather than an unexamined default.** Not built.

**Scope limit, stated so nobody over-reads this.** relerr is a FIT proxy and
E45 says it expires; this closes the fit-side case only. E25's output-scored
tail law is formally untested under VQ. But with zero fit-side support,
opposite-signed projections, and a transfer across quantization schemes
(affine bit-promotion -> VQ codebook size), the expected value did not justify
a ~2.9h fit plus scoring. If anyone revisits, the bar is an output-scored
matched-size build, not another proxy.

Cost of settling it: ~20 minutes of probing against a ~3h fit avoided.

**AMENDMENT (08-19, same day) — A DIRECT COUNTEREXAMPLE. relerr does not
merely expire at extremes (E45); it can RANK TWO BUILDS BACKWARDS.** The
sister Qwen3.6 line measured both quantities on the same pair of artifacts:

  | build | blended fit relerr | measured ppl |
  |---|---|---|
  | `tail30-d2K512` (L0-9 CHEAP d4K2048, L10-39 d2K512) | ~0.090 (WORSE) | **0.991x (BETTER)** |
  | `tail30-d2K256` (L0-9 CHEAP d4K2048, L10-39 d2K256) | ~0.109 (WORSE) | **1.002x (BETTER)**, 1.2 GiB SMALLER |
  | flat d2-K256 (uniform) | ~0.084 (best fit) | 1.016x (worst ppl) |

relerr ranks flat first; perplexity ranks it LAST, at essentially matched
size. **The ordering inverts, twice, in the same direction.** Whatever a
schedule exploits is invisible to reconstruction error — exactly the regime
this entry's own probe operates in.

**NAMING WARNING, and it cost the sister line a night of wrong theory.**
Those builds are named "tail30" for the layers being PROMOTED, but on a
`--k 2048 --dim 4` base `--tail-from 10 --tail-geom d2k512` yields layers
0-9 at CHEAP d4-K2048 and 10-39 rich. **They are HEAD-DOWNGRADE builds.**
Read as tail-promotions they suggest "spend more late"; read correctly they
suggest "spend LESS early" — the opposite lever, and the one that makes an
artifact SMALLER. Name a variant for the variable that MOVED, not for the
end you were thinking about.

Consequence for THIS entry, stated plainly: the five-layer sweep above
remains correct about what it measured — fit error has no usable depth
gradient on the 397B, and down_proj/gate_up genuinely cancel. But
"THE TAIL LEVER IS DEAD" was too strong a title for relerr evidence, and is
retitled. The honest claim is: **there is no FIT-SIDE reason to tilt a
schedule by depth, and no fit-side instrument can rule a tail IN or OUT.**
Settling it for the 397B would need an output-scored matched-size build, the
same bar E25 met on affine.

Related: the same probe found Qwen3.6 FLAT across depth at both d2-K512 and
d4-K2048 (largest spread 0.5%) — so two families reach "no fit-side depth
signal" by different mechanisms (397B: opposing profiles that cancel;
Qwen3.6: never diverging). Both are silent on the output side.

Method note worth keeping: the verdict logic in the first version of that
probe reported the SIGN of the depth difference without testing its
MAGNITUDE, so a 0.5% drift printed as "a depth effect is physically
available". A comparison script needs a threshold, not just a direction —
otherwise it manufactures a signal out of noise and reads as confirmation.

## E54 (08-19) — THE PACKING ASYMMETRY: d4 wins per BIT and loses per BYTE, on gemma only

**The contradiction.** E50's matched-bpw bracket showed d4 ahead of d2 by
1-2 agreement points. But plotted against SIZE, the gemma d2 curve sits
ABOVE the d4 curve at every point, with no crossover. Both are correct; the
difference is entirely packing.

**Cause.** `vq_pack` packs codes in blocks of 32, so a tensor packs only if
NSUB = in_features/d is a multiple of 32. gemma-4-26b has
moe_intermediate = 704:

| model | d | down NSUB | gate/up NSUB | result |
|---|---|---|---|---|
| gemma-4-26b | 4 | **176** | 704 | down_proj STRANDED at uint16 |
| gemma-4-26b | 2 | 352 | 1408 | all pack |
| qwen3.6-35b | 4 | 128 | 512 | all pack |
| qwen3.6-35b | 2 | 256 | 1024 | all pack |

down_proj is exactly 1/3 of gemma's code elements (down is [hidden,176],
gate/up are [704,704] — equal counts). So every gemma d4 build above K=256
carries a third of its codes at full uint16 width:

    effective bpw(d4, K>256) = [ (2/3)log2(K) + (1/3)(16) ] / 4 + 0.25

d4-K2048's nominal 3.00 bpw is really **3.42**, and it lands at 12.54 GiB
instead of 11.50. Adding this term made a one-parameter size model match
SIX measured artifacts to within 0.1 GiB (d4-K256 9.43, d4-K2048 12.54,
d2-K256 14.75, d2-K512 16.08, d2-K1024 17.41, d2-K2048 18.74). It is
arithmetic, not a fudge factor.

**CONSEQUENCE — and the scope limit that matters.** On gemma, d=4 is the
better geometry per BIT and the worse geometry per BYTE. Bits are what the
theory compares; bytes are what ships. **This does NOT generalize**: Qwen
packs fully at both d=2 and d=4 (NSUB 128/512 and 256/1024), so there the
bpw advantage does translate to size. The gemma penalty is an accident of
one architectural constant, not a property of d=4.

**WHY K IS THE CEILING.** bpw = log2(K)/d, so at fixed d each +0.25 bpw
DOUBLES K, while k-means assignment costs O(n*K*d). Extrapolated from the
measured gemma d4-K8192 fit (4255 s):

| d4 target | K | fit time |
|---|---|---|
| 3.50 bpw | 8,192 | 1.2 h  (measured) |
| 4.25 bpw | 65,536 | 9.5 h |
| 5.75 bpw | 4,194,304 | 605 h |

Exponential cost for linear quality. That is the wall — not memory (E50's
scatter-add removed that one) but the assignment search. Approximate NN or
hierarchical k-means would move it; nothing else will.

**PRACTICAL RULE.** Check `in_features/d % 32` BEFORE choosing a geometry.
If it fails, either pick a d where it passes, or accept a ~1 GiB penalty per
stranded third — and never compare geometries on nominal bpw alone when one
of them cannot pack.

## E55 (08-19) — DO NOT RANK VQ BUILDS BY relerr ACROSS DIFFERENT ALLOCATIONS

**THE DURABLE RESULT, stated first because it outlives the experiment that
produced it.** Reconstruction error ranked two builds BACKWARDS, twice, at
matched size, in the ordinary operating range. Both sessions then reasoned
from that proxy to a wrong conclusion within hours of each other — this one
to a deep-promotion mechanism that does not exist as stated, the 397B
session to "therefore do nothing" about a shallow anomaly sitting in its own
sweep. Neither error was careless; both were the instrument working exactly
as designed and measuring the wrong quantity.

    RULE: relerr compares builds that share BOTH geometry and allocation.
    Across allocations it is not a ranking instrument. Use an output-scored
    metric (ppl where valid, blind judging where not) or do not rank.

This is stronger than E45's "relerr expires": that was the proxy losing
resolution at the top of a ladder. This is the proxy inverting in the middle
of its range.

---

### The measurements behind it, and the depth effect only the output metric can see

**This is stronger than E45's "relerr expires".** E45 said the fit proxy
stops TRACKING quality at the top of a ladder. This says it ORDERS two
builds WRONG, in the ordinary operating range, at matched size. Qwen3.6-35B,
all measured (fit-log mean relerr; referee ppl vs bf16):

| build | mean relerr | packed | ppl vs bf16 |
|---|---|---|---|
| flat d2-K256 | **0.0834** (best fit) | 17.64G | **1.016x** (worst ppl) |
| tail30 d2-K512 tail | 0.0904 | 17.88G | **0.991x** (best ppl) |
| tail30 d2-K256 tail | **0.1091** (worst fit) | **16.47G** | 1.002x |

**Two inversions, same direction.** Both tail builds have WORSE
reconstruction error and BETTER perplexity than the flat build. Any
conclusion drawn from a fit proxy has to survive this.

**THE THIRD ROW IS THE INTERESTING ONE.** tail30-d2K256 *is* flat-d2-K256
with layers 0-9 DOWNGRADED to d4-K2048 — fewer bits (3.00 vs 4.25 bpw),
measurably worse fit — and it comes out better on ppl AND 1.2 GiB smaller.
Spending fewer bits on the shallow layers did not cost output quality.

**WHAT THIS IMPLIES, stated carefully.** Fit difficulty is FLAT with depth on
this model — probe_depth_profile measures 0.5% spread across L3/L20/L38 at
both d=2/K512 and d=4/K2048. So layers are equally hard to RECONSTRUCT but
apparently NOT equally costly to get wrong: the shallow layers tolerate more
error than the deep ones. That gradient is invisible to every fit-side
instrument, which is exactly why the 397B session's E53 (no fit-side depth
gradient there either, via opposing profiles that cancel) cannot settle
whether a schedule helps — and they have amended it to say so.

**WHAT IT DOES NOT SHOW.** These are three points, not a curve, and the two
tail builds differ from flat in TWO variables (depth allocation AND tail
codebook). E49's mechanism claim is still unproven. The clean control is
flat d2-K512 at matched geometry (~19.6G), queued; per the 397B session's
suggestion it will report relerr AND ppl so the inversion gets a third data
point either way. One instance is a counterexample; three consistent ones
would be a rule about when reconstruction error misleads, which is worth
more than the schedule question that produced it.

**PRACTICAL RULE UNTIL THEN.** Do not rank two VQ builds by relerr unless
they share geometry AND allocation. Across schedules it has now ordered
builds backwards twice.


### PRE-REGISTERED PREDICTIONS for arm 3 (written before the number exists)

Arm 2 (head-DOWN: L0-9 cheap d4-K2048, L10-39 rich d2-K512) = 17.88G, ppl
0.991x. Arm 3 mirrors it — identical bits, identical geometries, cheap end
moved to L30-39. Readings agreed by both sessions IN ADVANCE:

  arm2 > arm3   shallow layers are cheap; head-down is the lever, and it
                SHRINKS artifacts rather than growing them
  arm3 > arm2   deep promotion was the active ingredient; E49 survives as
                originally written
  arm2 ~ arm3   position is irrelevant; both beat flat only because d2-K512
                beats d2-K256, and E49 dies as a schedule claim

**397B session predicts arm2 > arm3**, reasoning from its own responsiveness
data (shallow layers gain 32.7% from a 16x codebook vs ~49% in the body, so
bits placed early buy less reconstruction than bits placed anywhere else).

**This session predicts arm2 > arm3 as well, with arm3 landing 0.005-0.015
worse on ppl (so ~0.996-1.006x)**, on independent evidence: tail30-d2K256 is
also a head-down build, and it beat flat d2-K256 on ppl (1.002x vs 1.016x)
while being 1.2 GiB smaller. Two head-down builds have now beaten flat; no
head-up build has been measured at all.

If arm3 wins instead, the shallow-layers-are-cheap reading is either
family-specific or does not survive the jump from fit to output, and BOTH
sessions' entries need amending rather than defending.
## E56 (08-19) — PRE-REGISTERED: is the gemma-small sidecar claim decidable? (PUBLISH HELD until this resolves)

**The problem Noah caught.** The sidecar falsification (e4b-8bit 84.62% vs
vq-K256-d4 79.81%, litbench cyclic generative n=104) was stated harder than
its error bars own it. The tell: e4b-8bit scores ABOVE its own bf16 teacher
(82.69%) on the same instrument — a quant beating its teacher is noise
announcing itself. Binomial SE at n=104 is ~±3.7pts, so the 4.8pt gap is
~1.3 SE. Suggestive, not decision-grade. Symmetrically: vq-K2048-d4's 86.54%
"above e4b-8bit" is the same size of noise and buys nothing.

**Burden of proof (measured, not assumed).** e4b-8bit is 8.38 GiB on disk;
gemma-small is 9.43 GiB. The smaller model is the incumbent. gemma-small
must therefore WIN outright to justify existing as a literary sidecar —
a tie keeps e4b-8bit. This asymmetry means the current card wording
("keep e4b-8bit") survives even a null result; what the card may NOT say
without significance is that e4b-8bit is *better*, only that gemma-small
was not shown better.

**Design: three paired instruments + one running control.** All pairing is
per-item — paired designs remove item-difficulty variance, which dominates
at this n. Instruments:

1. **Paired litbench McNemar** (`paired_litbench.py`). Re-run vq-K256-d4
   cyclic generative so it stores per_item (e4b-8bit already does; the
   storage landed after the older runs). Also re-run 26b-bf16 WITH --cyclic,
   closing the non-cyclic-row hazard flagged in CRUSH_RESULTS.
2. **Blind paired win-rate on 3 non-literary domains**
   (`winrate/prompts_domains.json`: 20 instruction-following, 20
   summarization, 20 dialogue) + the existing 60 literary = 120 paired
   prompts, judged blind per the E51 protocol, sign test on non-ties.
   litbench is one narrow lens; the sidecar ships all four of these.
3. **Deterministic constraint pass-rate** (`check_constraints.py`): the 20
   instruction prompts carry machine-checkable constraints (word caps,
   exact line counts, mandated openings). Pass/fail, zero judge variance,
   paired McNemar. (From Noah's Gemini triage — the one genuinely new,
   genuinely cheap instrument on that list; KL, top-1 match, and blind
   pairwise judging we already run.)
4. **Control (running, M4 ~/qlab/bf16_faceoff.log):** 26b bf16 vs e4b bf16
   blind win-rate — whether the base-model ordering is even real, zero
   quantization in the picture.

**Pre-registered readings — written before any number lands:**

- gemma-small "designed to outperform e4b-8bit" may return to the card ONLY
  if ≥2 instruments favor gemma-small at exact p<0.05 AND none favor
  e4b-8bit at p<0.05.
- "e4b-8bit is better — keep it" may be stated as MEASURED only if ≥1
  instrument favors e4b-8bit at p<0.05 and none favors gemma-small.
- Anything else → card says "statistically indistinguishable from e4b-8bit
  on our instruments (n and CI quoted); e4b-8bit is 1.05 GiB smaller, so it
  remains the default recommendation." That is a publishable, honest claim.

**Predictions (this session):** litbench McNemar discordant count 14-24,
p in 0.15-0.6 → alone inconclusive. Constraint pass-rate: both models pass
14-18/20, McNemar inconclusive. Pooled 120-prompt win-rate: e4b-8bit ahead
on literary (consistent with litbench point estimate), roughly even
elsewhere; pooled p likely 0.05-0.3 → final verdict most likely lands on
reading 3 (indistinguishable, keep the smaller incumbent). If instead
gemma-small wins pooled — the 26b base advantage (if the M4 control
confirms one) survived 2.25-bit quantization, which would be the more
interesting result.

**Status:** instruments built and committed; runs queued behind the M3
queue (verify_packed -> qwen_k8192 -> arm3_headup) as
`gemma_small_verdict.sh`. Nothing scored yet.

## E57 (08-19) — ARM 3 RESULT: HEAD-DOWN CONFIRMED, AND THE TAIL IS THE WRONG PLACE TO SPEND BITS

The three-arm experiment (design + pre-registered readings in E55/E56-era
notes) resolved:

| arm | L0-9 | L10-29 | L30-39 | size | ppl vs bf16 | agree |
|---|---|---|---|---|---|---|
| 2 head-DOWN | **d4-K2048** | d2-K512 | d2-K512 | 17.88G | **0.991x** | 90.75% |
| 3 head-UP | d2-K512 | d2-K512 | **d4-K2048** | 17.9G | **1.019x** | 89.28% |

Same bits, same size; only WHICH END gets the expensive geometry differs.
Head-down beats head-up by 0.028x ppl — larger than this session's
pre-registered magnitude (0.005-0.015) and in the direction BOTH sessions
predicted. KL agrees (50.944 vs arm2's lower; 89.28% vs 90.75%).

**Readings, as agreed in advance:** arm2 > arm3 → shallow layers are the
place the expensive geometry pays; deep "promotion" was never the active
ingredient; E49 dies as a deep-schedule claim and survives only as
"head-down works." Arm 3 also lands WORSE than what its bit budget should
buy — spending d4-K2048 on L30-39 bought less than spending it anywhere.

Mechanism note (consistent with the 397B responsiveness data): shallow
layers are the HARDEST to fit and feed every later layer, so error placed
there compounds; the tail's errors have nowhere to propagate. The fit-side
depth profile was FLAT on qwen (E53), which is exactly why relerr could
not see this — output sensitivity, not fit difficulty, is depth-graded.

Arm 1 (flat d2-K512, ~19.6G) fit on M4 remains unverified; it brackets the
schedule effect from below once scored, but the arm2-vs-arm3 question no
longer depends on it.

Instrument caveat recorded in STATE: verify_artifact --outlier flags the
d4-K2048 region of ANY mixed-geometry build (~0.19 vs d2-K512 ~0.09 body,
3.2x median) — a healthy geometry difference, not corruption. Read the
gate per-region on mixed builds.

### E56 addendum (08-19, before verdict benches ran) — control trending, question sharpened

M4 faceoff interim at 36/60 pairs: 26b bf16 leads e4b bf16 14-2 on decisives
(sign test already ~p=0.004). If it holds, litbench's contrary read
(e4b-bf16 82.69 "above" 26b) was instrument noise, and E56's question
sharpens to: does 2.25bpw quantization give back the real bf16 advantage
against an 8.38G incumbent that starts from a weaker base? The three paired
instruments answer that as designed — no protocol change, readings stand.

Candidate lever parked for AFTER baselines: E57 says head-down works on
qwen; a head-down gemma-small at the same bytes might retain more of the
26b advantage. Compute is Noah's call; do not start it before the E56
baselines land or it contaminates the comparison set.

### E57 AMENDMENT (08-19, caught by the 397B session within the hour) — the mechanism prose above is INVERTED; the table is correct

Read the bpw before the prose: d4-K2048 = 3.0 bpw (CHEAP), d2-K512 = 4.75
bpw (EXPENSIVE). Arm 2, the winner, puts the CHEAP geometry on L0-9 and the
expensive one on L10-39. So the correct reading is the one E55-era notes and
the 397B responsiveness data already carried: **shallow layers TOLERATE
cheapness; the expensive geometry pays on the deep end.** The sentences
above ("shallow layers are the place the expensive geometry pays" and the
error-compounding mechanism note) describe the LOSING arm and are wrong.

This is the naming trap's third strike: "head-down/head-up" invites reading
"which end is promoted" instead of "which end is cheap." From here, say it
only as CHEAP-SHALLOW (winner) vs CHEAP-DEEP (loser).

Gate rule (from the 397B session's gemma cheap-shallow FATAL): 
--relerr-abort is a STABILITY guard, not a quality guard — cheap-shallow
designs intentionally raise relerr where it is cheapest to be wrong, so a
fixed 0.35 bar vetoes the intended design using the exact proxy E55 proved
cannot rank allocations. For non-flat builds set it above the cheap
region's expected relerr and read the REFIT SPREAD (reproducibility) as
the stability signal instead; 0.4357/0.4360 across refits is stable-poor,
not broken.

---

## E58 (08-19) — bf16 BASE-MODEL FACEOFF: 26b-a4b beats e4b on prose (PARTIAL, stopped by Noah)

**Why this ran.** MODEL_CARD_GEMMA_SMALL states e4b-8bit beats our
gemma-small on literary work, on one litbench read (84.62 vs 79.81, n=104).
Noah doubted it and spotted the tell himself: **e4b-8bit scores ABOVE e4b
bf16 (82.69%) on the same instrument** — a quant cannot beat its own
teacher, so the instrument was resolving noise. At n=104 the binomial SE is
~±3.7 points: the 1.9-point "8bit wins" is 0.5 SE (noise), and the
4.8-point gap the sidecar falsification rests on is ~1.3 SE (suggestive,
not decision-grade). The card asserted harder than the error bars own.

**Design.** Strip quantization out entirely and ask the base-model question
directly: **26b-a4b bf16 vs e4b bf16**, 60 literary prompts, greedy,
winrate_bench dual-order blind judging (Qwen3.8-27B q4 judge, different
family). A "win" requires the judge to pick the same continuation in BOTH
orders; self-disagreement counts as no-decision. Gens on M4:
`gens_prose_26b-bf16.json` / `gens_prose_e4b-bf16.json`.

**Result — PARTIAL, 43 of 60 pairs.**

| | pairs |
|---|---|
| 26b-a4b bf16 wins | **15** |
| e4b bf16 wins | **4** |
| inconsistent (judge flipped with order) | 23 |
| ties / unparsed | 1 |

Exact two-sided sign test on the 19 decisives: **p = 0.0192**.

**Stopped deliberately at 43/60 by Noah**, not by failure and not on a
favorable interim — he judged the comparison decision-irrelevant ("two
models, one half the size of the other, released in the same family at the
same time") once its only real purpose, showing litbench cannot resolve
this question, was already established. Record it as a stopped-early
partial. It is NOT a publish gate; E56 is.

**What it establishes.** The base-model claim inverts: 26b-a4b bf16 is
better than e4b bf16 on literary prose, at p<0.05 even on two-thirds of the
planned pairs. litbench's contrary read was instrument noise. The publish
question therefore sharpens — it was never "is 26b a worse model", it is
"does 2.25 bpw give back enough of a real bf16 advantage to lose to a
smaller (8.38 GiB) incumbent built on a weaker base."

**Instrument standing.** litbench is a coarse SCREEN: decisive for large
gaps (the 42%-tier artifacts), unable to resolve ~5 points at n=104, and
saturating near the top — the same failure shape as the ~82% agreement
floor (E41) and bounded-metric extrapolation (E48). Every close call this
arc was settled by blind paired judging. Do not gate a publish on litbench
alone.

## E59 (08-19) — qwen d4-K8192 MEASURED: extrapolation vindicated to the third decimal, and d4's ceiling is now a data point

Noah's chart question: "by the looks of the qwen chart, d4-K8192 should be
similar to tail d2K256?" Measured:

| build | size | ppl | agree |
|---|---|---|---|
| vq-K8192-d4-packed | 14.8G | 1.013x | 89.37% |
| vq-tail30-d2k256-packed (cheap-shallow) | 16.5G | 1.002x | 89.92% |

The E48 extrapolation said ~14.8G / ~1.012x with a warned ±2pt window — it
landed at 14.8G / 1.013x, dead center. (One vindication does not repeal
E48: bounded metrics still cannot be extrapolated, ppl-ratio evidently can
be interpolated within a family's own ladder.)

Answer to the question: close, and instructively NOT similar. d4-K8192 is
1.7G smaller but 11 ppl-points worse; it is the best pure-d4 point on the
curve and still loses to a mixed cheap-shallow d2 build. This is the
measured version of E50's saturation argument: at d4 each further halving
of loss costs a DOUBLING of K (next rung K=65k ~ 9.5h fit), while d2
schedules reach the same quality region by allocation instead of brute K.
The d4 curve bends exactly where the geometry math said it must.

Verify gate PASSed (3.0x median) — a FLAT build, so the E57-amendment
mixed-geometry caveat does not apply here.

### E56 addendum 2 (08-19) — base-model control resolved (see E58): the conditional in the predictions is now unconditional

E58 (397B session, partial by Noah's decision, 43/60 pairs): 26b-a4b bf16
beats e4b bf16 15-4 on decisives, exact p=0.0192. The E56 prediction text
hedged "the 26b base advantage (if the M4 control confirms one)" — it is
confirmed. e4b was never the stronger base; litbench's contrary read is
demoted per E58's SE arithmetic (a coarse screen: decisive at 42%-tier
gaps, blind at ~5 points, saturating near the top).

E56's question is therefore in final form: 26b bf16 is the better base;
does 2.25 bpw give back enough of that real advantage to lose to a smaller
(8.38G) incumbent on the weaker base? Readings unchanged.

### E47 addendum (08-19, corrected same day) — M4 failure #5: a fit lost at the write step; cost was ~6 minutes, not 40

A gemma cheap-shallow fit on M4 died at the final mx.save_safetensors with
kIOGPUCommandBufferCallbackErrorSubmissionsIgnored. First report said "~40
minutes of completed work lost" — that number was pattern-matched from an
unrelated fit's duration, never measured, and Noah caught it: file mtimes
bound the crashed run at ~6-7 minutes (cheap-codebook fits are fast). Do
not build checkpointing machinery to protect 6-minute jobs.

Two mechanism hypotheses, deliberately BOTH held (neither proven):
1. Poisoned GPU context (from the error string): per-process, not
   persistent — a plain matmul passes immediately after; retry, no reboot.
2. SMB contention (Noah's): M4 reaches the Thunderbay SSD over SMB from
   M3, and the crash landed exactly at the save while M3 was hammering the
   same disk with the K8192 fit. Explains WHY it struck at the write,
   which hypothesis 1 does not.
Live discriminator RESOLVED same day: the K128 retry wrote its full
artifact cleanly while M3's E56 benches were hammering the same SSD — MORE
contention than the run that died (M3 was quiet then). Contention does not
explain the crash. Verdict: UNEXPLAINED TRANSIENT; neither hypothesis is
credited. (A "fits run slower under load" impression from partial logs is
also withdrawn — the finished retry took 511s, comparable to the crashed
run's <7 min; it was an eyeball, not a measurement.)

Related, measured here: the 6-bit pack path needed for cheap-shallow K64
regions is fully supported — vq_pack round-trips 6-bit exactly, and BOTH
Metal readers (fused packed6 d4 and decode packed6) match the unpacked
reference bit-identically (max rel diff 0.0e0, synthetic E2/OUT128/IN256).
No slow-path fallback: pack_bits is a template parameter, so 6-bit gets
its own compiled kernel, same as 9/11-bit.

## E60 (08-19, PRE-REGISTERED before scoring) — gemma cheap-shallow K128/K512: the fair bar is the INTERPOLATED flat line, not flat K256

Artifact (397B session's build, M4): gemma26b-rungs/
vq-headdown-k128-tail512-d4 — cheap-shallow L0-9 d4-K128 / L10-29 d4-K512,
90 tensors, mean relerr 0.3001 (which per E55 decides nothing). Nominal
2.33 bpw — NOT matched to shipping flat K256-d4 (2.25): the matched K64
tripped the old abort gate and K128 was substituted without re-deriving
size, so the build is ~3.6% richer than the incumbent.

Fair comparison, registered now: flat ladder gives K256 = 42.65% @ 2.25
and K512 = 45.04% @ 2.50, so the interpolated flat build at 2.33 bpw is
~43.4%. The cheap-shallow build must beat ~43.4% (by more than ~1 point,
given instrument noise) to demonstrate an allocation effect on gemma;
beating 42.65% alone is bought quality, not allocation. Score = KL/agree
vs bf16 (gemma ppl invalid), after E47 verification on M3.

## E61 (08-19) — RELEASE NEAR-MISS: text-only artifacts passed every gate because every gate measures the text path

Both Qwen3.6 publish artifacts reached 40 minutes INTO UPLOAD with zero
vision tensors (vision_config stripped) after passing guard, KL
reproduction, fused re-bundle, stock-venv smoke, and cards. Nothing failed
because nothing looks: every gate we built exercises the text path. Noah
caught it by asking. (397B releases unaffected — towers verified at debut;
gemma sighted builds fine.)

Fixed by the 397B session: graft_vision.py, KL gate reproduced exactly on
the grafted bytes (44.573 / 90.75), sizes 18.73 / 13.81 GiB, names
corrected to VQ-4.6bpw / VQ-3.4bpw — the old names divided text-only bytes
by a param count that INCLUDED vision, so the advertised bpw was quietly
inconsistent. Naming rule: total-bytes over total-params.

Mechanical gate added: check_vision.py — counts tensors by vision prefix
in artifact vs source indexes; FAILs on mismatch unless --allow-text-only
is passed AND the card states the decision. First run correctly failed the
gemma cheap-shallow artifact (356 missing: 355 vision_tower + 1
embed_vision — gemma needs BOTH prefixes grafted before it could ship).
Release checklist line: "vision tensor count == source count, or an
explicit text-only statement on the card."

The general lesson goes beyond vision: a gate suite verifies what it
MEASURES, and everything it does not measure ships unverified. When adding
a capability class (vision, future audio, adapters), add the corresponding
existence gate the same day.

## E62 (08-19) — FUSED PACKED-D2 KERNEL PRODUCTION-VALIDATED: ~3x decode on both quality artifacts, zero numeric drift

Gate = reproduce the recorded scores on the guarded artifact with the
fused runtime (397B session, M4, logs ~/qlab/kl_gate*.log; teacher cache
was never gone — a stale-path scare):

| artifact | KL / agree / ppl | old -> fused decode |
|---|---|---|
| qwen vq-tail30-d2k512-packed | 44.573 / 90.75% / ppl identical to every printed digit | 21.9 -> 66.2 (40-tok) / 54.9 (120-tok) |
| gemma vq-K2048-d2-packed-sighted | 537.302 / 77.89% reproduced | 8.4 -> 47.8 |

Identical to every printed digit on qwen incl. the raw ppl float; both
quality artifacts now ship the fused runtime (qwen build is what is on
the Hub as VQ-4.6bpw). The packed-d2 prefill fallback (E-noted at the d2
kernel's debut) is retired from production. Cards updated (gemma decode
~40 -> ~48 measured).

Hub verification note: `hf cache verify` false-alarms on the Hub's
auto-created .gitattributes and the local .cache state dir — content
verification is sha256 against the Hub file listing, done directly.

## E63 (08-19, PRE-REGISTERED before the run) — the capacity probe: testing where a small model should RUN OUT OF ROOM

**Why.** Every instrument returned parity between gemma-small (2.25bpw VQ
of 26b-a4b) and e4b-8bit: litbench McNemar p=0.33, quant-vs-its-own-teacher
p=0.15, constraint pass-rate 19/20 vs 19/20 (p=1.0), and my own read of 4
prose pairs scored ~2-1-1 with a clear STYLE difference (gemma-small more
structured and lands the sharper insight; e4b chattier, sometimes more
directly practical) and no competence gap. Noah's objection — a 6x
parameter gap cannot be invisible — is right, and the resolution is that
none of those tasks stress capacity. "Why does the moon follow the car"
needs one good analogy, not 26B parameters.

**Design.** 60 prompts, 4 families, THREE machine-scored so no judge
variance enters (winrate/prompts_capacity.json, score_capacity.py):
- multihop (20): constructed-world chains, all facts in-prompt so it is
  reasoning depth not recall; exact-match golds. An early wrong turn
  poisons the chain — the small-model failure mode.
- constraint (20): SIX simultaneous machine-checkable constraints per item
  (vs ONE in the easy set that both models aced), scored per-constraint so
  4-of-6 is distinguishable from 2-of-6.
- needle (10): a planted fact at 5 known depths in a 6k-17k token literary
  haystack; exact-match retrieval.
- sustain (10): 800-word continuations scored MECHANICALLY for
  degeneration (distinct-trigram ratio, repeated-sentence rate, longest
  repeated 8-gram, reached-length) — looping is what a model out of
  capacity does, and it is measurable without taste.

**Instrument hygiene applied up front, both caught before the run:**
1. A harness defect that would have faked the result: mlx-community's
   e4b-8bit does not register <turn|> as EOS under mlx_lm.generate, so
   59/60 of its domain generations trailed junk ("<turn|>msch<turn|>ichi")
   while our 26b builds did it 0/60. Judged blind and uncut, e4b loses on
   someone else's PACKAGING defect and we record it as quality.
   strip_thinking now cuts at end-of-turn markers (2f17bc2).
2. One of my own multihop items was UNSATISFIABLE (Dara<Eli<Faye<Gus forced
   Gus last while the item asserted "Gus is not last"). Found by
   hand-verifying all 20 golds before generating. RULE: verify every gold
   answer by hand; a wrong gold is an instrument that silently punishes the
   model that is right.

**Pre-registered readings.** If the parity is real capability parity, all
four families come back within noise and the gemma-small card's
"indistinguishable" language stands as the honest description of BOTH the
easy and the hard regime. If the 26b advantage is real but regime-specific,
the expected signature is: needle and sustain separate first (context and
looping are where parameters buy the most), multihop next, constraint last.
A gemma-small win on >=2 families at p<0.05 with none opposing would
justify a capability claim CONFINED TO THE STRESSED REGIME — never a
general "better model" claim, and never a bf16 comparison.

### E63 RESULT (08-19) — the capacity probe FAILED TO SEPARATE, because BOTH models are at ceiling

| family | gemma-small (2.25bpw of 26b) | e4b-8bit | paired |
|---|---|---|---|
| multihop | 20/20 | 20/20 | p=1.0 |
| needle (6k-17k tok, 5 depths) | 10/10 | 10/10 | p=1.0 |
| constraint (x6 each) | 120/120, 20/20 perfect | 118/120, 18/20 perfect | p=0.50 |
| sustain | 1128 w, distinct3 .969, rep-sent .000 | 1019 w, distinct3 .984, rep-sent .010 | no looping either |

**Pre-registered signature (needle and sustain separate first, then multihop,
constraint last) did not appear — nothing separated at all.** e4b's only
blemish was 2 missed required-vocabulary constraints out of 120 (p=0.50,
noise).

**The honest reading, and it is NOT "the models are identical."** This is a
CEILING result, and a ceiling cannot rank: both models scored at or within
2 items of perfect on every family, so the probe has no resolving power
left at the top. The data are equally consistent with (a) genuine
equivalence in this regime and (b) a probe that is still too easy. We
cannot distinguish those two from this run. Same failure family as E41
(agreement floor), E48 (bounded metrics), and E56's litbench demotion — the
fourth time this project has been bitten by an instrument that saturates.

**What IS established, and it is worth stating plainly:** 2.25bpw VQ
quantization of gemma-4-26b-a4b costs NOTHING measurable on multi-hop
reasoning depth, long-context retrieval to 17k tokens at all depths,
six-way simultaneous constraint satisfaction, or 1000+ word generation
without degeneration. That is a strong statement about the METHOD even
though it is a null result about the COMPARISON.

**Consequence for the publish decision (E56):** with litbench, domain
constraints, and now a purpose-built hard probe all returning parity, the
E56 third reading is the one the evidence supports —
"statistically indistinguishable from e4b-8bit on every instrument we
built; e4b-8bit is 1.05 GiB smaller and remains the default." Any claim of
a gemma-small capability advantage would now need a probe that first
demonstrates it has headroom (i.e. that some model FAILS it), which is the
correct bar and one no instrument here currently clears.

**Human instrument opened (chat_arena.py):** blind side-by-side browser
chat, A/B re-randomized every turn, votes logged with true identity to
winrate/human_verdicts.jsonl for a sign test. Noah's read is the only
signal not yet saturated.
