# GLM-5.3-Flash — family arc ledger

Authoritative from creation (2026-08-30). Survey/readiness history lives in
READINESS.md; this file records the arc's runs and verdicts.

## 2026-08-30 ~02:30 — teacher pass lands; memorization finding

First real-model run of the glm5_next scorer (M3, glm5vlm venv + mlx-lm
installed). Teacher (598.5 GiB bf16, streamed): prose ppl 1.9024, code
1.4888, literary 1.1580 — implausibly low; investigated before accepting:
tokenizer sane (~4.2 chars/tok, healthy diversity); attention mask causal
(lower-triangular, inspected); DIRECT-FORWARD CAUSALITY TEST on the tiny
rule-5 model: prefix logits bit-identical under suffix change (0.0e0) —
architecture causal at prefill. Verdict: the numbers are REAL —
MEMORIZATION. GLM-5.3 has near-verbatim absorbed WikiText/Gutenberg/mlx
code (teacher cache: mean top-1 prob 0.857, 68% of positions >0.9).
Consequences: (1) absolute ppl on public corpora is contamination-
dominated for this family — never compare cross-family; (2) KL-to-teacher
stays fully valid and is STRICTER here (sharp teacher); cache
glm53_teacher_topk_prose holds 99.06% mass. Consider a post-cutoff
held-out corpus if absolute anchors are ever needed.

Also tonight: struct base CPU-pinned on M4 after a Metal-watchdog kill
(GPU attempt); q4 affine baseline built on M3 (166 GiB, 4.524 bpw,
zai-org--GLM-5.3-Flash-4bit).

## 2026-08-30 ~06:00 — night queue lands COMPLETE

Everything queued finished: struct base (M4, CPU-pinned, 98.6 GiB, 19
shards, 4188s) and the affine ladder w/ KL (M3, glm5vlm venv; convert
config fix 4d1497d re-merges vision_config): q3 129 GiB KL 377.08 / q4
166 GiB KL 98.34 / q6 239 GiB KL 13.47. TABLE.md created with the
contamination note. The ladder's shape echoes Flash-Next: q3 collapses
(under-4-bit affine cliff, §4.1's law on a third family), q4 usable,
q6 near-teacher.

Ready for Noah at 09:00: VQ rung geometry choices (all machinery
validated; leverage probe code-complete via peer's 55864f9; 128GB-tier
target would be the first data-free GLM-5.3 to fit a single consumer
box — q3 affine at 129 GiB does NOT fit one and is the worst row).

## 2026-08-30 (evening) — d8/K16384 fit resumed; rung arithmetic priced

Fit RESUMED from the 11 banked shards (all skipped instantly, resume
path healthy) and ran clean at ~1900s/shard, relerr steady ~0.352 against
a 0.45 abort. Script: scratchpad/glm_fit_resume.sh (glm5vlm venv,
--vq-layers 3-45 --k 16384 --dim 8 --family glm5_next).

VISION PROVISIONAL CLEARED: all 347 vision tensors in the struct base are
non-zero (|mean| 0.0023 to 1.625, sane). READINESS's "graft_vision is NOT
needed for glm5_next" is now MEASURED, not provisional. GLM assembly is
therefore pack -> bundle -> check -> smoke -> score: no splice-ple/
pack-ple (no PLE in this family) and no graft. Staged as
scratchpad/glm_assembly.sh.

RUNG ARITHMETIC (measured header pass on the struct base, 98.55 GiB):
  experts (VQ'd)      756 tensors  89.64 GiB   2-bit markers
  attention          1351          5.82        8-bit affine
  embed/lm_head        72          1.44        8-bit affine
  vision              347          1.05        bf16
  norms/router        472          0.61        bf16
Protected (non-expert) mass is only 8.91 GiB, so raising ALL of it from
8-bit to bf16 buys just +7.3 GiB -> ~106 GiB. CONCLUSION: the protected-
set lever CANNOT reach the 120 or 140 rungs; expert mass (89.6 of 98.6
GiB) is the only dial big enough, so both higher rungs need their OWN
expert fits at richer geometry (lower d and/or different K). Those should
be CHEAPER than this run, since K dominates fit cost (Flash-Next: K256
~9 min vs K4096 ~80). 120/140 geometry is Noah's call and is NOT yet on
record -- it is the only thing blocking an overnight queue.

RUNTIME IS NOT A BLOCKER (unlike MTP): glm5_next ships in RELEASED
mlx-vlm 0.6.17 (pip, not a local edit), so published GLM VQ artifacts
land on a runtime users can install today.

## 2026-08-30 (night) — the 81 GiB rung is a DEAD RUNG (measured)

d8/K16384 packed to 80.87 GiB = 2.162 bpw overall, experts 1.983 bpw
(theory 2.000: 14 bits/8 weights + 0.250 group-64 fp16 scales). The FIT
hit its geometry almost exactly. What was wrong was the LABEL.

TWO LABEL ERRORS, neither in the fit:
  1. The fit log's "2.25 bpw stored" is the UNPACKED figure (codes in U32
     containers); pack crushes to true bit-width (0.901x observed).
  2. "~100 GiB rung" was anchored to the STRUCT BASE being 99 GiB, which
     carries 2-bit expert MARKERS as placeholders, not final packed codes.
     No one derived the target from a bpw budget.
THE CHECK THAT WOULD HAVE CAUGHT IT, before any compute: at d8, 100 GiB
is UNREACHABLE at any K. d8/K65536 -- an absurd codebook -- is 2.25 bpw
= ~91 GiB total. The size target and the geometry family were
incompatible from the start. Derive bpw from (log2(K)/d + scales) and
size from param counts BEFORE launching a fit.

SCORES (streamed, teacher cache glm53_teacher_topk_prose):
  KL 692.25 mnats | prose ppl 3.6339 | top-1 agreement 0.748
  corpus     teacher     VQ81    ratio   +nats
  prose       1.9024   3.6339    1.91x   0.647
  code        1.4888   1.9619    1.32x   0.276
  literary    1.1580   2.9562    2.55x   0.937
vs affine: q3 129 GiB KL 377.08 / q4 166 GiB 98.34 / q6 239 GiB 13.47.
VERDICT: 1.8x WORSE than the affine row that already collapsed. Not a
ship candidate. Kept as evidence only. Note there is no affine row at
81 GiB to compare against (q3 is the smallest affine and is 48 GiB
larger), so this does not establish a frontier point -- it establishes
that ~2 bpw is past the cliff for this family too.

DAMAGE IS NOT UNIFORM ACROSS CORPORA (added loss in nats, which is the
honest metric -- ratios are confounded by the teacher's very low
memorization-driven baselines): literary 0.937 > prose 0.647 > code
0.276. The most-memorized corpus takes the MOST absolute damage. Worth
a proper look: it suggests VQ damage falls hardest on precisely the
content the teacher had absorbed near-verbatim.

TWO GATE HOLES FOUND (both shipped-shaped, both caught by luck):
  1. config lacked vision_config -> exo would build no VisionCardConfig.
     Fixed by re-packing with --vision-config-from (which runs a
     CORRESPONDENCE assertion, not a presence check).
  2. The bundle template hardcoded mlx_lm.models.{model_type}; GLM's
     class lives in mlx_vlm, so the bundle died on import. It also
     hand-listed exports, and a VLM base is read for TextConfig,
     VisionConfig, VisionModel, LanguageModel. Fixed in VQLab 40a2855 +
     798977a (resolve either runtime; re-export the base's whole public
     surface; export the args class as BOTH ModelArgs and ModelConfig,
     since mlx_vlm calls ModelConfig.from_dict).
BOTH artifacts passed check-release AND check-bundle. Neither gate
EXECUTES the bundle. `smoke` is the only gate that catches this class
and it runs LAST -- it should run EARLY.

REQUEUED for Noah's actual targets (100/120/140), smallest K first:
  d4/K512  2.50 bpw -> 99.6 GiB   (~40 min fit, est.)
  d4/K2048 3.00 bpw -> 117.8 GiB  (~2.5 h, threadgroup-safe)
  d4/K8192 3.50 bpw -> 135.9 GiB  (~10 h, device-codebook path; needs
                                   bundle-accept before it ships)

## 2026-08-30 (night) — the 98.6 GiB rung: first frontier point

d4/K512 packed to 98.55 GiB (2.635 bpw overall, experts 2.471) against a
99.6 GiB projection -- within 1%, so the geometry model
(bpw = log2(K)/d + 0.25 group-64 fp16 scales; size from param counts) is
now VALIDATED ON TWO POINTS and should be used to price every future rung
BEFORE fitting. Fit 3102s (~52 min, ~170s/shard), relerr median 0.2634.

RELERR: d IS THE STRONGER LEVER THAN K at this scale. d4/K512 reconstructs
BETTER than d8/K16384 (0.2634 vs 0.3521) from a codebook 32x SMALLER --
halving the vector dimension beats a 32x larger codebook, because each code
has half as much to represent. (Does NOT overturn FINDINGS' "raise K first"
for a FIXED d; it says the d ladder was under-explored here.)

SCORES: KL 348.82 mnats | prose ppl 2.5743 | top-1 0.8398
  corpus     teacher    VQ81   VQ98.6 | +nats81 +nats98  drop
  prose       1.9024  3.6339   2.5743 |   0.647   0.302   53%
  code        1.4888  1.9619   1.7107 |   0.276   0.139   50%
  literary    1.1580  2.9562   1.6166 |   0.937   0.334   64%

FRONTIER: beats affine q3 on BOTH axes -- 98.6 GiB vs 129 GiB AND KL
348.82 vs 377.08. First such point in this family. TEMPERED: q3 is the row
already called collapsed, and 7% better than collapsed is not "good";
prose ppl is still 1.35x teacher. This is the AGGRESSIVE end of the
ladder, not the quality rung.

STEEPNESS: 17.7 GiB bought 345 mnats = ~19.5 mnats/GiB. The cliff sits
just below 100 GiB for this family -- §4.1's law on a third family, now
with VQ measured on BOTH sides of it.

MEMORIZATION COUPLING (both halves now measured): literary took the WORST
damage at 2 bpw (0.937 nats, the largest) and recovers the MOST at 2.5 bpw
(64%, the largest). Hit hardest, recovers fastest. Consistent with the
teacher-pass finding that this family absorbed Gutenberg near-verbatim:
memorization lives in precise weight values, which is exactly what coarse
VQ erases and finer VQ restores. Worth a proper writeup.

NEXT: d4/K2048 (~118 GiB, 3.00 bpw) fitting now, ~2.5h; then d4/K8192
(~136 GiB, 3.50 bpw, ~10h, device-codebook path -> needs bundle-accept).
STILL UNRUN on every rung: smoke (loads resident) and verify.

## 2026-08-30 (night) — layer-leverage on the 98.6 GiB rung

Probe run (1024 tokens, streamed, peak 29.2 GiB). ALLOCATION IS NOT FLAT:
local_rel min 0.0056 / mean 0.0765 / max 0.2611, max/mean 3.41x.

TOP BY LOCAL DAMAGE      TRAJECTORY JUMPS (leverage)
  L18  0.2611  3.41x       into L3   +0.1590  (0.0073 -> 0.1663)
  L3   0.1661  2.17x       into L21  +0.1515  (0.3550 -> 0.5065)
  L32  0.1350  1.76x       into L18  +0.0922  (0.2867 -> 0.3789)
  L29  0.1204  1.57x       into L17  +0.0379
  L43  0.1203  1.57x       into L32  +0.0331
  L41  0.1023  1.34x       into L29  +0.0269
  L42  0.1016  1.33x     final traj drift 0.5592
  L10  0.1011  1.32x

SHAPE DIFFERS FROM FLASH-NEXT — do NOT assume transfer between families.
Flash-Next was front-loaded (L1 a monster at 0.310, 2.4x any other). GLM
is a mid-model spike (L18) + the FIRST expert layer (L3) + a late cluster
(L29/L32/L41-43). Early layers here are the CHEAPEST (L0 0.0110, L1
0.0069, L2 0.0056 — those are pre-expert; experts are L3-45).

L21 IS A SEPARATE FAILURE MODE: 2nd-biggest trajectory jump (+0.1515) but
NOT in the top-8 local. It AMPLIFIES existing drift rather than injecting
its own, so bigger K there may not help — worth a distinct look.

PLAN (morning, Noah): each expert layer is ~2.08 GiB at 2.50 bpw;
upgrading one to 3.00 bpw costs ~0.42 GiB, so the ~1.4 GiB of headroom to
the 100 GiB target buys ~3 layers. Splice L18 + L3 + L32 from the
d4/K2048 fit (running tonight, produces exactly those tensors) into the
d4/K512 artifact -> ~99.9 GiB, then re-score. Probe is a RANKING
instrument, not a quality score: the referee + KL must confirm the mixed
build actually paid.

TOOL FIX (VQLab 90f32fe + f612f20): the probe crashed 40 min in with
"'NoneType' object is not callable". mlx-vlm's hyper_connection builds its
fused sinkhorn Metal kernel AT IMPORT TIME behind
`if mx.default_device() != mx.gpu: return None`, and a cpu-stream context
reports default_device() as CPU (VERIFIED). The 08-29 watchdog fix had
moved model loads inside `with mx.stream(mx.cpu)`, which imported the arch
under CPU default and nulled the kernel forever. Fix: import the arch
BEFORE the stream block — the watchdog needs the WEIGHT-READ ops on the
CPU stream, not the import. Note mlx-vlm dispatches
`hc_func = _hc_ops if use_ops else _hc_kernel`, i.e. on a FLAG, not on
whether the kernel exists — an upstream hole worth reporting.

## 2026-08-31 — the 116.3 GiB rung sweeps affine q3

d4/K2048 packed to 116.28 GiB (projected 117.8, within 1.3% -- geometry
model now holds on THREE points). 3.108 bpw overall, experts 2.959. Fit
9545s (~2.65h). vision_config present, gates PASS.

  KL 199.53 | prose 2.1954 | code 1.6187 | literary 1.3402 | top-1 88.6%

BEATS affine q3 (129 GiB) ON EVERY AXIS: 12.7 GiB smaller, 47% less KL,
better on all three corpora, +5.5pt top-1. Against q4 (166 GiB) it trades
~2x the KL for 50 GiB. Extrapolating the affine ladder down to 116 GiB
puts affine ABOVE q3's 377, so VQ is delivering roughly HALF the damage at
equal size. This is the rung most people would actually run and it is the
family's frontier claim.

MEMORIZATION DEFICIT CLOSES WITH BITS (both ends now measured): literary
is the corpus VQ handles worst -- it is the most memorized (teacher ppl
1.1580). At 80.9 GiB literary is catastrophic (2.9562, +0.937 nats, the
WORST of the three). At 98.5 GiB VQ still LOSES to q3 there (1.6166 vs
1.4731) while winning everywhere else. At 116.3 GiB it WINS (1.3402 vs
1.4731). So the near-verbatim recall VQ erases is RECOVERABLE with bits;
it is not a structural weakness of VQ against affine, just a steeper part
of its curve. Retracts any reading of the 98.5 result as "VQ is worse at
memorized content, full stop."

d4/K8192 (~136 GiB) fit STARTED 2026-08-31, ~10h. TABLE.md updated in
place (4da7c459).

## 2026-08-31 — MIXED-BIT ALLOCATION: bits help, TARGETING DOES NOT

Three splices off the d4/K512 base (98.55 GiB, KL 348.82), each swapping
whole expert layers to d4/K2048 codes taken PACKED from the K2048 artifact
(safe: identical weight_map, 2998 keys; entries differ only in k and
pack_bits; untouched shards hardlinked). Tool: scratchpad/glm_mix.py.

  build                     GiB     KL   prose   code    lit  top-1
  base d4/K512            98.55  348.82  2.5743 1.7107 1.6166 0.8398
  mix top-3 leverage      99.82  348.62  2.5795 1.6954 1.5747 0.8345
  mix top-8 leverage     101.93  333.59  2.5461 1.6798 1.5531 0.8462
  mix BOTTOM-8 (control) 101.93  331.93  2.4948 1.6986 1.5700 0.8481
  full uniform K2048     116.28  199.53  2.1954 1.6187 1.3402 0.8862

THE CONTROL IS THE RESULT. Top-8 (local_rel 0.101-0.261) vs bottom-8
(0.025-0.057) -- a 4x separation in measured leverage -- at IDENTICAL byte
spend (+3.38 GiB) and identical geometry. Outcome: control wins KL, prose
and top-1; top-8 wins code and literary. A 2-2 split with the control
ahead on the ranking column. THE LEVERAGE RANKING CARRIES NO USABLE
INFORMATION about output damage for this family.

(Stated precisely: NOT "anti-correlated" -- the metrics disagree in
direction, which is the signature of no signal, not inverse signal. An
earlier reading of mine that called it anti-correlated was corrected by
the code/literary rows arriving.)

EFFICIENCY vs JUST BUYING BITS: the full uniform K512->K2048 step is
+17.73 GiB for -149.29 mnats = 8.42 mnats/GiB. The targeted 8-layer mix
is +3.38 GiB for -15.23 mnats = 4.51 mnats/GiB -- 0.54x the uniform rate.
Spending 19.1% of the bytes on the "highest-leverage" layers captured only
10.2% of the gain (8 of 43 layers = 18.6%). Targeting returned BELOW its
proportional share.

CONSEQUENCES:
1. Do NOT use `vqlab layer-leverage` to drive allocation for glm5_next.
   Its own docstring called it a ranking instrument requiring referee
   confirmation; the referee has now REJECTED it. Choosing layers by
   local_rel would have cost quality against choosing nearly arbitrarily.
2. SCOPE OF THE DOUBT (corrected 2026-08-31 -- my first write-up here
   conflated two different Flash-Next experiments):
   - The headline "reallocation WINS" (flash-next LEDGER 2026-08-29, KL
     556.10 -> 419.88 byte-neutral) was reallocation ACROSS TENSOR CLASSES
     -- PLE dropped K4096->K256, experts raised K4096->K16384. That is a
     different hypothesis from layer selection and is NOT impugned by this
     control.
   - The SEPARATE L0/L1 mixed splice (same date, driven by layer-leverage:
     "L1 is a monster, local 0.310") IS the one sharing the assumption we
     just broke. It was never run against a low-leverage control. Re-test
     that one before it informs anything further.
   Noah's read that this may be a FAMILY difference is live: Flash-Next
   showed measurable improvement, GLM shows none.
3. The 3-layer mix (+1.27 GiB) moved KL by 0.2 mnats = nothing. Mixing
   needs scale before it registers at all.

CAVEAT STILL OPEN: seed-noise floor is RUNNING (unseeded d4/K512 refit,
--seed -1, FINDINGS III.12). It bounds how seriously to take the 1.65
mnat gap between the two mixes. It does NOT change the conclusion: if the
gap is noise, targeting is useless; if real, targeting is harmful.

ALL FITS WERE SEEDED (mx.random 1234, the --seed default, logged in every
fit header). The noise-floor run is a DELIBERATE unseeded comparison, not
a correction of an error.

## 2026-08-31 — SEED-NOISE FLOOR (FINDINGS III.12), finally measured

Unseeded refit of d4/K512 (--seed -1, fit 3161s), same geometry, same
pipeline. This is the yardstick every comparison above was missing.

  d4/K512            KL      prose    code     lit    top-1
  seed 1234 (ours) 348.82   2.5743  1.7107  1.6166  0.8398
  unseeded         342.51   2.5631  1.7518  1.6763  0.8433
  DELTA              6.32   0.0112 -0.0411 -0.0597  0.0035

SEED FLOOR ON KL = 6.32 mnats. Note the corpora do NOT move together:
the reseed is BETTER on prose/KL/top-1 and WORSE on code/literary. Seed
variation is a reshuffle, not a quality axis.

RESCALING EVERY MIX RESULT AGAINST IT:
  3-layer mix         d_KL  0.20   0.03x floor -> NOISE
  8-layer top-8       d_KL 15.23   2.41x floor -> REAL
  8-layer control     d_KL 16.89   2.67x floor -> REAL
  top-8 MINUS control d_KL  1.65   0.26x floor -> NOT RESOLVABLE
So: mixing at scale is a real effect; WHICH layers is not. Confirms the
control verdict with a calibrated scale rather than a hunch.

MEASUREMENT vs GENERALIZATION (important, and it rescues the sweep):
the floor bounds GENERALIZATION ACROSS SEEDS, not the precision of a
single comparison. All splice variants are built from the SAME two fits
and scoring is a deterministic forward over fixed weights, so a
splice-vs-splice delta is EXACT -- zero stochastic component. The 1.65
mnat top-8/control gap is therefore a real number for THESE codebooks
that simply would not survive a reseed (seed variation is 4x larger).

THE LEVER NOBODY PULLED: the unseeded refit scored 6.32 mnats BETTER than
the seed we shipped, i.e. our published rungs sit on the unlucky side of
the draw. Fitting a geometry under 3-4 seeds and keeping the best costs
~52 min/seed at K512 and ZERO BYTES, and plausibly captures much of what
the 8-layer mix bought for +3.38 GiB. For anything we publish this looks
like a strictly better trade than mixed-bit allocation. UNTESTED as a
procedure -- but the single data point is already suggestive.

NEXT (running): one-at-a-time layer sweep, all 43 expert layers, each
spliced alone K512->K2048 and scored on prose+KL (scratchpad/
glm_layer_sweep.sh, resumable JSONL, ~3.5h). Then the GO/NO-GO: re-run a
subset against the seedB fit. If the per-layer ranking does NOT correlate
across two independent seeds, layer sensitivity is a property of the
k-means draw rather than the architecture, and allocation optimization
(randomized trials, surrogate models) is dead for this family -- it would
be modelling the lottery. Noah's interaction concern (adjacent vs
alternating layers) is real and OFAT is blind to it, but the prize is
bounded: full uniform K512->K2048 is 149 mnats for 17.73 GiB, so even
perfect allocation at 3.38 GiB likely caps near 30-45 mnats against the
15 we measured -- versus 6.32 mnats free from the seed lottery.

## 2026-08-31 — SINGLE-LAYER SWEEP: targeting DOES work; the PROBE was the problem

RETRACTS my earlier line "bits help; targeting does not" (this ledger,
"MIXED-BIT ALLOCATION"). What the control experiment actually showed is
that the LEVERAGE PROBE's targeting does not work. Targeting by MEASURED
single-layer effect is a different and far better proposition.

Method: promote ONE expert layer K512->K2048 (+0.42 GiB), score prose+KL,
repeat for all 42. Splices come from fixed fits and scoring is a
deterministic forward, so every delta is EXACT. ~1.22 min/sample, 42
samples. scratchpad/glm_layer_sweep.{sh,jsonl}.

d_KL by layer (positive = better than the 348.82 base):
  L3   +0.30  L4   -5.23  L5   -4.15  L6   -3.16  L7   -2.21  L8   -0.02
  L9   -1.20  L10  -2.30  L11  +2.33  L12  +0.74  L13  -2.84  L14  -2.65
  L15  +2.20  L16  -0.24  L17  -1.49  L18  -4.79  L19  +1.38  L20  +6.17
  L21  +3.42  L22  -1.87  L23  -1.79  L24  +0.59  L25  +2.69  L26  +4.90
  L27 +19.08  L28  +3.13  L29  +7.92  L30  -2.55  L31  +9.70  L32  -2.39
  L33  +5.00  L34  +6.37  L35  +5.10  L36  +0.37  L37  +2.30  L38  +0.30
  L39  +6.35  L40  +4.58  L41  +2.91  L42  +2.63  L43  +4.46  L44  +2.97
  helps 26, hurts 16; mean +1.64, median +1.06, sum +68.99.

FINDINGS:
1. L27 = +19.08 mnats ALONE, for 0.42 GiB. That is 3x the seed floor and
   MORE than either full 8-layer mix achieved for 8x the bytes.
2. SOME PROMOTIONS ARE NET-NEGATIVE (16 of 42; L4 -5.23 is worst). Adding
   precision to one layer can HURT KL -- the surface is non-monotonic,
   presumably because a layer reconstructed more finely now mismatches
   neighbours still at K512 and breaks an error cancellation.
3. THE PROBE PICKED BADLY. Its top-8 contained L18 (-4.79), L32 (-2.39),
   L10 (-2.30), L3 (+0.30) -- three of its eight "highest-leverage"
   layers are among the HARMFUL ones. local_rel is not merely uninformative
   about output damage, it selected actively bad layers here.
4. SUPERADDITIVITY IS REAL AND CONSISTENT:
     probe top-8   additive +8.74 -> measured +15.23  (1.74x)
     control       additive +9.28 -> measured +16.89  (1.82x)
   Combinations beat the sum of their parts by a stable ~1.75-1.8x, which
   is encouraging for modelling: a multiplicative correction may suffice.
5. Best-8 by MEASURED effect [20,27,29,31,33,34,35,39] sums to +65.69
   additive -- 7x either built set at IDENTICAL bytes. At the observed
   superadditivity that projects to ~115 mnats. BUILDING AND SCORING NOW;
   the projection is untested and the surface is non-monotonic, so it may
   not hold.

L45 IS NOT A VQ LAYER: 126 modules / 3 projections = 42 layers, L3-L44.
L45 is GLM's MTP layer and was never quantized (the sweep's L45 KeyError
was a wrong loop bound, not missing data). Worth remembering if the MTP
arc is ever revived for this family.

## 2026-08-31 — BEST-8 BY MEASURED EFFECT: targeting is worth ~2x uniform

Built [20,27,29,31,33,34,35,39] (the 8 best single-layer d_KL), same
+3.38 GiB / 101.93 GiB as the two earlier 8-layer mixes.

  8-layer build @ 101.93 GiB        d_KL     KL   prose  top-1
  probe top-8 (leverage-ranked)   +15.23  333.59  2.5461 0.8462
  bottom-8 control                +16.89  331.93  2.4948 0.8481
  BEST-8 (measured effect)        +54.98  293.84  2.4014 0.8569

3.6x either earlier build at IDENTICAL bytes.

EFFICIENCY -- this REVERSES the morning's "targeting is 0.54x uniform":
  uniform K512->K2048, all layers  +17.73 GiB  -149.29 mnats   8.42 mnats/GiB
  probe-targeted 8                  +3.38 GiB   -15.23         4.51  (0.54x)
  MEASURED-targeted 8               +3.38 GiB   -54.98        16.27  (1.93x)
Targeting is worth ~2x the efficiency of simply buying bits -- it captures
37% of the full uniform step's gain for 19% of its bytes. Against affine
q3 (129 GiB, KL 377.08) this build is 27 GiB SMALLER and 22% better.

SUPERADDITIVITY IS NOT A CONSTANT (corrects my earlier "consistent 1.75x"):
  probe top-8  additive  +8.74 -> measured +15.23  (1.74x, SUPER)
  control      additive  +9.28 -> measured +16.89  (1.82x, SUPER)
  best-8       additive +65.69 -> measured +54.98  (0.84x, SUB)
Weak/negative sets beat their sum; strong sets fall short of it --
diminishing returns against the +149.29 ceiling. So the ratio cannot be
used as a fixed correction factor when predicting a combination.

NOTE: best-8 is GREEDY selection on a non-monotonic, interacting surface.
No optimality guarantee. The random-allocation phase may beat it.

FOUR-LEVEL DESIGN SPACE NOW OPEN (Noah's idea, extended): the d8/K16384
artifact splices into the d4 base -- weight_maps verified IDENTICAL (2998
keys), entries differ only in k/dim/pack_bits. So a layer can sit at:
  level 0  d8/K16384  2.00 bpw  -0.42 GiB   (DEMOTION, frees budget)
  level 1  d4/K512    2.50 bpw   base
  level 2  d4/K2048   3.00 bpw  +0.42 GiB
  level 3  d4/K8192   3.50 bpw  +0.84 GiB
Demotion is the half we never had: 16 of 42 promotions are net-NEGATIVE,
so dropping those layers to d8 should cost little while FUNDING promotions
elsewhere -- a better artifact at CONSTANT size, and the same logic that
might rescue the 80.9 GiB rung. Demotion sweep queued (42 runs).

TOOL NOTE: pipelining splice against scoring did NOT speed the sweep up
(1.31 vs 1.22 min/sample, though confounded -- K8192 splices move 2x the
bytes). The GPU gaps Noah spotted are disk-bound, not CPU-bound; both
stages stream multi-GiB shards off the same SSD, so overlapping makes them
contend. Any real speedup has to reduce I/O, not reorder it.

## 2026-08-31 — LAYER EFFECTS ARE BASE-SPECIFIC (single-layer proof)

The 42-layer sweep table is valid ONLY for the base it was measured on
(d4/K512, 98.55 GiB). Applying it to other rungs fails, and not subtly:

  116 rung, 34 layers rearranged byte-neutral
    predicted +121.71  measured +21.20   realisation 0.17
  134 rung, 9 "free shrink" demotions
    predicted  +11.99  measured -15.36   WRONG SIGN

Both changed many layers, so interactions were a live alternative
explanation. SETTLED WITH ONE LAYER: L33 demoted K8192->K2048 at the 134
base, no interaction confound.
    predicted  +4.49   measured  -6.87   WRONG SIGN
6.87 mnats is above the 6.32 seed floor, so the flip is real. Layer
effects depend on what precision the REST of the model carries -- the
same error-cancellation story that made 16 of 42 promotions net-negative
at the 98.55 base, seen from the other side. PER-RUNG SWEEPS ARE REQUIRED.

WHAT STANDS (measured at its own base, so unaffected):
  101.93 GiB greedy best-8   KL 293.84  (+54.98 vs the 98.55 base)
  101.94 GiB DP optimum      KL 291.69  (+57.13; beats greedy by exactly
                             2.15 -- splice-vs-splice is deterministic, so
                             that IS resolvable; only cross-SEED
                             persistence is unknown)
  116.28 GiB optimized       KL 178.33  (+21.20, BYTE-NEUTRAL) -- achieved
                             with mistransferred priors, so a proper
                             K2048-base sweep should beat it
  130.20 GiB shrink variant  KL 109.90  -- worse than the 134 base but
                             ~7 mnats BETTER than uniform interpolation at
                             that size, i.e. targeting helps even with bad
                             priors

TOOLING NOTE: reduced-token sweeps need their own teacher cache. The KL
scorer REFUSES a cache built at a different --tokens ("token ids differ
from the cache") -- a good gate that caught a silent-garbage path. A
512-token cache (glm53_teacher_topk_prose_512) was built for future
sweeps; deltas against it are internally consistent for RANKING but NOT
comparable to the 2048-token ladder numbers. Any shipped artifact must
still be scored at 2048 against the original cache.

PARKED HERE (Noah, 08-31): marginal value of further allocation work is
low against a V1 that is blocked on generation gates, not on quality.
Resume points if V2 wants them: per-rung sweeps at 512 tokens (~1h each),
a higher-K donor fitted for only the promotable layers (--vq-layers takes
a comma set, so ~8 layers at d4/K16384 is ~3h not ~18h), and the same
process applied to the 397B where the paper's claims live.

## 2026-09-01 — V2 RELEASE TRUNKS built to size targets (100/120/130 incl. head)

Standing convention now in force (METHOD.md §6.0): published size = trunk +
2.9 GiB reserved for the MTP head, so a downloader who picks a rung for their
machine still has room to add the head later. V1 was trunk-only.

Solved to SIZE TARGETS (not byte-neutral) from bases with 2048-token sweeps;
the 116 base was deliberately not used (its sweep is 512-token, R^2 0.438).

  build   trunk GiB     KL   prose   code    lit  uniform@size   vs uniform
  v2_100      97.30  338.32  2.5719 1.7114 1.5605      377.20      +10.3%
  v2_120     116.91  182.11  2.2162 1.5841 1.2396      183.01       +0.5%
  v2_130     126.19  136.13  2.0758 1.5532 1.2173      129.97       -4.7%

v2_100 IS THE BEST ALLOCATION RESULT OF THE ARC (+10.3%). It beats its own
98.55 base while being 1.25 GiB SMALLER (338.32 vs 348.82), and beats affine
q3 by 10% at 28.8 GiB less. It got there by shrinking: 11 layers demoted to
d8/K16384 funding 6 promotions.

v2_130 IS WORSE THAN UNIFORM and is DOMINATED by r134opt2, which is both
smaller (124.72) and better (KL 130.78). Discard v2_130.

NO RELIABLE PREDICTOR OF SUCCESS. Across eight solves the advantage spans
-4.7% to +10.3% with no clean relationship to layer count, displacement,
data quality, or predicted gain. Notably the 512-token r134opt2 (+4.7%) beat
BOTH 2048-token solves at the same base. §8's "displacement generates
advantage" does not survive v2_120 and v2_130 and should be read as one
observation, not a rule. The method produces good artifacts often enough to
be worth running, and the only way to know which is to build and score.

RECOMMENDED RELEASE SET (all beat uniform, all gated on check-release +
check-bundle):
  100 -> v2_100     trunk  97.30, KL 338.32, published 100.20   +10.3%
  120 -> r116opt    trunk 116.28, KL 178.33, published 119.18    +4.8%
         (or v2_120 if literary is weighted: 1.2396 vs 1.3402)
  130 -> r134opt2   trunk 124.72, KL 130.78, published 127.62    +4.7%

TWO BLOCKERS BEFORE UPLOAD:
1. RUNTIME DRIFT. vq_switch.py changed DURING the build run (v2_100 bundled
   1863 lines, v2_120/130 bundled 2019) and the file has UNCOMMITTED
   working-tree edits — the other session is fixing the dense 27B kernels in
   it. So v2_120/130 embed uncommitted code, and r116opt/r134opt2 carry the
   older 1366-line runtime. check-bundle passes a dirty tree silently; it
   should refuse, or record the commit + dirty flag in the manifest. Rebuild
   the chosen set against ONE pinned commit before publishing.
2. NOTHING HAS GENERATED A TOKEN. The 100 rung fits the M4 (128 GB) and must
   be smoked there; 120/130 are exo-cluster targets by design and only exo can
   execute them. Given three PUBLISHED dense 27B artifacts were found today to
   crash on first forward for downloaders, this is not optional.
   GLM's bundle was audited clean of that specific bug: it imports only
   stdlib/mlx/numpy and falls back to mlx_vlm.models.glm5_next, which IS in
   released mlx-vlm 0.6.17.

## 2026-09-01 — FINAL V2 RELEASE SET (100 / 120 / 140, head-inclusive)

Targets softened by Noah: "I don't want to ship a worse model over 1-2gb,
particularly at 120GB and above" — only the 100 sits near a hard constraint
(M4, 128 GB). Selection is therefore BEST-IN-CLASS, not nearest-to-target.

  rung      trunk  published      KL   prose   code    lit   total nats
  v2_100    97.30     100.20  338.32  2.5719 1.7114 1.5605      0.739
  v2_120   116.91     119.81  182.11  2.2162 1.5841 1.2396      0.283
  v2_140   135.06     137.96   89.21  2.0108 1.5490 1.1952      0.127
  affine reference: q3 129 GiB 0.766 | q4 166 GiB 0.155 | q6 239 GiB 0.024

BOTH ENDPOINTS BEAT THEIR AFFINE COUNTERPARTS ON EVERY CORPUS AT ~28 GiB LESS:
v2_100 vs q3 (0.739 vs 0.766 at 100.2 vs 129) and v2_140 vs q4 (0.127 vs 0.155
at 137.96 vs 166). v2_140 is the strongest artifact of the arc.

v2_140 = the 134 base plus the FIVE measured-positive promotions to
d4/K16384 (L6, L9, L33, L36, L38), +1.06 GiB for -29 mnats. Promotions
realised well, consistent with every other promotion-only solve (~0.84) and
in contrast to every demotion-heavy one (0.17-0.68).

METRIC CORRECTION — OUR "KL" IS PROSE-ONLY. The teacher cache is
glm53_teacher_topk_prose and --kl-cache is passed only on the prose run; code
and literary report perplexity alone. So ranking artifacts by KL weights prose
at 100%. Consequence: v2_120 LOSES to r116opt on KL (182.11 vs 178.33) but
carries 24% LESS TOTAL DAMAGE (0.283 vs 0.373 summed nats), because it is much
better on code and literary. Several earlier better/worse calls in this ledger
were made on KL alone and should be re-read as prose-only statements —
including "v2_130 is worse than uniform", which was a prose-only claim.
Use SUMMED ADDED NATS across the three corpora for artifact selection.

REJECTED FROM THE SET: v2_130 (dominated by r134opt2 on size AND prose-KL);
r116opt (worse total nats than v2_120); r134opt2 (superseded by v2_140).

BLOCKERS UNCHANGED, both required before upload:
1. Runtime drift — v2_100 bundles 1863 lines, v2_120 2019, v2_140 2035, all
   from an UNCOMMITTED vq_switch.py the other session is actively editing.
   Rebuild the three against ONE pinned commit once that work lands. Noah:
   "we'll add the corrected runtimes once it finishes."
2. No token has been generated. v2_100 (100.2 GiB) fits the M4 and must be
   smoked there. v2_120/v2_140 are exo-cluster rungs by design. Noah: "Not
   publishing without smoke anyway."
