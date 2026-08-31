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
