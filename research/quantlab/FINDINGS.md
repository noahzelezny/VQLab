# FINDINGS — settled laws, dead ends, and design rules

**Read this BEFORE proposing an experiment.** If your idea re-tests a law or
re-opens a retraction, cite new evidence or drop it. EXPERIMENTS.md is the
chronological lab notebook (E-numbers); this file is the distillation. Keep it
under ~150 lines; when you add a law, delete anything it supersedes.
Last updated: 2026-08-21 (through E110).

## I. Settled laws (each has survived at least one attempt to kill it)

1. **Quality tracks total bytes; packaging washes across K at d4.**
   Measured within the d4 family (E84, one instrument): KL 214.5 -> 85.5 ->
   68.5 -> 56.4 as bpw rises 2.00 -> 3.25, with the expected flattening at
   high K. Whether it also washes ACROSS d is UNSETTLED — the evidence that
   said otherwise was contaminated (E85). Rate (log2(K)/d) prices the bytes;
   whether it predicts the damage across d is untested.
   [E45, E50, E78, E84; E82 void]

2. **Position law: early layers tolerate cheap bits; enrichment pays only in
   the back of the network; knee ≈ layer 30 (of 60, affine) / L10 (VQ
   shallow-harvest).** Established in the affine era (tail30), re-confirmed in
   VQ clothing (cheap-shallow). It TRANSFERS across quantization families.
   Do not rediscover it a third time. [E12, E25, tail-ladder, E74/E78/E79]

3. **Escape the cheapest width broadly before enriching narrowly.** Under a
   byte budget, maximize non-cheapest layers; never buy the expensive width
   while any layer sits at the floor. Flat allocation at the target width is
   provably optimal at its size for 2/3/4-bit affine. [tail-ladder E-record]

4. **Harvest cost falls ~10x as base richness rises.** Taking shallow bits
   back costs 0.03 ppl/GiB off a K128 base, 0.003 off K256. It is NEVER free
   and never a quality win at a flat rung's own size — its value is hitting
   sizes BETWEEN flat rungs at ~2x the byte-efficiency of stepping down.
   [E78, E79 addendum; E80 tests the K2048 end tonight]

5. **Size model (6 hits + 1 in-band, 0 misses; stamped through E92/E93 post-graft, 08-21):**
   STAMP EVERY DATA POINT pre-graft or post-graft — the vision tower is a
   fixed 912,020,960 bytes (0.849 GiB, byte-identical across grafted
   artifacts) and historic points are all POST-graft; mixing the two is a
   units mismatch that briefly presented as a "flat-geometry bias" (n=2,
   killed by a 2-minute header read). Original form:
   `new_size = base_size − 1.87 GiB × shallow_bits_harvested` (397B; shallow
   region L0-9 = 1.87 GiB/bit, body L10-56 = 8.81 GiB/bit). Price a rung
   before fitting it. [E74 addendum, E78]

6. **Fit error ≠ output damage.** Weight-space relerr does not rank output
   quality across geometries or surfaces (e4b MLPs had excellent relerr and
   failed; embeddings the reverse). Only KL/ppl on the assembled model counts.
   [E55, E69, E76]

7. **Data-free k-means VQ is competitive with calibrated affine at matched
   size on MoE experts** — the core method claim, held on two families, and
   the reason our 100.9 GiB beats spicy2.6's larger artifact. [shipped rungs]

8. **Operational sweet spot: d4/K256.** NOTE: healthy relerr ranges SCALE
   WITH K — K2048-class fits sit ~0.19, K256-class ~0.31, K128 ~0.46. An
   abort threshold tuned at one K is wrong at another (0.35 would have
   killed the healthy E92 K256 refit; peer near-miss 08-21). Set
   --relerr-abort per geometry, not from this file's K2048-era numbers. Byte-aligned uint8 codes (zero unpack
   tax — the only rung that pays none), threadgroup-sized codebook, ~40 min
   fits at 397B, won both corpora at its size. Leave it only when the size
   budget forces you. K>256 costs uint16 + pack tax + ~6x fit time. [E71-79]

9. **Decode speed is a wash across all measured geometries; prefill is where
   geometry shows** (non-byte-aligned code widths pay bit-extraction: −8%
   measured at 6-bit). [E70, E71 speed rows — artifact-vs-artifact, unaffected
   by the E79 retraction]

10. **d4 beats d2 at matched information rate — measured, modest.** Clean
    pair (E87), both arms fit on M3 from one bf16 source, both outlier-gated
    before scoring, same instrument: at 2.00 bpw, d4-K256 = KL 210.7 /
    80.05% top-1 vs d2-K16 = 239.9 / 78.43%. **d4 wins by ~12% KL, NOT the
    3.3x that E82's corrupt arm produced (overstated ~25x).** Scope: one
    pair, at a bpw that forces d2 to a coarse K16 — a 2.5 bpw pair would
    test generality. Sizes are IDENTICAL when packed (13.83 GiB both); the
    21.3 GiB d2 figure is uint8 padding only. "Raise K first" = a measured
    preference, not a landslide. [E87 + correction; E82 void per E85]

11. **At LOW K, a better-fitting codebook can be a WORSE model — because
    k-means trades the tail for the bulk.** Measured on a matched K256 pair
    (identical size, identical geometry, only the fitter differs): the newer
    fit has lower mean relerr on all three projections and scores WORSE
    end-to-end (2.8057 vs 2.7655). Bucketing error by |w| percentile shows a
    monotonic crossover — better on the bottom 90% of weights, worse on the
    top 1%, consistent across layers:

        |w| 0-50  -0.0115 | 50-90 -0.0011 | 90-99 +0.0001 | 99-99.9 +0.0057 | 99.9-100 +0.0112

    k-means minimizes AVERAGE distortion, so when centroids are scarce it packs
    them into the dense middle and abandons the rare large-magnitude weights
    that dominate output. At K=2048/K=8192 there are enough centroids for both,
    so the same fitter change is a clean WIN. **This is why the 08-18 fitter
    change helps at large K and hurts at small K** — it is not a bug, it is the
    stated objective meeting a metric that weights the tail far more heavily.
    [E101, E102]

12. **Mean relerr is a BULK statistic and is the wrong gate at low K.** It is
    dominated by the 90% of weights that do not matter much, so it reports the
    bulk/tail trade above as an improvement. At low K it is not merely blind to
    output damage (E98: identical relerr, 6% KL apart) — it is
    ANTI-CORRELATED (E101). A tail-aware statistic (normalized error in the
    99-100th |w| percentile) would have flagged the bad artifact before
    scoring. Any fitter-tuning loop that reads reconstruction error will pick
    the worse artifact, confidently. [E98, E101, E102]

13. **The ++ SEEDING PENALTY IS DEPTH-STRUCTURED, and profiling it is a
    first-pass job for any new family.** 36 tensors on the 397B, init as the
    only variable: below L15 k-means++ is uniformly BETTER (L00 gate/up by
    -0.11/-0.13); from L20 on it sells the tail on 18 of 24 body tensors and
    is better on zero. Body `down_proj` is 8/8. Because the body is
    8.81 GiB/bit against shallow's 1.87, a body-only penalty IS an artifact
    penalty — which is why the K256 refit lost while K2048/K8192 won. Mean
    relerr across those body tensors moves -0.00033, so the gate sees nothing.
    **Run `probe_init_sweep.py` on a new family BEFORE fitting anything** —
    see PROCESS.md. [E107, E108, E109]

## II. Retracted / false leads — do NOT re-chase without new evidence

- **"Cheap-shallow beats the rung above it" — FALSE.** Root cause: E71 used a
  PROXY score (2.8197) as the shipped-2.4 column; the real 2.4 wins both
  corpora. The ladder is monotone in size. Every mechanism built on the
  anomaly (low-bit lever, floors, "shallow bits are wasted") is retracted
  with it. [E79]
- **"VQ beats 8-bit affine on embeddings" (e4b) — CONFOUNDED.** The KL win
  came from an accidental fp32 path (missing dtype cast), not the codebook.
  Fair test (affine given the same path) not yet run. [E76]
- **Fused row-gather GEMM prefill lever — DOES NOT EXIST.** MLX already fuses
  it; recoverable ≈ 0. The stale header note is the trap. [E52]
- **Byte-aligned packing — pure loss.** bits%8==0 saves 0 bytes, cost 37%
  decode. Packers now skip it. [E70]
- **The published e4b prefill −21% — instrument artifact** (21-token prompt,
  n=1). Real: −11% prefill, −17% decode. [E76]

## III. Instrument rules (violations produced every false result this week)

1. Pre-register predictions before fitting/scoring; falsified predictions are
   recorded as falsified, never reframed.
2. **A comparison row must name the artifact AND instrument that produced it;
   a number older than the artifact it faces gets re-measured, not cited.**
   (The proxy-score disaster.) Never compare a real artifact to a proxy score.
3. Comparators must pass `check_comparator.py` (or same-convention structural
   check) before their row is believed — a comparator that loads short scores
   worse and FLATTERS us.
4. Speed numbers: n≥3 with scatter, prompt length stated, one process per
   arm, never on a contended box, never on a model larger than RAM
   (streaming referee is fine; speed tests are not).
   **And quote a RATIO between arms measured in the same session, never an
   absolute.** At ~100 GiB the decode instrument is BIMODAL: the same
   artifact, same box, same script, back to back, gave 21.14 / 12.69 / 21.27
   / 21.20 tok/s. Swap (swapouts never moved), thermal (fan on the box), and
   storage path (both paths produce both modes) are ruled out by measurement
   — the cause is not yet known, which is exactly why absolutes are unsafe.
   Excluding a degraded sample is only legitimate when the mode was shown to
   exist independently of which arm produced it. Load time IS cleanly
   path-dependent (19s local vs 42-60s SMB) and belongs in no card. [E115]
5. Every new gate must FAIL on a known-bad input AND PASS on a known-good
   before it is trusted. (A gate that always fails is one nobody reads.)
6. Report nothing predicted as measured. Label ESTIMATE vs MEASURED in every
   table.
6b. **CHECK THE INPUTS BEFORE THE ALGORITHM. The cheapest check should come
   first, and on 08-21 it came last.** A ~0.04-0.06 ppl gap between the
   shipped 397B 2.4bpw and every refit of it was chased for two days as a
   property of the k-means: four explanations proposed INSIDE the algorithm
   (seeding, a K-crossover, float summation order, the fitter file itself),
   three falsified by experiment (E117, E118, E121) and the fourth voided.
   The actual cause was provenance — the BASE artifact had been silently
   rewritten Aug 19, three days AFTER the artifact built from it — and it was
   found with `ls -l` in about ninety seconds, once someone thought to look at
   the input rather than the algorithm. **Before designing an experiment to
   explain a difference between two runs, mtime and size every input both
   runs consumed: base, source, config, and the tool.** E117/E118/E120 are
   sound per-tensor physics aimed at the wrong object.
   Corollary on searching: when concluding an input is GONE, say where you
   looked. Two sessions searched five locations (SSD, HDD, external T7, and
   both boxes' exo caches — the caches existed but were 0 B). "We searched"
   over three locations would have been a weaker claim wearing the same words.
   [E121, E94]

6c. **KL AND PPL CAN RANK TWO ARTIFACTS OPPOSITELY. Report both, always.**
   First observed 08-21 (E124), one case, same model and same corpus:

       q4 (affine 4-bit)  14.094 G   KL 45.842   top-1 89.82%   ppl 5.2055
       E124 d2/K256       13.596 G   KL 40.327   top-1 90.10%   ppl 5.2330

   KL and top-1 say the VQ rung is CLOSER TO THE TEACHER; ppl says it is a
   WORSE LANGUAGE MODEL. Both are true and they measure different things: KL
   is agreement with one teacher's full distribution, ppl is absolute
   predictive quality on text. A quantisation can track the teacher's
   distribution more faithfully while being worse at the actual next-token
   job. This is NOT law 6 (weight error vs output damage) — it is one level
   higher, two OUTPUT metrics disagreeing.
   **Methodological consequence, which is the part that bites:** our entire
   27B ladder is ranked by KL. Any rung selected on KL alone is selected on a
   metric that has now demonstrably disagreed with the one Noah asks for. From
   here, every rung reports BOTH, and a winner is never declared on one.
   **Status: n=1.** One inversion is an existence proof, not a rate. The
   0.0275 ppl gap is small and no repeat has been measured. The M4's d2/K512
   (same d, same widths, 12.5% more code budget) is the natural test and was
   registered as such BEFORE it finished. Do not generalise until it lands.
   [E124]

6d. **FITS ARE STATISTICALLY REPRODUCIBLE, NOT BITWISE REPRODUCIBLE.** MLX's
   RNG is not seeded across processes, so the same fit twice gives different
   codebooks (measured: L30 d2/K512 -> 0.0590 vs 0.0592, codes and codebook
   differ, scales identical). But aggregate metrics reproduce to our reporting
   precision: a fresh fit of E94's recipe matched the original on every
   projection mean to 4 decimals and on KL to 3. **So "this artifact scores X"
   survives a re-run; "these exact bytes" does not. Say which you mean.** Two
   corollaries: a lost artifact does not necessarily cost you its NUMBER, and
   a difference at the 0.04+ ppl scale cannot be seed noise — look for an
   input difference instead. [E125, and it corroborates E121]

7. **Cite the commit when citing a law, and RESOLVE the citation.** This file
   moved twice in 25 minutes and a peer built an argument on the stale
   version. A law citation without a commit hash is as untrustworthy as a
   comparator row without an instrument, for the same reason.
   **A citation is only as good as the last time someone actually looked it
   up — and the most-cited rules get resolved least, because familiarity
   substitutes for lookup.** On 08-21 "III.10" was cited 14+ times across
   EXPERIMENTS, STATE, PROCESS, a compaction summary and a cross-session
   handoff, by two sessions, over several days. No such rule existed; section
   III had nine items. It survived not despite being repeated but BECAUSE it
   was: every repetition raised its apparent authority without adding a
   single check, and the handoff that told another session to "run III.10"
   was read by someone who already believed they knew what it said. A wrong
   entry has one author; a phantom citation recruits everyone who repeats it.
   It cost nothing only by luck — the rule was followed because its author
   happened to be the one running the gate. Resolve before you repeat,
   especially when you are sure. [smoke-gen rule now III.11, 3600888]
8. **Never compare artifact sizes before packing.** Stored bytes include
   whole-byte padding and are not a size. Quote packed bytes or analytic
   bpw. Three separate wrong conclusions today traced to this (E83's
   candidate-G rejection, the 397B pre-pack readings, the E87 overclaim).
9. **Before scoring ANY artifact, confirm it passed an outlier gate on a
   trusted box.** A corrupt artifact scores plausibly and silently, and the
   fitter's own log structurally cannot see it (it reports what it COMPUTED,
   not what reached disk). Cost of skipping this: E82. [E85]

10. **When a bug is rare and you have a reproducer, measure BEFORE you fix.**
   A fix landed while the evidence is still unmeasured is how "it stopped
   happening" gets mistaken for "we understood it" — and the rarer the bug,
   the longer that mistake survives undetected, because absence of the
   failure is exactly what both a real fix and a lucky run look like. The
   zeroed-tensor collapse was chased for two nights as three different
   diseases (write corruption, a cursed box, SMB) precisely because every
   encounter was met with a change rather than an instrument. Cost of the
   discipline: ~30 minutes of holding a known one-line fix. [E119]
   **Corollary, learned the same night at a cost of 41 minutes: INSTRUMENT
   THE REAL CODE, DO NOT REIMPLEMENT IT.** A probe written to mimic the
   failing path caught 0 in 240 tensors and yielded no verdict, because the
   reimplementation silently dropped the one variable under test: the real
   fitter accumulates every tensor's outputs so memory grows monotonically,
   while the probe discarded and cleared each iteration and never built the
   pressure the hypothesis names as the trigger. A probe that CANNOT fire is
   indistinguishable from a bug that is not there — the exact ambiguity this
   rule exists to prevent — so holding the fix bought nothing while the
   instrument was wrong. Correct form: patch the REAL file with a guarded
   diagnostic block, then diff the patched copy with the diagnostic stripped
   back against the original and confirm the guard is the ONLY difference.
   [E119]

11a. **THE SMOKE (III.11) NEEDS THE WHOLE MODEL RESIDENT — so it cannot run
   on a box smaller than the artifact.** The streaming referee scores models
   larger than RAM by design; GENERATION cannot. On 08-21 a publish-readiness
   chain was pointed at a 100.971 GiB and a 143.682 GiB artifact on the 96 GiB
   M3: the first thrashed (mlx warned "requires 102524 MB", max recommended
   86016 MB) and the second drove swap from 0 to 60 GiB of 61 GiB before it
   was stopped. Neither could have produced a verdict. **Current capability:
   d8 (100.97) smokes on the M4 (128 GiB) only; the 143.682 GiB rung fits
   NEITHER box and can only be smoked on the exo pair.** Check artifact bytes
   against box RAM before scheduling a smoke — the rule already existed for
   speed tests ("never on a model larger than RAM", III.4) and was not carried
   across to the smoke, because the smoke was written as a cheap correctness
   check rather than as a resident-memory operation.

11. **An artifact is not releasable until it has GENERATED ONE TOKEN through
   the fused path it will ship with.** Every byte-level gate we own can pass
   an artifact that cannot produce a token: the gate decodes weights, it does
   not execute the runtime the artifact bundles. Proven twice on 2026-08-21 in
   one evening on the first d=4 DENSE artifact (E95) — the dense fused kernel
   exists only for d=2 and raised, and the expert-kernel fallback then died at
   KERNEL LOAD at 27B mlp shapes (threadgroup 36864 B vs Metal's 32768 cap).
   Neither is visible to any relerr gate, and no bench would have touched
   either branch. Cost: seconds. [E100, E95]

   *(This rule was STATED at E100 and cited as "III.10" for days without ever
   being written here — a dangling citation that survived because everyone,
   including its author, trusted the number. III.10 was legitimately taken by
   the measure-before-you-fix rule on 08-21. Renumbered to III.11 and recorded
   properly; the citation rule (III.7, cite the commit) exists for exactly
   this failure and was not applied to a rule about gates.)*

## IV. MLX/Metal engineering rules (each cost ≥1 run to learn)

- **Any lazy read still pending when a save forces evaluation is paid inside
  a GPU command buffer** → watchdog kill "at the write step." Load under
  `with mx.stream(mx.cpu):` AND `mx.eval` inside the block — creation-binding
  alone is measured-insufficient; the eval is LOAD-BEARING. [E70 add. 5-6]
- **THE DEFERRED-READ FAMILY — sweep, do not discover one at a time.** Any
  script that `mx.load`s and later `save_safetensors` without an eval in
  between carries the IV.1 fault. Found on 08-21 by grepping the tree rather
  than by hitting them: `build_dense_vq.py` (013d2bb), `pack_dense.py`
  (3af8ed0), `graft_vision.py` and `build_e4b_vq.py` (this sweep).
  `graft_vision.py` was the worst exposure — it runs on EVERY published
  artifact, held its reads lazy across every source shard AND a `del data`,
  and `check_vision.py` verifies vision tensors are PRESENT, not non-zero, so
  a zeroed graft would have passed every gate we own. **Audited all 15
  on-disk graft shards including the three published artifacts: 333/333
  non-zero everywhere, no damage shipped.** Detection: `grep -l
  save_safetensors *.py` then check each for `mx.stream(mx.cpu)` + `mx.eval`.
  **STILL UNFIXED: ELEVEN files, and THREE sit on publish paths** — do not
  run any of them without applying the cure first:
      vq_35b_codes.py        <- produced the published 35B VQ artifacts
      fit_e4b_vq.py          <- produced the published gemma-4-26b artifact
      kl_damage.py           <- our SCORING instrument
      fitter_0816_cdcdeab.py <- E121's 08-16 fitter; deliberately unfixed to
                                stay faithful to that vintage, so it must be
                                GATED not cured (scan its log for
                                "relerr 1.0000", then zero-scan the artifact)
      fit_e4b_ple.py   vq_397b_fused.py   vq_fit.py
      assemble_gptq_35b.py   assemble_gptq_35b_v2.py
      dwq_assemble_tail.py   rotate_fuse.py
  **Detection must test the GUARD, not the file:** my first sweep counted
  `mx.eval` anywhere in the file and reported four, missing seven — a file
  can eval elsewhere and still leave the load-to-save window open. Correct
  test: has `save_safetensors` AND `mx.load(` AND NOT `mx.stream(mx.cpu)`.
  Caught by the M4 session re-running it independently.
- Never edit a script a running chain has not yet invoked (Python reads at
  invocation). Never `rm -rf` a fit output dir: fits RESUME (and the resume
  check now validates shard completeness, not existence).
- **A REFIT MUST NEVER AIM `--out` AT A SCORED ARTIFACT'S PATH.** Two
  independent consequences, and the second was learned the hard way on
  08-21: (a) resume-skip emits a repaired-LOOKING artifact containing the
  suspect bytes (below); (b) even a PERFECTLY CLEAN refit silently destroys
  the evidence for the number already published against that path — E94's
  53.022 mnats now describes bytes that no longer exist, unrecoverably. (b)
  fires even when nothing goes wrong, which is why the resume-skip framing
  did not prevent it. Corollary: a directory's sidecar files (README,
  tokenizer) keep their old timestamps through a refit, so a stale-looking
  `ls` is NOT evidence the weights are untouched — check the shards. [E94]
- **A REFIT MUST WRITE TO A NEW DIR.** Because fits resume, aiming a refit at
  the dir it is meant to repair makes it skip every existing shard as
  "complete" and emit a repaired-LOOKING artifact containing exactly the
  suspect bytes, with rc=0 and a log full of "exists, skip." Completeness
  validates shard *structure*, not shard *correctness*, so no downstream gate
  catches it. Keeping the original also lets you score old vs new on one
  cache. [caught pre-launch on the d2-K64/K128 repairs, 08-21]
- Stale scripts on a second box = silent divergence → `check_scripts_sync.sh`
  in every chain preamble.
- **Never register a duration derived from a probe. Probes measure the cheap
  half.** Three in a row now, all optimistic in the same direction: the d8
  fit-cost probe timed centroid updates and not assignment (4x fast); E89 was
  read off an elapsed-seconds counter instead of shard write stamps (1.5x
  fast); the scatter-add port predicted 30-45 min for E94/35B-K8192 against a
  measured 2.8 h (3.7x fast). A duration is MEASURED only from a completed run
  of the same shape; anything else gets stated as unmeasured, not as a number.
  Schedules built on probe timings put real deadlines at risk. [08-21]
- **Scheduling, settled [Noah, 08-21]: nothing gets killed to defend a
  schedule; an idle box is cheaper than lost work.** REFINEMENT: ending a
  QUEUE is not killing WORK. Stopping a probe wrapper so no further iterations
  launch, while the in-flight iteration runs to completion, throws nothing
  away — and a justification that rested on "this costs nothing" expires the
  moment it starts costing something. Re-examine standing decisions when their
  premise changes rather than when they fail. [peer, 08-21] Before treating a
  constraint as a deadline ask whether it is a TIME requirement or a
  COINCIDENCE requirement — the exo 2-box smoke needs both boxes quiet, which
  is satisfied tonight as well as at 14:30. Interrupt cost scales with shard
  granularity (a 3-shard 35B kill loses up to ~56 min; a 27-shard 397B, minutes).
- **Model names: Qwen3.5-397B-A17B and Qwen3.6-35B-A3B BOTH declare
  `model_type: qwen3_5_moe`** (60L/512exp vs 40L/256exp). "Different
  generation" is defensible; "different architecture" is FALSE. A real
  `Qwen--Qwen3.5-35B-A3B` also sits on the share, so an imprecise reference
  lands a reader on a model we never measured. Name models exactly.
- **A constraint that arrives as a given gets checked ONCE before anything is
  built on it.** The 14:30 deadline was never examined: one session asserted
  it, the other hardened it into a kill rule, then an unconditional kill rule,
  then an armed watchdog, each step locally sound. Careful execution downstream
  of an unexamined premise produces MORE confident wrongness, not less, because
  the machinery looks so sound that it draws scrutiny away from what it is
  protecting. Ask what makes a constraint binding before defending it. [08-21]
- **MEASURE THE OBJECT YOU MEAN. Today's recurring failure, seven instances,
  one shape: a real check run competently against the wrong thing, returning a
  number that looked right.** A verification scoped to the SOURCE checkpoint
  quoted as covering the BASE; a size read off the VQ subset quoted as the
  whole artifact; a pre-graft size compared with a post-graft one; durations
  extrapolated from probes that timed the cheap half (4-6x optimistic); an
  outlier gate run under the wrong `--family` and its KeyError read as "the
  gate cannot parse this family"; `pgrep` matching the zsh wrapper instead of
  the job; exo sizing a PACKED artifact from an index that described the
  UNPACKED one (+37%). None looked like an error. The remedy is boring: name
  what you measured and what you meant, and confirm they are the same before
  believing the number. State verifications as "X verified against Y", never
  bare. [08-21]
- **The clause added as a courtesy gets the least scrutiny.** A card sentence
  written to be generous about a result ("it also reproduced on...") carried
  two errors — wrong model family and a scale off by 10x — while the numbers
  around it were checked hard. Generous asides feel like they cost nothing, so
  they skip the verification the load-bearing claims get. Check the throwaway
  clause at the same standard, or cut it. [08-21]
- **A claim about a TOOL needs matched pairs at more than one geometry, made
  BEFORE the claim.** The 08-18 fitter change was called an improvement for
  hours on one matched pair (K2048) plus one unmatched corroboration. E92
  falsified the generalization only because it happened to be size-matched
  against a PUBLISHED artifact — had it been an unmatched rung like E93,
  nothing would have contradicted it and we would have published a general
  improvement that is false at a third of the ladder. Luck, not process.
  Measured: K2048 win, K8192 win, K256 LOSS, the loss 3.7x the win. Three
  points falsify "uniform"; they do NOT establish "helps at large K" — that
  needs a fourth matched pair at another K. [E92, 08-21]
- **A retraction must chase the CITATIONS, not just the claim.** E82 was
  voided on 08-20; on 08-21 its heading still read "LAW 10 SETTLED" above a
  table wrong by ~25x, and E83 still cited its 3.3x as live support — E83
  being the argument that justified the d8 fit. FINDINGS got updated because
  that is where laws live; a dependency sitting in prose in another entry was
  invisible. Citations are how a claim propagates, and a reader arriving by
  grep never sees a retraction filed 30 lines later. **When you void
  something, grep the notebook for its E-number and annotate every hit, and
  retract IN PLACE at the heading.** [E82/E83, 08-21]
- **GENERATE ONE TOKEN before calling anything releasable.** Every gate we own
  reads bytes; none runs the model. The d8-K16384 packed artifact passed the
  outlier gate, check_release, check_vision AND the referee — the referee
  scores through the REFERENCE decode path — then raised
  `NotImplementedError: no FUSED packed kernel for d=8` on its first real
  forward pass. Note the scope: unpacked d=8 kernels exist; only the PACKED
  path lacks d=8. A gate that passes tells you about the property it measures
  and nothing else. [E100, 08-21]
- Prefer local reads for big sources; reads outweigh writes ~6:1 in a fit.
- Make failure cheap (resume, quarantine, verified copies) — today's error
  budget held because failures were recoverable, not because reasoning was
  right.
- **To cancel the REST of a chain, kill the orchestrator — not the child.**
  When a queued rung stops being worth running, killing the shell that drives
  the loop leaves the in-flight job untouched: it is a child process, so it
  survives its parent, and nothing already computed is lost. Editing the
  script instead does NOT work — `sh` parses a `for` loop as one compound
  command, so the iteration list is fixed in memory the moment the loop is
  entered, and your edit changes nothing about what runs next. Verify both
  halves after (orchestrator gone, worker alive) rather than assuming. This
  is how a redirect respects a no-kill preference in substance and not just
  in letter. Used 08-21 to drop the d4/K2048 rung after Noah's target moved
  to q4-size, without disturbing the running K1024 fit.

## V. Open questions (the honest frontier)

- E80 (tonight): does harvest cost keep falling at K2048? Bar = 2.3997.
- Fair e4b embedding test: affine + fp32 path vs VQ + fp32 path. Cheap, decisive.
- Ship u8view (+33% prefill, bit-exact, E81) and re-splice bundled model.py
  in d4-K256 artifacts; then artifact-level 397B recheck of the E70 37% tax
  (does not reproduce at kernel level — laws 8/9 may need edits).
- Bundled-runtime gate: check_release should verify the artifact's model.py
  matches the runtime that produced its benchmark numbers (E81 finding 5).
- **d8/K65536 vs d4/K256 at matched 2.00 bpw** (E83): post-packing these are
  the same size with a 65,536 vs 256 codebook — the most favourable K-vs-K
  matchup available, and law 10 says K is what pays. Blockers: no kernel
  (1 MB codebook vs 32 KB threadgroup) and an unmeasured fit cost (the
  "169h" predates the k-means fixes). Decisive test is a 35B fit pair scored
  on kl_cache_qwen36 — quality first, kernel work only if it wins.
- Why is creation-binding insufficient for lazy loads? (MLX-side curiosity.)
