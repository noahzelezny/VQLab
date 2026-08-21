# FINDINGS — settled laws, dead ends, and design rules

**Read this BEFORE proposing an experiment.** If your idea re-tests a law or
re-opens a retraction, cite new evidence or drop it. EXPERIMENTS.md is the
chronological lab notebook (E-numbers); this file is the distillation. Keep it
under ~150 lines; when you add a law, delete anything it supersedes.
Last updated: 2026-08-20 (through E80).

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

5. **Size model (validated 3x out-of-sample, best err +0.05 GiB):**
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

8. **Operational sweet spot: d4/K256.** Byte-aligned uint8 codes (zero unpack
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
5. Every new gate must FAIL on a known-bad input AND PASS on a known-good
   before it is trusted. (A gate that always fails is one nobody reads.)
6. Report nothing predicted as measured. Label ESTIMATE vs MEASURED in every
   table.
7. **Never compare artifact sizes before packing.** Stored bytes include
   whole-byte padding and are not a size. Quote packed bytes or analytic
   bpw. Three separate wrong conclusions today traced to this (E83's
   candidate-G rejection, the 397B pre-pack readings, the E87 overclaim).
8. **Before scoring ANY artifact, confirm it passed an outlier gate on a
   trusted box.** A corrupt artifact scores plausibly and silently, and the
   fitter's own log structurally cannot see it (it reports what it COMPUTED,
   not what reached disk). Cost of skipping this: E82. [E85]

## IV. MLX/Metal engineering rules (each cost ≥1 run to learn)

- **Any lazy read still pending when a save forces evaluation is paid inside
  a GPU command buffer** → watchdog kill "at the write step." Load under
  `with mx.stream(mx.cpu):` AND `mx.eval` inside the block — creation-binding
  alone is measured-insufficient; the eval is LOAD-BEARING. [E70 add. 5-6]
- Never edit a script a running chain has not yet invoked (Python reads at
  invocation). Never `rm -rf` a fit output dir: fits RESUME (and the resume
  check now validates shard completeness, not existence).
- Stale scripts on a second box = silent divergence → `check_scripts_sync.sh`
  in every chain preamble.
- Prefer local reads for big sources; reads outweigh writes ~6:1 in a fit.
- Make failure cheap (resume, quarantine, verified copies) — today's error
  budget held because failures were recoverable, not because reasoning was
  right.

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
