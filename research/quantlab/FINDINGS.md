# FINDINGS — settled laws, dead ends, and design rules

**Read this BEFORE proposing an experiment.** If your idea re-tests a law or
re-opens a retraction, cite new evidence or drop it. EXPERIMENTS.md is the
chronological lab notebook (E-numbers); this file is the distillation. Keep it
under ~150 lines; when you add a law, delete anything it supersedes.
Last updated: 2026-08-20 (through E80).

## I. Settled laws (each has survived at least one attempt to kill it)

1. **Quality tracks total bytes; geometry packaging washes at matched size.**
   d/K combos at the same bits-per-weight land on the same size-quality curve
   (log2(K)/d IS the rate — this is rate-distortion, not a house discovery).
   No exotic geometry rescues a size class. [E45, E50, E78, gemma domain scan]

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

10. **At matched bytes, d4 + big codebook beats d2 + small codebook —
    SUPPORTED, not settled.** Admissible evidence is qwen-only (Noah's
    ruling 08-20: gemma-family scores are excluded as an instrument herring
    — see gemma4-loglikelihood note). On qwen: d2-K64 sits 1.26 pts below
    d4 at matched 3.25 bpw. Directionally clear, margin thin, one pair.
    The gemma corroboration (5.84 pts, widening with bpw) is set aside, not
    refuted. A second qwen matched-bytes pair would settle it. d2's real
    wins are operational either way: ~8x cheaper fits, faster decode at
    K≤256 (byte-aligned). "Raise K first" stands as the working default.
    [E-record ~L3380-3400, qwen table only]

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
- Qwen MoE speed candidates Q1-Q4 registered in E77, unmeasured.
- Settle law 10: score d2-K64 vs d4-K4096 (both exactly 3.0 b/w of codes,
  BOTH ARTIFACTS ALREADY EXIST in qwen36-35b-rungs) on one instrument.
  Scoring only, no fit — minutes, not an hour.
- Why is creation-binding insufficient for lazy loads? (MLX-side curiosity.)
