# Paper skeleton — data-free VQ for MoE experts on Apple Silicon

Author of record: Noah Zelezny. First reader: Dr. Saamer Saab Jr.
Working repo name "MoEMash" is ON HOLD pending E95 (dense result) — do not
bake it into the title or prose. Framing (Noah's, verbatim intent): **narrow
and thorough.** Exactly two claims, both scoped hard.

Every number in DRAFT.md traces to an E-entry; anything that doesn't is
marked `[[UNVERIFIED]]`. Five experiments land today (08-21) and get
explicit `[[PENDING]]` slots: E89 (d8 verdict), E92, E93, E94, E95.

---

## Title candidates (decide after E95)

1. "Data-Free Vector Quantization Beats Calibrated Affine Quantization for
   MoE Experts at 2–3.5 Bits" — the claim as title; safest.
2. "Flat Is the Peak: Codebook Ladders and Shallow Harvest for MoE-Expert
   Quantization on Apple Silicon" — leads with claim 2.
3. "Every Wrong Number Looked Plausible: Quantizing a 397B MoE on Two Macs"
   — leads with the methods story; riskier, memorable.
4. If E95 lands positive, a title that includes dense becomes available; if
   negative, the "MoE experts are the surface" scoping goes in the title's
   subtitle, not the title.

## The two claims (and their fences)

**Claim 1 — Data-free VQ (pure weight-space k-means, no calibration corpus)
beats calibrated affine quantization at matched bytes in the 2–3.5 bpw
MoE-expert regime.**
- Evidence, 397B (Qwen3.5-397B-A17B, all one instrument, referee
  re-scored same-day as each comparison; sizes packed + graft-stamped):
  - shipped VQ-2.2bpw: 100.9 GiB, prose 3.1706 / code 2.6988 vs
    spicyneuron 2.6bit (calibrated-affine-class community build, blind,
    text-only): 3.1830 @ ~120–121 GiB [E23/E24 instrument record; E74/E78
    ref re-scores; chart_397b_ladder.py uses 3.1843/121.0 — RECONCILE, see
    ambiguity #3 below].
  - E91 flat-K2048-refit: **2.3410 / 2.5963 @ 143.65 GiB post-graft** vs
    spicyneuron 3.5bit **2.3614 @ 165.6 GiB** — better on both corpora at
    21.9 GiB smaller [E91 RESOLVED; MODEL_CARD_397B_G].
  - E80 harvest rung: 2.3452 / 2.5969 @ 139.93 GiB — best-per-GiB [E80
    RESOLVED].
- Evidence, 35B (Qwen3.6-35B-A3B, kl_cache_qwen36, one instrument, E84):
  d4-K8192 (3.25 bpw analytic) = **56.4 mnats / 89.4%** vs mlx-community
  4-bit = **78.6 mnats / 85.61%** at larger size (19G) [E84 + E84
  addendum: comparator rows two-box verified, digit-identical].
- Fences (state all of them):
  - **At 8-bit the advantage vanishes** — mlx 8-bit is essentially lossless
    (7.4 mnats / 96.18%); the qwen3.6 8→4-bit cliff is 10.5x [E84
    addendum]. The claim lives strictly at 2–3.5 bpw.
  - **Dense is an open question**: `[[PENDING: E95 dense]]` (Qwen3.8-27B,
    flat d4/K256, registered expectations in E95 — if it loses, the claim
    stays "MoE experts are the surface where data-free VQ wins").
  - The e4b "VQ beats 8-bit affine on embeddings" result is CONFOUNDED
    (fp32 dtype accident, E76) — explicitly NOT cited as evidence.
  - 35B prefill deficit vs affine (~0.5x even with u8view, E81) is a real
    cost and goes in limitations, not buried.
  - Fitter-vintage caveat: shipped rungs are old-fitter; refit rungs are
    the honest frontier [E80 RESOLVED, E91, E94 mechanism caveat].

**Claim 2 — Flat codebook allocation at the target width is the peak;
"cheap-shallow harvest" prices sizes between flat nodes at ~2x the
byte-efficiency of stepping down, with measured exchange rates.**
- Flat-is-peak, affine era (transfers): matched-byte shape sweep at
  141.42 GiB — flat 2.3982 vs ramp 2.5042 vs spike 2.7224; a 4-bit layer
  costs exactly two 3-bit promotions, so under a byte budget you never buy
  the expensive width while any layer sits at the floor — flat is provably
  the best available shape at its size [E29-era tail-ladder record,
  EXPERIMENTS ~L920–945; FINDINGS law 3].
- VQ era: the ladder is MONOTONE — no harvest rung beats the flat rung at
  or above its size [E79; E78 dose-response; E74].
- The counter-design (fund a body upgrade from shallow harvest) is closed
  by the cost ratio: shallow = 1.87 GiB/bit, body = 8.81 GiB/bit — **4.7:1**
  — harvesting the entire shallow region cannot thread a single body-width
  step ("the needle has no eye") [E74 addendum; STATE 08-21 morning
  directive — see ambiguity #1: the phrase itself is not in the E-record].
- Measured exchange rates (ppl/GiB, prose, 397B):
  - flat-ladder slope in the region: ~0.0365 (K128→K256), 0.0129
    (K256→K2048) [E79 addendum]
  - harvest off K128 base: 0.0315 (1st bit), 0.0238 (2nd) [E78]
  - harvest off K256 base: 0.0033 [E79 addendum]
  - harvest off K2048 base: **0.0011** (−3.72 GiB for +0.0042) [E91]
  → harvest cost falls ~10–30x as base richness rises; ~2x byte-efficiency
  vs stepping down the flat ladder [FINDINGS law 4].
- Size model (pricing tool): new = base − 1.87 GiB × shallow bits
  harvested; 5–6 out-of-sample hits, 1 in-band, 0 misses; every point
  stamped pre-/post-graft, vision tower = exactly 912,020,960 bytes
  [E74 add., E78, E80, E91, night logs 08-21; FINDINGS law 5. NOTE
  FINDINGS says "5 hits + 1 in-band through E93", STATE says "6 hits 1
  in-band" — reconcile before print, ambiguity #4].
- Fence: harvest is NEVER free and never a quality win at a flat rung's own
  size [E78, E79]; its product value is "name a size, get the best
  artifact at it in one ~40-min fit" [E79 addendum].

---

## Section-by-section

### Abstract (drafted in full — DRAFT.md)
Two claims + the methods thesis + the pending-slot honesty.

### 1. Introduction (drafted in full — DRAFT.md)
- The gap: no shipped VQ MoE artifacts on Apple Silicon; community ladder
  is affine (mlx-community uniform, spicyneuron mixed); 2–3.5 bpw is where
  128–192 GB unified-memory machines live.
- Why data-free matters: no calibration corpus → no corpus-fit risk, no
  "calibrated on wikitext, deployed on code" asymmetry; also the honest
  costs (prefill, kernels).
- Contributions list = the two claims + the size model + the
  instrumentation gates + shipped artifacts on HF.

### 2. Method
- **Struct skeleton**: struct6-tail3x3 base — 6-bit structure, 4-bit
  qkv/z, bf16 routers; experts are the VQ surface [state-of-record header;
  E29]. Vision tower kept bf16, grafted (912,020,960 bytes exactly).
- **Flat VQ experts**: per-tensor k-means in weight space, d=4 subvectors,
  codebook K per rung (K128/K256/K512/K2048); unseeded k-means →
  refit/abort gate; relerr bars are PER-GEOMETRY (healthy relerr scales
  with K: ~0.19 @ K2048-class, ~0.31 @ K256, ~0.46 @ K128) [FINDINGS
  law 8; night log 08-21 near-miss].
- **Packing**: pack to true bit-width AFTER fit; bit-exact (verified on
  real artifacts, KL identical to 3 decimals); byte-aligned packing is a
  pure loss — skip [E70 addendum; 08-18 packing record]. NEVER quote
  stored bytes as size [FINDINGS rule III.8, E87 correction].
- **Size model + harvest**: the two-coefficient model; harvest = hold the
  body, take shallow bits back (NOT reallocation — E74 addendum).
- **Geometry choices**: d4/K256 operational sweet spot (byte-aligned uint8
  codes, threadgroup codebook, ~40-min 397B fits) [law 8]; "raise K
  first" — d4 beats d2 by ~12% KL at matched 2.00 bpw (E87, scoped to one
  pair); d8 = `[[PENDING: E89 d8 verdict]]` with the pre-registered
  reading grid (wash ≈3.1706 / pays 3.10–3.15 / worse / memorization-check
  <3.05) quoted verbatim [E89 + amendments].
- **The pipeline and gates** (forward-reference §5): fit → outlier gate on
  trusted box → pack → graft → verify → check_release/check_bundle →
  referee both corpora, serial, pre-registered.

### 3. Results — the ladders
- **397B table** (one instrument, referee prose/code, packed + post-graft
  stamped): flat K128 100.9 / 3.1706 / 2.6988; flat K256 112.0 / 2.7655 /
  2.6383; flat K512 `[[PENDING: E93]]` (~122.3 predicted, bar =
  interpolation 2.7655↔2.3519 at its size); flat K2048 (shipped 3.1)
  143.7 / 2.3519 / 2.5987; **K2048-refit 143.65 / 2.3410 / 2.5963**
  (flagship); harvest rungs 97.2/3.2730, 99.05/3.2289, 107.9/2.7790,
  **139.93/2.3452**; refit-K256 `[[PENDING: E92]]`; d8-K16384
  `[[PENDING: E89 d8 verdict]]`. Anchors: spicyneuron 2.6bit ~121/3.183,
  3.5bit 165.6/2.3614. Chart = chart_397b_ladder.png (regenerate with E92/
  E93/E91 points before submission).
- **35B table** (kl_cache_qwen36, mnats/top-1, E84/E85-cleaned): 8-bit
  7.4/96.18; **d4-K8192 56.4/89.4** (beats 4-bit, smaller); 4-bit
  78.6/85.61; d4-K4096 68.5; d4-K2048 85.5; d4-K256 210.7/80.05;
  d2-K16 239.9/78.43. Refresh: `[[PENDING: E94]]` (fitter-vintage third
  measurement; E94 amendment tool-identity rule applies).
- Fitter-vintage effect: −0.0109 prose at K2048, geometry matched [E91];
  mechanism UNIDENTIFIED inside the fitter [E94 caveat — do not repeat
  the model card's "corrected k-means" causal phrasing, ambiguity #2].
- Speed rows: decode is a wash across geometries; prefill is where
  geometry shows; u8view +25–33% prefill, bit-exact [E70, E81, E90].

### 4. Negative results (own section, not an appendix)
- d2-vs-d4: d4 wins ~12% KL at matched 2.00 bpw — NOT the 3.3x a corrupt
  artifact briefly manufactured [E87 + correction; E82 void per E85].
- Harvest is never free: monotone cost at every base, no floor above zero
  [E78, E79]; what survives is the exchange rate.
- Geometry washes within d4: packaging washes across K at matched bytes
  [E84 d4 line; law 1 — scope: across-d wash is UNSETTLED/measured-non-wash
  at 2.0 bpw].
- Dense/e4b embedding "win" retracted as dtype accident [E76].
- Calibration attempts that lost: OptiQ calibrated loses to uniform on
  dense 27B [E40/E42]; per-layer sensitivity probes fail mechanistically
  on MoE [headline finding 1, E7]; DWQ at 397B/2-bit falsified [E20/E27/
  E28]; fused row-gather prefill lever does not exist [E52].
- `[[PENDING: E95 dense]]` — if negative, it is reported HERE as a
  measured negative, per the E95 registration.

### 5. Methods & instrumentation — the gates and what each caught
(drafted in full — DRAFT.md; the paper's distinctive section)
Thesis: every wrong number LOOKED plausible; only pre-registration + cheap
measurement + cross-review caught them. Incident → gate table:
- E79 proxy-score poisoning → comparison-row rule + check_comparator.py
- E82/E85 corrupt artifact, 25x-overstated effect → outlier-gate-before-
  scoring rule (III.9)
- E76 dtype accident masquerading as a quality win → measure-KL-before-
  advocating; fair-test discipline
- E91 "exactly additive" algebraic identity → struck by peer review;
  decomposition-by-construction phrasing
- Vision-tower units mismatch → pre/post-graft stamping (law 5)
- Stale scripts / lazy-load watchdog kills → check_scripts_sync.sh,
  cpu-stream + load-bearing eval [E70 addenda]
- Placeholder gate (check_card_placeholders.sh), cite-the-commit rule
  (III.7), known-bad + known-good gate acceptance (III.5, E77 false-alarm
  lesson), stored-bytes rule (III.8).
Epigraphs: the two quotes (see ambiguity #5 on the second's provenance).

### 6. Limitations
- Two MoE families (Qwen3.5-397B, Qwen3.6-35B) + gemma-4 side evidence;
  dense pending [E95]. Single vendor stack (MLX/Metal).
- Prefill deficit vs affine at 35B (~0.5x even after u8view) [E81].
- Kernel gaps: d8/big-K has no fast kernel (codebook exceeds 32 KB
  threadgroup) [E83]; packed-d2 fused kernel history [E62].
- litbench statistical power: n=104, ~1.3 SE incidents; e4b-8bit scored
  above its own bf16 teacher [E56 hold]; blind win-rate n small
  (p=0.044–0.058 range results).
- gemma ppl invalid (model property, HF-verified) — instrument
  substitution documented, not hidden [GEMMA4_PPL_ANOMALY, memory].
- Fitter-vintage mechanism unidentified [E94].

### 7. Reproducibility
- Artifacts on HF (TheDrainFlorist), revision pinning: predecessor builds
  preserved at pinned revisions, `__PREDECESSOR_REVISION__` substituted at
  upload [card G, upload checklist]; bundled model.py must match the
  runtime that produced the numbers [E81 finding 5, check_bundle].
- exo sharding: codebooks replicate, not slice; upstream PR #2268 stalled;
  fork noahzelezny/exo:vq-codebook-replicate [card G].
- Scripts in-repo: vq_397b_codes.py, pack_artifact.py, verify_artifact.py,
  kl_damage.py, referee/score_streaming.py, gates. Referee corpus
  disjointness note [~L1232].

---

## Ambiguities / contradictions found while reading (fresh-eyes pass)

1. **"Needle-has-no-eye argument, E-record 08-21 morning" — not in the
   record.** The phrase appears nowhere in EXPERIMENTS/STATE/FINDINGS. The
   nearest substrate is the E74 addendum, which says the 4.7:1 ratio
   "correctly prices a reallocation nobody has attempted" — i.e. it CLOSES
   the counter-design in principle but was recorded as answering the wrong
   question. If an 08-21 morning E-entry formalizing the closure exists, it
   has not been written to EXPERIMENTS.md yet. The draft cites E74 addendum
   and leaves a slot.
2. **MODEL_CARD_397B_G attributes the fitter effect to "a corrected k-means
   implementation (fixed 2026-08-18)". E94 explicitly registers that the
   mechanism is UNIDENTIFIED** (scatter-add is math-identical and cannot be
   it; candidates: sampling, iteration completion, init draws,
   refit-on-abort). The card's causal phrasing is ahead of the record; the
   paper must use E94's phrasing, and the card may want the same fix.
3. **spicyneuron 2.6bit: 3.1830 @ 120.3 G (E23/E24 record) vs 3.1843 @
   121.0 in chart_397b_ladder.py.** Also "121G blind" at ~L343. Probably a
   re-score vs historic score; per rule III.2 the paper must use the
   re-measured value and name the instrument/date — but I could not find
   where 3.1843 was measured. Trace before print.
4. **Size-model tally: FINDINGS law 5 says "5 hits + 1 in-band, 0 misses
   through E93"; STATE 08-21 header says "6 hits 1 in-band 0 misses".**
   E91's 143.65 (err −0.05) is the 6th; the two graft-growth confirmations
   (E92→~111.62, E93→~122.31) were still pending at last write. Pick one
   tally and date-stamp it.
5. **Second epigraph — "accurately describing the wrong outcome is not the
   same as catching it" — is not verbatim anywhere in the repo.** The
   closest recorded events: E57 amendment (mechanism prose described the
   losing arm), E70 addendum 5 ("the write side was never the write side"),
   and the 08-21 night close. Included in the draft as directed, but its
   source needs pinning (or it's from an unlogged conversation).
6. **Shipped 3.1 size appears as both 144.0 GiB (E74 addendum, E79
   addendum ladder) and 143.7 GiB (E91 table, card G).** Likely stored-vs-
   packed or pre/post-graft drift from before the stamping rule; per law 5
   every printed point needs the stamp.
7. Minor: STATE headline still says "exactly additive" in its title line
   while carrying the E91-correction NOTE below it — the paper must use
   only the corrected "decomposition by construction" phrasing.
8. E84's 35B ladder was scored with fresh-from-bf16 arms for E86/E87
   (non-expert tensors bf16, absolute KL not comparable to historic struct6
   rungs) — the 397B-style caveat must ride any merged 35B table (E86
   provenance note).
