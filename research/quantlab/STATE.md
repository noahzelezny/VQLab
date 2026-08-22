# STATE (2026-08-22 ~13:25) — the publish landed; R3 is the open question

## WHERE THINGS STAND

**Headline: the 101 GiB swap is PUBLISHED and verified (E131).**
`TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw` now serves the d8/K16384 build.
v1 stays complete and reachable at `revision=4554635…` (08-19). Verified by
size diff (38/38, 0 mismatches), sha256 against the remote LFS oid on 3
shards, and the rendered card fetched back byte-identical with all 10 sections
intact. **The progress display reported healthy progress for 74 minutes while
committing nothing** — two residue files were moved out of the artifact after
the uploader registered them, and every commit batch containing them failed
wholesale. Full account and the two rules it earns: E131.

**Task benchmarks were NOT re-run** (Noah declined). The card's benchmark row
is relabelled `(v1 weights)`; v1 numbers are not presented as v2's.

## RUNNING NOW (strictly sequential — concurrency killed E120/E121)

    M3  80135  fit_dense_vq d2/K4096 (E128 run C)  L52 of 63, relerr 0.0205
    M3  81035  run_e130_rate_twin.sh               ARMED, waits on run C
    M4         e128-35b-d4K16384 (R2)              ~L19-L26 (peer session)
    M4         E129 vintage test                   QUEUED behind R2

## THE LADDER (E128)

    27B  R1 E124 d2/K256  13.596 GiB  KL 40.327   MET
         R2 E126 d2/K512  14.592 GiB  KL 33.095   MET
         R3 run C in flight — d2/K4096, 17.580 GiB packed (CONFIRMED vs 6e)
    35B  R1 e94b d4/K8192 14.838 GiB  KL 53.022   MET   (PACKED — see below)
         R2 e128 d4/K16384 15.783 GiB  KL unmeasured      (PACKED, fit clean)
         R3 NOT TO BE FITTED — run C closed R3; report the slope instead

**35B SIZE CORRECTIONS, 08-22, both MEASURED and both verified here from the
artifacts' own bytes:**
(1) **e94b's long-cited 17.651 GiB is an UNPACKED size** (no `pack_bits` in its
config; it was never packed) and was being compared against packed rungs — a
III.8 violation that this file previously repeated. PACKED it is **14.838
GiB**. On a like-for-like basis R1 14.838 < R2 15.783, which is what the
physics demands (14 bits vs 13). The mixed comparison made the BIGGER-codebook
rung look SMALLER; that was the tell.
(2) **The 35B non-MLP carry is 1.706 GiB MEASURED, not the 4.53 DERIVED** used
in HANDOFF/E128 — off by 2.82 GiB, so every 35B projection shifts DOWN by that.
Re-measured independently: codes 13.125 + vq_scales 0.938 + codebook 0.015 +
other 1.706 = 15.783, matching the file total exactly. **Measured ONCE, not
closed** — the 27B carry was wrong by 1.5 GiB until three builds agreed.

**R3 sizing is not the risk; quality is.** Run C's 17.580 GiB packed is
confirmed against the closed size model (6e: 11.953 codes + 0.498 + 5.129),
Assert the MEASURED packed size. **The unpacked figure is not a size and must
not be compared to a bar** (III.8) — stored bytes carry whole-byte padding.
Earlier text here cited run C's unpacked 21.564 GiB as headroom against the q8
bar; that was a III.8 violation and is withdrawn. The guard against a skipped
pack is asserting the packed number, not a margin computed from unpacked bytes.

**Registered expectation for run C: ~18 mnats against a 1.641 bar.** If it
lands there, say plainly that R3 is out of reach rather than burning the
weekend. The M4 session's recommendation, which I endorse: do NOT then fit a
35B R3 — report the measured slope honestly instead. "8-bit-class quality is
not reachable by this method at these rates, here is the slope" is a real
result, and the 35B has ZERO d2 points, so picking its R3 would rest entirely
on transferring the 27B slope across models — the exact move this project has
been burned by.

**Seed floor does NOT transfer to K4096.** 6f's floor (KL 2.085 mnats, ppl
0.0447) was measured at d2/K256, n=3. K4096 draws 16x more centroids from the
same sample budget; that is a different quantity and we have no data on it.
Moot for the R3 verdict (an 11x gap dwarfs any plausible floor), but it needs
its own n=3 before run C is ever compared to another K4096-class rung.
Report KL AND ppl regardless (6c).

## DOC GAP FOUND AND CLOSED

E130's pre-registration existed ONLY in `run_e130_rate_twin.sh` (d05d111),
never in EXPERIMENTS.md — despite HANDOFF §8 claiming "EXPERIMENTS.md through
E130". The paper session cites only committed EXPERIMENTS.md entries, so it
could not see E130 at all. Transcribed verbatim into EXPERIMENTS.md, marked as
a transcription, no branch altered.

## OPEN — NOAH'S CALL, NOT OURS

1. **K65536** — recommend NOT this weekend. The 40-60h estimate was taken
   under contention, and E130 may moot it. A Saturday start lands Monday at best.
2. **A second dense family** — genuinely blocked on Noah acquiring a model.
   Otherwise the paper states the single-model limitation.
3. **`chmod -R a-w` on scored artifacts** — must wait; run C and the M4 fit
   are both writing.
4. **`MODEL_CARD_397B_G.md`** (uncommitted, 3.1bpw flagship card, 35 added
   lines) attributes its improvement to "a later version of our k-means
   implementation" — **falsified by E121/E129**: running the actual 08-16
   fitter scored 2.8292, worst of four arms. Nothing live; must not ship as
   written.

## E129 CAVEAT THAT MUST BE IN THE PRE-REGISTRATION BEFORE THE NUMBER EXISTS

The M4's mlx stack was reinstalled Aug 17, AFTER the Aug 15-16 shipped fit. A
null result therefore excludes "the box as it is today", NOT "the box". H4 —
the mlx version the M4 carried on Aug 15 — remains unrecovered.

---

## SUPERSEDED — STATE (2026-08-21 ~18:35) — mechanism night

## WHERE THINGS STAND

**Tonight's headline: the 08-18 fitter mystery is SOLVED, and the obvious fix
for it FAILED.** Both are results; the second is a real negative, not a setback.

- **E107-E110: mechanism identified.** `kmeans++` seeding (commit 689e03c,
  labelled "robustness, NOT a quality lever") buys the bulk by selling the
  tail. Depth-structured: uniformly BETTER below L15, sells the tail on 18/24
  body tensors from L20, `down_proj` body 8/8. Explained by geometry (E110):
  shallow layers are heavy-tailed (excess kurtosis +1.25) so distance-
  proportional seeding lands ON the tail; body layers are SUB-Gaussian (-0.38)
  so it merely covers the bulk better. Body is 8.81 GiB/bit vs shallow's 1.87,
  so the body verdict is the artifact verdict — which is why K256 regressed
  while K2048/K8192 won. **Mean relerr across body tensors moves -0.00033: the
  gate is blind to all of it.**
- **E112: FALSIFIED.** Body-only magnitude-weighted k-means (p=4, from L20)
  scored **2.9945 / 2.6442** against the K256 refit's 2.8057 / 2.6447 at
  byte-identical 111.617 GiB. Wikitext +0.1888 WORSE — 4.7x the regression it
  was built to fix. Reading rule was pre-registered (d4f8da8) before any number
  existed; "code held steady" was the available spin and the pre-registration
  stopped it. **Kills "the bulk/tail trade is recoverable by reweighting the
  objective."** Strengthens laws 11-12: we engineered the exact error band
  E102 identified, succeeded in weight space, and made the model worse.
- **E113: packed d8 kernel WORKS.** Bit-identical to the unpacked d8 kernel on
  synthetic data, and byte-identical greedy text on the real artifact
  ('Paris.\nA. True\nB' both ways). Unlocks 3.0591/2.6728 @ ~101 GiB vs
  shipped 2.2's 3.1706.
- **E115: speed MEASURED — ~19% decode tax.** Clean-mode decode@2048 17.24 vs
  21.22 tok/s, ratio **0.812**, matched runtimes (md5-identical model.py),
  fan on the box, corroborated by adjacent-run pairings (0.817/0.837).
  **Confirms E83's device-memory-codebook warning** — d8-K16384's 1 MB
  codebook cannot sit in threadgroup memory. So the 101 GiB rung is a real
  quality win at a real speed cost, and which artifact belongs there is Noah's
  call. **Instrument caveat: decode at ~100 GiB is BIMODAL** — the same
  artifact ran 21.14 / 12.69 / 21.27 / 21.20. Swap (swapouts never moved off
  1,343,843), thermal, and storage path are all ruled out by measurement.
  Quote ratios from one session, never absolutes, never n=1.
- **E116: SMB write path CLEAN.** Five real shards, 16 GiB, md5 local -> share
  -> back, 0 mismatches. Exonerates the write path on this sample; does NOT
  exonerate the box and does NOT explain E95 (compute-time, proven from its own
  fit log). Kills "route M4 fits through local disk" — the last reason to pay
  that permanent tax is gone. The post-hoc outlier gate stays mandatory.
- **E114 (running, M3): dense 27B refit.** The first dense artifact was
  DEFECTIVE — L60 up_proj had codebook, codes AND scales all exactly zero
  against a healthy source. Fitter printed relerr 1.0000 and carried on
  because `fit_dense_vq.py` had NO abort. Fixed (129f66d). Refitting on M3 per
  standing policy (M4: 4 collapse incidents; M3: 0).

## OPEN DECISIONS — NOAH'S, NOT OURS

1. **[RESOLVED 08-21 19:00, Noah authorized the fetch]** Published model.py
   fetched from all three HF repos: 2.2/3.1 are md5-identical to our 717-line
   copies, 2.4 to our 1093-line copy. The "695-line local copy" does not
   exist anywhere on the volume — downloaders run exactly our code. Nothing
   needs correction. Noah leans toward pushing the tested 1093 runtime to the
   2.2/3.1 repos (it is a superset, one added dense path); needs a III.11
   smoke on each repo's artifact with the 1093 bundle first. NOT pushed.
1b. **The original HuggingFace fetch question.** M4's local copy of the published 2.2bpw has a
   695-line bundled `model.py`; the share copy has 717. Same weights, same
   index, same config. Fetching the published `model.py` from HF and md5-ing
   it against both would settle whether **downloaders are running different
   code than our benches** (E81 in its published form). Neither session pulled
   it — it reaches an external service and the answer could imply a
   published-artifact correction. Blocks nothing.
2. **Whether any published artifact needs re-splicing.** Downstream of (1).
   All three published bundles are older than HEAD, which is EXPECTED (they
   bundle the runtime they shipped with; line counts track publication dates).
   That is not a defect. The 695-vs-717 discrepancy is.

## NEXT WORK, IN ORDER

- **Gate + score the dense 27B refit** (M3, ~15 min after fit). The
  does-this-generalise-past-MoE question. Compare against the q2/q3/q4 ladder
  at its ACTUAL size (9.61 GiB, between q2's 7.83 and q3's 10.96). NOT
  size-matched to q4 — the readings are the ablation vs q4 and the ladder
  placement, never "VQ beats 4-bit".
- **Score E94** (35B-K8192 refresh, fit done 12:01, 17.7 GiB on the Thunderbay
  SSD, 120 tensors mean relerr 0.1323). Waiting on the M3.
- **Understand the bimodal decode instrument** before any speed number goes on
  any card. Every published card's speed table was produced by the method E115
  just showed to be unreliable. This is a hole in the instrument, not a defect
  in any artifact — but it blocks speed claims.
- **DONE, was pending here:** d8 speed A/B (E115) and the SMB round-trip
  (E116). "Route M4 fits through local disk" is dead — E116 removed the
  remaining reason.

## THE TWO FAILURE SHAPES (keep these separate)

- **Compute-time:** fit log SHOWS the bad value (E95 L60: 1.0000, printed).
  `--relerr-abort` catches these.
- **Write-time:** fit log is CLEAN and the disk bytes are not (gemma d2-K512).
  The abort is structurally blind. **Only the post-hoc outlier gate, on a
  different box, covers both.** Which one fires is diagnostic.

## DOCS

- `FINDINGS.md` — laws, through E110. Laws 11-13 are tonight's.
- `PROCESS.md` — NEW. Family-onboarding pass (profile geometry, sweep init,
  THEN fit), the new-fitter guard list, box policy, and the standing gates.
  This is the artifact Noah asked for toward shaping MoEMash practically.
- `EXPERIMENTS.md` — chronology through E118 (E117/E118 pre-registered, running).

## FIXES THAT LANDED TODAY AND NOW FIRE AUTOMATICALLY

vision_config copied by default (graft_vision); packed index `total_size`
recomputed from the packed shards (pack_artifact, was declaring the UNPACKED
size — 37-61% overstatements that made exo refuse to place the flagship); dense
code width chosen from K (uint16 at K256 doubled an artifact); packed d8
dispatch; row-width validation on the fused packed path; `--relerr-abort` on
the dense fitter; verify_artifact reads dense artifacts (vq_linear key, 2D
codes, qwen3_8_dense family).


## NIGHT OF 08-21 (M3 session, Noah asleep) — running ship

- **E95 CONCLUDED: dense VQ carries — not MoE-only.** 27B dense, flat
  d4/K256: 325.6 mnats / 76.5% top-1 / ppl 6.403 @ 9.7 GiB. Above the affine
  q2->q3 line at its size (predicts ~439 mnats). Gate PASS, III.11 smoke
  PASS. Paper session unblocked. Full entry + two runtime defects III.11
  caught (d!=2 dense dispatch; expert kernel exceeds threadgroup memory at
  27B shapes) in EXPERIMENTS E95 RESULT.
- **L60 ROOT-CAUSED — it was a DEFERRED READ, not write corruption.**
  (Superseding this file's earlier framing and E95's original paragraph,
  both corrected 7da7399.) `build_dense_vq.py` loaded the fit file lazily
  and left it unevaluated until save_safetensors forced it ~1300 tensors
  later, across mx.clear_cache() — the read was paid inside a GPU command
  buffer under memory pressure and returned zeros. FINDINGS IV.1 on an
  unchecked path. Independently corroborated: the M4 session hit the same
  disease in fit_dense_vq.py 65 s into E119 (L02 gate_proj, relerr exactly
  1.0000 — a zero RECONSTRUCTION against a correctly-read tensor).
  FIXED 013d2bb: cpu-stream load + mx.eval inside the block, all-zero
  assertion on read, read-back scan of the written shard.
  Two consequences: (1) the SMB framing is dead — this instance was a LOCAL
  SSD read on the M3, so the transport is incidental and the invariant is
  "any lazy read left pending across a memory-pressure boundary";
  (2) the "M4 fits are cursed / M3 is clean" box policy was never evidence
  about the fitter — the M3 had simply never been exposed on that path.
- **E117 running on M3** (K256 random-init, chain self-gates/packs/scores;
  logs say E115 — numbering collision, see EXPERIMENTS E117 note).
- **E118 armed** behind it via run_night_queue.sh (K512 random; logs say
  E116). Watcher launches only on a clean E117 DONE banner.
- **HF fetch RESOLVED** (Noah authorized): published model.py md5-identical
  to our copies on all three repos; no 695-line file exists; no correction
  needed. Pushing the 1093 runtime to 2.2/3.1 awaits a III.11 smoke per repo
  and Noah's go.


## NIGHT OF 08-21 -> 08-22: armed schedule and standing duties

**M3, strictly sequential (run_night_final2.sh -> run_night_ladders.sh):**
1. E121 (08-16 fitter, cdcdeab) — running. NO abort by design, so its log is
   the compute-side catcher.
2. COLLAPSE-SCAN GATE: `grep -c "relerr 1.0000"` on E121's log. If > 0 the
   chain does NOT gate, score or compare it, and that is REPORTED. A refit of
   that fitter is a re-roll, not a fix.
3. E120 accumulation probe (fixed, smoke-tested on a 1-expert case).
4. E124 — 27B dense d2/K256, 13.594 GiB projected, Noah's target band.
5. Ladder fills: every 27B d4 rung the M4 has finished (build -> gate ->
   pack_dense -> III.11 on the PACKED artifact -> score, logging what it
   skips), then the 35B second-family point via e94b under its OWN name.

**M4 (peer):** in-flight d4/K1024, then d2/K512 (14.590 GiB, inside the
relaxed band). Blocking and unrun: the d8 clean-venv smoke, which is the only
thing gating the 101 GiB swap. Holds till morning — Noah did not answer and an
unanswered question is not an approval.

**E124 criterion, relaxed by Noah 23:15 BEFORE any fit (29ddfb2):**
size <= 14.80 GiB, KL <= 36.7 mnats, ppl < 5.2055 by >= 0.02. Quote the commit
with the numbers. A 14.7 GiB marginal result gets read against THIS bar, never
a friendlier one invented afterward.

**Open cross-check:** the 5.129 GiB non-MLP carry was measured on a d4
artifact only. Both sessions report the four-way byte split (codes / codebook
/ vq_scales / other) from their d2 builds. If `other` != 5.129, BOTH size
projections move and no size goes to Noah until re-derived.

**STANDING DUTIES WHEN THE SCHEDULE COMPLETES (Noah, 08-21 23:30):**
1. Record every result in EXPERIMENTS/FINDINGS — results, not just runs, and
   the falsified branches too.
2. Then contact the PAPER SESSION and ask what further experiments would
   strengthen the paper. Do not idle the boxes; do not invent work either —
   ask the session that knows what the argument needs.

**Do not repeat tonight's three:** preflight_ram.py before any resident-memory
op; a small-case run before any launch; nothing concurrent.


## 08-21 23:45 — THE VINTAGE HUNT IS OVER, AND THE ANSWER IS PROVENANCE

The base artifact `struct6-tail3x3` was rewritten Aug 19 22:07-22:15. The
shipped 2.4 was built Aug 15-16. The bf16 source is untouched since Aug 8.
So every 397B refit since Aug 19 used a base that is NOT what the shipped
artifact was built from — E92, E93, E117, E118, E121, flatk2048-refit, all of
them. That is a uniform input change across every arm and it is outside the
k-means entirely.

E121 is therefore VOID as a vintage test, not a result about the fitter. It
held the fitter fixed and assumed the base was constant; the base was the
thing that moved.

**The original base does not survive** — not on the SSD, not on the HDD, never
published to HF. So the shipped 2.4bpw at 111.617 GiB is unreproducible by
construction. Say that plainly; do not run a fifth variant hunting it.

Three mechanism experiments (E117 seeding, E118 K-crossover, E120 accumulation
order) were all chasing a k-means explanation for what is now most likely an
input difference. They stand as measured per-tensor physics. They were aimed
at the wrong object.

**TWO silent in-place overwrites confirmed tonight** (E94's scored artifact,
this base), both caught by mtime checks rather than by any failure. Artifacts
have no write protection and no provenance stamp. TOMORROW, not tonight:
publish-time manifest (path -> mtime + size + first-shard hash) and
`chmod -R a-w` on scored artifacts.
