# Flash-Next (Qwen3.8-Flash-Next, qwen4_exp) — family arc ledger

Arc opened 2026-08-27. Prior entries for this arc live in paper/LEDGER.md
(2026-08-27 through 2026-08-28, commits 03194f53..96f83dc4); this file is
authoritative from here forward.

## 2026-08-28 — state at ledger creation; overnight ladder run

MODEL: 180B total / 10-of-512 active MoE + 51.2B ngram PLE (one bank, 128
shards, [~2.5M, 160]) + MTP head + vision tower. BF16 teacher and FP-side
facts in paper/LEDGER.md entries.

INSTRUMENTS (all VQLab, all public): prose = pinned WikiText; code =
public mlx corpus (canonical since fd5336c); stream_score (layer-streamed
ppl + KL-to-cached-teacher, validated to all printed decimals); teacher
top-64 cache at Exo Models/flashnext_teacher_topk_prose.

AFFINE LADDER (2048 tok): q3 3.649bpw/75GiB 12.8502 prose; q4 4.649/96
6.4534, KL 293.9; q5 5.649/116 5.2434, 91.7; q6 6.649/137 4.9155(-4.9%,
slice artifact, KL 52.8 ranks it truly); q8 8.649/178 5.1968, 27.1; bf16
5.1662 / 1.9015 code. Family is ~20x more quant-sensitive than the 27B at
matched rungs (best rung 27 mnats vs 1.25).

VQ RUNG 1 — d4/K2048, 66.5 GiB (~3.1 bpw), THE 96GB TIER. Experts mean
relerr 0.1875 (144/144, zero refits); PLE 0.1813 (128/128); PLE codes
row-packed 55 B/row. Scores: prose 5.2911 (+2.4%), code 1.9384 (+1.9%),
KL 146.6 mnats, top-1 86.6%. Beats q4 on every column at 30 GiB less;
within 0.9% of q5 at 49.5 GiB less. ALL SIX GATES PASS (incl. cross-box
verify on M4, worst tensor 0.1912; vision 333/333 grafted). 16.6 tok/s
decode resident on the M3, 71.8 GB peak.

RUNG SIZING RULE (Noah): build to the box, not the bpw — headroom ratio
~0.69 of RAM (66.5:96 ≈ 101:128). The d8/K16384 (~54 GiB) was cut mid-fit
as an awkward tier (partial fit kept, 27 tensors, resumable); geometry
lesson from its abort: relerr thresholds are GEOMETRY-DEPENDENT (d8
normal ~0.32-0.36 per the 397B's 0.3156 mean; 0.35 default is
d4-calibrated — documented in the fitter, e0a9a4a).

OVERNIGHT QUEUE (self-driving, in value order):
1. d2/K256 fits -> auto-assembly -> gates -> scores (~102 GiB, 128GB tier,
   the q4-slot quality rung; uint8 codes = no pack step)
2. d8/K4096 fits (~48 GiB, 64GB tier; PLE rows 20x12=30 B aligned)
3. d2/K1024 fits (~124 GiB, diminishing-returns tier; PLE 100 B/row,
   needs pack-ple at assembly) — killable if morning has better uses

GLM-5.3 readiness survey spun off as a separate session (task_378d8069);
its deliverable lands as GLM53_VQ_READINESS.md and belongs in
research/glm53-flash/ when it arrives.

## 2026-08-29 (night) — rung 2 assembled; a decode bug caught by the identity gate; table cells filled

RUNG 2 (d2/K256): fits mean relerr experts 0.0837 / PLE ~0.0795 (best
geometry yet). Assembled 115.4 GiB -> PLE pack to 8-bit rows (80 B/row,
round-trip-verified) -> 92.4 GiB (~4.3 bpw, 0.72 of 128 GB — on the
sizing rule). Pre-pack scores (M3 streamer): prose 5.3825 (+4.2%), code
1.9033 (+0.1%), KL 59.04, top-1 91.85% — q6-class fidelity 45 GiB under
q6. Cross-box verify (M4): 144/144 from artifact bytes, means
0.0834-0.0840. Gates: check-release/check-bundle PASS; smoke PASS on M4.

DECODE BUG, caught by the packed-path identity check: post-pack streamed
scores read NaN — VQPLEEmbedding's unpack HARDCODED 11-bit strides, so
8-bit K256 rows were read misaligned (indices past the codebook; junk
gather). SMOKE PASSED on the same broken decode — the sharpest proof yet
that generation is a weak gate and the identity re-score is not optional.
Codes on disk always correct (pack-time round trip). Fixed: stride+mask
derived from codebook size (VQLab commit "unpack stride from codebook
size"); bundle regenerated; re-score running — rung 2 is QUARANTINED
until the packed path reproduces prose 5.382537 exactly.

LITERARY COLUMN (M4 sweep, 2048 tok) + q3 KL — table now dense:
  q3 19.4794 | q4 9.0975 | q5 7.8895 | q6 7.7097 | q8 7.6695
  teacher 7.6643 | VQ d4/K2048 7.8077 (+1.9%)
  q3 prose KL: 1083.35 mnats, top-1 61.9% (completes the KL column;
  affine at 3.65 bpw is not merely worse, it is a different model)

## 2026-08-29 (night, cont.) — rung 2 QUARANTINE LIFTED: packed path exact

Fixed decode reproduces every printed digit: prose 5.382537, KL 59.0428,
top-1 0.9185, code 1.903324 — bit-equivalent to pre-pack. Literary
7.7112 (+0.6% vs teacher; q6-level). RUNG 2 IS DONE: 92.4 GiB, ~4.3 bpw,
all gates, three corpora, KL — the 128 GB-tier quality rung.

Two shipped rungs now:
  d4/K2048  66.5 GiB  KL 146.6  top-1 86.6%  (96 GB tier)
  d2/K256   92.4 GiB  KL  59.0  top-1 91.9%  (128 GB tier)

## 2026-08-29 — rung 3 (d8/K4096) assembled and scored: the 64 GB floor, decision pending

43.8 GiB (~2.0 bpw whole; PLE rows packed to 30 B). All gates PASS incl.
smoke; splice succeeded after splice_ple was pinned to CPU (4th watchdog
instance; first M4 attempt corrupted the packed dir mid-splice — rebuilt
clean from intact fit dirs). d8 down_proj (NSUB=80) rides UNPACKED
pending the padded-tail GPU acceptance (below).

Scores (2048 tok): prose 7.0390 (+36%), code 2.2605 (+19%), literary
10.5480 (+38%), KL 556.10 mnats, top-1 74.3%. Reading: keeps the VQ
quality-per-byte line above affine (half of q3's 1083 mnats at 58% of
its size) but is a FLOOR rung in absolutes — worse than q4 at less than
half q4's bytes. Precedent for shipping honest floors: 397B VQ-2.2bpw
(~+31% prose) is the family's most-downloaded artifact. Ship / mixed-
allocation retry / hold — Noah's call.

Also this morning (walk window): padded-tail pack format landed (VQLab
e4d5a6a + 1992ff6) — unaligned NSUB packs via zero-padded tail block,
ceil-WPR in all 7 packed kernels, aligned output byte-identical to HEAD
(verified), NSUB=80/12-bit round-trip exact; gated behind
--pack-unaligned until bundle_accept's new unaligned cases pass on
metal. GLM arc (peer session): mlx-vlm venv staged, scorer line-verified
against INSTALLED code (3 fixes), glm5_next confirmed in the pip
release; GPU-window checklist agreed (selftest, bundle_accept, rule-5).

## 2026-08-29 — 64GB-rung lever: diagnostic hybrid queued, prediction registered

Plan (Noah-approved): (1) DIAG hybrid — d8/K4096 experts + d4/K2048 PLE
(~57 GiB, instrumental only, never ships) — one streamed KL isolates
whether experts or PLE tables drive rung 3's 556 mnats. (2) Apply the
lever where diagnosis points: experts -> tail-weight-pow p~2 with
--tail-weight-from ~11 (E102's scarce-centroid regime is exactly
d8/K4096; E106/E110 shallow-layer guard); PLE -> weighted PLE fit or
byte reallocation.

PREDICTION, registered before the run (Noah): the experts, not the PLE,
carry the damage — the hybrid's KL stays high (~500). If instead KL
collapses toward ~200, the tables were the bleeder and the hunch is
wrong. Arithmetic note either way: expert relerr 0.4156 vs PLE 0.4094 at
this geometry are nearly equal, so relerr alone cannot adjudicate —
which is the point of measuring at the output.

## 2026-08-29 — DIAGNOSTIC VERDICT: experts carry the damage (prediction confirmed)

Hybrid (rough d8/K4096 experts + good d4/K2048 PLE, 58.7 GiB,
instrumental): KL 544.28 vs rung 3's 556.10, top-1 75.1% vs 74.3%,
prose 6.9739 vs 7.0390. A 2.3x PLE reconstruction improvement moved
~2% of the divergence: THE EXPERTS ARE THE BLEEDER. Noah's registered
prediction confirmed. Corollary worth bits: PLE tables are nearly
damage-free at 2.0 bpw — low rungs can push them cheaper and spend on
experts. Also: equal relerr (0.416 vs 0.409) produced wildly unequal
output damage — §4.3's law again, measured at component level.

Lever fires: d8/K4096 expert refit with --tail-weight-pow 2
--tail-weight-from 11 (E102 scarce-centroid regime; E106/E110 shallow
guard). Same bytes; also repacking down_proj via --pack-unaligned
(criterion met this morning) -> target ~41.3 GiB.

## 2026-08-29 ~10:45 — compaction anchor: what is in flight

M3: weighted expert refit d8/K4096 (--tail-weight-pow 2 --tail-weight-from
11) -> Exo Models/qwen4exp_vq_fit_d8k4096_tw, ~2h. EXPECTATION SET: its
relerr will read WORSE than 0.4156 by design (weighted objective trades
mean for tail); judge by KL only. On completion: assemble via the standard
chain, pack experts WITH --pack-unaligned (criterion met), PLE from the
existing d8/K4096 PLE fit (diagnostic proved PLE near-damage-free), target
~41.3 GiB, score, compare KL vs 556.
M4: d2/K1024 assembly+scores -> fills TABLE.md's last row (~124 GiB rung,
no smoke possible on owned boxes — 192GB-class artifact).
Then: 64 GB ship/hold call (Noah's), release block (cards/HF/VQLab tag),
DIAG hybrid dir is deletable after the refit scores land.
GLM: fully validated stack, ready for teacher pass + affine ladder + fits
whenever GPU frees; peer session had VQLab commit permission denials —
surfaced to Noah, unresolved.

## 2026-08-29 — d2/K1024 rung lands (M4)

Assembled + scored clean on the M4: packed 144 expert tensors 174.6 ->
132.4 GiB, PLE row-packed at row_bytes=100 -> 111.6 GiB final (better
than the ~124 estimate). check-bundle PASS; verify + smoke still owed
(192GB-class — no owned box can smoke it; same caveat as recorded for
the class).

Scores: prose 5.2449 / code 1.8975 / literary 7.6358 / KL 34.14 mnats /
top-1 94.1%. Sits between q6 (52.76 @ 137 GiB) and q8 (27.06 @ 178 GiB):
beats q6 on every column at 25 GiB less, and gets within 1.26x of q8's KL
at 66 GiB less. Literary reads below bf16 — slice artifact, noted in
TABLE.md. Ladder table now complete; only the weighted d8 refit remains
in flight.

## 2026-08-29 — weighted refit scored: THE LEVER FAILS AT THIS RUNG

d8/K4096 tail-weighted refit (--tail-weight-pow 2 --tail-weight-from 11),
assembled with --pack-unaligned (first production use), smoke PASS:

  prose 7.2576 (was 7.0390)   KL 581.30 (was 556.10)   top-1 73.3% (74.3%)
  code  2.2162 (was 2.2605)   literary 10.9853 (was 10.5480)   43.7 GiB

Verdict: WORSE on KL, top-1, prose, literary; only code improved slightly.
The E102 scarce-centroid lever does not transfer to this family/geometry —
at d8/K4096 on Flash-Next experts, tail emphasis buys the tail less than
it costs the body. Negative result recorded; the BASELINE d8/K4096
(43.8 GiB, KL 556.10) remains the 64GB-tier candidate. Refit fit dir
(90.2 GiB) + tw artifact (44 GiB) + DIAG hybrid (58.7 GiB) now cleanup
candidates — Noah's call.

Silver lining: this run took --pack-unaligned through a full model load
for the first time and caught two stale floor-WPR defects the GPU kernel
acceptance could not see (bundle shim allocation; lossy WPR->NSUB in
input_dims/from_weights). Fixed in VQLab ad6918c, smoke-verified;
value-identical for all aligned artifacts. It also established that the
original d8/K4096 pack had left down_proj UNPACKED (aligned packer
skipped NSUB=80); an unaligned repack of the baseline would shave ~0.1
GiB — not material.

## 2026-08-29 — reallocation play launched: d8/K16384 experts + d8/K256 PLE

Noah: "the best possible configuration for a 64gb tier model... might be
the thing to do." The diagnostic's corollary, applied: move bytes FROM the
near-damage-free PLE TO the experts. tw artifacts deleted (lever failed).

M3: resume d8/K16384 expert fit (27/144 tensors already on disk from the
mis-aborted run; --relerr-abort 0.45, d8-calibrated). M4: fresh PLE fit
at d8/K256 (16x smaller K than the 12-bit fit -> fast; geometry otherwise
matches the K4096 manifest: group 32, iters 12, seed 1234).

Target: ~45.5 GiB (experts 12->14 bits, PLE 12->8 bits). Bet: KL
meaningfully below the baseline's 556.10. Assembly will use
--pack-unaligned end-to-end (load path fixed in VQLab ad6918c).
