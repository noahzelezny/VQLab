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
