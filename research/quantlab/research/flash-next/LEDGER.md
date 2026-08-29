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
