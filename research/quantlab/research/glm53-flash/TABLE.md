# GLM-5.3-Flash ladder — the one table (newest numbers win; update in place)

2048 tok. prose=WikiText referee, code=public mlx corpus, literary=Gutenberg.
KL vs bf16 teacher top-64 cache (captured mass 0.9906 all rows). Sorted by size.

CONTAMINATION NOTE (see LEDGER 2026-08-30): this teacher has near-verbatim
memorized the public corpora (mean top-1 prob 0.857 on prose). Absolute ppl
is contamination-dominated — NEVER compare cross-family. KL-to-teacher is
the ranking column and is stricter here, not weaker.

| artifact | bpw | GiB | prose | code | literary | KL mnats | top-1 | fits |
|---|---|---|---|---|---|---|---|---|
| VQ d8/K16384 | 2.162 | 80.9 | 3.6339 | 1.9619 | 2.9562 | 692.25 | 74.8% | 128GB |
| VQ d4/K512 | 2.635 | 98.5 | 2.5743 | 1.7107 | 1.6166 | 348.82 | 84.0% | 128GB |
| **VQ mix best8 (SHIPPED 2.7bpw)** | 2.73 | 101.9 | **2.3978** | **1.6696** | **1.4843** | **291.46** | **86.2%** | 128GB |
| VQ d4/K2048 | 3.108 | 116.3 | 2.1954 | 1.6187 | 1.3402 | 199.53 | 88.6% | 192GB |
| q3 affine | 3.524 | 129 | 2.6824 | 1.7842 | 1.4731 | 377.08 | 83.1% | 192GB |
| VQ d4/K8192 | 3.582 | 134.0 | 2.0379 | 1.5475 | 1.2154 | 94.54 | 92.1% | 192GB |
| q4 affine | 4.524 | 166 | 2.0263 | 1.5718 | 1.2025 | 98.34 | 91.9% | 256GB |
| q6 affine | 6.524 | 239 | 1.9285 | 1.4929 | 1.1660 | 13.47 | 97.1% | — |
| bf16 teacher | 16 | 598.5 | 1.9024 | 1.4888 | 1.1580 | 0 | 100% | — |

SHIPPED ROW ADDED 2026-09-03, AND ITS INSTRUMENT OFFSET IS STATED. The
2.7bpw row above is the first full three-corpus score of the published
artifact; all six of its numbers were measured in ONE session on ONE
instrument (`python -m vqlab.stream_score --model <dir> --corpus <c>
--tokens 2048 [--kl-cache glm53_teacher_topk_prose]`, mlx-vlm 0.6.17, M3).
That scorer has no chunk/traversal/seed/dtype flags at all, so "matching
the instrument" reduces to matching corpus + token count, which is exact.

Two reconciliation facts, both measured rather than assumed:

1. THE REPACK WAS VALUE-NEUTRAL. The shipped artifact and the local build
   `glm53_vq_packed_mix_best8` score BIT-IDENTICAL on prose (2.397798 /
   291.4628 / 0.8618 / mass 0.9906, both). The rows=8 packed-d8 re-bundle
   changed the kernel and not the outputs. That is a release finding.

2. TODAY'S ENVIRONMENT SCORES ~0.1-0.2% BETTER THAN THE ROWS ABOVE, and
   the shipped row therefore carries a small favourable bias against them.
   Re-measured comparators, today vs published:

     d4/K512    prose 2.5722 vs 2.5743 | KL 347.17 vs 348.82 | code 1.7077 vs 1.7107
     d4/K2048   code  1.6168 vs 1.6187 | literary 1.3406 vs 1.3402

   Prose, KL and code drift consistently negative and small (-0.08% to
   -0.18%, KL -1.65 mnats) — above the ~0.04% cross-instrument floor but
   far below every margin this table argues from. The published prose-only
   numbers for the mix (2.4014 / 293.84 / 85.7%) sit in the same relation
   to today's 2.3978 / 291.46 / 86.2%. NO CLAIM IN THIS TABLE CHANGES SIGN
   OR ORDER under the offset.

   OPEN DISCREPANCY, flagged not smoothed: d4/K512 LITERARY does not
   reproduce. Today 1.6285 vs published 1.6166 — +0.74%, wrong-signed
   against every other cell, and ~18x the drift seen anywhere else, while
   d4/K2048 literary reproduces to +0.03%. One of the two 1.6166 or 1.6285
   is wrong. The narrative below ("at 98.5 GiB VQ LOSES to q3 on literary")
   survives either value, so nothing downstream is blocked, but the K512
   literary cell should be re-measured before it is cited on its own.

VQ d4/K2048 (116.3 GiB) BEATS q3 affine (129 GiB) ON EVERY AXIS: 12.7 GiB
smaller, 47% less KL damage, better on all three corpora, +5.5pt top-1.
Extrapolating the affine ladder down to 116 GiB puts affine ABOVE q3's 377,
so VQ delivers roughly half the damage at equal size.

VQ d8/K16384 (80.9 GiB) is a DEAD RUNG, kept as evidence only: 1.8x worse
than the affine row that already collapsed. ~2 bpw is past the cliff for
this family too.

MEMORIZATION DEFICIT CLOSES WITH BITS: literary is the corpus VQ handles
worst (it is the most memorized, teacher ppl 1.1580). At 98.5 GiB VQ LOSES
to q3 there (1.6166 vs 1.4731); at 116.3 GiB it WINS (1.3402). Whatever VQ
erases in near-verbatim recall is recoverable with more bits.

VQ d4/K8192 (134.0 GiB) MATCHES OR BEATS affine q4 (166 GiB) ON EVERY
AXIS: 32 GiB smaller, KL 94.54 vs 98.34, top-1 92.1% vs 91.9%, code 1.5475
vs 1.5718, literary 1.2154 vs 1.2025 (a hair behind), prose 2.0379 vs
2.0263 (a hair behind). Call it q4-equivalent quality at 81% of the size.
This is the rung that reaches a KNOWN-USABLE operating point, not just a
win over the collapsed row.

SUPERSEDED IN PART (2026-09-03): the shipped 2.7bpw row HAS generated
tokens — plain mlx-lm decode 19.7 tok/s and vqlab mtp-generate 19.99 tok/s
at acceptance 0.827 on the M4. The paragraph below still holds for every
OTHER rung above ~84 GiB.

NOTHING HERE HAS GENERATED A TOKEN. Every number above comes from the
STREAMED scorer, which never holds the model resident. The build box is
96 GiB (84 GiB wired limit), so smoke/verify/generation CANNOT run on any
rung above ~84 GiB -- i.e. on any rung worth publishing. Needs a 128GB+
box or the exo cluster before release. d4/K8192 additionally needs
`bundle-accept` (its K is past the d4 threadgroup ceiling -> device-
codebook path). Rung geometry choices are Noah's.
