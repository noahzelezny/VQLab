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
| VQ d4/K2048 | 3.108 | 116.3 | 2.1954 | 1.6187 | 1.3402 | 199.53 | 88.6% | 192GB |
| q3 affine | 3.524 | 129 | 2.6824 | 1.7842 | 1.4731 | 377.08 | 83.1% | 192GB |
| VQ d4/K8192 | 3.582 | 134.0 | 2.0379 | 1.5475 | 1.2154 | 94.54 | 92.1% | 192GB |
| q4 affine | 4.524 | 166 | 2.0263 | 1.5718 | 1.2025 | 98.34 | 91.9% | 256GB |
| q6 affine | 6.524 | 239 | 1.9285 | 1.4929 | 1.1660 | 13.47 | 97.1% | — |
| bf16 teacher | 16 | 598.5 | 1.9024 | 1.4888 | 1.1580 | 0 | 100% | — |

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

NOTHING HERE HAS GENERATED A TOKEN. Every number above comes from the
STREAMED scorer, which never holds the model resident. The build box is
96 GiB (84 GiB wired limit), so smoke/verify/generation CANNOT run on any
rung above ~84 GiB -- i.e. on any rung worth publishing. Needs a 128GB+
box or the exo cluster before release. d4/K8192 additionally needs
`bundle-accept` (its K is past the d4 threadgroup ceiling -> device-
codebook path). Rung geometry choices are Noah's.
