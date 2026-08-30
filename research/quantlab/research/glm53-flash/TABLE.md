# GLM-5.3-Flash ladder — the one table (newest numbers win; update in place)

2048 tok. prose=WikiText referee, code=public mlx corpus, literary=Gutenberg.
KL vs bf16 teacher top-64 cache (captured mass 0.9906 all rows). Sorted by size.

CONTAMINATION NOTE (see LEDGER 2026-08-30): this teacher has near-verbatim
memorized the public corpora (mean top-1 prob 0.857 on prose). Absolute ppl
is contamination-dominated — NEVER compare cross-family. KL-to-teacher is
the ranking column and is stricter here, not weaker.

| artifact | bpw | GiB | prose | code | literary | KL mnats | top-1 | fits |
|---|---|---|---|---|---|---|---|---|
| q3 affine | 3.524 | 129 | 2.6824 | 1.7842 | 1.4731 | 377.08 | 83.1% | 192GB |
| q4 affine | 4.524 | 166 | 2.0263 | 1.5718 | 1.2025 | 98.34 | 91.9% | 256GB |
| q6 affine | 6.524 | 239 | 1.9285 | 1.4929 | 1.1660 | 13.47 | 97.1% | — |
| bf16 teacher | 16 | 598.5 | 1.9024 | 1.4888 | 1.1580 | 0 | 100% | — |

VQ rungs: pending (struct base ready at 99 GiB, 2-bit expert markers +
8-bit protected). Rung geometry choices are Noah's.
