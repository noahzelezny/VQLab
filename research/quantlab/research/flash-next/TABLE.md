# Flash-Next ladder — the one table (newest numbers win; update in place)

2048 tok. prose=WikiText referee, code=public mlx corpus, literary=Gutenberg.
KL vs bf16 teacher top-64 cache (captured mass 0.9626 all rows). Sorted by size.

| artifact | bpw | GiB | prose | code | literary | KL mnats | top-1 | fits |
|---|---|---|---|---|---|---|---|---|
| VQ d8/K16384 + K256 PLE | ~2.0 | 43.7 | 6.0216 | 2.1018 | 9.1096 | 419.88 | 77.8% | 64GB (decision pending) |
| VQ d4/K2048 | ~3.1 | 66.5 | 5.2911 | 1.9384 | 7.8077 | 146.61 | 86.6% | 96GB, shipped-grade |
| q3 affine | 3.649 | 75 | 12.8502 | 3.0522 | 19.4794 | 1083.35 | 61.9% | 128GB |
| VQ d2/K256 | ~4.3 | 92.4 | 5.3825 | 1.9033 | 7.7112 | 59.04 | 91.9% | 128GB, shipped-grade |
| q4 affine | 4.649 | 96 | 6.4534 | 2.0638 | 9.0975 | 293.86 | 79.6% | 128GB |
| q5 affine | 5.649 | 116 | 5.2434 | 1.9528 | 7.8895 | 91.66 | 87.5% | — |
| VQ d2/K1024 | ~5.5 | 111.6 | 5.2449 | 1.8975 | 7.6358* | 34.14 | 94.1% | 128GB (tight), 192GB-class |
| q6 affine | 6.649 | 137 | 4.9155* | 1.9116 | 7.7097 | 52.76 | 91.6% | — |
| q8 affine | 8.649 | 178 | 5.1968 | 1.9138 | 7.6695 | 27.06 | 94.9% | — |
| bf16 teacher | 16 | 335 | 5.1662 | 1.9015 | 7.6643 | 0 | 100% | — |

*d2/K1024 literary reads below bf16 (7.6358 vs 7.6643) — same below-teacher
slice artifact class as q6 prose; KL is the ranking column.

*q6 prose is a below-teacher slice artifact (confirmed slice-specific on the
literary corpus); KL is the ranking column.

Reading: every VQ rung dominates every affine rung at-or-above its size on
KL and top-1. q3, the only affine inside VQ territory, is the worst row on
the board.
