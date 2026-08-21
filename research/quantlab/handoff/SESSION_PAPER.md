# Session brief: the paper

Repo: /Users/noahzelezny/Documents/AgenicAI/quantlab   Author of record: Noah.
Work ONLY in paper/. Do not touch EXPERIMENTS.md, FINDINGS.md, STATE.md,
model cards, or scripts — another session owns those and is running live
experiments against them. Do not run anything on the GPU.

Start by reading: paper/OUTLINE.md and paper/DRAFT.md (already drafted:
abstract, intro, and the full instrumentation section), then FINDINGS.md
(settled laws + retractions — nearly an outline already), then the last
~800 lines of EXPERIMENTS.md (E73-E95, the dense week).

Framing, Noah's words: "narrow and thorough." Two claims only:
1. Data-free VQ beats calibrated affine at matched bytes in the 2-3.5 bpw
   MoE-expert regime. Scope it: at 8-bit the advantage vanishes; dense is
   an open question (E95 lands today).
2. Flat codebook allocation is the peak; harvest prices sizes between flat
   nodes at ~2x byte-efficiency. The 4.7:1 shallow:body ratio closes the
   counter-design.

The unusual feature is the honesty: retractions and instrument failures are
IN the paper (E79 proxy-score, E82/E85 corrupt artifact, E76 dtype
confound, E91 algebraic-identity strike, vision-tower units). Each produced
a gate. Thesis: every wrong number LOOKED plausible.

FIVE PENDING SLOTS land today — E89 (d8), E92 (K256 refit), E93 (K512
rung), E94 (35B refresh), E95 (dense 27B). Ask Noah for results; do not
guess. The other session will publish them into EXPERIMENTS.md.

The prior agent flagged 7 record inconsistencies — they are listed at the
bottom of paper/OUTLINE.md. Do not fix them in the source files; report
them to Noah so the lab session can.
