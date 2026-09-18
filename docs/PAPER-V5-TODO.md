# Paper v5 — audit and TODO

Audit of *Data-Free Vector Quantization Beats Affine Quantization at Matched
Bytes Below 6 Bits* (Zenodo 10.5281/zenodo.22119017) against everything
measured through F125. Written 2026-09-18.

**Verdict up front: the thesis is intact and the evidence for it got
stronger. What has to change is the artifacts behind one family, the
instrument behind every number, and the scope of what the paper can claim
about metrics.** This is a rewrite because three independent layers moved
(artifacts, instrument, runtime), not because the argument failed.

---

## A. What survives — do NOT redo

1. **The central claim.** The one matched-byte pair re-measured on the
   current instrument came out *stronger*: 27B VQ-4.5bpw (14.45 GiB) against
   affine q4 (14.95 GiB, stock `mlx_lm.convert`, community recipe, vision
   tower grafted, `check-comparator` gated) is **0.50 GiB smaller and better
   on all three corpora — prose -19.4%, code -10.2%, literary -8.5% KL**,
   every one significant paired. The published claim was 12% on prose alone.
2. **§4.3, reconstruction error cannot steer design.** Strengthened hard:
   F118 measured the drift-map proxy at **Spearman -0.24** against KL over
   42 layers. It is not merely uninformative, it is anti-correlated.
3. **The measurement-discipline section (§5).** Every rule in it earned its
   place again this week. The comparator gate caught a dropped 333-tensor
   vision tower in a fresh q4 build on 2026-09-17 — that is a live example
   to cite, not a hypothetical.
4. **Excluding mixed geometry, and excluding Flash-Next.** Both hold up.
   Mixed geometry would expose the paper to a selection critique (allocation
   chosen by measuring on the same corpora it reports); uniform rungs are
   immune. Flash's PLE tables are ~26% of its bytes and F121 priced those
   bytes at roughly **zero mnats/GB at the margin**, so they do not behave
   like expert bytes under a matched-byte claim.

---

## B. Artifact defects — the only ones found

**B1. The 397B "flat" rungs are not flat.** §3.2 lists `flat d4/K256`,
`flat d4/K512`, `flat d4/K2048` (VQ-2.4 / 2.6 / 3.1). All three carry
**affine 3-bit, group 64 on the expert modules of layers 57, 58, 59** —
9 modules each, described in the paper as uniform VQ. The 2.2bpw is the only
clean 397B rung (60/60 VQ) and it is the *mixed-geometry* one the paper
deliberately excludes.

Direction of the error: those layers carry ~3.5 effective bpw against the
rung's 2.0-2.75 VQ body, so they are better-preserved than the nominal rate
— mildly flattering, on 3 of 60 layers. Size accounting is unaffected (bytes
are measured from the files).

*Repair, mostly free:* d4-K256 and d4-K2048 tails for layers 57/58/59 are
already in `vqlab-fits/qwen3.5-397b/demote_fit-d4/` — **2.4 and 3.1 are
zero-fit rebuilds**. 2.6 needs three d4-K512 fits (~1 h). Then re-gate and
re-smoke (397B needs the cluster path).

**B2. GLM has the same bug at the other end** (expert layers 0,1,2 on
affine) — out of scope for this paper, but it should be recorded so the
class is known and the audit is standing, not one-off.

**B3. No other family is contaminated.** Verified across all 20 rungs: the
27B (3/3), 35B (3.4/3.8/5.4) and Flash-5.5 uniform rungs are clean.

---

## C. Instrument — every number needs re-measuring

**C1. The paper's instrument is heterogeneous and none of it is current.**
Per §2.6: the 397B used streaming referee perplexity, prose + code, first
**8192** tokens; the 35B and 27B used KL to cached bf16 logits plus referee
ppl on the 27B at **2048**. So it is three conventions across three families
— defensible at the time, but no row is comparable to any row measured now.

**C2. The 397B referee ran the whole sequence with `cache=None`**
(`research/quantlab/referee/score_streaming.py:72`). On this architecture
(GatedDeltaNet on 3 layers in 4) the metric is chunk-dependent, and the
house standard since F111 is chunk 512. The old numbers are internally
consistent; they simply cannot be mixed with new ones. Not a correctness
scandal — a convention change that forces a full re-measure.

**C3. Re-measure everything on one instrument.** Three-corpus (prose / code
/ literary) KL to cached top-64 teacher logits at **12288** tokens, chunk
512, plus top-1 and ppl reported alongside. *This is the cheap part:*
teacher caches for all four families already exist and are permanent, so
each rung is a ~1-4 min scoring cell.

**C4. Rebuild the affine comparators** with the stock community recipe,
each gated by `check-comparator`, each carrying the same vision tower, each
scored on the same cache. The 397B ones need `stream-convert` (teacher is
751 GiB, exceeds RAM).

**C5. State the runtime on every row.** F120 measured that *where a weight
read is bound* moves Flash-3.2 prose KL by 1.6 mnats (156.7034 -> 155.1233).
Runtime is part of the instrument now, not a footnote.

---

## D. New results to fold in

**D1. §5 gains a real metric result (F125), and it is the most portable
thing here.** Not "ppl is junk" — measured, ppl ranks correctly in **9 of 12
family x corpus cells** and gets all three 397B corpora exactly right. The
sharper, defensible claim:

> ppl orders quantized models correctly only while every rung sits on the
> **same side of the teacher**. Damage can push ppl *below* bf16 on finite
> text; once it does, "lower ppl" and "closer to bf16" point opposite ways
> and the ranking inverts.

Predicts **8 of 9** cells with a recorded teacher ppl. The 35B code column
inverts perfectly (rho = -1.000, all four rungs under the teacher's 3.2428);
the 27B prose column inverts at 4/5 under. The practical point: **the failure
condition is one number almost nobody reports** — the teacher's own ppl on
the same corpus — so an inverted ladder is invisible from the page.
`stream_score` now stores `teacher_ppl` in every cache meta (commit 217d38b)
so the check is free forever.

This also *supersedes* the paper's existing hedge in §6 ("perplexity cannot
rank quantizations on the instruct family") — replace an observation with a
mechanism and a test.

**D2. Runtime/speed numbers are stale in the paper's favour.** §3.5 and the
cards are arc6-era (27B: 23.7 tok/s decode, ~103 prefill on M3 Ultra). The
v2 stack is +8.1% prefill / +12.2% decode bit-exact, more with bf16-I/O, and
VQ-vs-affine prefill parity moved 82% -> 90.5% (F60). **Caveat that splits
the table: `vq_dense.py` has none of the v2 work and predates the freeze, so
the 27B is still arc6 while the MoE families are v2.** Either port dense to
v2 first or report two runtime generations explicitly.

**D3. Byte accounting — int8 codebooks are NOT a prerequisite.** Checked:
codebook tables are **0.000%-0.050% of artifact bytes** (11 MB on a 45 GB
Flash-2.1; 0.3 MB on the 397B-2.4). F123's q8-entry result is real but it is
a *speed/cache-footprint* lever, and F123's stated payoff (d4-K2048 dropping
to the threadgroup arm) is contradicted by F124 measuring **device beating
threadgroup by 20.9% at that exact geometry**. Leave it out of the paper;
resolve it as a kernel thread.

**D4. Packing exactness is the real byte risk, and it is unaudited here.**
F96/F97 found Flash-2.1 wasting **1.69 GB to padding** — 150x anything
codebooks can move, and it lands directly on the matched-bytes axis.
`check-release` already knows how to test this. Audit all ten uniform rungs
before regenerating any size column.

---

## E. Open scope decisions (Noah)

1. **Does the 2.2bpw enter v5?** It is the only clean 397B rung but it is
   mixed-geometry. Including it means either widening scope or a stated
   exception.
2. **Dense v2 port before or after?** Affects whether §3.5 reports one
   runtime generation or two.
3. **Held-out corpus?** Not needed for uniform-geometry claims (nothing is
   selected from the corpora), but it would future-proof the paper against
   the allocation work arriving later. Note prose and code are ~58 KB each —
   at 12288 tokens a run consumes essentially the whole file, so there is no
   in-corpus holdout available; literary (1.2 MB) can be split.
4. **397B affine crossover above 3.5 bits** remains untestable (~225 GB for
   4-bit). §6 already discloses this; keep it.

---

## F. Order of work, with measured costs

| # | task | cost | blocks |
|---|---|---|---|
| 1 | Packing audit, 10 uniform rungs | minutes | size columns |
| 2 | Repair 397B 2.4 + 3.1 tails (zero-fit) | ~40 min | §3.2 |
| 3 | Fit 3x d4-K512, repair 2.6 | ~1 h + build | §3.2 |
| 4 | Re-gate + cluster-smoke the three 397B rungs | ~1 h | release claims |
| 5 | Rebuild affine comparators, all families, gated | ~2-3 h (397B needs stream-convert) | every claim-1 row |
| 6 | Re-measure every row, 3 corpora @ 12288 | ~1 h (caches exist) | all tables |
| 7 | Rewrite §2.6, §3.2, §3.5, §5, §6 | writing | — |

Compute is roughly one day. The rewrite is the long pole.

**Do not start at 5 or 6.** Measuring before the 397B artifacts are repaired
means measuring twice.
