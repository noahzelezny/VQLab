# The Flash geometry campaign — consolidated findings (2026-09-12 → 09-13)

One narrative for the F67–F85 arc. The findings log holds the per-number
record; this is the readable account and the paper's spine. Status: R1
gate-complete, R2 ladder in flight, NOTHING SHIPPED (Noah's call).

## The one-paragraph version

We tried to beat k-means three principled ways and lost three times —
activation-aware code re-selection improves every teacher-proximity
metric while *degrading* real-text ppl (invariant to teacher precision
and calibration source: it's the objective); full-Gram re-selection
overfits catastrophically; annealing has nothing to fix. The wins were
hiding elsewhere: the never-optimized max-abs **scale heuristic**
(alternation: +2.8% distortion, gate-safe) and, an order of magnitude
above everything, **per-layer geometry allocation** — Flash's depth-cost
curve is a true U (unlike GLM's and the 397B's), its shipped uniform d8
mass wastes bits in an early-mid trough while starving the tail, and
moving them (R1) beats the shipped 2.1bpw at identical size on every
gate instrument.

## Part 1 — the re-selection negative (F67–F78)

The idea: keep k-means centroids, re-pick code assignments under an
activation Gram from ~8k self-generated tokens. Held-out output-space
error improved +9–11% per module (F67–F70); whole-model teacher-KL
improved on both models, tails most (F71/F73/F75). Then the release
gate: **ppl worse on both models, every corpus** (F76). Four arms —
{self-gen, corpus} Grams × {affine-8bit, bf16} targets — regress
identically (~+0.45% 35B, +1.9% Flash-2.1): the damage is the
objective, not its ingredients (F77/F78).

Mechanism, stated once: minimizing E‖(W−Ŵ)x‖² under any activation
second moment spends the fixed code budget matching the teacher on
high-energy directions and pays in low-energy directions, where
real-text NLL partly lives. k-means' direction-isotropic MSE is
accidentally the better hedge. Output-space proxies invert against ppl
at 2–3.4 bpw exactly as weight-space proxies inverted in the 397B DWQ
arc. Corollaries measured en route: full-Gram ICM +56% train / −20%
held-out (F74, rank-starved Gram); the F72 "tail regression" was one
order statistic — quantiles, not the max, are the tail instrument (F73).

## Part 2 — what does improve the fit (F79–F81)

* Annealed k-means: NULL (−0.06% vs Lloyd; the landscape is benign).
* **Scale↔codebook alternation** (per-group least-squares scale
  s\*=⟨w,c⟩/⟨c,c⟩ alternated with s²-weighted Lloyd): +2.83% held-out
  distortion, 10/10 modules, and whole-model **gate-safe** (2 of 3
  corpora better, wikitext +0.18% sub-noise). The max-abs scale was the
  one never-optimized recipe component. Ships as data in existing slots.
* Incidental: refitting from bf16 targets alone beats the production
  fits (−0.73% corpus-B) — the shipped codebooks are not at the current
  pipeline floor.

## Part 3 — geometry is where the real money was (F82–F85)

Method rule that made this work: **no proxies in the loop.** Every probe
is a whole artifact scored end-to-end on the ppl instrument (isolation
probes measured exactly backwards on GLM — E10's own falsification).

Flash-2.1's shipped mix (6× d2-K256 layers 0–1 + 138× uniform
d8-K16384) was hand-set. Probes, all iso-instrument vs shipped:

| probe | wikitext | corpus-B | reading |
|---|---|---|---|
| R0: front d2→d8 (remove protection) | +3.21% | +3.40% | front protection REAL |
| demote 12–19 →K4096 | +0.13% | +1.19% | trough — nearly free |
| demote 32–39 →K4096 | +2.32% | +2.27% | late — expensive |
| promote 44–47 →d4-K4096 | **−2.47%** | **−1.23%** | tail was starved |

**Flash's depth-cost curve is a true U** — the third distinct family law
measured (GLM: front is ballast/tail-monotone, E10–E13b; 397B:
tail-graded, E24–E29; Flash: costly both ends, cheap trough). Depth laws
flip sign across families; per-family measurement is not optional.

**R1** (demote 12–19 + promote 46–47, ~iso-byte): wikitext **−1.08%**
(pre-registered −1.0), corpus-B flat, code −0.30%; benchmarks within
noise; generation smoke clean. First artifact to beat the shipped
2.1bpw at its own budget. **R2** (in flight): fund promotion of 44–47
with band 20–31 and/or deeper trough demotion (K1024); climb to the
knee, then ship once.

## Standing method rules this campaign earned

1. Ppl (and only ppl-class end-to-end instruments) graduates a fit
   change. Teacher-KL, output-space error, distortion: hypothesis
   generators, never gates.
2. Pre-register the bar before the probe (the annealing null cost 5
   minutes; the alternation pass was believable because its bar was
   named first). R1's −1.0% prediction hit within 0.08.
3. Depth/geometry laws are family-local. Three families, three shapes.
4. Diff-style candidates (unchanged modules keep shipped bytes) make
   whole-artifact probes cheap enough to replace proxies entirely.
5. Ops: geometry refits under HDD contention crash ~1/3 modules — the
   retry-supervisor + per-module-checkpoint pattern absorbs it.

## Assets

bf16 teachers archived: `/Volumes/Thunderbay HDD/Teacher Models/`
(retention policy supersedes delete-and-redownload). Builders:
`scratchpad/flash_geo_build.py` (GEOMAP-driven, pack_bits-aware),
`scalealt_whole.py` (ALT/DST/PARTS env). Instruments:
`scripts/score_ppl_resident.py` + corpora B/C, `flashbench_venv`
(exo's patched mlx_vlm + lm_eval 0.4.12 overlay), per-position KL
tooling (`tail_probe.py`). Artifacts (scratch, unshipped):
art_reselect*, art_scalealt, art_lloydctl, art_flash_{r0,r1,banda,
bandb,tailpromo,p1,p2}.
