# Model-family ledger — what exists, per family, WITH the numbers

Families as rows; per family the released rungs, config-verified
geometry, and every ppl / task-bench number that exists on disk, with
its harness and WHICH VERSION of the artifact it measured. Geometry
cells read from artifact configs 2026-09-13. Numbers copied from the
named result files — nothing reconstructed from memory.

## Version vocabulary (Noah, 2026-09-13 — one scheme, all families)

* **v1** — first-generation codebooks: uniform or hand-set geometry,
  weight-MSE k-means fits. What most shipped rungs still are.
* **v2** — **mixed codebooks**: per-layer geometry chosen by the measured
  sensitivity ladder (end-to-end probes, ppl-gated). GLM (no v1 existed)
  and the 397B small rung shipped as v2; Flash-2.1's v2 is BUILT and
  awaiting release.
* **v3** — reserved for gradient-tuned codebooks (through-the-model NLL
  loss; the one unexplored fit-side road after F78 killed proxy-objective
  re-selection).
* **arc6** — orthogonal: the 2026-09-09 runtime republish (all 20 repos).
  A runtime version, not a codebook generation.

Candidate artifacts carry no private letters — a rung's improved build is
"<rung> v2", not "R1". (Probe arms inside a campaign may be lettered in
the findings log; artifacts are not.)

## Summary matrix

| family | rungs (bpw) | mixed-geo status | ppl data | task-bench data | fit assets | pending |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 2.7/3.1/3.6 | **v2 mixed** (ladder) | **NONE — never benched** | **NONE — never benched** | full fit ladder on SSD | bench pass is the open gap |
| Qwen3.5-397B | 2.2/2.4/2.6/3.1 (+3bpw card G) | **ONLY 2.2 = v2 mixed**; 2.4/2.6/3.1 and card G are FLAT | card-G one-harness table (below) | **V1 artifacts only** (below) | in-artifact + HDD demote fits + bf16 | re-bench v2 artifacts; flagship swap (Noah) |
| Qwen3.6-35B | 3.4/3.8/4.6/5.4 | 4.6 mixed (unaudited); rest flat | 3.4 rung, this week (below) | 3.4 rung, this week (below) | in-artifact only + bf16 teacher | geometry ladder never run; alternation candidate (F81) |
| Qwen3.8-27B (dense) | 3.9/4.5/4.8 | flat (vq_linear schema) | none | none | unaudited | audit before touching |
| Qwen3.8-Flash-Next | 2.1/3.2/4.4/5.5 | hand-set front mixes, NOT ladder | 2.1 rung full grid (below) | **one-harness incl 8bit ref** (below) | full fit set on SSD + bf16 teacher | **Flash-2.1 v2 ship candidate**; other rungs' knees unmeasured |
| gemma-4-26b | 6.2 | flat | anomaly-caveated (GEMMA4_PPL_ANOMALY) | struct-experiment rows only (results_crush); shipped rung unbenched | unaudited | none open |
| gemma-4-e4b PLE | 1 repo | n/a | none | none | PLE fit family on SSD | none open |

---

## GLM-5.3-Flash — geometry (v2 mixed, ladder-derived)

2.7bpw: d4 K512×102 + K2048×24 · 3.1: d4 K512×51 + K2048×24 + K8192×51
· 3.6: d4 K8192×111 + K16384×15.
**No ppl or task benchmark has ever been run on any GLM VQ rung** (per
Noah, confirmed: nothing on disk). Fits survive (glm53_vq_fit d4
K512(+seedB)/K2048/K8192/K16384-partial, d8k16384 — SSD), and the
resident ppl instrument + streaming bench harness both apply directly.
This is the ledger's largest open gap.

## Qwen3.5-397B

Geometry: 2.2bpw = d8-K16384×151 + d4-K256×20 + d4-K2048×9 — **the only
v2 mixed rung in this family** · 2.4/2.6/3.1 = flat d4 K256/K512/K2048.

**Card G is NOT a v2 mixed artifact** (corrected 2026-09-13, Noah). Its
own card says it replaces VQ-3.1 at the "same size and same geometry —
143.682 GiB, flat d4/K2048 experts". G is a better FIT of the same flat
geometry (−0.46% wikitext), i.e. a v1-geometry refit, not mixed
codebooks. Do not cite it as evidence for the mixed-geometry process.

**ppl — card G one-harness table** (unmodified mlx-lm, prefix-8192 +
mixed-language code corpus; quantlab/MODEL_CARD_397B_G.md; artifact = the
FLAT-geometry 3bpw refit "G", not a mixed rung):

| | G (143.7 GiB, flat refit) | spicyneuron 3.5bit (165.6) | prior VQ-3.1 (143.7) |
|---|---|---|---|
| wikitext | **2.3410** | 2.3614 | 2.3519 |
| code | **2.5963** | 2.6005 | 2.5987 |

(Championship-ladder era rows, same wikitext family: tail30 2.3982 —
E24–E29 in EXPERIMENTS.md.)

**Task benches — V1 ARTIFACTS ONLY** (lm-eval, 1000 items, 0-shot;
AgenicAI/quantlab/results_tasks/; hs/piqa = acc_norm, wino = acc). The
repo names match current rungs but these scored the PRE-mixed builds:

| artifact (v1) | hellaswag | piqa | winogrande |
|---|---|---|---|
| VQ-2.2bpw | 0.861 | 0.841 | 0.787 |
| VQ-2.4bpw | 0.883 | 0.844 | 0.784 |
| VQ-3.1bpw | 0.903 | 0.840 | 0.780 |
| spicyneuron 2.6bit | 0.880 | 0.841 | 0.771 |
| spicyneuron 3.5bit | 0.904 | 0.846 | 0.767 |

No task rows exist for any v2 397B artifact.

## Qwen3.6-35B

Geometry: 3.4 flat d4-K2048 · 3.8 flat d4-K8192 · 4.6 = d4-K2048×30 +
d2-K512×90 (provenance unaudited) · 5.4 flat d2-K1024.

**3.4 rung, current artifact, this week's instruments**
(scripts/score_ppl_resident.py; benches F71 card protocol, 1000 items):

| | shipped 3.4 | notes |
|---|---|---|
| wikitext-12k ppl | 5.4101 | refit-from-bf16 control 5.4153; alternation 5.4248 (F81) |
| corpus-B ppl | 11.7484 | refit 11.6625; alternation 11.5961 |
| hellaswag / piqa / winogrande | 0.741 / 0.828 / 0.736 | re-selected arm 0.735/0.835/0.747 (F71; both within noise) |

3.8 / 4.6 / 5.4 rungs: no rows.

## Qwen3.8-Flash-Next

Geometry: 2.1 = d2-K256×6(front) + d8-K16384×138 · 3.2 = d2×18 +
d4-K2048×126 · 4.4 = d2-K1024×18 + d2-K256×126 · 5.5 = flat d2-K1024.
Hand-set front-protection mixes — R0 (F82) proved the front protection
CORRECT, but no rung is ladder-derived.

**2.1 rung ppl grid** (resident instrument, this campaign):

| arm | wikitext-12k | corpus-B | code |
|---|---|---|---|
| shipped 2.1 | 5.8327 | 8.3372 | 1.4106 |
| **v2 (measured knee, ship candidate)** | **5.7698 (−1.08%)** | 8.3394 | 1.4063 |
| probes: front-demote | +3.21% | +3.40% | +0.56% |
| trough 12-19→K4096 / late 32-39→K4096 | +0.13% / +2.32% | +1.19% / +2.27% | — |
| tail-promote 44-47 (non-iso) | −2.47% | −1.23% | — |

**Task benches — ONE-HARNESS TABLE incl the 8bit reference** (streamed
qwen4_exp harness b256, 1000 items; results_bench3/; F87 forbids mixing
harnesses on this family):

| | 8bit ref (178 GB) | shipped 2.1 (v1) | **2.1 v2** |
|---|---|---|---|
| hellaswag (acc_norm) | 0.792 | 0.747 | 0.738 |
| piqa (acc_norm) | 0.835 | 0.819 | 0.825 |
| winogrande | 0.719 | 0.743 | 0.744 |

3.2 / 4.4 / 5.5 rungs: no rows.

---

## Depth-law card (why mixes can't be copied across families)

| family | measured law | evidence |
|---|---|---|
| GLM-5.3 | front lobe = ballast; value in tail; attention protected everywhere | E10/E12/E13b (affine ladders) |
| 397B | tail-graded, knee ~tail30 | E24–E29 |
| Flash-Next | **true U**: front (0-1) and late (32+) expensive; trough 12-19 cheap, one K-step deep | F82–F86 (end-to-end VQ probes) |
| 35B / 27B / gemma | **unmeasured** | — |

## Codebook portability

Codebooks are per-module tensors inside each artifact — mix-and-match
within a family = copying module tensors (`scratchpad/flash_geo_build.py`,
GEOMAP-driven; set pack_bits). Nothing is unrecoverable since the bf16
teachers were archived (HDD `Teacher Models/`: 35B + Flash-Next; 397B
bf16 on the SSD): missing geometries refit at ~1-5 min/module.

## PPL corpus protocol (house standard — use these, not ad-hoc corpora)

`src/vqlab/referee/` ships THREE corpora and card numbers are quoted on
them: **prose** (`referee_corpus.txt`, wikitext), **code**
(`referee_corpus_code_public.txt`, mlx @ v0.30.0, public/Apache-2.0 so it
is citable), **literary** (`referee_corpus_literary.txt`).

CAVEAT ON THIS WEEK'S FLASH/35B NUMBERS: the prose column above IS the
house corpus (the scorer's default), but "corpus-B" and "corpus-C" were
ad-hoc substitutes I built (quantlab FINDINGS.md; an mlx_lm source dump)
— NOT the house code/literary corpora. They are internally consistent
(one instrument, all arms) so the DELTAS stand, but they are not
protocol-comparable to any card number. Re-run on the house three before
anything goes on a card.

## Bench-row comparability (F87)

qwen4_exp loglikelihoods shift 0.1-0.7 nats with batch composition IN
THE UPSTREAM FORWARD. Flash rows compare only within one harness + one
batching; the streamed harness (score_tasks_q4exp.py, b256) is
canonical and the only way to score the 178 GB 8bit ref.
