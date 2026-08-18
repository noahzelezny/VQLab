# Gemma-4 ppl anomaly — investigation log (2026-08-17)

**Status: RESOLVED — it is a MODEL PROPERTY, not an mlx bug.**

The independent referee (HF transformers 5.5.4, fp32 compute, on the
unquantized `mlx-community/gemma-4-e2b-it-bf16` weights, text-only extraction
loaded as `Gemma4ForCausalLM`) reproduces the inflation:

| probe | mlx_vlm e2b-6bit | HF transformers e2b-bf16-fp32 |
|---|---|---|
| plain English | 96.62 | 115.61 |
| Austen 300ch | 729.15 | 695.07 |

Same magnitudes from an implementation sharing zero code with mlx. The mlx
ports are EXONERATED. gemma-4-it genuinely assigns these probabilities:
RL/distillation sharpening has collapsed its raw-text distribution (own
greedy output ppl 1.42; everything external heavily deflated; ranking mostly
survives, which is why generation is fluent, the production sidecar
summarizes well, and hellaswag lands above chance but far below capability).

## Standing consequences (the new methodology facts)

1. **Raw-loglikelihood instruments are INVALID on gemma-4-it.** No wikitext
   ppl, no literary-corpus ppl, no raw-continuation MC (hellaswag-style,
   litbench-as-shipped) may be cited for this family, absolutely or
   cross-family. The scorer itself is fine — it is the instrument/model
   pairing that is broken.
2. **Qwen numbers are untouched** (sane ppl; streamed==direct parity).
3. **The ladder's quality gate is now `kl_damage.py`** — KL to the model's
   own bf16 output distribution. Sharpening is common-mode between teacher
   and student and cancels exactly, so it is immune to the pathology above.
   BUILT AND VALIDATED 2026-08-17 on gemma-4-e2b:

   | rung | size | mean KL (millinats/tok) | top-1 agreement |
   |---|---|---|---|
   | bf16 (self) | 5.2G | -0.002 (noise floor) | 100.00% |
   | q8 | 4.6G | 8.4 | 95.69% |
   | q4 | 2.5G | 635.8 | 65.98% |
   | q2 | 1.4G | 15437.1 | 0.28% |

   Monotonic across seven orders of magnitude, self-KL at zero, and it
   independently reproduces EXPERIMENTS.md headline 4 (dense collapses under
   extreme quant) from a different instrument. Top-k truncation is *helped*
   by the sharpening: captured_mass 0.969 at k=64, reported every run.
4. **The bf16 shootout as designed would be garbage** — gemma would lose
   litbench for instrument reasons. The shootout needs a chat-native
   instrument: score the SAME litbench items presented through each model's
   chat template with lettered options (A-D), scoring the letter token.
   Sharpened chat models are in-distribution there. Not yet built.

## The anomaly

Every gemma-4 checkpoint on disk, through every available mlx runtime,
assigns absurdly high perplexity to external text while generating fluent
text and ranking multiple-choice above chance:

| model | runtime | plain English | Austen 300ch | self-generated |
|---|---|---|---|---|
| 31b-it-8bit | mlx_lm 0.31.3 | 27.15 | 10,449 | **1.42** |
| 31b-it-8bit | mlx_vlm 0.5.0 | 27.15 (identical) | 10,449 (identical) | — |
| e2b-it-6bit (PRODUCTION sidecar) | mlx_vlm | 96.62 | 729 | — |
| 26b-a4b-it-4bit | mlx_lm | (hellaswag 46% acc_norm; a 26B should be ~75%) | | |

Healthy references on the same harness: Qwen3.6-35B-4bit hellaswag acc_norm
0.76, identical between the streamed and direct scorer paths.

## What is ruled out (each by direct experiment)

- **Scoring math** — self-generated text scores ppl 1.42 through the same
  code; the math is fine.
- **BOS handling** — was a real bug (encode() does not prepend BOS; without
  it gemma degenerates), fixed in `score_tasks_streaming.py`, moved gemma
  hellaswag 40.5→46.0. Not the main effect.
- **Sliding-window mask length effects** — broken at 64 tokens, far inside
  window 512.
- **Cache vs no-cache prefill** — bit-identical ppl.
- **Chat-format OOD** — scoring inside `<start_of_turn>user` framing makes
  it WORSE (27→180 plain, 10k→62k Austen).
- **Rare-token collapse** — the damage is uniform: median token nll 8.96;
  ordinary continuations (' departed' after 'her daughters then') get nll 22.
- **Order-blindness / broken rope** — shuffled words score 92k vs 27:
  the model uses order strongly.
- **Logit mis-scaling** — raw pre-softcap logits are healthy (max 36, std
  3.4). Softcap implementation is the standard tanh.
- **Two mlx ports as independent evidence** — mlx_lm and mlx_vlm agree to
  the DECIMAL; they share lineage and referee nothing.

## Also found on the way (separate, real)

1. **The gemma-4 E-series ships dead shared-KV tensors.** e2b/e4b quants
   carry k_proj/v_proj/k_norm for their KV-shared layers (e2b 140 = 20
   layers x 7, e4b 126 = 18 x 7) that mlx_lm 0.31.3 never builds, so most
   fail to load strict (e4b-6bit happens not to; 26b/31b/all-bf16 are
   clean). **They are provably dead**: mlx_vlm 0.5.0 builds them, mlx_lm
   drops them, and both score e2b-6bit at ppl 96.62 — identical to the
   decimal. So `--allow-unmatched` is safe for this family.

   CORRECTION: an earlier revision of this log, and of
   `score_tasks_streaming.py`, blamed these dropped tensors for gemma's bad
   benchmark scores. That was wrong. The cause is the sharpening above; the
   drop is output-neutral. The scorer still refuses by default because a
   LIVE unmatched tensor would degrade silently on some other checkpoint.
2. **The streamed scorer cannot run gemma4** (tuple returns, PLE threading,
   shared KV, alternating masks). `--direct` added: whole-model forward via
   upstream's own `__call__`, validated to the digit against the streamed
   path on Qwen, 2x faster. Use for anything that fits in RAM.
3. **BOS fix** benefits any BOS-requiring family, harmless for Qwen
   (verified 0.52/0.76 unchanged).

## The two live hypotheses

**A) Model property.** gemma-4-it is RL/distillation-sharpened until raw
loglikelihood is uninformative: its own outputs get ppl 1.42, everything
else is heavily deflated, ranking survives (fluent generation, above-chance
hellaswag, and the production sidecar summarizes well daily — consistent
with everything observed). If true: no code fix exists; raw-ppl and
raw-loglikelihood instruments are simply invalid for this family, and the
quant ladder's quality gate must be built on something else (chat-formatted
scoring, generation-based checks, or KL-to-bf16 divergence, which stays
valid regardless).

**B) Shared mlx-port bug.** Some subtlety wrong in the shared lineage
(candidates not yet individually falsified: proportional rope details,
RMSNormNoScale, v_norm, attn_output_gate, shared-KV offset threading,
per_expert_scale handling in the MoE block). If true: fixable, and all
gemma numbers regenerate afterward.

## The referee experiment (staged, waiting on download)

`gemma4_ppl_referee_hf.py` — HF transformers 5.5.4 (genuinely independent
implementation), CPU, fp32, on `mlx-community/gemma-4-e2b-it-bf16`
(unquantized, same repo family; quantization damage excluded by
construction). Same three probes.

- transformers gives sane ppl (~10-25 plain) → **B**: hunt the port diff.
- transformers reproduces the inflation → **A**: change instruments.

google/gemma-4-* originals are gated (no HF_TOKEN in the environment), so
the mlx-community bf16 conversion is the accessible referee weightset.

## Consequences while open

- The bf16 shootout result would be garbage today: gemma would lose litbench
  to Qwen for instrument reasons, not capability reasons.
- litbench numbers for gemma-4 (e4b 17.3%, 26b 21.2% — both ~chance) are
  VOID, tagged instrument-invalid, kept out of results_literary/.
- Qwen numbers are unaffected (streamed==direct parity, sane ppl).
