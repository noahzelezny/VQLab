# MTP speculative decoding — findings per family

Working notes for the MTP arc. The README carries the user-facing summary;
this is where per-family evidence, falsified predictions and open questions
live, so the next family does not re-derive them.

Status 2026-08-31: one family shipped and measured (`qwen4_exp`), four
identified and unimplemented, one measurement instrument built.

---

## 1. What we depend on

A drafting head predicts token *t+2* from (trunk hidden state at *t*,
embedding of *t+1*). Each step verifies one speculative token inside a single
2-token trunk forward: accepted gives two tokens for one forward, rejected
rolls the caches back and replays.

The only mlx-lm contract used is `load() -> model` and
`model(tokens, cache=cache) -> logits`, plus one per-family capture point for
the pre-lm_head activation. Everything else is VQLab's decode loop
(`vqlab/mtp/`), so we are not forking a runtime.

**Head precision cannot affect output quality.** The trunk verifies every
drafted token, so a worse draft costs a rejection, never a different token.
This is the single most useful property of the whole technique: head
quantization is a pure speed/memory search with no quality gate to defend.
It is why `--expert-bits 3` is safe and why a trunk bit traded for a head bit
is NOT a symmetric trade — see §6.

---

## 2. Three rules that are settled by measurement

**(a) The RMSNorm gain convention differs per family, and getting it wrong
gives exactly 0.0 acceptance.** Both families studied store MTP norm gains as
a delta needing +1.0, and they resolve it in *opposite places*:

| family | how the +1.0 arrives | what the head must do |
|---|---|---|
| `qwen3_5` | conventional `nn.RMSNorm`; mlx-lm's trunk sanitize adds 1.0, but `mtp.*` loads OUTSIDE sanitize | **shift the stored gains** (measured 0.0000 -> 0.7285) |
| `qwen4_exp` | the arch's OWN zero-centered RMSNorm (`y = norm(x) * (1 + weight)`) adds it | **do not shift** — apply via `arch.RMSNorm` (0.0000 -> 0.6992) |

Pre-shifting a qwen4_exp head double-counts. Hand-rolling `n * w` for either
drops the +1.0. This one mistake was the entire original bug.

**(b) The head's cache offset IS its position signal.** `qwen4_exp`'s
attention takes rotary positions straight from `cache.offset`. Keep one head
row per *committed* token — seed over the prompt, advance two positions per
step after verification — and the offset is the true position by
construction. Worth **+5.9pp acceptance** (t=6.34, better on 12/12 prompts).
MTPLX documents the same invariant for this architecture.

**(c) Attention caches roll back by the offset DELTA, not a fixed count.**
Trimming a hardcoded 1 leaves a stale key while recurrent caches roll back 2,
and the streams drift silently.

---

## 3. Measurement methodology (learned the hard way)

**Acceptance is the reliable instrument; wall-clock is not.** Greedy decode is
deterministic, so acceptance reproduces to four decimals regardless of machine
state. Speedup on a laptop swung 1.135x-1.723x for the *same configuration*
until active cooling brought the baseline spread from 26% to 0.32%.

- **Quote the baseline spread next to any speedup**, or do not quote the
  speedup. It is the readout for whether the machine held still.
- **Repeats add nothing.** Re-running a greedy prompt reproduces its
  acceptance exactly. Independent PROMPTS are the replicates.
- **Steps within one generation are not independent trials.** They share a
  prefix; a binomial interval over one trajectory's N steps is far too narrow.
- **Pair across prompts.** Prompt difficulty dominates the spread; pairing
  removes it. `vqlab mtp-accept` does this.
- **Do not use a palindromic run order.** `(a,b,c,d,d,c,b,a)` pins the middle
  configuration to both hottest slots — worse than no design. Use balanced
  blocks shuffled within each, plus a cooldown.

### Falsified predictions, recorded as falsified

| prediction | what happened |
|---|---|
| alignment worth +12.5pp (n=48, one prompt) | 1.53 sigma. Did not survive. |
| alignment "measured neutral / falsified" (n=256, one prompt) | ALSO wrong — no power to see a 5pp effect. Failure to reject is not evidence of absence. |
| the +5.9pp acceptance would convert to a large speedup | it converts to **+1.58%** wall-clock. The committed scheme spends most of it on the extra head forward. |

---

## 4. Family status

| family | models | head in source? | status |
|---|---|---|---|
| `qwen4_exp` | Qwen3.8-Flash-Next | graft on disk | **DONE** — 0.817 acceptance, 1.25 GiB head, 1.58x |
| `qwen3_5_moe` | Qwen3.5-397B-A17B, Qwen3.6-35B-A3B | **YES**, 397B bf16 has 1553 tensors / 12.29 GiB | not implemented |
| `qwen3_5` | Qwen3.8-27B | needs bf16 re-download | not implemented; 0.7285 measured once on a dense 27B |
| `glm5_next` | GLM-5.3-Flash | needs bf16 re-download | not implemented; no mlx-lm class (mlx-vlm has one) |
| `deepseek` | — | we have no build | not a target: oMLX and MTPLX already serve it natively |

**Every VQ artifact we publish declares `mtp_num_hidden_layers: 1` (or
`num_nextn_predict_layers: 1`) and ships zero MTP tensors.** This is not
specific to us — Qwen's own MLX uploads and the GLM-5.3 MLX conversions also
carry none. MLX conversion strips `mtp.*` systematically, which is why oMLX
and MTPLX publish their own re-added checkpoints, and why `mtp-pack` /
`mtp-graft` exist.

### `qwen3_5_moe` head shape (397B, read from the index)

Structurally a different animal from `qwen4_exp`, so it needs its own head
module rather than a parameter:

- single fused `mtp.fc.weight` (qwen4_exp splits `fc_embedding`/`fc_hidden`)
- 1541 of 1553 tensors are `mtp.layers.0.mlp` — **unfused per-expert**, like
  `glm5_next`, not qwen4_exp's fused `[E, 2I, H]` stack
- conventional `input_layernorm` / `post_attention_layernorm` + `mtp.norm`,
  not hyper-connections
- but it DOES share `pre_fc_norm_embedding` / `pre_fc_norm_hidden` naming
- 512 experts, 10 active, hidden 4096 — the head is itself a large MoE layer,
  so it must ship quantized (the qwen4_exp bf16 head cost ~44ms/forward and
  ate the entire speedup until quantized)

---

## 5. Open questions

**Prefill cost is UNMEASURED.** The committed alignment seeds the head over
the whole prompt: a full MoE head forward over N positions. Every benchmark so
far used ~62-68 token prompts where this is noise. At 4k-32k it may not be,
and that is exactly what a server sees. Measure before shipping to anyone with
long prompts.

**The MTPLX strategy.** MTPLX decouples the rope offset from the cache offset
(passing `position_offset` explicitly) instead of making the offset true by
construction. That keeps ONE head forward per step with correct positions,
where our scheme needs two — plausibly capturing the +5.9pp acceptance at half
the head cost, and turning the +1.58% into something larger. It also enables
two things we currently cannot do:

- **windowed prompt seeding** (bounding the prefill cost above)
- **head-cache reset on long generations.** MTPLX measured an uncapped draft
  cache decaying 86 -> 25 tok/s within a single 34k-token request. Our head
  cache grows one row per committed token with nothing trimming it, so we
  inherit this bug and have simply never generated long enough to hit it.

Both are correctness-free by the verify contract — head state conditions
acceptance only.

**Serving with prompt-cache reuse.** Cross-request prefix reuse is disabled
because a reused trunk cache would mis-position the head. Pairing a head cache
with each cached prefix is the fix; the loop currently RAISES rather than
decoding at wrong positions.

**oMLX interop.** Tested 2026-08-31 against oMLX 0.6.4: a VQ artifact does not
load. Two causes were identified — mlx-vlm tensor namespace (`language_model.*`
vs our `model.*`), and `model_file` apparently not honoured on the mlx-vlm
path. **The second needs re-testing**: it was measured against a bundle
predating VQLab 40a2855, which made the bundle resolve its base arch from
either runtime.

---

## 6. The v2 question: is trading trunk bits for a head a good deal?

The proposal is to tighten the mixed-bit trunk recipe to reclaim the ~1.25 GiB
the head needs, and re-release each model with MTP.

The asymmetry that matters: **head bits are quality-free, trunk bits are not.**
A coarser head costs a rejection; a coarser trunk costs output quality. So this
trades measurable quality for measurable speed, and both sides are already
instrumented — `vqlab kl` and `vqlab score` price the trunk damage, and
`mtp-accept` plus a cooled `mtp-bench` price the gain.

Do not ship this on intuition. The specific number to produce is: KL-to-bf16
damage from reclaiming N GiB of trunk budget, against the decode speedup that
N GiB of head buys, at the same total residency.

Note also that the head is **rung-independent** — it is grafted from the
upstream bf16 MTP tensors, not derived from the trunk's quantization, so one
head file serves every rung of a model.
