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
- **Never measure acceptance on synthetic repeated text.** A prompt built by
  repeating filler drove acceptance to exactly 1.0 at 1k-16k tokens: the model
  continues the pattern and the head predicts it perfectly. Prefill timings
  from such a prompt are still valid (prefill does not care what the text
  says); every decode, acceptance and speedup number from it is worthless.
- **Do not use a palindromic run order.** `(a,b,c,d,d,c,b,a)` pins the middle
  configuration to both hottest slots — worse than no design. Use balanced
  blocks shuffled within each, plus a cooldown.

### Falsified predictions, recorded as falsified

| prediction | what happened |
|---|---|
| alignment worth +12.5pp (n=48, one prompt) | 1.53 sigma. Did not survive. |
| alignment "measured neutral / falsified" (n=256, one prompt) | ALSO wrong — no power to see a 5pp effect. Failure to reject is not evidence of absence. |
| the +5.9pp acceptance would convert to a large speedup | it converts to **+1.58%** wall-clock — which is near the depth-1 ceiling, not a defect. See §7. My first explanation (the extra head forward eats it) was only a third of the story. |

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

**The MTP prefill tax is MEASURED and small** (2026-08-31, 2.1bpw, e3q8 head,
time-to-first-token including one decode step):

| prompt | plain TTFT | MTP TTFT | tax | prefill tok/s (plain -> MTP) |
|--------|-----------|----------|-----|------------------------------|
| 256    | 2.36s  | 2.58s  | +0.22s (**9.2%**)  | 108 -> 99 |
| 1024   | 3.28s  | 3.78s  | +0.50s (**15.3%**) | 313 -> 271 |
| 4096   | 10.03s | 11.17s | +1.14s (**11.4%**) | 408 -> 367 |
| 16384  | 42.82s | 47.38s | +4.56s (**10.6%**) | 383 -> 346 |

Seeding the head over the prompt costs a flat ~10-15% of prefill and does NOT
grow super-linearly, so long-context prompts are not disproportionately
punished. Windowed seeding would recover it but is not urgent.

**The VQ prefill tax remains unquantified.** Decoding codebooks costs more per
token at prefill than an affine kernel. The obvious comparison —
Flash-Next-VQ-2.1bpw (46G) against the stock affine 4-bit (96G) — is
confounded by size, though it is decisive in one direction: if the 46G VQ
artifact prefills SLOWER than the 96G affine one, the kernel tax is
unambiguous.

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

## 6. The v2 plan: fund the head from better technique, not a worse trunk

The proposal is NOT to spend trunk quality on the head. It is to pull more
quality out of the same bits — better fitting, better per-layer allocation,
better codebook use — so the reclaimed GiB funds the head while output quality
goes UP, not down. That is the size-targeting thesis applied to a new budget
line, and it is the version worth doing.

Two asymmetries make it favourable:

- **Head bits are quality-free.** The trunk verifies every drafted token, so
  the head can be quantized as hard as acceptance tolerates. Measured: experts
  at 3-bit is 1.25 GiB against 2.14, indistinguishable in both acceptance
  (-0.2pp, t=0.54) and speed (0.5%).
- **The head is rung-independent.** It is grafted from the upstream bf16 MTP
  tensors, not derived from the trunk's quantization, so ONE head file serves
  every rung of a model. Technique improvements to the trunk compound across
  rungs; the head cost is paid once.

What this needs before it becomes a release plan:

1. A trunk recipe that is measurably BETTER at equal or smaller size —
  priced with `vqlab kl` and `vqlab score` against the current rungs, not
  assumed. `docs/ONBOARDING.md` and `layer-leverage` are the existing tools.
2. The head's true cost per family (measured, not projected).
3. **Both prefill taxes quantified** (see §5). A decode speedup that is paid
  for at prefill is a different product for a chat user than for an agent.

The failure mode to avoid is shipping a v2 whose headline is "now with MTP"
while quality quietly regressed to pay for it. The gates that prevent that
already exist and are cheap to run.

---

## 7. Why we are stuck near 1.6x, and what actually moves it

At depth 1 every step emits exactly two tokens whatever happens — the
committed `t1`, plus either the accepted draft or the trunk's own `t2`.
Acceptance therefore only controls how often a rejection costs a replay
forward. That caps what acceptance can buy:

    tokens per unit cost = 2 / (1 + (1 - alpha) + h)      h = head cost in
                                                          trunk-forward units

The measured +1.58% from a 2.7pp acceptance change fits this with h ~ 0.5:
the ceiling for that delta is +2.24% with a free head, and the head absorbs
about a third of it. **The depth-1 loop is close to its own ceiling.** Chasing
acceptance further, or making the head cheaper, buys single-digit percentages.

### The empirical depth-1 ceiling is 1.95x

Measured by accident and worth more than the run it came from: with a
repetitive synthetic prompt the head drafts perfectly, and at **acceptance
1.0 the depth-1 loop hits 1.95x** (1.93-1.95x at 1k/4k/16k). That is the
ceiling of the CURRENT design. We sit at 1.58x with acceptance 0.78, so
roughly **+0.37x is available from acceptance alone**, with no depth-k work.

It also falsifies the analytic head-cost estimate above: at alpha = 1.0 that
model predicts 2/(1+h), so 1.95x implies h ~ 0.03, not the ~0.5 inferred from
the alignment delta. The model assumed a speculative seq=2 forward costs the
same as a baseline seq=1 forward, and the ledger measured seq=2 as CHEAPER
(49ms vs 61ms). Trust the empirical ceiling, not the algebra.

### Depth is the lever — but only at high acceptance

Tokens per trunk-forward, modelled (head = 0.5 trunk-forwards each, geometric
acceptance along the chain):

| acceptance | depth 1 | depth 2 | depth 3 |
|-----------|---------|---------|---------|
| 0.70 | 1.13 | 1.09 | 1.01 |
| **0.78 (us)** | **1.19** | 1.19 | 1.15 |
| 0.85 | 1.23 | 1.29 | 1.27 |
| 0.92 | 1.28 | 1.38 | 1.42 |
| 0.97 | 1.31 | 1.46 | 1.53 |

**At our acceptance, going deeper is worthless** — each extra drafted token
costs a head forward and is probably rejected. Depth only pays above ~0.85.
This is the model, not a measurement, but the shape of it is robust: it is why
oMLX reports 2.33-2.62x with **96.8-97.9%** acceptance and we report 1.58x
with 78%. Their win is acceptance first, depth second.

### So the question is why their acceptance is 97% and ours is 78%

Hypotheses, cheapest first:

1. **Trunk quantization damage.** The head was trained against a bf16 trunk.
   Ours drafts from a 2.1bpw VQ trunk whose hidden states are damaged, and the
   head is asked to predict what that damaged trunk will do. If acceptance
   rises with trunk quality, this is the dominant term — and it makes the v2
   thesis (§6) compounding rather than merely affordable: better technique
   buys quality AND acceptance AND, above ~0.85, the option of depth.
   **Test: measure acceptance on the 2.1 / 3.2 / 4.4bpw rungs. Cheap, decisive.**
2. **Hidden-state variant.** MTPLX exposes `mtp_hidden_variant`
   (fc / pre_norm / post_norm / embedding / prev / mix) as a knob because it
   matters. We feed the hyper-connection mixer's input and have never swept
   the alternatives on this family. The dense-27B ablation put pre_norm 0.7285
   against post_norm 0.7188 — a near-tie there, but untested here.
3. **Their checkpoint differs.** oMLX's headline is on their own oQ4e build.
   Independent numbers on a third-party Flash-Next MTP checkpoint report
   58.3-89.5% acceptance, which brackets ours rather than theirs.

Note (1) is the interesting one and nobody has tested it.
