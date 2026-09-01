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
| the 397B's MTP norms use a different convention from its trunk (their gains sit outside the trunk's range, by inconsistent offsets) | wrong. The offset is a uniform +1.0; a different layer simply has different learned magnitudes. Comparing magnitudes across layers was never evidence about a convention. See §8. |
| head-cache growth would decay throughput inside a long request (the bug MTPLX documents) | not reproduced: over 8192 tokens peak memory moved 47.34 -> 47.60 GiB and throughput was flat within noise. Caveat below. |
| the +5.9pp acceptance would convert to a large speedup | it converts to **+1.58%** wall-clock — which is near the depth-1 ceiling, not a defect. See §7. My first explanation (the extra head forward eats it) was only a third of the story. |

---

## 4. Family status

| family | models | head in source? | status |
|---|---|---|---|
| `qwen4_exp` | Qwen3.8-Flash-Next | graft on disk | **DONE** — 0.817 acceptance, 1.25 GiB head, 1.58x |
| `qwen3_5_moe` | Qwen3.5-397B-A17B, Qwen3.6-35B-A3B | **YES**, 397B bf16 has 1553 tensors / 12.29 GiB | **head works** — 0.9023 acceptance, our best. But **1.000x speedup**: see below |
| `qwen3_5` | Qwen3.8-27B | **YES**, 15 tensors / 0.79 GiB | **DONE** — 0.7399 pooled acceptance (12 prompts, real loop), 0.49 GiB head |
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

**The MTP prefill tax is small — but the first measurement of it was wrong.**

A first attempt compared mlx-lm's `stream_generate` against ours and reported
10-15%, attributing all of it to the head. That was two errors in one number.
It charged to MTP (a) a difference between two prefill implementations, and
(b) an outright bug in ours: the loop did `mx.eval(model(chunk, cache=cache))`,
forcing the full `lm_head` projection for every prefill position and throwing
it away. MLX is lazy; mlx-lm evaluates `[c.state for c in cache]` precisely to
avoid that. Fixed in b8e2430.

The correct control is our OWN loop with seeding on and off, head resident in
every condition (2.1bpw, e3q8, TTFT = prefill + first token):

| prompt | mlx-lm | ours, no seed | ours, seeded | tax_seed | tax_loop |
|--------|--------|---------------|--------------|----------|----------|
| 256    | 2.26s  | 2.40s  | 2.49s  | 0.10s (**4.2%**) | 0.14s (6.1%) |
| 1024   | 2.84s  | 3.15s  | 3.20s  | 0.04s (**1.3%**) | 0.32s (11.2%) |
| 4096   | 9.73s  | 10.63s | 11.12s | 0.49s (**4.6%**) | 0.91s (9.3%) |
| 16384  | 42.95s | 45.31s | 47.63s | 2.32s (**5.1%**) | 2.36s (5.5%) |

**Seeding the head over the prompt costs 1.3-5.1% of TTFT.** That is the
actual MTP prefill tax and it does not grow super-linearly, so windowed
seeding would be a small optimisation rather than a fix.

`tax_loop` (5.5-11.2%) is our prefill path against mlx-lm's, and its cause is
**NOT ISOLATED**. It was predicted to be a fixed constant — two extra forwards
before the first token, since our first speculative step draws a draft and
runs a 2-token verify where mlx-lm runs one 1-token forward — but it grows
with prompt length (0.14 -> 0.32 -> 0.91 -> 2.36s), so that explanation is
insufficient. What is established is that it is not the head: seeding is
controlled separately above. Total MTP prefill overhead against stock mlx-lm
is 7-15%, of which seeding is the smaller half.

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

### Trunk quantization does NOT affect acceptance (tested, negative)

The obvious hypothesis was that our 0.78 is capped by drafting from a damaged
VQ trunk — the head was trained against bf16, and is being asked to predict
what a 2.1bpw trunk will do. **Measured on three rungs, 12 paired prompts
each, and it is false:**

| rung | pooled acceptance | within-rung sd |
|------|------------------|----------------|
| 2.1bpw | 0.8151 | 0.076 |
| 3.2bpw | 0.7823 | 0.089 |
| 4.4bpw | 0.8057 | 0.044 |

| paired | delta | t | verdict |
|--------|-------|---|---------|
| 2.1 - 3.2 | +3.01pp | 1.50 | not significant |
| 2.1 - 4.4 | +0.74pp | 0.34 | not significant |
| 3.2 - 4.4 | -2.27pp | -0.97 | not significant |

Not monotonic, nothing significant, and the effect is not even ordered by
trunk quality. **Prompt-to-prompt spread within one rung dwarfs anything
between rungs.**

Two consequences, one of them load-bearing for §6:

- **Trunk improvements and MTP are independent.** Better technique buys
  quality without giving back speedup, and without buying extra acceptance
  either. They simply do not interact. (This also retires the worry that a
  better trunk would be HARDER to draft for and would cost speed — it does
  not.)
- **Acceptance is a property of the WORKLOAD, not of our quantization.** It
  ranged 0.64-0.95 across twelve ordinary prompts on one model, and hit
  exactly 1.0 on repetitive synthetic text.

### Single-prompt acceptance is not comparable across rungs either

`bench_plan` / `mtp-bench` use ONE fixed prompt. On that prompt, acceptance
came out 0.7695 on Flash-Next 2.1bpw and 0.6445 on 3.2bpw -- a 12.5pp gap in
the direction that would say a better trunk drafts worse.

It is not a real effect, and the reason is structural rather than statistical:
two rungs decoding greedily from the same prompt produce DIFFERENT TEXT, so
their acceptance figures are measured on different content. Acceptance is a
property of the text being generated (§7), so comparing one trajectory to
another compares workloads, not models. The 12-prompt paired instrument says
what the single prompt cannot: 0.7823 at 3.2bpw against 0.8057 at 4.4bpw,
alongside 0.78-0.82 at 2.1bpw -- all one band, consistent with the N=3 paired
result that trunk quantization does not move acceptance.

Read speedup out of `mtp-bench`. Read acceptance out of `mtp-accept`.

### So cross-project acceptance numbers are close to meaningless

Given the above, comparing our 0.78 against oMLX's reported 96.8-97.9% says
almost nothing: the spread from workload alone is larger than the gap being
discussed. An easy prompt gets 1.0 on our own stack. Independent numbers on a
third-party Flash-Next MTP checkpoint report 58.3-89.5%, which brackets ours
rather than theirs.

**Any acceptance comparison across projects needs the same prompts.** Until
someone runs that, the honest statement is that the numbers are not
comparable, not that theirs is better.

Still untested and now the most plausible remaining lever:

- **Hidden-state variant.** MTPLX exposes `mtp_hidden_variant`
  (fc / pre_norm / post_norm / embedding / prev / mix) as a knob because it
  matters. We feed the hyper-connection mixer's input and have never swept the
  alternatives on this family. The dense-27B ablation put pre_norm 0.7285
  against post_norm 0.7188 — a near-tie there, untested here.

---

## 8. Working state (2026-08-31, end of session)

### Where things live

| what | where |
|------|-------|
| qwen4_exp runtime venv | `~/.venvs/qwen4exp` on **both** M3 and M4 (mlx-lm 0.32.0 from the unmerged PR ml-explore/mlx-lm#1788; README inside explains why) |
| repo copy on M4 | `~/vqlab-mtp` (rsync target; `PYTHONPATH=~/vqlab-mtp/src`) |
| heads (M4) | `~/heads/mtp-head-e3q8.safetensors` (1.25 GiB), `mtp-head-e4q8.safetensors` (1.55) |
| q6 head | `/Volumes/Thunderbay SSD/Exo Models/mtp-head-q6.safetensors` (2.14 GiB) |
| **397B graft** | `~/heads/mtp-graft-397b-bf16.safetensors` on **both** M3 and M4 — 1553 tensors, 12.29 GiB, all-zero gated |
| **397B sidecar** | `~/heads/mtp-397b-e3q8.safetensors` on both — 3.19 GiB, experts 3-bit / rest 8-bit, `fc_order=eh`, `norm_shift=1.0` |
| **27B graft / sidecar** | `~/heads/mtp-graft-27b-bf16.safetensors` (0.79 GiB, M3), `~/heads/mtp-27b-q8.safetensors` (0.49 GiB, both) |
| corpora | `~/corpora/referee_corpus_{literary,code_public}.txt` on M3, `~/referee_corpus_*.txt` on M4 |
| diagnostics | `~/seqcost.py` on M4 (t(seq=2)/t(seq=1)); `~/bench_plan.py`, `~/soak.py`, `~/decay_test.py`, `~/longctx_accept.py` |
| oMLX (for interop testing) | `~/omlx-src` + `~/.venvs/omlx` on M4; removable with two `rm -rf` |
| overnight results | `~/overnight/` on M4 (`campaign.log` + per-stage json) |

Machine limits: **M3 = 96 GB / 84 GiB wired**, **M4 = 128 GB / 120 GiB wired**.
Flash-Next runs to 4.4bpw on M4; 397B fits ONLY at 2.2bpw (measured: trunk
100.12 + head 3.13 = **103.25 GiB resident**), and 2.4bpw upward needs the M3
clustered in. The 397B does NOT fit on the M3 at any rung.

**Everything on the Thunderbay reaches the M4 over SMB**, and that link is the
single biggest source of bad measurements in this project: it produced a 522s
cold start in the soak and a 0.43 tok/s "baseline" in the discarded 27B run.
Do not run two model loads at once and then trust a wall-clock number from
either.

### Settled

**Long-request decay: not reproduced, with one caveat.** An 8192-token single
request on Flash-Next 2.1bpw grew peak memory by 0.26 GiB total (47.34 ->
47.60) and held throughput flat (35.7 tok/s at 2304 tokens, 32.0 at 8192 --
consistent with ordinary trunk KV growth, not an MTP-specific leak). The
caveat is real and limits what the second half of that run measures: from
about token 2000 the unattended generation degenerated into repetition and
window acceptance pinned at exactly 1.000, which is the known
degenerate-text artifact. So the MEMORY result stands (it is structural), and
the throughput result past ~2000 tokens is measuring the easy case.

- qwen4_exp shipped: 1.25 GiB head, acceptance 0.78-0.82, **1.58x** measured
  with a 0.32% baseline spread.
- Alignment (`align="committed"`) is worth +5.9pp at short prompts and
  **grows with context** (+3.9pp at 512, +9.8pp at 2048 on real prose).
- Trunk quantization does NOT affect acceptance (N=3 rungs, paired).
- Acceptance is a workload property: 0.64-0.95 on ordinary prompts, 1.0 on
  repetitive text. Depth-1 ceiling measured at **1.95x**.
- MTP prefill tax (seeding) is **1.3-5.1%**.
- `vqlab serve` works, and has now been soaked: **643 requests over 60
  minutes, zero errors, and every single one drafted.** The two numbers that
  matter are the ones that caught the earlier silent failure --- 644 `MTP:
  acceptance` lines (one per request) and **0** `Prompt processing progress`
  lines, which only stock mlx-lm emits. The patches held for the full hour.
  Throughput was flat: mean 26.76 tok/s, median 26.79, p05 25.18, p95 28.20,
  and first-half to second-half drift of **+0.41%**. Cold start was 522s to
  page 46 GiB over SMB, which is a storage number, not a serving one.

- Speedup by rung on Flash-Next, same fixed prompt: **1.58x** at 2.1bpw
  (acceptance 0.77), **1.52x** at 3.2bpw (0.64), **1.65x** at 4.4bpw (0.70).
  It tracks acceptance, not bit-width --- and since each rung writes different
  text, those three acceptance figures are three workloads, not three models
  (see SS7).

### The 397B (qwen3_5_moe): head module written, wiring measured

`vqlab/mtp_head_qwen35.py` covers both `qwen3_5` (dense 27B) and
`qwen3_5_moe` (397B); they differ only in the mlp the stock `DecoderLayer`
builds from `args.num_experts`, so one module serves both. Structurally it is
much simpler than the qwen4_exp head: one residual stream, so no
hyper-connections and no per-stream norm statistics, and the head's block is a
stock full-attention `DecoderLayer` that owns its own rope.

Three things about the head are **not recoverable from the checkpoint**, and
each wrong choice is a silent near-zero-acceptance failure with no error --
the same failure mode that made the qwen4_exp head look dead for a day. So
`vqlab mtp-probe35` sweeps them against a single trunk load. Measured on the
dense 27B (VQ-3.9bpw, 512 positions, literary corpus, control 0.596):

| norm_shift | fc_order | h_source | acceptance vs main greedy |
|---|---|---|---|
| 1.0 | **eh** | pre_norm | **0.6562** |
| 1.0 | **eh** | post_norm | **0.6582** |
| 1.0 | he | pre_norm | 0.0020 |
| 1.0 | he | post_norm | 0.0020 |
| 0.0 | any | any | **0.0000** (all four) |

- **`norm_shift = 1.0` is settled.** The family stores RMSNorm gains as
  deltas. Without the shift the head is exactly dead, in every other wiring.
- **`fc_order = "eh"` is settled** — `[embedding | hidden]` into the fused
  `fc`. The other order is chance.
- **`h_source` is NOT settled** and probably cannot be by this experiment: a
  one-token difference over 512 positions. `pre_fc_norm_hidden` is applied
  immediately afterwards and an RMSNorm of an already-normed vector is close
  to idempotent, so the two arms are nearly the same computation. Default
  `pre_norm`, on the argument that the head carrying its own hidden norm
  expects a raw hidden state.

The real decode loop -- cache rollback, alignment, verification, none of
which the probe exercises -- confirms it on the dense 27B: **0.7399 pooled
acceptance** over 12 prompts / 1511 steps (per-prompt 0.617 to 0.852), which
sits in the same band as qwen4_exp's 0.78-0.82. The teacher-forced probe read
0.656 on a literary corpus; the loop reads higher on chat prompts, consistent
with acceptance being a workload property (SS7) rather than the two
instruments disagreeing.

#### The 27B wall-clock, and a problem that is NOT about MTP

The 27B VQ-3.9bpw artifact benchmarks at a **0.43 tok/s baseline** on the M4 ---
roughly forty times too slow for a 12 GiB model on that machine. My first
explanation was SMB contention. That was wrong: a re-measure with the link
quiet reproduced it exactly (0.43 and 0.42 tok/s, 1063s per config both
times). Reproducible is not contention.

So the number to carry forward is not the 1.43x ratio --- which is a real
ratio inside a broken regime, and not comparable to Flash-Next's 1.58x --- but
the baseline itself. The control settles it: on the same machine, same
harness, same prompt, no head involved, stock `Qwen3.8-27B-8bit` generates at
**16.687 tok/s** and our `27B-VQ-3.9bpw` at **0.426 tok/s** --- 39x slower,
while using 7 GB LESS memory.

**That is a VQ finding, not an MTP finding, and it is the most consequential
thing this campaign turned up.** It is written up separately in
[DENSE-VQ-DECODE.md](DENSE-VQ-DECODE.md). The MoE VQ path measures clean
(Flash-Next 18.85 tok/s, 397B 17.59), so it points at the dense read path
specifically.

The general rule: a speedup ratio is only a compute speedup if the absolute
throughput is plausible for the model. Check the baseline against what the
machine should do BEFORE reading the ratio --- and when something looks like
contention, reproduce it before believing that.

#### The recurrent-cache trap (cost 30 minutes of a run going nowhere)

`cache_semantics` is not just tidiness for this family, it is the difference
between working and unusable. The trunk's cache list is MOSTLY recurrent --
48 `ArraysCache` to 16 `KVCache` on the 27B -- and under `"copy"` every one of
those GatedDeltaNet states is deep-copied once per speculative step. It does
not fail; generation just crawls. `check_snapshot_semantics` against a loaded
27B returns True (GatedDeltaNet reassigns its slots, `cache[0] = ...`, and mlx
arrays are immutable), so the family earns `"reassign"`. Prompt 0 gives
0.6328 under both paths, so this is a pure speed fix with the answer
unchanged.

The lesson generalises: for any family whose trunk is mostly linear/recurrent
attention, run `check_snapshot_semantics` BEFORE concluding anything about
speed. The conservative default is correct and slow, and slow looks like
broken.

The 397B head packs to **3.19 GiB** at experts-3-bit / rest-8-bit (from 12.29
GiB bf16, ratio 0.257); e4q8 is 3.91 GiB. Against the 101 GiB 2.2bpw rung
that is ~104 GiB resident, which fits the M4's 120 GiB wired limit but is the
one 397B rung that does.

#### The 397B result: the best acceptance we have measured, and no speedup

Both instruments agree the head is correct, and the wiring measured on the
dense 27B transferred to the MoE unchanged:

| instrument | result |
|---|---|
| probe, 512 positions, literary corpus (control 0.850) | eh/pre_norm **0.8320**, eh/post_norm **0.8516**; he 0.0039 / 0.0000 |
| real decode loop, 12 prompts / 1536 steps | **0.9023** pooled, per-prompt 0.836 to 0.953 |
| wall-clock, 2.2bpw rung, 103.25 GiB resident | spec 17.61 vs base 17.59 tok/s = **1.001x** |

0.9023 is the highest acceptance in this document --- higher than qwen4_exp's
0.78-0.82, which is what you would expect from a much stronger model
predicting its own next token. At that acceptance the depth-1 ceiling (SS7)
says roughly 1.8x. We measured 1.000x, twice, with baselines agreeing to
0.3%.

**The mechanism is measured.** Depth-1 swaps two seq=1 trunk forwards for one
seq=2 forward, so it pays only when the second token is nearly free.
`seqcost.py` measures that directly, and the two models could hardly differ
more:

| | Flash-Next 2.1bpw | 397B 2.2bpw |
|---|---|---|
| t(seq=1) | 52.30 ms | 54.48 ms |
| t(seq=2) | 46.66 ms | **81.24 ms** |
| **ratio seq2/seq1** | **0.892** | **1.491** |
| head / trunk1 | 0.099 | 0.168 |
| predicted at its acceptance | 1.80x | 1.15x |
| measured | 1.58x | 1.00x |

On Flash-Next the second token rides along free --- single-token decode is
overhead-bound, so a 2-token forward is *cheaper* than a 1-token one. On the
397B the second token costs half a forward again.

That ratio is fatal on its own. At 1.491 with a head at 0.168, the ceiling at
PERFECT acceptance is 2 x 54.48 / (81.24 + 9.17) = 1.205x predicted --- and
the cost model overpredicts by 12-14% on both models, so the real ceiling is
about 1.06x. No acceptance rate rescues this, and neither does a cheaper head:
the head is only a tenth of the denominator.

**Memory pressure was the obvious explanation, and it is wrong.** The 397B
sits at 103.25 GiB against a 120 GiB wired limit, so bandwidth-bound decode
was the natural guess. Flash-Next tests it directly, being the same
architecture at three residencies:

| model | resident | ratio seq2/seq1 |
|---|---|---|
| Flash-Next 2.1bpw | 46 GiB | 0.892 |
| Flash-Next 3.2bpw | 70 GiB | 0.874 |
| Flash-Next 4.4bpw | **95 GiB** | **0.838** |
| 397B 2.2bpw | 103 GiB | **1.491** |

The ratio does not climb with residency --- it falls slightly, and it is still
0.838 at 95 GiB, within 8 GiB of where the 397B sits. Memory pressure on this
machine does not produce a 1.49 ratio.

**So this is not a "needs a bigger machine" problem, and clustering the 397B
across M3+M4 will not fix it.** That is worth knowing before anyone spends a
day on exo for this.

**Our VQ path is not the cause either.** The dense-VQ investigation
(DENSE-VQ-DECODE.md) made "our kernels behave badly at some shapes" a live
hypothesis for the MoE side too. The 35B-A3B settles it, being the same model
in both forms:

| model | t(1) | t(2) | ratio |
|---|---|---|---|
| Flash-Next 2.1bpw (VQ) | 52.27 | 46.78 | **0.895** |
| 27B 8-bit (stock, dense) | 62.28 | 63.41 | 1.018 |
| 35B-A3B 8-bit (stock, MoE) | 12.27 | 14.02 | 1.143 |
| 35B-A3B 3.8bpw (VQ, MoE) | 16.94 | 20.07 | 1.185 |
| 397B 2.2bpw (VQ, MoE) | 54.54 | 81.35 | **1.492** |

VQ costs ~38% of absolute time on the 35B (16.94 vs 12.27) and barely touches
the ratio (1.185 vs 1.143). So VQ is not eating the n=2 discount.

**The real lesson is that Flash-Next is the outlier, not the 397B.** It is
the only model measured whose 2-token forward is CHEAPER than its 1-token
forward. Everything else sits at 1.02-1.19, and the 397B's 1.49 is the
extreme end of a normal spectrum rather than a pathology --- consistent with
it having by far the most active parameters (17B), so the marginal token
costs more and there is less fixed overhead to amortise. **Some of our
headline 1.58x is Flash-Next collecting a kernel-selection discount at n=1
that other models do not offer.**

#### Codebook residency does NOT explain it (a wrong claim, corrected)

This document briefly recorded that Flash-Next behaves differently because
its codebook is small enough for threadgroup memory (d=2, K=256, 1 KB) while
the 397B's (d=8, K=16384, 256 KB) must stream from device memory. That was
wrong, and it came from reading the FIRST entry of `vq_modules` and
generalising it to the model. The real distributions:

| model | expert modules |
|---|---|
| Flash-Next 2.1bpw | **138 x d=8 K=16384 pack=14**, 6 x d=2 K=256 |
| 397B 2.2bpw | 171 x d=8 K=16384 pack=14 |

The six d=2 modules are layer 0 alone. **Both models are dominated by the
same geometry and both stream the same 256 KB codebook from device memory**,
so residency cannot be what separates 0.895 from 1.492. The seq2/seq1 gap
remains unexplained; the active-parameter argument above (17B vs far fewer)
is still the only account that survives, and it is an argument rather than a
measurement.

The methodological error is the same one this document already records twice:
reading a property off one sample and asserting it of the population. Check
the distribution.

### Depth is the lever, and the 397B is the best candidate we have

SS7 already argued depth pays only at high acceptance. The 397B has the
highest acceptance in this document (0.9023), and at depth-1 that asset is
wasted. Working the measured scaling curves --- a depth-d step is ONE trunk
forward of length d+1, d head forwards, and 1 + a + ... + a^d committed
tokens in expectation:

| model | depth | seq | E[commit] | cost | predicted |
|---|---|---|---|---|---|
| Flash-Next | 1 | 2 | 1.78 | 51.99 ms | 1.790x |
| Flash-Next | **2** | 3 | 2.39 | 68.00 ms | **1.836x** |
| Flash-Next | 3 | 4 | 2.86 | 87.57 ms | 1.709x |
| 397B | 1 | 2 | 1.90 | 90.70 ms | 1.142x |
| 397B | **2** | 3 | 2.71 | 114.14 ms | **1.295x** |
| 397B | 3 | 4 | 3.44 | 156.14 ms | 1.201x |

Depth-2 is optimal for both, and it is worth far more to the 397B (+13%
relative) than to Flash-Next (+3%), precisely because acceptance is higher.

Two corrections to apply before believing 1.295x: the cost model overshot
depth-1 by ~13% on both models, and geometric decay (a^2) flatters a
multi-step draft, since step 2 feeds the head its own hidden state rather
than the trunk's. Realistically depth-2 on the 397B is **~1.1-1.15x** --- not
the 1.6x Flash-Next gets, but no longer nothing.

**So: "the 397B buys nothing" is true only of depth-1.** The head is correct,
drafts better than any other family we have, and is currently being run in
the one configuration that cannot use it. Implementing depth-2 is real work
(multi-step drafting off the head's own hidden state, per-step acceptance
decay to measure rather than assume) and is a scoping decision, not something
to start unasked.

#### Flash-Next: the depth-1 cost model, measured rather than assumed

`seqcost.py` measures the trade depth-1 actually makes. On Flash-Next 2.1bpw
at 256 context:

| quantity | measured |
|---|---|
| t(seq=1) trunk forward | 52.30 ms |
| t(seq=2) trunk forward | 46.66 ms |
| **ratio seq2/seq1** | **0.892** |
| head forward | 5.17 ms = **9.9%** of a trunk forward |
| predicted speedup at acceptance 0.78 | 1.796x |
| measured speedup | 1.58x |

Two things worth keeping. First, a 2-token forward is *cheaper* than a 1-token
forward (ratio 0.892) --- single-token decode is overhead-bound, so the second
token rides along free. That is exactly the condition depth-1 needs, and it is
why this model gets 1.58x. Second, the head costs 9.9% of a trunk forward,
which finally puts a measured number on the head-cost term that SS7 could only
bound (it inferred h ~ 0.03 from the 1.95x ceiling; the direct measurement says
0.099).

The model still overpredicts --- 1.80x against a measured 1.58x, a 12% gap ---
so it is a mechanism, not a calibration. Do not quote the predicted number.

#### h_source is now worth a second look

On the 27B, `post_norm` beat `pre_norm` by one token in 512 --- noise. On the
397B the gap is ten tokens (436 vs 426, paired on the same positions), still
small but no longer obviously nothing. The shipped default is `pre_norm`.
This is cheap to settle properly (it is a flag on an already-packed sidecar,
so it costs one model load) and should be settled before the family is
described as tuned.

#### A wrong inference, recorded

Earlier the same evening I compared the 397B's MTP norm gains against their
trunk counterparts, found no single offset that explained all of them, and
concluded in this document that "the ledger's +1.0 rule does not survive
contact with this checkpoint" and that the convention had to be swept because
it might not be a uniform shift.

The sweep says the offset **is** a uniform +1.0. The magnitudes differ from
the trunk's because it is a different layer doing a different job, not
because it uses a different convention. Comparing a norm's magnitude to
another layer's was never evidence about the convention; the only evidence
was `has_unsanitized_conv1d` (the 397B source stores `conv1d.weight` as
`[12288, 1, 4]`, i.e. unsanitized, so the trunk norms ARE shifted at load)
and, decisively, acceptance.

Sweeping was still the right call -- it just was not right for the reason I
gave.

