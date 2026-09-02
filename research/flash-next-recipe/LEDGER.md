# Flash-Next recipe arc — dense-tensor bit-width ledger

Arc opened 2026-09-02. Scope: the NON-EXPERT (dense) side of
`TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw`. The VQ expert fits and the
PLE tables are inputs here, never touched.

---

## 2026-09-02 — E-DENSE6: 8-bit -> 6-bit dense. NO-SHIP. Costs quality, buys no speed.

**Question.** The 2026-09-02 stub-ablation found the 2.1bpw artifact reads
5.878 GiB of dense weights per token against the stock 3-bit comparator's
2.111, and attributed 56% of the 17.6-vs-27.0 tok/s gap to that dense read.
If that attribution is causal, dropping the dense tensors from 8-bit/g64 to
6-bit/g64 should return a healthy share of the +11 ms/token.

**Answer: it returns nothing measurable, and it costs quality on both
corpora. The ablation's byte accounting is right and its causal reading is
REFUTED.**

### Provenance — established before building anything

The shipped artifact's config carries **726 dense modules, every one at
`{bits: 8, group_size: 64}`** (plus 144 VQ expert modules under `vq_modules`
and the PLE under `vq_ple`).

Those 726 modules are **byte-identical to
`Qwen--Qwen3.8-Flash-Next-VQ-BASE`** — the struct base cut by
`vqlab stream-convert --struct --family qwen4_exp --protect-bits 8` from the
bf16 Qwen3.8-Flash-Next checkpoint. sha256 over the raw tensor bytes,
artifact vs base:

    model.layers.0.linear_attn.in_proj_qkv.weight   7569e42715eb7421  IDENTICAL
    model.layers.20.linear_attn.out_proj.scales     3d30a6a7d3f5036a  IDENTICAL
    lm_head.weight                                  cfc9b05743c84088  IDENTICAL
    model.embed_tokens.weight                       ceee23032697acc6  IDENTICAL

So the dense side was quantized **directly from bf16 at 8-bit/g64**, and a
6-bit rebuild must come from bf16 too, not from those 8-bit bytes.

**The bf16 checkpoint is gone.** `/Volumes/Thunderbay SSD/Exo Models/` holds
the Flash-Next affine ladder (3/4/5/6/8-bit) and the VQ base; there is no
`Qwen--Qwen3.8-Flash-Next-bf16` (the 397B bf16 survives, this one does not),
and nothing on the HDD.

**Source used instead: `Qwen--Qwen3.8-Flash-Next-6bit`** (137 GiB), the
affine q6 rung of the ladder the model card compares against. It qualifies
as a bf16 source, not a requantization:

- its `quantization` map is the same 873-key shape as the struct base's,
  with the same recipe — group 64 on the body, bf16 router/norms/conv1d;
- every one of the 2178 dense tensor keys (weight/scales/biases across the
  726 modules) is present, same dtype, same shape, only the packed U32
  width differing (640 -> 480 words, i.e. 8 -> 6 bits);
- it was written **2026-08-28 11:25, before the VQ base (14:39)**, from the
  bf16 snapshot that was still on disk then.

So: **6-bit dense tensors quantized from bf16, not from the 8-bit bytes.**
Stated explicitly because it was the one way this experiment could have
gone quietly wrong.

### Build

`scripts/build_flashnext_dense6.py` ->
`/Volumes/Thunderbay SSD/Exo Models/flashnext_dense6_experiment`

Dense tensors occupy 9 of the 138 shards (`model-00001`, `model-00012..19`);
those 9 are rewritten tensor-by-tensor — dense keys taken from the q6 rung,
**every other key copied as raw bytes** — and the remaining 129 shards (128
PLE + the bf16 vision graft), `model.py`, the tokenizer and the chat
template are **hardlinked**, which makes byte-identity a property of the
filesystem rather than a claim. `config.json`'s 726 dense entries go
8 -> 6 bits; `vq_modules`, `vq_ple`, `vision_config` and the index's
`weight_map` are untouched. `metadata.total_size` recomputed from the
written shards (METHODOLOGY §2), never copied.

    dense weights on disk   4.831 GiB (8b/g64)  ->  3.694 GiB (6b/g64)
    artifact total         45.780 GiB           ->  44.643 GiB   (-1.137)
    resident (lazy=False)  44.96 GiB            ->  43.82 GiB    (-1.14)

Gates: **`check-bundle` PASS** (bundle carries the current runtime, 2048
lines, verbatim). Generation smoke **PASS** — 377 coherent greedy tokens
through the shipped `model.py`, four times over.

`check-release` FAILS, and **not on the artifact**: `src/vqlab/smoke.py`
calls `mlx_lm.utils.load(..., trust_remote_code=True)`, a kwarg the exo
env's mlx-lm 0.31.9 does not have. Pre-existing env drift; it fails
identically on the shipped artifact. Not fixed here (out of scope, and the
env is the grafted one — see the exo runtime map).

### Measurement 1 — decode tok/s. NO RECOVERY.

Warmed A/B, M3 Ultra, same prompt, `mlx_lm.utils.load(lazy=False)`, greedy,
377-token runs timed first-token-to-last. Each load burns a full throwaway
run first. Two independent A/B pairs, interleaved, 3 timed runs each ->
6 samples per side (`scripts/bench_decode_ab.py`).

| side | samples (tok/s) | median | mean | ms/token |
|---|---|---|---|---|
| shipped 2.1bpw (8-bit dense) | 17.278 17.028 17.217 17.260 17.265 16.668 | **17.239** | 17.119 | 58.41 |
| dense6 experiment (6-bit dense) | 17.443 17.148 17.241 16.931 17.022 17.411 | **17.195** | 17.199 | 58.14 |

**Delta: +0.08 tok/s on the mean, -0.04 on the median — 0.3% either way,
inside a per-side spread of 0.6 tok/s.** In latency: **-0.27 ms/token
against a predicted -6 ms** (56% of the 11 ms gap). The two distributions
overlap almost completely; there is no effect here to size.

**Why the ablation's inference failed.** Its bytes are right — 1.137 GiB of
per-token dense read really did go away. But at 58 ms/token, 1.137 GiB is
~1.4 ms even at a *conservative* 800 GB/s of M3 Ultra bandwidth, so the
ceiling on this lever was ~2.4%, not 56%, before a token was generated. We
did not even get that. The dense read is not on the critical path: 48
layers of small dispatches, the VQ expert decode kernels and the PLE gathers
hide it. **Per-token bytes read is a budget, not a latency model** — the
gap is dispatch/kernel-bound, and byte-share accounting cannot allocate it.

### Measurement 2 — referee perplexity. A CONSISTENT REGRESSION.

`vqlab score` could not run: `src/vqlab/referee/score_streaming.py` streams
blocks with `blk(h, mask=mask, cache=None)` against a `blk.is_linear` flag,
and the arch installed in the exo env (the ml-explore PR #1788 `qwen4_exp`
graft) renamed that to `layer_type` and changed the block signature to
`(h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx)`, with PLE
layers needing `ids`/`prev_ctx`. Shimming that would have made the number
unvalidatable. These artifacts fit in 96 GiB, so the streaming trick is not
needed: `scripts/score_ppl_resident.py` runs the whole model resident,
through the runtime the artifact ships, same metric (prefix NLL, exp-mean),
same corpora, chunked prefill with a KV cache.

**Validated against the known answer** (METHODOLOGY): on the shipped
artifact at the published prefix-2048 it reads **5.9003 prose** vs the
card's 5.9033 (0.05%) and **2.0780 code** vs 2.0762 (0.09%). Same
instrument, both sides, so the deltas below are on one ruler.

prefix-2048 (the published setting), chunk 512:

| corpus | shipped 2.1bpw | dense6 | delta | % |
|---|---|---|---|---|
| wikitext (prose) | **5.9003** | **5.9228** | +0.0225 | +0.38% |
| mlx public code | **2.0780** | **2.1116** | +0.0336 | +1.62% |

prefix-8192 (the referee's default), chunk 512:

| corpus | shipped 2.1bpw | dense6 | delta | % |
|---|---|---|---|---|
| wikitext (prose) | **6.9869** | **7.0138** | +0.0269 | +0.38% |
| mlx public code | **1.7191** | **1.7350** | +0.0159 | +0.92% |

**Four rows, four regressions, same sign, and the prose penalty is +0.38%
at both prefix lengths.**

### Reading it against a floor

**The fit-to-fit noise floor does not apply to this comparison, and saying
so is the honest move rather than borrowing a number.** Both artifacts carry
the *same VQ expert codebooks and codes, byte for byte* (hardlinked /
raw-copied). The only difference is a deterministic affine requantization of
726 dense tensors. There is no stochastic draw anywhere in the delta, so
there is nothing for a seed-noise floor to bound — and inheriting the 397B
d4/K2048 floor (0.0056 prose / 0.0104 code) across model *and* geometry
would be a METHODOLOGY §3 violation twice over.

The applicable floor is the instrument's own repeatability, and it was
measured: re-running the identical configuration reproduced
`nll = 0.7474676594138145` **bit for bit**. Repeat floor **0.0000**.

Two things do bound how much to make of it:

- The referee's chunk size is a real, deterministic bias: chunk 512 -> 1024
  moved code ppl by **0.0110** on a fixed artifact (2.1116 -> 2.1007).
  Common-mode across both sides here (both scored at chunk 512), so the
  deltas survive it — but it is the natural scale for "an instrument choice
  can move this number by".
- Against that 0.0110 scale the code delta is 3.1x and the prose delta 2.0x;
  against the indicative 397B floors they are 3.2x and 4.0x.

So the regression clears the 3x bar on some readings and not on others.
**It does not matter, and this is the point: the bar is there to decide
whether a trade is worth taking, and there is no trade.** The speed side
measured zero. A quality change that is consistently negative across two
corpora and two prefix lengths, bought with nothing, is a loss whether or
not it is individually claimable.

### VERDICT — NO-SHIP

6-bit dense recovers **0%** of the decode gap (measured -0.27 ms/token
against a predicted -6) and costs **+0.38% prose / +0.9-1.6% code**
perplexity. It is a strict loss.

The 1.137 GiB it frees is real and speed-neutral, so it exists as a pure
*headroom* lever if a future rung is bytes-constrained — but at ~20 mnats/GiB
on this family's frontier (ledger 2026-08-29), those bytes are worth more
spent on expert bits than saved on dense bits.

### Refuted / recorded

1. **REFUTED: "56% of the Flash-Next decode gap sits in the dense read."**
   The byte accounting is correct; the causal attribution is not. Removing
   24% of the dense read returned nothing. Byte-share is not a latency
   model on this architecture.
2. **The dense read is not bandwidth-bound.** 1.137 GiB/token at M3 Ultra
   bandwidth is ~1.4 ms of a 58 ms token — the lever's ceiling was ~2.4%
   before it was pulled. Anything claiming more than that from a dense
   byte reduction was over-budget from the start; **check the lever against
   the bandwidth ceiling before building the artifact.**
3. **The remaining gap is dispatch/kernel-bound, not read-bound.** That is
   where the next Flash-Next decode work belongs. The VQ expert kernels
   remain untested by this experiment — it did not touch them.
4. **Two instruments have drifted off the installed runtime** and cannot
   score or gate a qwen4_exp artifact in the exo env today:
   `src/vqlab/smoke.py` (`trust_remote_code` kwarg, mlx-lm 0.31.9) and
   `src/vqlab/referee/score_streaming.py` (`is_linear` / block signature vs
   the PR #1788 graft). Both are pre-existing and fail on the *shipped*
   artifact too. Worth a repair pass before the next flash-next number.
5. **The referee's chunk size is a reportable setting, not an
   implementation detail** — 0.0110 ppl on code between 512 and 1024. Any
   cross-run comparison must pin it.
6. `mtp-head-q6.safetensors` (2.14 GiB) is **back in the shipped 2.1bpw
   artifact dir**, mtime 2026-09-02 10:52, after the 2026-08-30 decision to
   remove the staged sidecars from all four dirs so no upload could sweep
   them. Not touched here (SSD source artifacts are read-only, and it is
   not in the index so it does not affect any number above) and
   deliberately **not** copied into the experiment dir. Flagging it for
   Noah: it is staged again.

### Artifacts

`/Volumes/Thunderbay SSD/Exo Models/flashnext_dense6_experiment` (44.643
GiB on disk; ~35 GiB of new bytes, the rest hardlinked to the shipped
artifact). Uncommitted by design. **Release for deletion** — the verdict is
negative and `scripts/build_flashnext_dense6.py` rebuilds it in minutes.
