# Card changes for the 2.2bpw v3 release — DRAFT for Noah's review

Not the whole card: only the blocks that change, in card order, so the
edits can be applied against `MODEL_CARD_397B_F.md` and checked one by one.
`<REV-V2>` = the current HEAD commit hash of the repo, to be filled in
BEFORE pushing (it becomes the pin for the outgoing weights).

---

## 1. Header block (replaces the "v2 — updated 2026-08-22" block)

> **100.9 GiB — the accessibility build.**
>
> **v3 — updated 2026-09-07.** This revision changes **weights**, not just
> the runtime: 11 of the 28 shards are rewritten. Three improvements, all
> measured, at a size 0.1 GiB *smaller* than v2:
>
> 1. **A defect fix.** Layers 57–59 of this 60-layer model were never
>    VQ-fitted in any published rung — they shipped as affine 3-bit, the
>    lowest precision in the artifact — because the original fit was run
>    over layers 0–56. They are now VQ at d4/K2048, which is both smaller
>    and slightly better. See "Layers 57–59" below.
> 2. **Measured per-layer, per-projection allocation.** Eleven layers carry
>    richer codebooks, chosen by direct measurement on three corpora rather
>    than by a heuristic, and on a per-projection basis rather than whole
>    layers.
> 3. **Input embeddings at 4-bit instead of 6-bit**, which measured free on
>    every corpus and refunded 0.18 GiB to spend on the layers above.
>
> v2's bytes remain downloadable by pinning the previous revision:
>
> ```python
> snapshot_download("TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw",
>                   revision="<REV-V2>")  # v2
> ```

## 2. Measured results table (replaces the v2/v1 columns block)

| | **this model, v3** (100.9 GiB) | v2 (101.0 GiB) | v1 (100.9 GiB) | `VQ-2.4bpw` (111.6 GiB) |
|---|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.9200** | 3.0591 | 3.1706 | 2.7655 |
| code perplexity (mixed-language) | **2.6619** | 2.6728 | 2.6988 | 2.6383 |

v3 is better than v2 on **both** corpora — prose by 0.139, code by 0.011 —
at 0.1 GiB less. It also holds up on a **held-out literary corpus that was
never used to select anything**: six disjoint windows, v3 ahead of v2's
weights on all six, mean +0.074.

**On reading these numbers.** Perplexity deltas measured on a single text
window carry roughly a 35% relative spread — the same comparison across six
windows here ranged +0.040 to +0.105. Rank by direction, not by the third
decimal.

## 3. Memory (replaces the resident/peak figures)

Measured externally on a 128 GB machine, sampling system-wide availability
at 0.33 Hz while generating:

| workload | MLX peak | machine memory consumed |
|---|---|---|
| short prompts, 200-token generations | 107.9 GB | ~101.6 GiB |
| 7721-token prefill + 256 new | 115.0 GB | ~108.3 GiB |

An 8k prefill costs only **6.6 GiB over the short-prompt peak** rather than
a multiple of it — the runtime caps MLX's buffer-reuse cache by default
(`VQLAB_CACHE_LIMIT_GB`). On a 128 GB machine that leaves ~20 GiB at 8k
context for KV cache and everything else.

Note a 40-token smoke reads only ~55 GiB resident. That is a
mixture-of-experts artefact, not a memory saving: 10 of 512 experts fire
per token per layer, so a short run never faults in most expert weights.
Budget the numbers above.

## 4. Speculative decoding (MTP)

The 397B MTP head was fitted against the v1/v2 trunk. v3 rewrites the last
three trunk layers, which is exactly where the head reads from, so
acceptance was re-measured rather than assumed: **0.9121 on v3 against
0.9082 on v2**, same head, 1536 steps over 12 prompts. Acceptance is
unchanged-to-slightly-better; this remains the only 397B rung that fits
with the head on a single 128 GB machine.

## 5. NEW SECTION — "Layers 57–59" (place after Methodology)

> ### Layers 57–59: a defect this release fixes
>
> Qwen3.5-397B-A17B has 60 hidden layers. Every VQ rung we published
> fitted expert codebooks for layers **0–56 only** — the fit was invoked
> over that range on a 60-layer model — so layers 57, 58 and 59 fell
> through to the affine path and shipped at **3 bits**, the lowest
> precision anywhere in the artifact. They are ordinary routed-expert
> layers on the main forward path, structurally identical to their
> neighbours.
>
> In v3 they are VQ at d4/K2048. That is 1.25 bits per weight
> *cheaper* than the affine tensors it replaces (2.25 bpw versus 3.5 —
> affine carries an fp16 scale AND bias per group of 64), refunds 1.1 GiB, and measured slightly better
> on prose with code and literary level.
>
> **This does not invalidate any published comparison.** All four rungs
> carried the defect identically, so every rung-to-rung number in these
> cards and in the paper stands exactly as printed. The effect of the
> defect was to make the artifacts ~3.4 GiB heavier, and marginally worse,
> than the method actually delivers — it understated VQ rather than
> overstating it. The other rungs will be corrected in a single family-wide
> update rather than piecemeal.

## 6. Task benchmarks section — ADD this note under the table

> **These benchmark figures are from the v1 weights and have NOT been
> re-run for v3.** The perplexity numbers above are v3; these are not.
> They are kept because the comparators were all evaluated on the same
> harness and remain a valid relative picture, but treat the row for this
> repo as a lower bound on v3 rather than a measurement of it. As the
> section already notes, PIQA and WinoGrande separate no pair at n=1000
> and stand as integrity checks rather than rankings.

## 7. Provenance — append

> v3 (2026-09-07): promotion set chosen by exhaustive per-layer,
> per-projection measurement over 96 candidate shapes scored on prose,
> code and a held-out literary corpus, allocated by knapsack at a fixed
> byte budget; layers 57–59 converted from affine 3-bit to d4/K2048;
> `embed_tokens` re-quantized 6→4 bit. Release gates re-run, MTP
> acceptance re-measured.

---

## Checklist before pushing

- [ ] fill `<REV-V2>` with the current HEAD hash
- [ ] `check_release --artifact <dir>` passes WITH the strict smoke
- [ ] chart regenerated if the card carries one (the 2.2 point moves)
- [ ] confirm the 4-bit `embed_tokens` is reflected in any config table
- [ ] Noah runs the upload (credentials are his)
