# Card changes for the 2.2bpw v3 release — DRAFT for Noah's review

Changed blocks only, in card order, to be applied against
`MODEL_CARD_397B_F.md`. `<REV-V2>` = `9cc3212e300227ded585669d35be025d98b04d55`
(current HEAD — the outgoing weights).

---

## 1. Header block (replaces the "v2 — updated 2026-08-22" block)

> **100.9 GiB — the accessibility build.**
>
> **v3 — updated 2026-09-07.** This revision changes **weights**, not just
> the runtime: 11 of the 28 shards are rewritten. Better on both perplexity
> corpora than v2, at 0.1 GiB smaller. What changed:
>
> - **Measured per-layer allocation.** Eleven layers carry richer codebooks,
>   chosen by direct measurement over 96 candidate shapes scored on prose,
>   code and a held-out literary corpus — and allocated per *projection*
>   rather than per whole layer, which is finer-grained than any previous
>   build in this family.
> - **Layers 57–59 now carry VQ codebooks**, where they were previously left
>   in affine 3-bit. Smaller and slightly better.
> - **Input embeddings at 4-bit instead of 6-bit**, which measured free on
>   every corpus and paid for part of the above.
>
> v2's bytes remain downloadable by pinning the previous revision:
>
> ```python
> snapshot_download("TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw",
>                   revision="<REV-V2>")  # v2
> ```

## 2. Measured results table

| | **this model, v3** (100.9 GiB) | v2 (101.0 GiB) | `VQ-2.4bpw` (111.6 GiB) |
|---|---|---|---|
| wikitext perplexity (raw, prefix-8192) | **2.9235** | 3.0591 | 2.7655 |
| code perplexity (mixed-language) | **2.6632** | 2.6728 | 2.6383 |

v3 is better than v2 on **both** corpora — prose by 0.135, code by 0.010 —
at 0.12 GiB less, and it holds on a **held-out literary corpus never used to
select anything**: six disjoint windows, v3 ahead on all six.

**On reading these numbers.** A perplexity delta measured on one text window
carries roughly a 35% relative spread — the same comparison across six
windows here ranged +0.040 to +0.105. Rank by direction, not by the third
decimal.

## 3. Memory (replaces the resident/peak figures — SHORTER)

> **~101 GiB resident; ~108 GiB peak at 8k context**, measured externally on
> a 128 GB machine with **runtime r3** (the `model.py` in this revision).
>
> Peak is a property of the runtime, not the weights: r3 caps MLX's
> buffer-reuse cache by default, so an 8k prefill costs ~7 GiB over the
> short-prompt peak instead of a multiple of it. `VQLAB_CACHE_LIMIT_GB`
> moves that ceiling. Note that `ps` RSS reads ~57 GiB for this model at
> every workload and is not a useful guide — budget the figures above.

*(Suggestion: stamp `__runtime_version__ = "r3"` in `model.py` and quote it
here. The bundle currently carries no version marker, so cards describe
runtime behaviour without being able to name which runtime — which is why
memory figures go stale in them. A stamp also lets a user check what they
actually have.)*

## 4. Speculative decoding (MTP)

The MTP head was fitted against the earlier trunk, and v3 changes late trunk
layers, so acceptance was re-measured rather than assumed: **0.9121 on v3
against 0.9082 on v2**, same head, 1536 steps over 12 prompts. This remains
the only 397B rung that fits with the head on a single 128 GB machine.

## 5. Task benchmarks — extend the existing v1/v2 note

The card already says these rows were measured on v1's weights. Add:

> v3 has likewise not been re-evaluated on this harness. Its perplexity
> improvements over v2 are measured; its task scores are not, and are not
> presented as such.

## 6. Provenance — append one line

> v3 (2026-09-07): per-layer, per-projection promotion set chosen by
> exhaustive measurement and allocated at a fixed byte budget; layers 57–59
> moved from affine to VQ; `embed_tokens` re-quantized 6→4 bit.

---

## On the layers-57–59 wording

The header bullet is deliberately plain, per your call — no section, no
forensics. One thing worth keeping somewhere, though not necessarily in
this card: **because every rung carried the same configuration, no
published rung-to-rung comparison changes.** That belongs in the paper's
errata whenever it is next touched. It protects the existing numbers rather
than drawing attention to the cause, and a reader who notices the weights
changed will otherwise wonder whether the comparisons still hold.

## Checklist before pushing

- [ ] `<REV-V2>` = `9cc3212e300227ded585669d35be025d98b04d55`
- [x] `check_release --artifact <dir>` passes WITH strict smoke
- [ ] decide on the `__runtime_version__` stamp
- [ ] chart regenerated if the card carries one (the 2.2 point moves)
- [ ] Noah runs the upload


---

## POST-DRAFT CORRECTION (2026-09-07, after the exo failure)

The first build of this artifact carried **float32** `embed_tokens`
scales/biases: `affine_requant.py` upcast the weight to fp32 before
`mx.quantize`, and mx.quantize returns scales in the INPUT dtype. That
made the embedding output fp32, which propagated through the whole
forward pass, which at `head_dim: 256` asked Metal for a 53 KB
threadgroup attention kernel against a 32 KB limit -- crashing every exo
prefill while single-box `mlx_lm` worked fine.

Fixed (scales cast back to the source dtype). Numbers above are the
CORRECTED build: 100.855 GiB, prose 2.9235, code 2.6632. The fp32 scales
were also 0.059 GiB of pure waste.

**Two gate gaps this exposed, both still open:**

1. **No gate checks dtypes.** `check_release`, the strict smoke and every
   perplexity run passed on the broken artifact, because they are all
   single-box and never reach the distributed prefill path. A dtype
   census against the source artifact belongs in `check_release`.
2. **No gate exercises exo.** The only failure mode was in exo's
   sharded prefill. `check_release` has a `--cluster-smoke` flag; it was
   not run here and should be mandatory for any artifact that ships for
   distributed serving.
