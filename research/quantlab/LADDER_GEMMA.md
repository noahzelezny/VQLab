# Gemma-4-26B-A4B ladder — affine baseline DONE, VQ pending

Target: replace the `gemma-4-e4b-it-8bit` sidecar (8.96 G on disk) with a
quantized `gemma-4-26b-a4b-it` at the same or smaller footprint, keeping the
audio tower and the family's literary character.

## STATUS (corrected 2026-08-17 evening — this section was stale)

An earlier revision of this file said "nothing here has been run" while
`results_crush/` already held twelve scored rungs. Both halves were true of
different things; stated precisely:

- **AFFINE ladder: RUN AND SCORED.** `convert_gemma_struct.py` built 12
  rungs and `kl_damage.py` scored all of them. Best at target size:
  `struct8-e2`, 9.1G, 34.90% top-1 agreement, against a 79.95% 8-bit
  ceiling. Full table + analysis in **CRUSH_RESULTS.md**. The converter is
  verified against real tensors (struct8-e8 reproduces uniform-q8).
- **VQ ladder: NOT RUN.** No VQ fit has touched gemma. This is the part
  still awaiting a go-ahead, and it is blocked on a decision from Noah —
  see the `down_proj` packing constraint below.
- **Teacher cache: BUILT** (`kl_cache_gemma26b`, 24x512 chat-wrapped
  literary, top-64, captured_mass 0.965).

The affine numbers are the BASELINE VQ has to beat, not the deliverable.
The 80% -> 35% gap at 9.1G is exactly the gap VQ closed at 397B.

## Why this model and not Qwen3.6-35B

| | addressable weights | total | audio |
|---|---|---|---|
| gemma-4-e4b-it-8bit (current) | ~6.0 G (2.8 G is PLE tables) | 8.96 G | yes |
| gemma-4-26b-a4b @ 2.4bpw | 7.8 G | ~7.8 G | yes |
| gemma-4-26b-a4b @ 3.1bpw | 10.1 G | ~10.1 G | yes |
| Qwen3.6-35B-A3B @ 2.3bpw | ~10 G | ~10 G | **no** |

`gemma-4-26b-a4b-it` is MoE (`enable_moe_block: True`, 128 experts, top_k 8,
30 layers, hidden 2816, moe_intermediate 704) — the regime the VQ pipeline is
proven in — and unlike e2b/e4b it has **no per-layer embeddings**
(`hidden_size_per_layer_input: 0`), so the whole weight surface is
addressable. Qwen3.6-35B has vision+video configs but no `audio_config`;
swapping to it drops a capability rather than upgrading one.

## Preflight (run before the VQ fit)

```bash
./gemma_preflight.py --model "/Volumes/Thunderbay SSD/Exo Models/mlx-community--gemma-4-26b-a4b-it-6bit"
```

Current result: **11 checks, 1 FAIL, 1 WARN.** Cleared statically —

- `mlx_lm.models.gemma4` exists and exports `Model`/`ModelArgs`, so the
  artifact's `model.py` shim has something to subclass (`add_model_file.py:77`).
- Experts are bias-free (`gemma4_text.py:164` `SwitchGLU(bias=False)`), which
  `vq_switch.py:552-553` requires.
- `sanitize()` drops **both** towers by `continue` (`gemma4.py:61-71`:
  `vision_tower`, `audio_tower`, `multi_modal_projector`, `embed_audio`,
  `embed_vision`). This is the guarantee `graft_vision.py:14-16` rests on, so
  **grafting the audio tower is as safe as grafting vision was** — text-only
  referee numbers cannot move.
- gate/up split is `mx.split(v, 2, axis=-2)` (`gemma4_text.py:627`) on a
  rank-3 `[E, 2*I, H]` tensor — that is axis 1, the same OUT-dim half-slice
  `vq_397b_codes.py:158-160` already does. Transfers unchanged.

### The one FAIL — `down_proj` cannot be sub-byte packed at d=4

`moe_intermediate_size = 704`, so at d=4 `NSUB = 176` and `176 % 32 = 16`.
`vq_pack.py:42` (`words_per_row`) asserts `NSUB % 32 == 0`. Three ways out,
in preference order:

1. **Skip packing for `down_proj`.** Packing is explicitly an optional final
   pass (`vq_pack.py:26-28`: "`vq_packed` absent means read codes as
   before"). Byte-aligned K≤256 codes stay valid. Costs size, not quality.
2. **d=2 for `down_proj`** → `NSUB = 352`, `352 % 32 = 0`. Packs cleanly, but
   halves the subvector dim, and E36 found down_proj *prefers* larger d — so
   this trades quality for the packing win. Measure before adopting.
3. Generalize `BLOCK` to 8 (`176 % 8 = 0`). Touches the Metal kernel's word
   alignment math (`vq_switch.py:111-112`, `:382-383`). Highest risk; last resort.

Note d=8 does **not** rescue it: `704/8 = 88`, `88 % 32 = 24`.

## What to download: bf16, for both models

The pipeline needs **bf16 sources**, not the community quants. Two consumers,
and both would be silently wrong on a pre-quantized source:

- `convert_gemma_struct.py` calls `mlx_lm.convert(src, quantize=True)` —
  handing it a 6-bit artifact quantizes a quantization.
- `vq_397b_codes.py:154-161` loads the **bf16 source tensor** for the k-means
  fit. The base artifact supplies only *which* tensors get VQ'd and their
  shapes; the values come from bf16.

`gemma-4-e4b-it-bf16` matters for a second reason: the incumbent sidecar on
disk is *mlx-community's* 8-bit quant. Comparing our VQ of 26b against
someone else's 8-bit of e4b is exactly the cross-publisher mixing the
standing methodology rule forbids (`score_tasks_streaming.py:23-26`, the 21%
harness-gap precedent). With e4b bf16 we can produce our own e4b reference
**and** our own e4b quant on one harness — which is also the control for
"does the method transfer to a dense, PLE-heavy gemma at all."

```bash
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"
hf download mlx-community/gemma-4-26b-a4b-it-bf16   # ~52G, the VQ source
hf download mlx-community/gemma-4-e4b-it-bf16       # ~18G, the control + reference
```

**BOTH DOWNLOADED** (2026-08-17). 26b bf16 = 48G, 11 shards, loads
strict. e4b bf16 = present. The 26b transfer wedged once with the process
alive but idle at ~14G; killing and re-running `hf download` resumed from
the `.incomplete` files at 44 MB/s.

### Disk state as of staging

| artifact | on disk | role |
|---|---|---|
| `gemma-4-e4b-it-8bit` | 8.4 G | incumbent sidecar |
| `gemma-4-26b-a4b-it-4bit` | 15 G (under `hub/` only) | candidate, stock shootout |
| `gemma-4-26b-a4b-it-6bit` | **config.json only** | unusable — do not point anything at it |
| `Qwen3.6-35B-A3B-4bit` | 19 G | code-leaning comparator |
| `gemma-4-26b-a4b-it-bf16` | **48 G** | VQ source — DOWNLOADED |
| `gemma-4-e4b-it-bf16` | **present** | control + reference — DOWNLOADED |
| `gemma26b-rungs/` | 12 rungs | affine ladder, all scored |
| `kl_cache_gemma26b/` | built | teacher cache for `kl_damage.py score` |

Note the 26b-a4b-4bit has no top-level symlink, only the HF cache tree;
`run_literary_bench.sh` resolves both layouts and requires actual
`*.safetensors` before accepting a directory.

After downloading, re-run the preflight against the bf16 dir — the
source-key checks (the `vq_397b_codes.py:156` template) stay skipped until
`model.safetensors.index.json` exists, and those are the ones that confirm
the re-targeted names against real tensors rather than against a reading of
the mlx_lm source.

## Pipeline order (mirrors the 397B lane)

1. **Structure base** — `./convert_gemma_struct.py --src <bf16 dir> --name struct8-e2`
   DONE. Recipe re-targeted (`switch_glu` not `switch_mlp`, `router.proj`
   not `mlp.gate`, q/k/v/o instead of Qwen3-Next `in_proj_*`) and VERIFIED:
   struct8-e8 reproduces uniform-q8 (441 vs 472 mnats) and edges it via the
   bf16 router. Use `--structure-bits 8 --qkv-bits 8`: at 4-bit qkv the
   artifact collapses to 45% agreement regardless of expert bits (E8's
   attention cliff, already known — EXPERIMENTS.md). Non-expert at full
   8-bit costs only 2.54G of the 8.4G budget.
2. **VQ fit** — `vq_397b_codes.py` **still needs the family re-target wired in.**
   Deliberately NOT edited: it is the script that produced the shipped 397B
   artifacts, and changing it silently risks that proven path. The three sites,
   with gemma values already derived in `gemma_preflight.py:FAMILY`:
   - `:99` `if "switch_mlp" not in name` → `switch_glu`
   - `:87-88` `PROJ` — unchanged for gemma (same fused stack, same axis)
   - `:156` `f"model.language_model.layers.{li}.mlp.experts.{key}"` →
     drop `.mlp` → `f"model.language_model.layers.{li}.experts.{key}"`
3. **Pack** — `pack_artifact.py`, subject to the `down_proj` decision above.
4. **Graft towers** — `graft_vision.py`, with `:38-39` extended to
   `audio_tower`/`embed_audio`/`embed_vision` and `:65` to check `audio_config`.
5. **Score** — `referee/score_streaming.py` ×2 corpora, plus the new
   `run_literary_bench.sh`.

## Ladder points to measure

The 397B tail knee (~tail30 of 60 layers) **does not transfer by ratio** to 30
layers — EXPERIMENTS.md:934-952. Re-ladder it. Suggested first three:

| name | schedule | predicted |
|---|---|---|
| `struct6-e2` | flat 2-bit experts | ~7 G |
| `struct6-tail10` | `0-19:2,20-29:3` | ~8.5 G |
| `struct6-tail20` | `0-9:2,10-29:3` | ~10 G |

Score each on wikitext + code + litbench before extending the ladder.

## Before any of it: the bf16 shootout

The whole premise — that a bigger gemma beats e4b for the jobs the sidecar
actually does — is untested. The clean test is **bf16 vs bf16**:

```bash
./run_literary_bench.sh          # awaiting Noah's all-clear; do not run early
```

At bf16 there is no quantization confound on either side, so the result
answers the capability question outright. Any community-quant comparison
would stay ambiguous between "the 26B is better" and "their quantizer was
kinder to it" — and mixing two publishers' quants is what the standing
methodology rule forbids anyway (`score_tasks_streaming.py:23-26`).

Memory is not the obstacle it looks like: the scorer streams one block at a
time, flat ~15 GB resident regardless of artifact size, so the 52G bf16 costs
no more than the 9G one.

This is the kill-shot. If 26b-a4b does not beat e4b on litbench **at full
fat**, no amount of quantization skill will rescue it downstream, and the
ladder is not worth the GPU. Only if it wins does the size-matched question
(can we hold that win at ~8-10G) become worth answering.

`INCLUDE_QUANTS=1 ./run_literary_bench.sh` adds the community quants as a
secondary sanity check that the bf16 ordering survives compression. Never
report those beside the bf16 pair.

Caveat on reading it: litbench at n=104 resolves ~10.5pp, enough to separate
these two models but **not** enough to separate two quants of one of them
(`literary/README.md`).
