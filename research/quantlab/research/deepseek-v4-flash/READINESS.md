# DeepSeek-V4-Flash — VQ readiness pass (2026-09-05)

Status: PARKED behind the 397B v2 arc (Noah's call). This is the
investigation so the build is turnkey when the GPUs free.

## Source situation (verified against HF 2026-09-05)

- **No bf16 exists anywhere.** The chat release (`deepseek-ai/
  DeepSeek-V4-Flash`, 148.7 GiB — the model Noah swaps nightly, i.e. the
  official artifact itself) ships **fp4 experts** (`expert_dtype: "fp4"`)
  under an fp8 e4m3 block-quant config (128x128); `-0731` same;
  `-Base` (274.4 GiB) has **fp8 experts but is pretrain-only** — wrong
  weights for a chat rung, usable only for codebook-transfer experiments.
- Teacher = the fp4 chat release. Card framing when shipped: "compresses
  the official release's 4-bit experts to ~2-bit", KL vs ITS logits.

## Parameter accounting (from config.json)

256 routed experts x 3 mats x 4096x2048 = 6.44B/layer x ~44 MoE layers
~= 270B routed (~96% of ~280B total). MLA/compressed-attn + shared
expert + embeddings ~= 10B -> stays affine/bf16 (~18.6 GiB), VQ-for-MLA
NOT required to hit targets.

## Size targets

| experts recipe | total |
|---|---|
| d8/K16384 (1.75b, Flash-2.1 recipe) | ~74 GiB |
| d4/K256 (2.0b) | ~82 GiB |
| + leverage-mix budget to ~100 GiB | ~15-25 GiB of promotions |

Sub-100 GiB with a real promotion budget is comfortably reachable
(nightly swap 145 -> ~95-100).

## Layout (verified from the shipped index, 69187 tensors)

`layers.{li}.ffn.experts.{e}.{w1,w2,w3}.{weight,scale}` — UNFUSED
per-expert 2D, the glm5_next pattern. Family entry needed:

    "deepseek_v4": {
        "src_key": "layers.{li}.ffn.experts.{e}.{key}.weight",
        "proj": {"gate_proj": ("w1", ...), "up_proj": ("w3", ...),
                 "down_proj": ("w2", ...)},   # DeepSeek naming
        ...
    }

## The actual new work (bounded)

1. **fp4 dequant loader** in expert_src: each expert weight is fp4 with a
   `.scale` sibling — load_expert_stack must dequant to fp32 before the
   fit sees it. [TO VERIFY: exact fp4 packing (e2m1? nibble order) and
   scale granularity from a shard header — read one expert tensor.]
2. Family entry above + w1/w3/w2 mapping.
3. Runtime: which mlx class serves deepseek_v4 (mlx-lm? exo fork?) —
   determines the model.py bundle story. [TO VERIFY]
4. Everything else (fit math, packing, referee, release gates) is stock.

Disk: chat release already local (M4 nightly) + fits well under
Thunderbay free. No TB-class anything.
