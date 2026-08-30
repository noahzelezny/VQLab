"""Generate the four Flash-Next VQ release cards from measured numbers.

Single source of truth for the card tables; regenerate after any rescore.
PENDING placeholders are caught by check_card_placeholders.sh — a card
with one must never be pushed.
"""
import pathlib

AFFINE_ROWS = [
    ("affine q3 (ours)", "75 GiB", "1083.4", "61.9%", "12.850"),
    ("affine q4 (ours)", "96 GiB", "293.9", "79.6%", "6.453"),
    ("affine q5 (ours)", "116 GiB", "91.7", "87.5%", "5.243"),
    ("affine q6 (ours)", "137 GiB", "52.8", "91.6%", "4.916"),
    ("affine q8 (ours)", "178 GiB", "27.1", "94.9%", "5.197"),
]

MODELS = [
    dict(bpw="2.1", size="45.0 GiB", kl="390.1", top1="78.8%", ppl="5.903",
         code="2.076", lit="8.945", fits="64 GB machines",
         art="qwen4exp_vq_packed_mixL01",
         geometry=("MoE experts at d=8/K=16384 (14-bit codes, padded-tail "
                   "packed), PLE n-gram tables at d=8/K=256 (8-bit rows), "
                   "and layers 0–1 upgraded to d=2/K=256 experts (the "
                   "leverage mix — see below)."),
         insert_after="affine q3 (ours)"),
    dict(bpw="3.2", size="69.4 GiB", kl="123.5", top1="87.0%", ppl="5.211",
         code="1.939", lit="7.823", fits="96 GB machines",
         art="qwen4exp_vq_packed_31mix6",
         geometry=("MoE experts at d=4/K=2048, PLE at d=4/K=2048, and the "
                   "six highest-leverage layers (0, 1, 31, 35, 36, 39) "
                   "upgraded to d=2/K=256 experts."),
         insert_after="affine q3 (ours)"),
    dict(bpw="4.4", size="94.1 GiB", kl="50.3", top1="92.8%", ppl="5.223",
         code="1.916", lit="7.698", fits="128 GB machines",
         art="qwen4exp_vq_packed_92mix6",
         geometry=("MoE experts at d=2/K=256, PLE at d=8/K=4096, and the "
                   "six highest-leverage layers upgraded to d=2/K=1024 "
                   "(10-bit packed) experts."),
         insert_after="affine q4 (ours)"),
    dict(bpw="5.5", size="111.6 GiB", kl="34.1", top1="94.1%", ppl="5.245",
         code="1.898", lit="7.636", fits="128 GB machines (tight)",
         art="qwen4exp_vq_packed_d2k1024",
         geometry=("MoE experts at d=2/K=1024 (10-bit packed rows), PLE at "
                   "d=8/K=4096. Flat allocation — at this size the leverage "
                   "mix measured as a wash and is not shipped."),
         insert_after="affine q5 (ours)"),
]

TEMPLATE = """---
language:
- en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- qwen3.8
---

# TheDrainFlorist/Qwen3.8-Flash-Next-VQ-{bpw}bpw

**{size} — a 335 GiB frontier MoE on {fits}.**

A data-free vector-quantized build of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(180B total / 10-of-512 active, 51.2B n-gram PLE, vision) for Apple
Silicon. Stock `mlx-lm`, no patches — the VQ runtime ships inside the
checkpoint as `model.py`. Built with [VQLab](https://github.com/noahzelezny/VQLab).

{geometry}

The affine builds compared against below are our own conversions made with
the same tooling, scored on the same instrument.

![where these releases sit](chart_ladder.png)

## Measured results

Prose referee, 2048 tokens; KL against the bf16 teacher's cached top-64
(captured mass 0.963 for every row — same cache, same positions). All
sizes include the 333-tensor bf16 vision tower (0.84 GiB).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | perplexity |
|---|---|---|---|---|
{table}
| bf16 teacher | 335 GiB | 0 | 100% | 5.166 |

Additional corpora (perplexity): code {code} (public mlx corpus,
pinned manifest), literary {lit} (Gutenberg). Teacher reads 1.902 / 7.664.

**Rank these by KL, not perplexity.** Perplexity is an aggregate over
finite text and absorbs offsetting errors; KL measures distance to the
teacher's distribution directly. Several rungs here read within noise of
the teacher on perplexity while differing by an order of magnitude in KL.

## The leverage mix

Quantization damage is not uniform across layers. A one-pass probe
(teacher and student streamed together, per-layer local damage measured
with no compounding) shows the same hot set on every rung of this family:
layer 1 dominates, a late band (31–39) follows, and the map is identical
across geometries (rank correlation 0.905, identical top-10). Upgrading
only those layers buys 15–24% KL for 3–4% size on the lower rungs; the
probe, the mixing, and the verdicts are all reproducible with VQLab
(`vqlab layer-leverage`, scatter fits via `fit-moe --vq-layers`).

## Provenance and gates

Fitted data-free from the bf16 checkpoint (k-means / Lloyd on weights,
seed 1234, recipes in the VQLab repo). Release gates passed on this
artifact: file/index/tokenizer checks, bundle-runtime verbatim match, and
a generation smoke through the shipping runtime on Apple Silicon.
exo-ready: config carries vision_config + image_token_id; the vision
tower is grafted bf16. The teacher's MTP (multi-token-prediction) head is
not included — no MLX runtime implements MTP decoding; standard decoding
is unaffected.

Local artifact: `{art}`.
"""

def rows_for(m):
    rows = []
    for r in AFFINE_ROWS:
        rows.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
        if r[0] == m["insert_after"]:
            rows.append(f"| **this model** | **{m['size']}** | "
                        f"**{m['kl']}** | **{m['top1']}** | **{m['ppl']}** |")
    return "\n".join(rows)

for m in MODELS:
    name = f"MODEL_CARD_FLASHNEXT_{m['bpw'].replace('.', '_')}bpw.md"
    body = TEMPLATE.format(table=rows_for(m), **{k: v for k, v in m.items()
                                                 if k != "insert_after"})
    pathlib.Path(name).write_text(body)
    print("wrote", name, "(PENDING)" if "PENDING" in body else "")
