#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Mixed-precision rungs for Qwen3.8-27B (dense, Qwen3-Next linear attention).

WHY MIXED AND NOT UNIFORM. Measured uniform baseline on this model
(kl_ppl_calibrate.py, referee wikitext, 2048 tok):

    bf16 55.6G  ppl 5.2249  1.000x
    q4   14.0G  ppl 5.2055  0.996x   <- free
    q3   11.0G  ppl 5.8323  1.116x   <- the target to beat
    q2    5.7G  ppl 16.4349 3.146x   <- cliff

E4 (EXPERIMENTS.md) found calibrated/mixed allocation beats uniform on DENSE
models at matched budget in exactly this steep zone. This script is how we
test that here: hold the byte budget at ~11G and spend the bits differently.

WHERE THE BYTES ARE (measured from safetensors headers, 27.78B total):
    mlp Linears          17.11B  61.6%   <- the bulk; crush this
    linear_attn Linears   5.56B  20.0%   <- expensive to protect
    embed + lm_head       2.54B   9.2%   <- untied, so 2x vocab
    full_attn Linears     1.68B   6.0%   <- only 16 of 64 layers
    vision tower          0.46B   1.7%   <- dropped by sanitize; graft later
    mtp head              0.43B   1.5%   <- speculative decoding; free bytes
    conv1d/A_log/dt_bias  ~0.00B  0.01%  <- keep bf16, negligible

NOTE THE ASYMMETRY VS THE MoE WORK. On 397B/gemma-26b the experts are ~90%
and structure is nearly free, so "protect attention, crush experts" costs
almost nothing. Here attention is 26% of the model, so protecting it at
6-bit is a real expense and the tradeoff is genuinely contested. That is
why this sweep exists rather than a single recipe.

E8's cliff warning still applies: at low uniform targets the allocator is
forced to drop attention to minimum bits, and attention starvation alone
reproduced the catastrophe. Every rung here keeps an explicit attention
floor for that reason.

Usage:
  ./convert_qwen38_mixed.py --name m2-a4 --mlp-bits 2 --linattn-bits 4 \
      --fullattn-bits 6 --embed-bits 4
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.8-27B")
ap.add_argument("--out-root", default="/Volumes/Thunderbay SSD/Exo Models/qwen38-27b-rungs")
ap.add_argument("--name", required=True)
ap.add_argument("--mlp-bits", type=int, default=3)
ap.add_argument("--mlp-group-size", type=int, default=64)
ap.add_argument("--linattn-bits", type=int, default=4)
ap.add_argument("--fullattn-bits", type=int, default=6)
ap.add_argument("--embed-bits", type=int, default=4)
ap.add_argument("--head-bits", type=int, default=6)
args = ap.parse_args()

OUT = f"{args.out_root}/{args.name}"
hits = {}


def predicate(path, module):
    # bf16: norms, the tiny linear-attention state projections, conv1d.
    # in_proj_a/in_proj_b are per-head scalars-ish (kept bf16 in the 35B
    # recipe too); conv1d is ndim=3 and only 2M params total.
    if ("norm" in path or path.endswith(("in_proj_a", "in_proj_b", "conv1d"))):
        hits["bf16"] = hits.get("bf16", 0) + 1
        return False

    if path.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
        b, gs = args.mlp_bits, args.mlp_group_size
    elif path.endswith(("in_proj_qkv", "in_proj_z", "out_proj")):
        b, gs = args.linattn_bits, 64
    elif path.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
        b, gs = args.fullattn_bits, 64          # E8 attention floor
    elif path.endswith("embed_tokens"):
        b, gs = args.embed_bits, 64
    elif path.endswith("lm_head"):
        b, gs = args.head_bits, 64
    else:
        b, gs = args.fullattn_bits, 64          # anything unclassified: safe side
        hits.setdefault("_unclassified", []).append(path)

    hits[f"{b}b"] = hits.get(f"{b}b", 0) + 1
    return {"group_size": gs, "bits": b, "mode": "affine"}


from mlx_lm.convert import convert

convert(args.src, mlx_path=OUT, quantize=True, q_group_size=64, q_bits=4,
        quant_predicate=predicate)
unc = hits.pop("_unclassified", [])
print("bit histogram:", hits)
if unc:
    print(f"UNCLASSIFIED ({len(unc)}) -> got {args.fullattn_bits}b:",
          sorted({p.rsplit('.', 1)[-1] for p in unc}))
print("done:", OUT)
