#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Rung-2 quantizer: 35B MoE at the 397B struct6 recipe, from any bf16 src.

Same predicate for baseline and rotated sources — rotation must be the ONLY
variable. Recipe mirrors the 397B binding constraint: experts 2-bit,
structure 6-bit, linear_attn in_proj_qkv/z 4-bit, routers/gates/a/b bf16.

Usage: ./convert_35b_struct.py --src <bf16 dir> --name <suffix>
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--expert-bits", type=int, default=2)
ap.add_argument("--expert-group-size", type=int, default=64,
                help="group size for the 2-bit expert region only (E33: "
                     "finer scales; gs32 costs ~6%% more bytes there)")
args = ap.parse_args()

OUT = f"/Volumes/Thunderbay SSD/Exo Models/rotlab-35B-{args.name}"

hits = {}


def predicate(path, module):
    import mlx.nn as nn
    from mlx.nn.layers.quantized import QuantizedEmbedding  # noqa
    # bf16: routers, gates, a/b projections, norms, anything tiny
    if path.endswith(("mlp.gate", "shared_expert_gate", "in_proj_a",
                      "in_proj_b")) or "norm" in path:
        b = None
    elif "switch_mlp" in path or "experts" in path:
        if args.expert_bits == 0:
            # E35 M0: leave experts bf16 — the VALUES are the VQ
            # reconstruction (or the bf16 control). Quality proxy only:
            # bytes are computed analytically, not stored.
            hits["expert_bf16"] = hits.get("expert_bf16", 0) + 1
            return False
        hits["gs"] = args.expert_group_size
        hits[args.expert_bits] = hits.get(args.expert_bits, 0) + 1
        return {"group_size": args.expert_group_size,
                "bits": args.expert_bits, "mode": "affine"}
    elif path.endswith(("in_proj_qkv", "in_proj_z")):
        b = 4
    else:
        b = 6  # structure: embed, q/k/v/o, out_proj, shared_expert, lm_head
    hits[b] = hits.get(b, 0) + 1
    if b is None:
        return False
    return {"group_size": 64, "bits": b, "mode": "affine"}


from mlx_lm.convert import convert

convert(args.src, mlx_path=OUT, quantize=True, q_group_size=64, q_bits=2,
        quant_predicate=predicate)
print("bit histogram:", hits)
print("done:", OUT)
