#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Structure quantizer for gemma-4 MoE — the struct6 recipe, re-targeted.

Forked from convert_35b_struct.py. The RECIPE is unchanged and deliberately
so: experts low, structure 6-bit, routers/norms bf16. That shape is the one
thing the 397B ladder established as family-independent (EXPERIMENTS.md
headline 2, and E8's "the cliff is attention, not total bits"). What changes
here is only the NAMES it matches, because gemma is not Qwen3-Next:

  - Qwen's `mlp.switch_mlp.*` experts   -> gemma's `experts.switch_glu.*`
    (mlx_lm gemma4_text.py:625-634 renames experts.gate_up_proj into
     switch_glu.{gate,up}_proj during sanitize)
  - Qwen's `mlp.gate` router            -> gemma's `router.proj`
    (+ `router.per_expert_scale`, a [num_experts] vector — tiny, keep bf16)
  - Qwen3-Next linear attention (`in_proj_qkv`, `in_proj_a/b`, `in_proj_z`)
    HAS NO GEMMA ANALOGUE. Gemma uses standard q/k/v/o attention, so that
    branch is replaced by an explicit qkvo rule rather than silently
    falling through to the 6-bit structure default.

NOTE ON THE UPSTREAM PREDICATE. mlx_lm's gemma4_text.py:641-646 already
ships a quant_predicate that puts `router.proj` at 8-bit/gs64. We override
to bf16: the router is ~128*2816 params per layer (~10 MB total across 30
layers) and E7/E8 showed routing damage is the expensive kind. Buying that
back at bf16 costs almost nothing.

Usage:
  ./convert_gemma_struct.py --src <bf16 or 6bit dir> --name struct6-e2
  ./convert_gemma_struct.py --src <dir> --name struct6-tail --expert-schedule '0-19:2,20-29:3'
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out-root", default="/Volumes/Thunderbay SSD/Exo Models")
ap.add_argument("--expert-bits", type=int, default=2)
ap.add_argument("--expert-group-size", type=int, default=64,
                help="group size for the low-bit expert region only (E33: "
                     "finer scales pay in proportion to remaining 2-bit loss)")
ap.add_argument("--structure-bits", type=int, default=6)
ap.add_argument("--qkv-bits", type=int, default=4)
ap.add_argument("--expert-schedule", default=None,
                help="per-layer expert bits, e.g. '0-19:2,20-29:3'. "
                     "Inclusive ranges; overrides --expert-bits where it hits.")
args = ap.parse_args()

OUT = f"{args.out_root}/rotlab-gemma26B-{args.name}"

# --- expert schedule ----------------------------------------------------
# Same grammar as convert_variant.py:89-106, validated eagerly: a typo here
# costs a full conversion to discover.
SCHEDULE = {}
if args.expert_schedule:
    for part in args.expert_schedule.split(","):
        rng, _, bits = part.strip().partition(":")
        lo, _, hi = rng.partition("-")
        lo, hi, bits = int(lo), int(hi or lo), int(bits)
        for li in range(lo, hi + 1):
            if li in SCHEDULE:
                raise SystemExit(f"--expert-schedule: layer {li} listed twice")
            SCHEDULE[li] = bits

hits = {}


def layer_of(path):
    try:
        return int(path.split("layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def predicate(path, module):
    # bf16: routers, per-expert scales, norms, anything tiny. Gemma's router
    # is `router.proj`; `per_expert_scale` is a bare parameter, not a module,
    # but match it defensively in case mlx_lm wraps it later.
    if path.endswith(("router.proj", "router.per_expert_scale")) \
            or "norm" in path:
        b = None
    elif "switch_glu" in path or ".experts" in path:
        eb = SCHEDULE.get(layer_of(path), args.expert_bits)
        if eb == 0:
            # M0-style quality proxy: leave experts bf16 so the VALUES are
            # the VQ reconstruction. Bytes computed analytically, not stored.
            hits["expert_bf16"] = hits.get("expert_bf16", 0) + 1
            return False
        hits["gs"] = args.expert_group_size
        hits[f"e{eb}"] = hits.get(f"e{eb}", 0) + 1
        return {"group_size": args.expert_group_size, "bits": eb,
                "mode": "affine"}
    elif path.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
        b = args.qkv_bits
    else:
        b = args.structure_bits  # embed, dense MLP, lm_head, altup/laurel
    hits[b] = hits.get(b, 0) + 1
    if b is None:
        return False
    return {"group_size": 64, "bits": b, "mode": "affine"}


from mlx_lm.convert import convert

convert(args.src, mlx_path=OUT, quantize=True, q_group_size=64, q_bits=2,
        quant_predicate=predicate)
print("bit histogram:", hits)
print("expert schedule:", SCHEDULE or f"flat {args.expert_bits}-bit")
print("done:", OUT)
