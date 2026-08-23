"""Shared model-family registry — the single source of truth.

MoE/verify families (FAMILY): tensor-name templates and projection maps for
the MoE fitter and the outlier gate. Dense fitter families (DENSE_FAMILIES):
key template + default layer range for fit_dense_vq. Historically this dict
lived inline in vq_397b_codes.py, was text-scraped-and-exec'd by
verify_artifact.py, and was duplicated (with a diverged name) in
fit_dense_vq.py; one table, imported everywhere, ends that.
"""

FAMILY = {
    "qwen3_5": {
        "target_substr": "switch_mlp",
        # HF-layout source: gate and up live FUSED in one [E, 2I, H] stack,
        # taken as halves along the OUT axis.
        "src_key": "model.language_model.layers.{li}.mlp.experts.{key}",
        "proj": {"gate_proj": ("gate_up_proj", 0),
                 "up_proj": ("gate_up_proj", 1),
                 "down_proj": ("down_proj", None)},
    },
    "gemma4": {
        "target_substr": "switch_glu",
        # mlx-community's gemma bf16 is an MLX-FORMAT conversion, so
        # mlx_lm's sanitize has ALREADY split experts.gate_up_proj into
        # switch_glu.{gate,up}_proj (gemma4_text.py:625-634). There is no
        # fused stack in the checkpoint — verified: 'gate_up_proj' appears
        # in zero keys. So every projection is direct, no half-slicing, and
        # the prefix is language_model.model.* not model.language_model.*.
        "src_key": "language_model.model.layers.{li}.experts.switch_glu.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "gemma4_e4b": {
        # gemma-4-e4b-it: DENSE mlp (no experts), weights are 2D [OUT, IN].
        # Only used for VERIFICATION of e4b VQ artifacts (fit_e4b_vq.py) —
        # the main fitter's is_vq_target() wants a 2-bit-marked struct BASE,
        # which dense e4b builds do not have. Consumers must treat a 2D
        # source tensor as [1, OUT, IN] (E=1).
        "target_substr": ".mlp.",
        "src_key": "language_model.model.layers.{li}.mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "qwen3_8_dense": {
        # DENSE Qwen3.8-27B. Not an MoE: the mlp trio lives directly on the
        # layer with no .experts./.switch_mlp. segment, and each tensor is 2D
        # ([OUT, IN]) rather than [E, OUT, IN] — verify_artifact adds the E=1
        # axis for dense families. Source is the HF-format bf16 checkpoint,
        # whose prefix is model.language_model.* (the ARTIFACT uses
        # language_model.model.*; build_dense_vq.py owns that remap).
        "target_substr": "mlp",
        "src_key": "model.language_model.layers.{li}.mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
    "qwen3_5_mlx": {
        # SAME architecture as qwen3_5 (qwen3_5_moe, switch_mlp, shared
        # expert) but sourced from an mlx-community MLX-FORMAT bf16
        # conversion rather than the original HF-format one. Verified on
        # mlx-community/Qwen3.6-35B-A3B-bf16: sanitize already split
        # gate_up_proj (zero fused keys), prefix is language_model.model.*
        # (no leading "model."), and there is no .experts. path segment —
        # the module IS switch_mlp.{gate,up,down}_proj directly. Do not
        # point this at an HF-format source (use "qwen3_5" for that).
        "target_substr": "switch_mlp",
        "src_key": "language_model.model.layers.{li}.mlp.switch_mlp.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
                 "down_proj": ("down_proj", None)},
    },
}

# Dense fitter families: (src key template, default layer range "LO-HI").
DENSE_FAMILIES = {
    "gemma4_e4b": ("language_model.model.layers.{li}.mlp.{key}.weight", "0-41"),
    # named to match FAMILY's verify entry; the pre-release name was "qwen3_8"
    "qwen3_8_dense": ("model.language_model.layers.{li}.mlp.{key}.weight", "0-63"),
}
