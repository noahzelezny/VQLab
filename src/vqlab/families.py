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
    "qwen4_exp": {
        # Qwen3.8-Flash-Next (180B, 10-of-512). VERIFIED IDENTICAL to
        # qwen3_5 on every axis this table encodes (2026-08-28, from the
        # checkpoint index + the mlx-lm PR #1788 module tree): same fused
        # [E, 2I, H] gate_up stack halved along OUT, same HF source key
        # template, same runtime member name (SparseMoeBlock.switch_mlp).
        # All 48 layers are MoE — no dense head despite the Qwen4 lineage.
        # NOT covered here: the per-layer ngram PLE tables (28.4% of
        # params, [*, 160] rows) — those take the PLE fitter, not this
        # one — and the per-layer shared_expert MLP, which stays in the
        # protected budget with attention and the router.
        "target_substr": "switch_mlp",
        "src_key": "model.language_model.layers.{li}.mlp.experts.{key}",
        "proj": {"gate_proj": ("gate_up_proj", 0),
                 "up_proj": ("gate_up_proj", 1),
                 "down_proj": ("down_proj", None)},
    },
    "glm5_next": {
        # GLM-5.3-Flash (320B, 8-of-288 routed + 1 shared, 43 MoE layers on
        # indices 3-45 — 42 main + the MTP layer 45, which carries its own
        # full expert stack; layers 0-2 are dense). MEASURED from the BF16
        # checkpoint's safetensors headers 2026-08-28 (see quantlab
        # GLM53_VQ_READINESS.md): experts are UNFUSED per-expert 2D tensors
        # — gate/up [2048, 4096], down [4096, 2048], no gate_up stack, no
        # [E, out, in] stack anywhere. The "{e}" in src_key marks that:
        # expert_src.load_expert_stack gathers experts.{0..E-1} into a
        # stack, discovering E from the index (288 measured; never trust
        # the config's count over the index).
        # target_substr is PROVISIONAL: it matches mlx-vlm's shipped
        # glm5_next (DeepseekV32MoE.switch_mlp = SwitchGLU, sanitize stacks
        # to mlp.switch_mlp.{key}.weight) and mlx-lm's own deepseek_v32
        # convention, but mlx-lm has no glm5_next class yet (checked
        # 2026-08-28) — re-verify the module name against the real class
        # before the first struct base is built.
        "target_substr": "switch_mlp",
        "src_key": "model.language_model.layers.{li}.mlp.experts.{e}.{key}.weight",
        "proj": {"gate_proj": ("gate_proj", None),
                 "up_proj": ("up_proj", None),
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

# Dense assembly remap: family -> (fitter module template, base module
# template). The fitter names modules with the SOURCE checkpoint's
# convention because that is what it read; the mlx base uses another. These
# are different strings and nothing downstream catches a mismatch except a
# strict load failure at score time, so build_dense_vq --dry-run asserts
# every fitted module lands on a real base module before bytes are written.
DENSE_REMAP = {
    "qwen3_8_dense": ("model.language_model.layers.{li}.mlp.{key}",
                      "language_model.model.layers.{li}.mlp.{key}"),
    "gemma4_e4b": ("language_model.model.layers.{li}.mlp.{key}",
                   "language_model.model.layers.{li}.mlp.{key}"),
}
