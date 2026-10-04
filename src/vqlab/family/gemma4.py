"""gemma4 (Gemma 4 e4b and 26B-A4B, text + vision): the reference behaviours
VQLab checks.

Reference: transformers 5.18 `models/gemma4/modeling_gemma4.py` (HF:<line>)
plus the maker checkpoints' config.json. Runtime: knurlogic's vendored
`engine/families/gemma4/architecture/gemma4_text.py` (gt:<line>), served as
mlx_lm.models.gemma4_text by vqlab.family.arch, and its vision tower
`engine/families/gemma4/vision/{vision.py,__init__.py}`. Line numbers are
knurlogic 6a2ab49.

Not here: the FAMILY fit entry (core/families.py "gemma4" / "gemma4_e4b")
and audio (knurlogic serves no audio tower; the reference's audio path is
out of scope).
"""
from __future__ import annotations

from vqlab.family import FamilyPlugin, Parity

_T = "tests/test_gemma4_parity.py::"
_GT = "knurlogic gemma4/architecture/gemma4_text.py"
_V = "knurlogic gemma4/vision/vision.py"

SPEC = FamilyPlugin(
    name="gemma4",
    model_types=("gemma4", "gemma4_text"),
    notes="Gemma 4 e4b (dense, PLE, KV sharing) and 26B-A4B (MoE, K=V full layers); "
          "architecture + vision tower from knurlogic",
    reference="transformers 5.18 models/gemma4/modeling_gemma4.py (+ modeling_rope_utils.py "
              "proportional rope, masking_utils overlays, image_processing_gemma4.py patchify); "
              "maker config.json (mlx-community gemma-4-{e2b,e4b,26b-a4b}-it-bf16)",
    parity=(
        Parity("rmsnorm_plain_weight_eps",
               "Gemma4RMSNorm (HF:197-216): x * pow(mean(x^2)+eps, -0.5) * weight -- weight "
               "as-is, NOT 1+weight; eps = rms_norm_eps (1e-6); v_norm/router norm/"
               "embedder pre-norm without scale",
               f"nn.RMSNorm / mx.fast.rms_norm (plain weight) and RMSNormNoScale ({_GT}:73-81)",
               _T + "test_rmsnorm_plain_weight_and_eps"),
        Parity("embed_scale_sqrt_hidden",
               "Gemma4TextScaledWordEmbedding (HF:1444-1455, 1576): embed * sqrt(hidden), the "
               "scale cast to the weight dtype",
               f"h * embed_scale ({_GT}:402, 576-580; the python scalar takes the array dtype)",
               _T + "test_e4b_like_forward_matches_reference"),
        Parity("final_logit_softcap",
               "logits = tanh(logits / 30) * 30 (HF:1864-1867); no attention-logit softcap on "
               "text attention (eager_attention_forward is called without softcap, HF:1262-1272)",
               f"logit_softcap ({_GT}:84-86, 660-661); attention has none",
               _T + "test_e4b_like_forward_matches_reference"),
        Parity("attention_scale_one_qk_norm",
               "scaling = 1.0 (HF:1177): q_norm/k_norm (scaled RMSNorm) and v_norm (no scale) "
               "stand in for 1/sqrt(d) (HF:1192-1250)",
               f"Attention.scale = 1.0 ({_GT}:203), q/k/v norms ({_GT}:214-217, 241-263)",
               _T + "test_attention_scale_and_kv_layout"),
        Parity("layer_types_and_sliding_window",
               "layer_types from config (5 sliding : 1 full); sliding layers mask kv > q - "
               "sliding_window (masking_utils.sliding_window_overlay), full layers causal",
               f"layer_types read from config ({_GT}:64-70); create_attention_mask(window_size) "
               f"({_GT}:561-564)",
               _T + "test_text_only_masks_match_reference"),
        Parity("rope_per_layer_type",
               "full: proportional rope, theta 1e6, partial_rotary_factor 0.25 over "
               "global_head_dim (512), unrotated pairs get freq 0 (modeling_rope_utils.py:"
               "193-262); sliding: default rope, theta 1e4 over head_dim (256)",
               f"initialize_rope per type ({_GT}:219-229) -> mlx_lm rope_utils ProportionalRoPE "
               "(inf period on the unrotated pairs)",
               _T + "test_rope_per_layer_type"),
        Parity("k_eq_v_global_kv_heads",
               "26B: attention_k_eq_v on FULL layers -> no v_proj, V = raw k_proj output "
               "(v_norm, no rope), num_global_key_value_heads KV heads, global_head_dim "
               "(HF:1173-1209, 1243-1250)",
               f"use_k_eq_v / n_kv_heads ({_GT}:196-212, 251-263)",
               _T + "test_26b_like_forward_matches_reference"),
        Parity("mlp_gelu_tanh_double_wide",
               "down(gelu_pytorch_tanh(gate x) * up x) (HF:1061-1077); 2x intermediate on "
               "KV-shared layers when use_double_wide_mlp (false on e4b/26B)",
               f"geglu = nn.gelu_approx(gate) * x ({_GT}:94-114)",
               _T + "test_mlp_gelu_tanh"),
        Parity("moe_router",
               "26B: router on the PRE-norm residual: norm (no scale) * scale * H^-0.5 -> proj "
               "-> softmax over all experts -> top-8 -> renormalize -> * per_expert_scale "
               "(HF:1318-1352)",
               f"Router: rms_norm(x, scale * H^-0.5) -> argpartition top-k -> softmax over the "
               f"k -> * per_expert_scale ({_GT}:117-143); same selection and weights",
               _T + "test_router_topk_softmax"),
        Parity("moe_block_combination",
               "26B: h1 = post_ffn_ln_1(mlp(pre_ffn_ln(h))); h2 = post_ffn_ln_2(experts("
               "pre_ffn_ln_2(h))) with gelu-tanh GLU experts; post_ffn_ln(h1 + h2) + residual "
               "(HF:1406-1429, 1279-1316)",
               f"DecoderLayer.enable_moe branch ({_GT}:353-369), SwitchGLU(GeGLU)",
               _T + "test_26b_like_forward_matches_reference"),
        Parity("per_layer_embeddings",
               "e4b PLE: tok = embed_per_layer(ids) * sqrt(256); proj = rms(W h_scaled * H^-0.5); "
               "pli = (proj + tok) * 2^-0.5; per layer h += rms(W_p(gelu(W_g h) * pli_i)) "
               "(HF:1590-1604, 1755-1788, 1431-1438)",
               f"_get_per_layer_inputs / _project_per_layer_inputs ({_GT}:444-501), "
               f"DecoderLayer gate ({_GT}:371-384)",
               _T + "test_e4b_like_forward_matches_reference"),
        Parity("kv_sharing",
               "e4b: layers >= num_hidden_layers - num_kv_shared_layers (18) reuse the K/V of "
               "the LAST non-shared layer of the same type (HF:1182-1188, 1236-1253)",
               f"previous_kvs ({_GT}:433-442, 601-625)",
               _T + "test_kv_sharing_source_layers"),
        Parity("layer_scalar",
               "hidden_states *= layer_scalar at the end of every layer (HF:1367, 1440)",
               f"h * layer_scalar ({_GT}:386-387)",
               _T + "test_e4b_like_forward_matches_reference"),
        Parity("image_mask_26b_sliding_bidirectional",
               "26B (use_bidirectional_attention='vision') with images: FULL layers causal "
               "only; SLIDING layers AND(window, OR(causal, same-image-block)) "
               "(create_masks_for_vision_model HF:2080-2141, used at HF:2407-2420)",
               f"KNOWN BUG in 6a2ab49: _make_masks ({_GT}:503-566) ORs the block overlay into "
               "the FULL layers and leaves sliding layers causal. Measured on 12 tokens with "
               "images of 6 and 2 tokens: full layers 16 (q,k) pairs allowed that the reference "
               "masks, sliding layers 16 pairs masked that the reference allows. Image prompts only",
               _T + "test_image_mask_26b_bidirectional_on_sliding_layers"),
        Parity("image_mask_e4b_causal",
               "e4b (use_bidirectional_attention=None): image prompts use the plain causal / "
               "sliding-causal masks (HF:2408-2420, create_masks_for_generate)",
               f"KNOWN BUG in 6a2ab49: gemma4_text ModelArgs has no use_bidirectional_attention, "
               f"so the overlay ({_GT}:529-559) fires whenever the vision family passes mm_mask "
               "(vision/__init__.py:240-242): 16 extra (q,k) pairs on full layers in the "
               "same fixture. Image prompts only",
               _T + "test_image_mask_e4b_stays_causal"),
        Parity("vision_patchify_and_positions",
               "processor patchify (C,H,W)->(pH*pW, p*p*C) channel-last, (x,y) grid "
               "(image_processing_gemma4.py:88-98, 251-258); patch embed 2*(x-0.5) -> "
               "input_proj + x/y position tables (HF:579-621)",
               f"VisionPatchEmbedder._patchify / _position_embeddings, _patch_positions ({_V})",
               _T + "test_vision_tower_e4b_like_matches_reference"),
        Parity("vision_rope_2d",
               "axial 2D rope, theta 100, freqs over head_dim//2 laid out h,h,w,w, x part "
               "first (HF:707-770, 848-903)",
               f"apply_multidimensional_rope ({_V}:95-137)",
               _T + "test_vision_rope_2d"),
        Parity("vision_encoder_layer",
               "bidirectional attention at scaling 1.0 with q/k norm (scaled) and v norm "
               "(no scale), sandwich norms, gelu-tanh GLU MLP (HF:904-1015)",
               f"VisionAttention / VisionTransformerBlock / VisionMLP ({_V}:140-227)",
               _T + "test_vision_tower_e4b_like_matches_reference"),
        Parity("vision_clipped_linears",
               "e4b: use_clipped_linears=True -> clamp input and output of every tower linear "
               "to the checkpoint's input_/output_ min/max (HF:168-194)",
               f"ClippableLinear ({_V}:26-48); VisionConfig default False, but every "
               "gemma-4 config on the box sets the field explicitly",
               _T + "test_vision_tower_e4b_like_matches_reference"),
        Parity("vision_pool_and_standardize",
               "k*k position-average pool, * sqrt(vision hidden), 26B standardize "
               "(x - std_bias) * std_scale (HF:624-689, 2041-2048)",
               f"VisionPooler / VisionModel.__call__ ({_V})",
               _T + "test_vision_tower_26b_like_matches_reference"),
        Parity("vision_pool_precision",
               "pooler and standardize computed in float32, cast to the working dtype after "
               "(HF:684-689, 2044-2048)",
               "knurlogic casts the pooled average back to the activation dtype, then scales "
               "and standardizes in it (bf16 in serving). Precision only; not measured",
               required=False),
        Parity("multimodal_embedder",
               "embed_vision: RMSNorm without scale then Linear to text hidden (HF:2053-2077)",
               "MultimodalEmbedder (knurlogic gemma4/vision/__init__.py:51-65)",
               _T + "test_multimodal_embedder"),
        Parity("image_features_unscaled",
               "image features are scattered unscaled into the sqrt(H)-scaled text embeddings "
               "(HF:2306-2370)",
               "encode() divides by sqrt(H), the trunk multiplies everything by it "
               "(vision/__init__.py:202-213): exact in fp32, one extra rounding in bf16 "
               "(bounded < 1e-2 relative)",
               _T + "test_image_feature_prescale_roundtrip", required=False),
        Parity("ple_image_placeholder",
               "PLE for image positions looks up pad_token_id (HF:2311-2321)",
               "vision embed() zeroes sentinel ids (vision/__init__.py:235-238): equal while "
               "pad_token_id == 0, which every gemma-4 config on the box has",
               _T + "test_real_configs_read_as_reference"),
        Parity("config_fields_read",
               "layer_types, rope per type, softcap, eps, KV sharing, k_eq_v, MoE, PLE sizes, "
               "vision clip/standardize as the maker config.json states them",
               "gemma4.ModelArgs -> gemma4_text.ModelArgs.from_dict, VisionConfig.from_dict",
               _T + "test_real_configs_read_as_reference"),
        Parity("decode_cache_path",
               "incremental decode: rotating sliding cache, rope offset, shared-KV layers read "
               "the source layer's cached K/V (HF:1252-1255)",
               "make_cache / RotatingKVCache / offset handling (gemma4_text.py:256-269, "
               "601-625, 729-742): prefill 5 then decode 1 at a time past the window",
               _T + "test_cached_decode_matches_reference"),
    ),
)
