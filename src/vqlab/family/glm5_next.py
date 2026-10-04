"""glm5_next (GLM-5.3-Flash): the reference behaviours VQLab checks.

The architecture is knurlogic's (engine/families/glm5/architecture/glm5_next,
vendored from mlx-vlm 0.6.17), served as mlx_lm.models.glm5_next. The
reference is the maker's transformers code (5.18,
models/glm5_next/modeling_glm5_next.py, self-contained after modular
expansion) plus the maker checkpoint's config.json and stored dtypes.

Paths below: M = transformers/models/glm5_next/modeling_glm5_next.py,
K = knurlogic/engine/families/glm5/architecture/glm5_next/ (language.py
unless named), V = K/_mlx_vlm/models/. Lines are at transformers 5.18 and
knurlogic 6a2ab49. Every test is tests/test_glm5_next_parity.py; it
transcribes M into numpy and runs the knurlogic module on tiny CPU inputs.

Found at knurlogic 6a2ab49 (each test FAILED there and printed the
measurement, recorded in the item's `runtime`), all fixed in f03ec21 where
every test passes: the SwiGLU clamp at swiglu_limit=10 is missing in the dense MLP, the shared
expert and the routed experts; router logits are not fp32; the MLA latent
norms use eps 1e-6 (reference rms_norm_eps 1e-5) and the indexer k_norm
1e-5 (reference 1e-6); the indexer scores in the activation dtype
(reference fp32).
"""
from __future__ import annotations

from vqlab.family import FamilyPlugin, Parity

_T = "tests/test_glm5_next_parity.py::"

SPEC = FamilyPlugin(
    name="glm5_next",
    model_types=("glm5_next",),
    notes="GLM-5.3-Flash (KDA linear attention + DSA/NoPE-MLA, mHC streams, 288-expert MoE); "
          "architecture from knurlogic (stock mlx-lm has none)",
    reference="transformers 5.18 models/glm5_next/modeling_glm5_next.py; "
              "zai-org/GLM-5.3-Flash config.json + stored tensors",
    parity=(
        # -- SwiGLU clamp ------------------------------------------------
        Parity(
            name="dense_mlp_swiglu_clamp",
            reference="Glm5NextTextMLP clamps gate <= swiglu_limit and |up| <= swiglu_limit "
                      "(10) before silu(gate)*up, M:100-106; layers 0-2 are dense",
            runtime="6a2ab49: V/mlp.py:29 DeepseekMLP, no clamp (measured max rel error 1.71); "
                    "f03ec21: Glm5NextMLP clamps, test passes",
            test=_T + "test_dense_mlp_swiglu_clamp",
        ),
        Parity(
            name="shared_expert_swiglu_clamp",
            reference="the shared expert IS a Glm5NextTextMLP (M:197-199), so clamped",
            runtime="6a2ab49: V/deepseek_v32/language.py:300 DeepseekV32MLP, no clamp (measured 1.52); "
                    "f03ec21: Glm5NextMoE builds Glm5NextMLP, test passes",
            test=_T + "test_shared_expert_swiglu_clamp",
        ),
        Parity(
            name="routed_expert_swiglu_clamp",
            reference="Glm5NextTextExperts._apply_gate clamps the same way, M:138-143",
            runtime="6a2ab49: V/switch_layers.py:162 SwitchGLU default SwiGLU(), no clamp "
                    "(measured 1.26); f03ec21: _ClampedSwiGLU, test passes",
            test=_T + "test_routed_expert_swiglu_clamp",
        ),
        # -- routing -----------------------------------------------------
        Parity(
            name="router_semantics",
            reference="sigmoid scores; e_score_correction_bias added for the CHOICE only; "
                      "n_group/topk_group group mask; weights from unbiased scores, "
                      "norm_topk_prob (+1e-20), * routed_scaling_factor 2.5; M:146-184",
            runtime="V/deepseek_v32/language.py:226-283 group_expert_select (n_group>1 masks "
                    "with 0 instead of -inf; the checkpoint has n_group=1)",
            test=_T + "test_router_semantics_fp32",
        ),
        Parity(
            name="router_logits_fp32",
            reference="logits = F.linear(x.float(), W.float()), M:161 (config moe_router_dtype float32)",
            runtime="6a2ab49: V/deepseek_v32/language.py:276 x @ W.T in the activation dtype, "
                    "upcast after (bf16: max routing-weight error 2.4e-4, 0/256 set flips); "
                    "f03ec21: Glm5NextMoEGate fp32, test passes",
            test=_T + "test_router_logits_fp32",
        ),
        # -- norms -------------------------------------------------------
        Parity(
            name="mla_latent_norm_eps",
            reference="q_a_layernorm / kv_a_layernorm eps = rms_norm_eps (1e-5), M:1129, M:1137",
            runtime="6a2ab49: K:524, K:531 nn.RMSNorm(eps=1e-6) (0.47 rel error at row RMS 3e-3); "
                    "f03ec21: eps=config.rms_norm_eps, test passes",
            test=_T + "test_mla_latent_norm_eps",
        ),
        Parity(
            name="indexer_k_norm_eps",
            reference="indexer k_norm = nn.LayerNorm(head_dim, eps=1e-6), M:801",
            runtime="6a2ab49: K:256 nn.LayerNorm(head_dim) = mlx default eps 1e-5 (0.31 rel error "
                    "at RMS 3e-3); f03ec21: eps=1e-6, test passes",
            test=_T + "test_indexer_k_norm_eps",
        ),
        Parity(
            name="block_norms_eps",
            reference="input/post-attention/final RMSNorm, HC input norm (unweighted) and "
                      "KDA gated o_norm all at rms_norm_eps; M:1295-1296, M:1439, M:278, M:660",
            runtime="K:635-637, K:682, V/deepseek_v4/hyper_connection.py:225, K:136",
            test=_T + "test_block_norms_eps",
        ),
        # -- hyper-connections ------------------------------------------
        Parity(
            name="hyper_connection",
            reference="mHC: fp32 RMS-normed flattened streams @ fn; pre = sigmoid+hc_eps, "
                      "post = 2*sigmoid, comb = softmax+hc_eps then 20 Sinkhorn iterations; "
                      "collapse sum(pre*x); expand post*out + comb^T @ residual; "
                      "M:220-331, M:1337-1348",
            runtime="V/deepseek_v4/hyper_connection.py:184-254 (CPU ops path; the GPU Metal "
                    "kernel :9-180 uses fast::exp and is NOT exercised here)",
            test=_T + "test_hyper_connection_matches_reference",
        ),
        Parity(
            name="hc_head_mean",
            reference="final collapse is an UNWEIGHTED mean over the hc streams, then model.norm "
                      "(M:334-338, M:1511) -- not DeepSeek-V4's learned HyperHead",
            runtime="K:712-713 h.mean(axis=2) then norm",
            test=_T + "test_hc_head_is_unweighted_mean",
        ),
        # -- KDA linear attention ---------------------------------------
        Parity(
            name="kda_linear_attention",
            reference="q/k/v proj -> depthwise causal conv K=4 + silu (M:431-450) -> fp32 "
                      "l2norm(eps 1e-6) q,k, q*Dk^-0.5 (M:453-462, M:478-487) -> safe forget gate "
                      "lower_bound(-5)*sigmoid(exp(A_log)*(f_b(f_a(x))+dt_bias)) (M:341-371), "
                      "beta=sigmoid(b_proj) (M:735), delta rule (M:465-516; chunked M:519 is the "
                      "same function) -> gated RMSNorm with sigmoid(g_b(g_a(x))) (M:376-395)",
            runtime="K:109-239 + V/gated_delta.py (ops path on CPU; the Metal kernel is not "
                    "exercised), conv weights [C,1,K] fused by sanitize K:757-789",
            test=_T + "test_kda_linear_attention_matches_reference",
        ),
        # -- DSA indexer + NoPE MLA -------------------------------------
        Parity(
            name="dsa_indexer_selection",
            reference="k-pool compression (softmax over gate+ape per pool of 4), relu(q.k*hd^-0.5) "
                      "weighted by weights_proj*nH^-0.5, a pool selectable once its LAST token is "
                      "visible, top index_topk//kpool pools, visible incomplete tail appended; "
                      "M:774-1062",
            runtime="K:242-442 Glm5NextIndexer",
            test=_T + "test_indexer_selection_fp32",
        ),
        Parity(
            name="dsa_indexer_short_context_bypass",
            reference="for T <= index_topk the reference selects every visible token (= dense causal)",
            runtime="K:342 returns None (dense attention) while T <= index_topk",
            test=_T + "test_indexer_short_context_is_dense",
        ),
        Parity(
            name="dsa_indexer_fp32",
            reference="pool scores in fp32: q.float() @ pool_keys.float() (M:863), weights "
                      ".float() (M:867), pool softmax over fp32 logits (M:1000-1004)",
            runtime="6a2ab49: K:286-290, K:409-412 in the activation dtype (bf16: 2/288 query "
                    "rows select a different token set); f03ec21: fp32, 0/288, test passes",
            test=_T + "test_indexer_fp32_scoring",
        ),
        Parity(
            name="nope_mla_prefill",
            reference="NoPE MLA (qk_rope_head_dim=0: no rope in attention or indexer), scale "
                      "qk_head_dim^-0.5 (M:1149), kv_b_proj split k_nope|v per head, attention "
                      "masked to the indexer selection (M:1239-1277), fp32 softmax (M:1077-1099)",
            runtime="K:498-615 expanded path (L > SMALL_L); kv_b_proj -> embed_q/unembed_out "
                    "in V/deepseek_v32/language.py:479-513",
            test=_T + "test_dsa_mla_prefill_matches_reference",
        ),
        Parity(
            name="nope_mla_decode",
            reference="same function, row by row with a cache",
            runtime="K:569-612 absorbed path + K:460-494 gathered selection + incremental "
                    "pool cache K:366-391",
            test=_T + "test_dsa_mla_decode_matches_reference",
        ),
        Parity(
            name="decoder_layer_wiring",
            reference="hc collapse -> input_layernorm -> mixer -> hc expand -> hc collapse -> "
                      "post_attention_layernorm -> mlp -> hc expand; M:1280-1350",
            runtime="K:618-669",
            test=_T + "test_decoder_layer_wiring",
        ),
        Parity(
            name="checkpoint_config",
            reference="maker config.json: swiglu_limit 10, rms_norm_eps 1e-5, hc_eps 1e-6, "
                      "sigmoid/noaux_tc n_group 1, scaling 2.5, NoPE, KDA lower bound -5, every "
                      "indexer_types 'full' (shared-indexer layers M:1151 are not implemented by "
                      "the runtime), dense/MoE layer split; fp32 stored bias/A_log/dt_bias/hc base+scale",
            runtime="K/config.py TextConfig.from_dict; K:628-632",
            test=_T + "test_checkpoint_config_matches_runtime",
        ),
        # -- untested ----------------------------------------------------
        Parity(
            name="mtp_layer",
            reference="NONE in transformers: layers.45 (MTP) and shared_head are dropped at load "
                      "(M:1380). The checkpoint stores layer 45 as an MLA+indexer+MoE block with "
                      "eh_proj/enorm/hnorm/shared_head.norm and no hc_* tensors",
            runtime="trunk sanitize drops mtp. keys (K:749) and DSV32 drops layers >= 45; "
                    "knurlogic/engine/families/glm5/heads/glm5.py builds the draft head "
                    "(plain residual, no mHC). Affects draft acceptance, not trunk logits",
            required=False,
        ),
        Parity(
            name="hc_param_load_dtype",
            reference="hc_*_base/scale stored F32, hc_*_fn BF16; the reference keeps neither in "
                      "fp32 at a bf16 load (not in _keep_in_fp32_modules_strict, M:1379) but "
                      "casts fn to float in forward (M:305)",
            runtime="HyperConnection declares fp32 params; whether the loader casts them to bf16 "
                    "follows cast_predicate (K:797-801, excludes only e_score_correction_bias). "
                    "Unchecked against a real load",
            required=False,
        ),
        Parity(
            name="gpu_kernels",
            reference="the functions above",
            runtime="Metal kernels for HC sinkhorn+collapse, gated_delta and SwitchGLU gather "
                    "run only on the GPU; these CPU tests do not traverse them",
            required=False,
        ),
    ),
)
