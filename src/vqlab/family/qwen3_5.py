"""qwen3_5 / qwen3_5_moe (Qwen3.5-397B-A17B, Qwen3.6-35B-A3B, Qwen3.8-27B,
Qwen3.5-2B): the reference behaviours VQLab checks.

Hybrid stack: three gated delta-net (linear attention) layers to every
gated full-attention layer; the MoE variant swaps the dense MLP for a
softmax top-k router with a sigmoid-gated shared expert. The architecture
is knurlogic's qwen3_5.py (served as mlx_lm.models.qwen3_5 by `import
vqlab`), which imports its attention, MLP, gated norm and MoE block from
mlx_lm.models.qwen3_next and the recurrence from mlx_lm.models.gated_delta.

Reference paths below: Q35 = transformers 5.18
models/qwen3_5/modeling_qwen3_5.py, Q35M = models/qwen3_5_moe/
modeling_qwen3_5_moe.py. Runtime: K = knurlogic engine/families/qwen/
architecture/qwen3_5.py (commit 6a2ab49), KM = its qwen3_5_moe.py, QN =
mlx_lm/models/qwen3_next.py, GD = mlx_lm/models/gated_delta.py.

Known bug (2026-10-03): the q/k l2norm inside the delta-net uses an
effective eps of Dk * 1e-6 (128x the reference's at Dk=128);
tests/test_qwen3_5_parity.py::test_gdn_l2norm_eps fails until knurlogic's
family-audit fix is pinned.
"""
from __future__ import annotations

from vqlab.family import FamilyPlugin, Parity

_T = "tests/test_qwen3_5_parity.py::"

SPEC = FamilyPlugin(
    name="qwen3_5",
    model_types=("qwen3_5", "qwen3_5_moe"),
    notes="Qwen3.5/3.6/3.8 hybrid gated-delta-net + gated attention, dense and MoE; "
          "architecture from knurlogic",
    reference="transformers 5.18 models/qwen3_5/modeling_qwen3_5.py + "
              "models/qwen3_5_moe/modeling_qwen3_5_moe.py; the maker checkpoints' "
              "config.json and stored tensor layout (Qwen/Qwen3.5-2B, Qwen/Qwen3.6-35B-A3B)",
    parity=(
        Parity(
            name="rmsnorm_zero_centred",
            reference="Q35:840-854 Qwen3_5RMSNorm: x*rsqrt(mean(x^2)+eps)*(1+w), eps=rms_norm_eps "
                      "(1e-6); input/post-attention, final, q_norm, k_norm",
            runtime="K:488-511 sanitize adds 1.0 to those norm weights (only when the checkpoint's "
                    "conv1d is unsanitized [C,1,K]), then nn.RMSNorm(w, eps); the gated norm is "
                    "NOT shifted; mtp.* dropped",
            test=_T + "test_rmsnorm_zero_centred",
        ),
        Parity(
            name="gated_rmsnorm",
            reference="Q35:217-233 Qwen3_5RMSNormGated: plain weight (ones-init), norm before gate, "
                      "* silu(z) in fp32",
            runtime="QN:58-71 Qwen3NextRMSNormGated: rms_norm(x, w, eps) then precise_swiglu",
            test=_T + "test_gated_rmsnorm",
        ),
        Parity(
            name="gdn_causal_conv",
            reference="Q35:268-288 causal_conv1d_fn: depthwise, left pad K-1, silu; weight [C,1,K]",
            runtime="K:276-287 zero conv_state of K-1 rows concatenated, nn.Conv1d(groups=C, "
                    "padding=0), silu; sanitize moveaxis(2,1) of the weight",
            test=_T + "test_gdn_causal_conv",
        ),
        Parity(
            name="gdn_l2norm_eps",
            reference="Q35:293-296 l2norm x*rsqrt(sum(x^2)+1e-6) on q and k (Q35:343-344, :463-464), "
                      "then q /= sqrt(Dk) (Q35:467)",
            runtime="K:299-301 q = Dk^-1 * rms_norm(q, 1e-6), k = Dk^-0.5 * rms_norm(k, 1e-6) == "
                    "x*rsqrt(sum(x^2) + Dk*1e-6): eps 128x too large at Dk=128. KNOWN BUG on "
                    "knurlogic 6a2ab49 (test fails); fixed by the family-audit pin",
            test=_T + "test_gdn_l2norm_eps",
        ),
        Parity(
            name="gdn_gating_and_recurrence",
            reference="Q35:548-662 + Q35:437-497: g=-exp(A_log)*softplus(a+dt_bias), "
                      "beta=sigmoid(b), q/k repeat_interleave to Hv heads, state*=exp(g), "
                      "delta=(v-S^T k)*beta, S+=k delta^T, o=S^T q; gated norm with z; out_proj",
            runtime="K:262-326 + GD:19-21 compute_g, GD:605-650 gated_delta_update, "
                    "GD:385-426 step ops (mx.repeat interleaves heads)",
            test=_T + "test_gdn_forward",
        ),
        Parity(
            name="gdn_metal_kernel",
            reference="as gdn_gating_and_recurrence",
            runtime="GD:545-555 gated_delta_kernel (Metal) is what runs on the GPU; the parity tests "
                    "run on the CPU stream and reach only the ops path GD:557-602. The kernel is "
                    "not compared with the reference here",
            test="tests/test_qwen3_5_parity.py::test_gdn_metal_kernel_matches_ops",
        ),
        Parity(
            name="full_attention",
            reference="Q35:748-820: q_proj per head [q | gate], (1+w) q/k norm, partial rotary "
                      "(head_dim*partial_rotary_factor dims, rotate_half), rope_theta from "
                      "rope_parameters, GQA, scale head_dim^-0.5, out * sigmoid(gate), o_proj",
            runtime="QN:74-151 Qwen3NextAttention (nn.RoPE dims=head_dim*partial, traditional=False, "
                    "base=rope_theta via K:70-85 __post_init__)",
            test=_T + "test_full_attention",
        ),
        Parity(
            name="mrope_sections",
            reference="Q35:142-212 interleaved MRoPE: freq i on axis h if i%3==1 and i<3*sec[1], "
                      "w if i%3==2 and i<3*sec[2], else t (text: all three axes equal)",
            runtime="K:112-144 mrope_selector / apply_mrope (image chunks); text uses 1-D nn.RoPE",
            test=_T + "test_mrope_sections",
        ),
        Parity(
            name="dense_mlp",
            reference="Q35:823-836 down(silu(gate(x)) * up(x))",
            runtime="QN:154-162 Qwen3NextMLP (swiglu)",
            test=_T + "test_dense_mlp",
        ),
        Parity(
            name="moe_router_and_shared_expert",
            reference="Q35M:884-923: softmax (fp32) router, top-k, top-k weights ALWAYS renormalised "
                      "(no norm_topk_prob switch), experts.gate_up_proj chunked gate-first, "
                      "shared expert * sigmoid(shared_expert_gate(x))",
            runtime="QN Qwen3NextSparseMoeBlock (softmax precise, argpartition top-k, renormalised "
                    "when norm_topk_prob, which K:53 defaults True and the maker configs omit); "
                    "KM:23-52 sanitize splits gate_up_proj [..., :I, :] = gate",
            test=_T + "test_moe_block",
        ),
        Parity(
            name="decoder_and_head",
            reference="Q35:860-913 pre-norm decoder, two residuals; Q35:1221-1300 final (1+w) norm; "
                      "lm_head tied to embed_tokens when tie_word_embeddings",
            runtime="K:329-367 DecoderLayer, K:369-499 text model, K:524-527 tied head",
            test=_T + "test_full_forward",
        ),
        Parity(
            name="cached_decode",
            reference="Q35:567-603 cached conv state / recurrent state; KV cache with rope offset",
            runtime="K:268-326 ArraysCache conv_state (last K-1 rows) + state; QN:138-141 KVCache "
                    "offset rope",
            test=_T + "test_cached_decode",
        ),
        Parity(
            name="maker_config_read",
            reference="maker config.json: layer_types, rope_parameters (theta 1e7, partial 0.25, "
                      "mrope_section [11,11,10]), rms_norm_eps, tie, num_experts / top-k",
            runtime="K:26-86 TextModelArgs; K:332 is_linear = (i+1) % full_attention_interval != 0 "
                    "(layer_types itself is not read)",
            test=_T + "test_maker_config_read",
        ),
        Parity(
            name="maker_stored_layout",
            reference="maker safetensors: conv1d [C,1,K], fused experts [E,2I,D] / [E,D,I]",
            runtime="K:488-511 / KM:23-52 sanitize converts exactly this layout; the +1.0 norm "
                    "shift keys off the [C,1,K] conv shape",
            test=_T + "test_maker_stored_layout",
        ),
        Parity(
            name="mtp_head",
            reference="the checkpoints ship mtp.* (fc, pre_fc_norm_embedding/hidden, one layer); "
                      "transformers ignores them (Q35:924, Q35M:1013 _keys_to_ignore_on_load_"
                      "unexpected) and has no MTP forward to compare against",
            runtime="K:492 sanitize drops every mtp.* key; greedy text never runs the head",
            test="",
            required=False,
        ),
    ),
)
