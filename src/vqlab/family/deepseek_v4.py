"""DeepSeek-V4-Flash (deepseek_v4): the family as one plugin.

Fit layout (official FP4 release read natively), the streamed KL/ppl scorer
(rule-5 validated), and the tokenizer registration check-release needs until
transformers registers model_type deepseek_v4 (huggingface/transformers#45616).
Moved here from core/families.py, score/stream_score.py and
gate/check_release.py on 2026-10-02; the scorer body is unchanged.
"""
from __future__ import annotations

import gc
import time

import mlx.core as mx

from vqlab.family import FamilyPlugin, Parity

FIT = {
        # DeepSeek-V4-Flash (~280B, 6-of-256 routed + 1 shared, 43 layers).
        # MEASURED from the official release's safetensors headers: experts
        # are UNFUSED per-expert 2D tensors `layers.{li}.ffn.experts.{e}.
        # {w1,w2,w3}.weight` (I8 = two FP4 e2m1 codes per byte) with an
        # F8_E8M0 `.scale` sibling, one scale per 32 inputs -- the mxfp4
        # layout. No bf16 release exists; the FP4 chat release IS the
        # teacher. expert_src dequantizes it through mx.dequantize(mode=
        # "mxfp4"), which is exact (an e2m1 value times a power of two is
        # representable in bf16). w1 = gate, w3 = up, w2 = down. The runtime
        # module (mlx-lm deepseek_v4) is ffn.switch_mlp.
        "runtime": "mlx_lm",
        "model_type": "deepseek_v4",
        "target_substr": "switch_mlp",
        "src_key": "layers.{li}.ffn.experts.{e}.{key}.weight",
        "src_quant": "mxfp4",
        "proj": {"gate_proj": ("w1", None),
                 "up_proj": ("w3", None),
                 "down_proj": ("w2", None)},
}


def score_deepseek_v4(model, ids_list, args):
    """Streamed deepseek_v4 scorer (DeepSeek-V4-Flash). Rule 5: see SCORERS.

    LINE-MIRRORED against mlx_lm.models.deepseek_v4 (DeepseekV4Model.__call__
    / Model.__call__), in the reference's order:
      - embed_tokens, then BROADCAST h to (B, S, hc_mult, D) + mx.contiguous
        (hyper-connection bookend #1; the hc mixing itself lives inside each
        block's hc_attn / hc_ffn).
      - block signature is (h, cache, input_ids): the first num_hash_layers
        route experts by a hash of the TOKEN IDS, so each chunk passes its
        own ids slice -- a hidden-state-only loop would silently mis-route
        those layers.
      - after the stack: model.hc_head(h) (bookend #2: collapses hc), THEN
        model.norm, then lm_head (never tied on this family).
    CHUNKED with each layer's own DeepseekV4Cache: the attention keeps a
    sliding window plus compressed / indexed long-range state, so the metric
    is chunk-dependent and must be measured the way generation computes it.
    hc_head and norm are per-position, so applying them per chunk is exact.

    Validated (rule 5) streamed vs a direct resident forward that walks the
    same chunks with one shared cache list: bitwise at 2048 and 12288.
    """
    lm = getattr(model, "language_model", model)
    core = lm.model
    _apply_variant(core)
    ids = mx.array([ids_list[:-1]])
    S = ids.shape[1]
    C = max(1, int(getattr(args, "chunk", 512) or 512))
    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    h = core.embed_tokens(ids)
    h = mx.contiguous(mx.broadcast_to(
        h[:, :, None, :], (*h.shape[:2], core.args.hc_mult, h.shape[-1])))
    mx.eval(h)

    caches = lm.make_cache()               # before any layer is dropped
    n = len(core.layers)
    for i in range(n):
        blk = core.layers[i]
        with mx.stream(mx.cpu):
            _, skipped = _budgeted(
                blk, getattr(args, "lazy_over_gb", 8.0))
        if skipped:
            print(f"  layer {i}: {skipped / 1024 ** 3:.1f} GiB left lazy",
                  flush=True)
        t0 = time.time()
        c = caches[i]
        parts = []
        for s0 in range(0, S, C):
            e0 = min(s0 + C, S)
            parts.append(blk(h[:, s0:e0], c, ids[:, s0:e0]))
        h = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        mx.eval(h)
        core.layers[i] = None
        caches[i] = None
        del blk, parts
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1} {time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)

    with mx.stream(mx.cpu):
        mx.eval(core.hc_head.parameters(), core.norm.parameters(),
                lm.lm_head.parameters())
    lg = []
    for s0 in range(0, S, C):
        o = core.norm(core.hc_head(h[:, s0:min(s0 + C, S)]))
        lg.append(lm.lm_head(o).astype(mx.float32)[0])
        mx.eval(lg[-1])
    return mx.concatenate(lg, axis=0) if len(lg) > 1 else lg[0]


def _variant():
    """The scorer's numerics variant, stamped into every cache and result.
    DEFAULT: DeepSeek's reference numerics, which apply the SwiGLU clamp
    (limit = swiglu_limit, 10) to the SHARED expert too. mlx-lm's
    deepseek_v4 builds it unclamped (F195: the clamp fires on ~0.026% of
    shared activations and moves that expert's output 2% on average);
    VQLAB_DS4_SHARED_CLAMP=0 opts into mlx-lm's variant, for comparison only."""
    import os
    return None if os.environ.get("VQLAB_DS4_SHARED_CLAMP") == "0" else "shared-clamp"


def _apply_variant(core):
    if _variant() == "shared-clamp":
        lim = float(getattr(core.args, "swiglu_limit", 10.0) or 10.0)
        n = 0
        for blk in core.layers:
            sh = getattr(getattr(blk, "ffn", None), "shared_experts", None)
            if sh is not None and hasattr(sh, "swiglu_limit"):
                sh.swiglu_limit = lim
                n += 1
        print(f"  [deepseek_v4] shared-expert SwiGLU clamped at {lim:g} on {n} layers "
              f"(reference parity, F195)", flush=True)


def _budgeted(blk, lazy_over_gb):
    from vqlab.core.mem_budget import eval_params_budgeted
    return eval_params_budgeted(blk, lazy_over_gb)


def _register_tokenizer(model_type: str) -> None:
    # Knurlogic's vendored architecture registers the missing HF config at
    # import (max_position_embeddings, rope_theta): exactly what serving does.
    from knurlogic.engine import register as kreg
    kreg.register(model_type)


# --- reference parity (operator notes 1.4) ---------------------------------
# The official inference code ships with the release, under inference/
# (model.py, kernel.py, config.json). Line numbers below are the
# DeepSeek-V4-Flash-Vision-Exp copy read 2026-10-03; mlx-lm is the exo env's
# mlx_lm/models/deepseek_v4.py.
REFERENCE = ("deepseek-ai/DeepSeek-V4-Flash (and -Vision-Exp): "
             "inference/model.py, inference/kernel.py, inference/config.json")

_T = "tests/test_ds4_parity.py::"
PARITY = (
    Parity("shared_expert_swiglu_clamp",
           "MoE.shared_experts = Expert(..., swiglu_limit=args.swiglu_limit) (model.py:682): "
           "the SHARED expert clamps like the routed ones",
           "mlx-lm builds shared_experts with swiglu_limit=0.0 (UNCLAMPED, F195); the scorer's "
           "default variant sets it to swiglu_limit (_apply_variant). Serving through stock "
           "mlx-lm is still unclamped",
           "tests/test_ds4_shared_clamp.py::test_plugin_applies_and_stamps",
           probe="shared"),
    Parity("routed_expert_swiglu_clamp",
           "Expert.forward (model.py:653-658): up = clamp(up, -limit, limit); "
           "gate = clamp(gate, max=limit); silu(gate) * up; limit = swiglu_limit (10)",
           "_limited_swiglu (min(gate, limit), clip(up, -limit, limit)) via "
           "SwitchGLU(activation=_DSV4SwiGLU(args.swiglu_limit))",
           _T + "test_routed_clamp_matches_reference", probe="routed"),
    Parity("routing_score_and_bias",
           "Gate.forward (model.py:611-639): scores = sqrtsoftplus(x.float() @ W.float()); "
           "bias added for top-k SELECTION only; weights = unbiased scores at the chosen "
           "experts / their sum, * route_scale (1.5)",
           "MoEGate fallback path (float32): same, with e_score_correction_bias (renamed from "
           "ffn.gate.bias in sanitize) and a +1e-20 in the normalizer",
           _T + "test_gate_matches_reference"),
    Parity("routing_fused_gate_kernel",
           "scores computed in float32 (model.py:612)",
           "non-hash layers on Metal take _moe_gate_kernel, which forms x @ W.T in the "
           "ACTIVATION dtype (bf16) before sqrtsoftplus + bias + top-k: a lower-precision "
           "score, so near-tied experts may be chosen differently. Never measured",
           required=False),
    Parity("hash_routing_layers",
           "layers < n_hash_layers (3): indices = tid2eid[input_ids]; weights from the "
           "unbiased scores, normalized, * route_scale (model.py:599-636)",
           "MoEGate.hash = layer_id < num_hash_layers; inds = tid2eid[ids]; the scorer "
           "passes each chunk's own ids to every block",
           _T + "test_hash_gate_matches_reference"),
    Parity("norm_eps",
           "RMSNorm / per-head q rsqrt / hc_pre all use norm_eps from inference/config.json "
           "(1e-20 in the release, not the dataclass default 1e-6)",
           "mlx-lm reads rms_norm_eps from the HF config.json (1e-20 in the release); "
           "hc_eps 1e-6 both. Config-driven; no test reads both configs",
           required=False),
    Parity("hyperconnection_sinkhorn",
           "hc_split_sinkhorn (kernel.py), hc_sinkhorn_iters=20, hc_eps",
           "mlx-lm _hc_split_sinkhorn_ops / fused Metal kernel, same iters and eps from config",
           required=False),
    Parity("activation_quant_simulation",
           "every fp8 Linear quantizes its INPUT to fp8 (act_quant, ue8m0 block scales) "
           "before the GEMM; kv gets act_quant(64) and the indexer fp4_act_quant (QAT "
           "simulation, model.py:125-133, 410-459, 547)",
           "mlx-lm does NOT simulate activation quantization: it dequantizes fp8 weights to "
           "bf16 at sanitize and runs bf16 activations. A known divergence, never measured",
           required=False),
    Parity("image_token_routing",
           "Vision-Exp: Gate.bias_vl replaces bias for image tokens (input_ids >= vocab_size) "
           "and overrides hash routing for them (model.py:609-633)",
           "text-only mlx-lm has no bias_vl; irrelevant on the text corpora, owned by the "
           "vision runtime (graft-extras carries the tensors)",
           required=False),
)

# Projections mlx-lm's sanitize fuses ("Fuse wq_a + wkv -> wqkv_a, and
# Compressor wkv + wgate -> wkv_gate"); the release stores them apart.
# stream-convert --skeleton-from reads this through the plugin registry.
FUSED = (("attn.wq_a", "attn.wkv", "attn.wqkv_a"),                       # attention
         ("compressor.wkv", "compressor.wgate", "compressor.wkv_gate"))  # its compressors


def act_stats(model_dir, corpora, tokens, args):
    """How often each SwiGLU clamp in PARITY fires: per corpus, the routed
    experts (_DSV4SwiGLU, clamp live) and the shared expert (counted at
    swiglu_limit whether or not the scorer variant applies it). Streams the
    teacher through the validated scorer. GPU; yields one dict per probe."""
    import pathlib
    import types

    import mlx_lm.models.deepseek_v4 as A
    from mlx_lm.utils import load_tokenizer

    from vqlab.core import runtime_load
    from vqlab.score.act_stats import ClampCounter, corpus_ids

    tok = load_tokenizer(pathlib.Path(model_dir))
    counters = {}
    orig_mlp, orig_act = A.DeepseekV4MLP.__call__, A._DSV4SwiGLU.__call__

    def mlp_call(self, x):                        # the shared expert
        g, u = self.gate_proj(x), self.up_proj(x)
        counters["shared"].add(g, u)
        return self.down_proj(A._limited_swiglu(g, u, self.swiglu_limit))

    def act_call(self, x, gate):                  # routed: SwitchGLU passes (up, gate)
        counters["routed"].add(gate, x)
        return orig_act(self, x, gate)

    A.DeepseekV4MLP.__call__, A._DSV4SwiGLU.__call__ = mlp_call, act_call
    try:
        for name in corpora:
            ids = corpus_ids(tok, name, tokens)
            model, _ = runtime_load.load_for_family("deepseek_v4", model_dir,
                                                    lazy=True, cpu_stream=True)
            core = getattr(model, "language_model", model).model
            lim = float(getattr(core.args, "swiglu_limit", 10.0) or 10.0)
            counters["shared"], counters["routed"] = ClampCounter(lim), ClampCounter(lim)
            score_deepseek_v4(model, ids, types.SimpleNamespace(
                chunk=getattr(args, "chunk", 512),
                lazy_over_gb=getattr(args, "lazy_over_gb", 16.0)))
            for probe, c in counters.items():
                yield {"corpus": name, "tokens": len(ids) - 1, "probe": probe,
                       "variant": _variant(), **c.result()}
            del model, core
            mx.clear_cache()
    finally:
        A.DeepseekV4MLP.__call__, A._DSV4SwiGLU.__call__ = orig_mlp, orig_act


SPEC = FamilyPlugin(
    name="deepseek_v4",
    model_types=("deepseek_v4",),
    fit=FIT,
    # Rule-5 run 2026-10-02 on a 4-layer slice of the exact teacher
    # (sanitize-stream output; layers 0-2 hash-routed, 3 score-routed,
    # compressor/indexer attention), chunk 512, vs a direct resident forward
    # over the same chunks with one shared cache list: logits BITWISE
    # identical (max|d| 0.0) at 2048 and 12288 tokens. Scope: 4 of 43 layers,
    # real weights; the rest of the stack is the same block class.
    # cpu_stream_load: the converted teacher's blocks are ~3.5 GiB.
    scorer={"fn": score_deepseek_v4, "family": "deepseek_v4",
            "validated": True, "cpu_stream_load": True, "variant": _variant},
    tokenizer_register=_register_tokenizer,
    reference=REFERENCE,
    parity=PARITY,
    fused=FUSED,
    act_stats=act_stats,
)
