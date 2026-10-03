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

from vqlab.family import FamilyPlugin

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
    VQLAB_DS4_SHARED_CLAMP=1 applies DeepSeek's reference SwiGLU clamp
    (limit = swiglu_limit, 10) to the SHARED expert too; mlx-lm's
    deepseek_v4 builds it unclamped (F195: the clamp fires on ~0.026% of
    shared activations and moves that expert's output 2% on average)."""
    import os
    return "shared-clamp" if os.environ.get("VQLAB_DS4_SHARED_CLAMP") == "1" else None


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
)
