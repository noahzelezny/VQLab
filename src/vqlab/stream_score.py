"""Streaming referee: score a model that cannot be RESIDENT on any box.

Layer-streamed forward over a frozen corpus — embed once, then materialize
one DecoderLayer at a time (weights eval'd on the CPU stream, so nothing
hits Metal until the layer runs), advance the activations, free the layer.
Flat memory: peak is one layer + activations, so a 598 GiB bf16 teacher
scores on a 96 GB box. The per-layer release pattern (slot = None, del,
gc, clear_cache) is the E18 machinery from the 397B teacher cache.

Two outputs from one pass:
  - referee perplexity over the corpus (the ladder number), and
  - optionally --save-topk K: the model's top-k logprobs per position, in
    the same {indices, logprobs, tokens} safetensors layout the KL scorer
    reads — so a single teacher pass arms every later KL comparison.

FAMILY SUPPORT IS EXPLICIT. A streamed loop re-implements the model's
forward; a family whose per-layer signature it does not know is a silent
wrong answer waiting to happen, so unknown model_type is a hard error.
  qwen4_exp: layers take (h, rope, mask, conv_mask, cache, idx_cache,
    ids, prev_ctx) — ids/prev_ctx feed each layer's own n-gram PLE lookup,
    which is why a hidden-state-only loop (kl_damage.py's warning) would
    silently starve the PLE path. h is tiled x hc before the stack and
    resolved by hyper_connection_mixer after it; there is no final norm.

    python -m vqlab.stream_score --model <dir> --corpus <txt> [--tokens N]
        [--save-topk K --out <dir>]
"""
import argparse
import gc
import json
import math
import pathlib
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runtime_load
from mem_budget import eval_params_budgeted


def score_qwen4_exp(model, ids_list, args):
    """Streamed qwen4_exp scorer — CHUNKED, with per-layer recurrent state.

    Rule 5 (F111): the earlier version pushed the WHOLE sequence through each
    layer in one call and no cache. On this architecture that is not the same
    computation as the house standard: qwen4_exp's linear-attention layers
    carry recurrent state, so the metric is chunk-DEPENDENT, and every
    published number on this family was measured at chunk 512
    (score_streaming.py's own note). The unchunked pass read 5.8857 against a
    direct chunk-512 forward's 5.9056 at 2048 tokens, and degraded
    monotonically with length -- 6.18 at 3072, 7.10 at 6144, 274 at 12288
    (F109).

    So: layers stay streamed (one materialised at a time, the whole point),
    but each layer now walks the sequence in --chunk blocks carrying its OWN
    cache, which is exactly the per-layer state the resident chunked forward
    maintains. prev_ctx (the PLE n-gram history) depends only on `ids`, so it
    is derived per chunk directly rather than threaded through a cache.
    """
    from mlx_lm.models.qwen4_exp import create_attention_mask, create_ssm_mask

    core = model.model
    ids = mx.array([ids_list[:-1]])
    S = ids.shape[1]
    C = max(1, int(getattr(args, "chunk", 512) or 512))
    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    h = core.embed_tokens(ids)

    # PLE n-gram context per chunk: the ctx_len ids immediately before the
    # chunk, EOS-padded at the start of the sequence.
    prev_ctxs = None
    if core.ple_layers:
        ctx = core.args.ngram_size - 1
        eos = core.args.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        pad = mx.full((ids.shape[0], ctx), eos, ids.dtype)
        hist = mx.concatenate([pad, ids], axis=1)
        prev_ctxs = [hist[:, s:s + ctx] for s in range(0, S, C)]

    h = mx.tile(h, (1, 1, core.hc))
    mx.eval(h)

    caches = model.make_cache()
    n = len(core.layers)
    for i in range(n):
        blk = core.layers[i]
        with mx.stream(mx.cpu):
            # Budgeted: this family's layer 1 is 100.3 GiB of PLE tables in
            # bf16 and no eager eval of it fits (F115). See mem_budget.
            _, skipped = eval_params_budgeted(
                blk, getattr(args, "lazy_over_gb", 8.0))
        if skipped:
            print(f"  layer {i}: {skipped / 1024 ** 3:.1f} GiB left lazy",
                  flush=True)
        t0 = time.time()
        c = caches[i]
        parts = []
        for k, s0 in enumerate(range(0, S, C)):
            e0 = min(s0 + C, S)
            hc_ = h[:, s0:e0]
            ids_c = ids[:, s0:e0]
            # Each layer consumes only the mask its own type needs, and each
            # mask is built from THAT layer's cache (a linear-attention cache
            # has no make_mask, so it must never reach create_attention_mask).
            is_lin = blk.layer_type == "linear_attention"
            mask = None if is_lin else create_attention_mask(hc_, c)
            conv_mask = create_ssm_mask(hc_, c) if is_lin else None
            pc = prev_ctxs[k] if prev_ctxs is not None else None
            # Re-read the indexer EVERY chunk: the reference reads it once per
            # __call__, and one __call__ is one chunk. Hoisting it out of the
            # loop holds a stale object if the cache replaces rather than
            # mutates it.
            idx_c = c.indexer if (c is not None and
                                  hasattr(c, "indexer")) else None
            parts.append(blk(hc_, core.rope, mask, conv_mask, c, idx_c,
                             ids_c, pc))
        h = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        mx.eval(h)
        core.layers[i] = None
        caches[i] = None
        del blk, parts
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1} {time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)

    mixer = core.hyper_connection_mixer
    head = model.lm_head if not model.args.text.tie_word_embeddings else None
    with mx.stream(mx.cpu):
        mx.eval(mixer.parameters(),
                head.parameters() if head is not None
                else core.embed_tokens.parameters())
    # Head in CHUNKS too. vocab is 248320 here, so a whole-sequence projection
    # is 9.5 GiB of fp32 logits at 10240 tokens -- the largest length-scaling
    # op in the pass, and the resident scorer never builds it (it accumulates
    # NLL per chunk). Chunking keeps each matmul the size the model is
    # normally run at.
    lg = []
    for s0 in range(0, S, C):
        o = mixer(h[:, s0:min(s0 + C, S)])
        lg.append((head(o) if head is not None
                   else core.embed_tokens.as_linear(o)).astype(mx.float32)[0])
        mx.eval(lg[-1])
    logits = mx.concatenate(lg, axis=0) if len(lg) > 1 else lg[0]
    return logits


def score_glm5_next(model, ids_list, args):
    """GLM-5.3-Flash streamed scorer. RULE-5 RUN IS STALE — see F113.

    The 2026-08-29 validation below was run on the UNCHUNKED version at 33
    tokens, where chunking cannot matter. The qwen4_exp twin carried the same
    structure and was wrong by 0.03 at 2048 and catastrophically above 8192
    (F112); this function has now been given the same fix (per-layer cache
    across --chunk blocks, chunked head projection) but has NOT been re-run
    against a direct forward. Re-validate before any GLM number from it enters
    a ladder or a card.


    Written 2026-08-29; LINE-VERIFIED the same day against the installed
    release (mlx-vlm 0.6.17, venv glm5vlm) — every step below mirrors
    Glm5NextModel.__call__ / LanguageModel.__call__ one-to-one: masks built
    on the pre-broadcast h (attention mask with return_array=True),
    broadcast to hc_mult + mx.contiguous, per-layer mask picked by the
    layer's own `is_linear`, h.mean(axis=2), model.norm, then lm_head or
    tied as_linear. Still NEVER EXECUTED. Validation standard before the
    flag comes off (house rule 5 / III.11): the streamed pass must
    reproduce a direct full-model forward to all printed decimals — on a
    small resident model or a known artifact — and the DSA-indexer-with-
    fresh-cache question must be answered by that same run.

    Shape per the readiness design note (research/glm53-flash/READINESS.md):
      - text stack one level deeper: model.language_model.model.layers
      - layer signature is (x, mask, cache) — no ids/prev_ctx (no PLE)
      - hc bookends: broadcast h to (B, S, hc_mult, D) BEFORE the stack,
        h.mean(axis=2) AFTER — hc mixing itself lives inside each layer
      - final norm EXISTS (core.norm) — opposite of qwen4_exp
      - KDA layers take an ssm mask, DSA layers an attention mask; helper
        names are resolved from the model's own module (III.13: score with
        the copy that loaded, never a parallel import).
    """
    # VALIDATED 2026-08-29 (rule 5): streamed pass vs direct forward on a
    # tiny random-init glm5_next LanguageModel (4 layers = 3 KDA + 1 DSA,
    # dense+sparse MLP, hc_mult=4, 33 tokens; glm5vlm venv, M3 metal):
    # logits BITWISE IDENTICAL (max|diff| 0.0), ppl equal to all printed
    # decimals. Same run answered the open unknown: the DSA indexer
    # accepts cache=None at full-sequence prefill. Scope: tiny geometry,
    # random weights — the first REAL-model score should still be
    # cross-checked against another instrument once one exists.
    lm = getattr(model, "language_model", model)
    core = lm.model
    ids = mx.array([ids_list[:-1]])
    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    h = core.embed_tokens(ids)

    import importlib
    lang = importlib.import_module(type(core).__module__)
    make_attn = getattr(lang, "create_attention_mask", None)
    make_ssm = getattr(lang, "create_ssm_mask", None)
    if make_attn is None:
        raise SystemExit("FAIL: create_attention_mask not found in "
                         f"{type(core).__module__} — the runtime's helper "
                         "names moved; update score_glm5_next against the "
                         "resolved module before scoring.")
    h = mx.broadcast_to(h[:, :, None, :],
                        (*h.shape[:2], core.hc_mult, h.shape[-1]))
    h = mx.contiguous(h)
    mx.eval(h)

    S = ids.shape[1]
    C = max(1, int(getattr(args, "chunk", 512) or 512))
    caches = lm.make_cache() if hasattr(lm, "make_cache") else [None] * len(core.layers)

    n = len(core.layers)
    for i in range(n):
        blk = core.layers[i]
        with mx.stream(mx.cpu):
            mx.eval(blk.parameters())
        t0 = time.time()
        c = caches[i] if caches else None
        parts = []
        for s0 in range(0, S, C):
            e0 = min(s0 + C, S)
            hc_ = h[:, s0:e0]
            # masks rebuilt per chunk against THIS layer's cache, mirroring the
            # per-call construction in Glm5NextModel.__call__
            hp = hc_[:, :, 0, :] if hc_.ndim == 4 else hc_
            m = (make_ssm(hp, c) if (blk.is_linear and make_ssm is not None)
                 else make_attn(hp, c, return_array=True))
            parts.append(blk(hc_, mask=m, cache=c))
        h = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        mx.eval(h)
        core.layers[i] = None
        if caches:
            caches[i] = None
        del blk, parts
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1} {time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)

    h = h.mean(axis=2)                      # hc bookend #2
    with mx.stream(mx.cpu):
        mx.eval(core.norm.parameters())
    out = core.norm(h)                      # final norm EXISTS here
    head = getattr(lm, "lm_head", None)
    tied = getattr(getattr(lm, "args", None), "tie_word_embeddings", False)
    with mx.stream(mx.cpu):
        mx.eval(head.parameters() if (head is not None and not tied)
                else core.embed_tokens.parameters())
    # Head in CHUNKS: a whole-sequence projection is the largest
    # length-scaling op in the pass and is what broke qwen4_exp above 8192.
    lg = []
    for s0 in range(0, out.shape[1], C):
        o = out[:, s0:min(s0 + C, out.shape[1])]
        lg.append((head(o) if (head is not None and not tied)
                   else core.embed_tokens.as_linear(o)).astype(mx.float32)[0])
        mx.eval(lg[-1])
    logits = mx.concatenate(lg, axis=0) if len(lg) > 1 else lg[0]
    return logits


def score_qwen3_5_moe(model, ids_list, args):
    """Streamed qwen3_5_moe scorer (Qwen3.5-397B-A17B, Qwen3.6-35B-A3B).

    VALIDATED 2026-09-17 (rule 5), on a REAL shipped artifact rather than a
    random-init toy: Qwen3.6-35B-A3B-VQ-3.4bpw (same qwen3_5_moe
    architecture, 40 layers, 14 GB so it fits resident), prose referee,
    chunk 512, this streamed pass vs a DIRECT full-model resident forward
    with one shared cache list (scratch/direct_qwen35.py):

        2048 tokens   direct 4.842402   streamed 4.842402
        12288 tokens  direct 5.414175   streamed 5.414175

    Both lengths matter and 12288 is the load-bearing one: F111/F112 showed
    this class of error is length-GROWING (qwen4_exp's broken pass was off
    by 0.02 at 2048 and read 274 at 12288), and 12288 is the length the KL
    caches and every ladder cell use. Scope: 40 layers, VQ weights, one
    corpus; the 60-layer 397B is the same module tree by config and the
    same code path, but the first 397B number should still be sanity-checked
    against another instrument if one appears.

    LINE-MIRRORED against mlx_lm.models.qwen3_5 (Qwen3_5TextModel.__call__ /
    TextModel.__call__), each step below in the reference's order:
      - embed_tokens, then layers with `mask=` chosen by the layer's own
        `is_linear` (GatedDeltaNet gets create_ssm_mask, full attention gets
        create_attention_mask); layer signature is (x, mask=, cache=).
      - A FINAL NORM EXISTS (`return self.norm(hidden_states)`) -- opposite
        of qwen4_exp, same as glm5_next. RMSNorm is per-position, so applying
        it per chunk is identical to applying it to the whole sequence.
      - head: tied -> embed_tokens.as_linear, else lm_head.
      - no PLE, no hyper-connection bookends: h is NOT tiled or averaged.

    CHUNKED, and it must be. `is_linear = (layer_idx + 1) %
    full_attention_interval != 0` puts GatedDeltaNet recurrent state on most
    layers, so this metric is chunk-DEPENDENT exactly as qwen4_exp's is
    (F111/F112: an unchunked pass on that family read 5.8857 against 5.9056
    and degraded to 274 at 12288 tokens). Each layer walks the sequence in
    --chunk blocks carrying its OWN cache.

    MASK EQUIVALENCE, the one place this is not a literal transcription: the
    reference builds fa_mask/ssm_mask ONCE per __call__ from a representative
    layer of each type (`cache[self.fa_idx]`, `cache[self.ssm_idx]`) and
    shares them across every layer of that type. Streaming runs one layer per
    pass, so each mask is built from THAT layer's cache. Equivalent because
    every layer walks the same chunks in the same order, so all caches of a
    type sit at the same offset when their mask is built -- and it keeps a
    linear-attention cache (ArraysCache, no make_mask) out of
    create_attention_mask, which is what the qwen4_exp scorer does too.
    """
    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

    lm = getattr(model, "language_model", model)
    core = lm.model
    ids = mx.array([ids_list[:-1]])
    S = ids.shape[1]
    C = max(1, int(getattr(args, "chunk", 512) or 512))
    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    h = core.embed_tokens(ids)
    mx.eval(h)

    # Built BEFORE any layer is dropped: make_cache walks lm.layers, which is
    # a property over core.layers.
    caches = lm.make_cache()
    n = len(core.layers)
    for i in range(n):
        blk = core.layers[i]
        with mx.stream(mx.cpu):
            _, skipped = eval_params_budgeted(
                blk, getattr(args, "lazy_over_gb", 8.0))
        if skipped:
            print(f"  layer {i}: {skipped / 1024 ** 3:.1f} GiB left lazy",
                  flush=True)
        t0 = time.time()
        c = caches[i]
        parts = []
        for s0 in range(0, S, C):
            hc_ = h[:, s0:min(s0 + C, S)]
            mask = (create_ssm_mask(hc_, c) if blk.is_linear
                    else create_attention_mask(hc_, c))
            parts.append(blk(hc_, mask=mask, cache=c))
        h = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        mx.eval(h)
        # pipeline_layers is a property over this list, so dropping the entry
        # here drops the only persistent reference to the block.
        core.layers[i] = None
        caches[i] = None
        del blk, parts
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1} {time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)

    head = None if lm.args.tie_word_embeddings else lm.lm_head
    with mx.stream(mx.cpu):
        mx.eval(core.norm.parameters(),
                head.parameters() if head is not None
                else core.embed_tokens.parameters())
    # Norm + head in CHUNKS: a whole-sequence fp32 projection over this
    # family's vocab is the largest length-scaling allocation in the pass,
    # and the resident scorer never builds one.
    lg = []
    for s0 in range(0, S, C):
        o = core.norm(h[:, s0:min(s0 + C, S)])
        lg.append((head(o) if head is not None
                   else core.embed_tokens.as_linear(o)).astype(mx.float32)[0])
        mx.eval(lg[-1])
    logits = mx.concatenate(lg, axis=0) if len(lg) > 1 else lg[0]
    return logits


# family -> scorer entry. `runtime` names the loader (runtime_load), and
# `validated` is house rule 5: a scorer is validated only once its streamed
# pass has reproduced a direct forward to all printed decimals. Unvalidated
# scorers refuse without --allow-unvalidated and stamp their output record.
SCORERS = {
    "qwen4_exp": {"fn": score_qwen4_exp, "family": "qwen4_exp",
                  "validated": True},
    # F113: the 2026-08-29 rule-5 run predates the chunking fix and was done
    # at 33 tokens, where chunking cannot matter. Re-validate before trusting.
    "glm5_next": {"fn": score_glm5_next, "family": "glm5_next",
                  "validated": False},
    # Registry key is the checkpoint's model_type; "family" names the
    # runtime_load loader (qwen3_5 -> mlx_lm). Rule-5 run 2026-09-17 on the
    # 35B-A3B twin at 2048 AND 12288 tokens, exact to all printed decimals.
    # cpu_stream_load: 12.2 GiB blocks on the 397B teacher cannot be read
    # inside a GPU command buffer without tripping the watchdog. Neutral on
    # this family's numbers (35B-3.4 @12288 reads 5.414175 either way).
    "qwen3_5_moe": {"fn": score_qwen3_5_moe, "family": "qwen3_5",
                    "validated": True, "cpu_stream_load": True},
    # The DENSE variant (Qwen3.8-27B) is the SAME mlx_lm module: qwen3_5's
    # DecoderLayer picks SparseMoeBlock or MLP on `num_experts > 0`, and that
    # choice lives inside `.mlp`, which this scorer never touches -- it walks
    # embed_tokens, the layers with the is_linear mask dispatch, the final
    # norm and the head. Same GatedDeltaNet on 3 layers in 4, so the same
    # chunked cache is required. cpu_stream_load stays on so the whole
    # qwen3_5 architecture is measured through one load path (its blocks are
    # small enough not to need it; consistency is the reason).
    # Rule-5 run 2026-09-17 on Qwen3.8-27B-VQ-4.5bpw (dense, 64 layers,
    # 14.5 GB so it fits resident) vs a direct full-model forward, chunk 512:
    # 5.227517/5.227517 at 2048 and 5.829932/5.829932 at 12288, exact.
    "qwen3_5": {"fn": score_qwen3_5_moe, "family": "qwen3_5",
                "validated": True, "cpu_stream_load": True},
}


def main():
    # A 335 GiB teacher streamed through a 96 GB box sits at the edge: MLX's
    # buffer cache holding freed allocations is enough to push it into swap,
    # where the GPU command buffer then times out waiting on paging. Cap the
    # cache so freed layer weights go back to the OS promptly.
    mx.set_cache_limit(2 << 30)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512,
                    help="prefill chunk. qwen4_exp carries recurrent state, so "
                         "the metric is NOT chunk-invariant; 512 is what every "
                         "published number on this family used (F111).")
    ap.add_argument("--save-topk", type=int, default=None,
                    help="also dump top-k logprobs per position (teacher "
                         "cache for KL) to --out")
    ap.add_argument("--kl-cache", default=None,
                    help="teacher top-k cache dir (from --save-topk): also "
                         "report KL-to-teacher in millinats + top-1 "
                         "agreement. Token ids must match the cache exactly.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lazy-over-gb", type=float, default=8.0,
                    help="per-BLOCK budget for the eager parameter eval; the "
                         "rest is left lazy, largest first (F115).")
    ap.add_argument("--kl-per-position", default=None,
                    help="write the per-position KL (millinats) here, so two "
                         "rungs on the same cache can be compared PAIRED.")
    ap.add_argument("--stream-ple", action="store_true",
                    help="stream the PLE n-gram gather instead of holding the "
                         "tables resident. REQUIRED to run this family's bf16 "
                         "teacher: its tables are 96 GiB. Changes no "
                         "arithmetic, only when buffers are freed; costs a "
                         "re-read of the tables per call, so put the teacher "
                         "on fast storage first.")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="run a scorer that has NOT yet reproduced a direct "
                         "forward (rule 5). The output record is stamped "
                         "\"unvalidated\": true; such a number must never "
                         "enter a ladder or a card.")
    a = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    mp = pathlib.Path(a.model)
    cfg_peek = json.load(open(mp / "config.json"))
    mt = cfg_peek.get("model_type") or \
        cfg_peek.get("text_config", {}).get("model_type")
    if mt not in SCORERS:
        raise SystemExit(f"FAIL: no streaming scorer for model_type={mt!r}. "
                         f"Supported: {sorted(SCORERS)}. A generic loop "
                         f"would silently mis-run this family.")
    entry = SCORERS[mt]
    if not entry["validated"] and not a.allow_unvalidated:
        raise SystemExit(f"FAIL: the {mt!r} scorer is UNVALIDATED (rule 5: "
                         "it has never reproduced a direct forward). Pass "
                         "--allow-unvalidated to run it anyway; the record "
                         "will be stamped.")
    # Load via the family's declared runtime (runtime_load; mlx_lm families
    # behave exactly as before, incl. the in-checkpoint model.py bundle —
    # both runtimes honour model_file). III.13: print what resolved.
    # cpu_stream_load: only families whose blocks are too big for the read to
    # finish inside a GPU command buffer (see runtime_load). It is NOT
    # arithmetic-neutral, so it stays off for families whose published
    # numbers were measured without it.
    model, config = runtime_load.load_for_family(
        entry["family"], mp, lazy=True,
        cpu_stream=entry.get("cpu_stream_load", False))
    print(runtime_load.resolved_runtime_note(model), flush=True)
    if a.stream_ple:
        import ple_stream
        if not ple_stream.install(model, str(mp)):
            print("[ple_stream] no sharded n-gram tables found", flush=True)
    tok = load_tokenizer(mp)
    ids = tok.encode(open(a.corpus).read())[: a.tokens + 1]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids[: a.tokens]

    logits = entry["fn"](model, ids, a)
    tgt = mx.array(ids[1:])
    lse = mx.logsumexp(logits, axis=-1)
    pk = mx.take_along_axis(logits, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
    nll = lse - pk
    ppl = math.exp(float(mx.mean(nll).item()))
    rec = {"model": str(mp), "corpus": a.corpus,
           "tokens": len(ids) - 1, "ppl": round(ppl, 6)}
    if not entry["validated"]:
        rec["unvalidated"] = True           # rule 5: never enters a ladder

    if a.kl_cache:
        cd = pathlib.Path(a.kl_cache)
        cache_tok = mx.load(str(cd / "tokens.safetensors"))["tokens"][0]
        if cache_tok.tolist() != ids:
            raise SystemExit("FAIL: token ids differ from the cache — the "
                             "KL would compare different positions. Same "
                             "corpus, same --tokens, same tokenizer required.")
        t = mx.load(str(cd / "teacher_topk.safetensors"))
        t_idx = t["indices"][0].astype(mx.int64)          # [S, k]
        t_lp = t["logprobs"][0].astype(mx.float32)        # [S, k]
        s_lp_all = logits - lse[:, None]
        s_lp = mx.take_along_axis(s_lp_all, t_idx, axis=-1)
        # truncated KL(teacher || student) over the teacher's top-k
        kl = mx.sum(mx.exp(t_lp) * (t_lp - s_lp), axis=-1)
        top1 = mx.mean(
            (mx.argmax(s_lp_all, axis=-1) == t_idx[:, 0]).astype(mx.float32))
        mass = mx.mean(mx.sum(mx.exp(t_lp), axis=-1))
        # ERROR BARS, because a KL mean without one cannot answer the only
        # question anyone asks of it: is this rung DIFFERENT from that rung.
        # The 2026-09-16 Flash-3.2 sweep had four arms inside 3 mnats of each
        # other and no way to say whether that meant anything (F116); F51
        # asked for this and it never got built. Positions are the sample:
        # S of them, so the standard error of the mean is sd/sqrt(S). They are
        # not independent -- adjacent tokens share context -- so this is a
        # LOWER BOUND on the true uncertainty, and it is reported as such.
        n_pos = int(kl.size)
        kl_mn = kl * 1000.0
        mean_mn = float(mx.mean(kl_mn).item())
        sd_mn = float(mx.sqrt(mx.var(kl_mn, ddof=1)).item())
        sem = sd_mn / math.sqrt(n_pos)
        if getattr(a, "kl_per_position", None):
            # EXPORT THE PER-POSITION KL. Two rungs scored against the same
            # cache see the SAME positions and the SAME teacher, so their
            # comparison is PAIRED -- and a paired test cancels the
            # position-to-position variance that dominates this SEM. Without
            # this array the only available test is the unpaired one, which
            # on a 10 mnat difference between arms whose own spreads are
            # ~3 mnat cannot resolve what a paired test resolves easily.
            mx.eval(kl_mn)
            mx.save_safetensors(a.kl_per_position, {"kl_millinats": kl_mn})
        rec.update(mean_kl_millinats=round(mean_mn, 4),
                   kl_sem_millinats=round(sem, 4),
                   kl_ci95_millinats=[round(mean_mn - 1.96 * sem, 4),
                                      round(mean_mn + 1.96 * sem, 4)],
                   kl_sd_millinats=round(sd_mn, 4),
                   kl_positions=n_pos,
                   kl_sem_note="sd/sqrt(n) over positions; positions are "
                               "correlated, so this UNDERSTATES the true "
                               "uncertainty",
                   top1_agreement=round(float(top1.item()), 4),
                   captured_mass=round(float(mass.item()), 4))
    if a.stream_ple:
        import ple_stream as _ps
        st = _ps.STATS
        print(f"[ple_stream] {st['calls']} calls, {st['rows']} rows, "
              f"{st['secs']:.1f}s in the gather", flush=True)
    print(json.dumps(rec), flush=True)

    if a.save_topk:
        C_ = max(1, int(getattr(a, "chunk", 512) or 512))
        outd = pathlib.Path(a.out or (mp.name + "_topk"))
        outd.mkdir(parents=True, exist_ok=True)
        lp = logits - lse[:, None]
        idx = mx.argpartition(-lp, kth=a.save_topk - 1, axis=-1)[:, : a.save_topk]
        top = mx.take_along_axis(lp, idx, axis=-1)
        mx.eval(idx, top)
        mx.save_safetensors(str(outd / "teacher_topk.safetensors"),
                            {"indices": idx[None].astype(mx.int32),
                             "logprobs": top[None].astype(mx.float16)})
        mx.save_safetensors(str(outd / "tokens.safetensors"),
                            {"tokens": mx.array([ids])})
        # CURRENT kl_damage SCHEMA, not the 2026-08 one. The pre-existing
        # flashnext cache was written with {model, tokens} and kl_damage
        # reads {teacher, seq_len, num_samples, batch_size}; it therefore
        # could not be read at all without a shim, and the shim was deleted
        # in a disk cleanup (F51, and it cost an hour on 2026-09-16). Write
        # what the reader wants, and keep the old keys as aliases so nothing
        # that consumed the old format breaks.
        # teacher_ppl is the whole point of storing this: a ppl ladder is
        # only ordered correctly while every rung sits on the SAME SIDE of
        # the teacher, and quantization damage can push a student BELOW bf16
        # on finite text -- at which point "lower ppl" and "closer to the
        # teacher" point opposite ways and the ranking inverts (F125: 8 of 9
        # cells predicted by this one comparison). The number was always
        # printed here and never kept, so the test needed a build log.
        (outd / "meta.json").write_text(json.dumps(
            {"teacher": str(mp), "corpus": a.corpus, "top_k": a.save_topk,
             "num_samples": 1, "seq_len": len(ids) - 1, "batch_size": 1,
             "chunk": C_, "streamed": True, "teacher_ppl": round(ppl, 6),
             "model": str(mp), "tokens": len(ids)}, indent=1))
        print(f"top-{a.save_topk} cache -> {outd}", flush=True)


if __name__ == "__main__":
    main()
