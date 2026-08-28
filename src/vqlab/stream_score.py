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
import time

import mlx.core as mx


def score_qwen4_exp(model, ids_list, args):
    from mlx_lm.models.qwen4_exp import create_attention_mask, create_ssm_mask

    core = model.model
    ids = mx.array([ids_list[:-1]])
    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    h = core.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    lin = [i for i, l in enumerate(core.layers)
           if l.layer_type == "linear_attention"]
    conv_mask = create_ssm_mask(h, None) if lin else None
    prev_ctx = None
    if core.ple_layers:
        ctx = core.args.ngram_size - 1
        eos = core.args.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        prev_ctx = mx.full((ids.shape[0], ctx), eos, ids.dtype)
    h = mx.tile(h, (1, 1, core.hc))
    mx.eval(h)

    n = len(core.layers)
    for i in range(n):
        blk = core.layers[i]
        with mx.stream(mx.cpu):
            mx.eval(blk.parameters())
        t0 = time.time()
        h = blk(h, core.rope, mask, conv_mask, None, None, ids, prev_ctx)
        mx.eval(h)
        core.layers[i] = None
        del blk
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
    out = mixer(h)
    logits = (head(out) if head is not None
              else core.embed_tokens.as_linear(out)).astype(mx.float32)[0]
    return logits


SCORERS = {"qwen4_exp": score_qwen4_exp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--save-topk", type=int, default=None,
                    help="also dump top-k logprobs per position (teacher "
                         "cache for KL) to --out")
    ap.add_argument("--kl-cache", default=None,
                    help="teacher top-k cache dir (from --save-topk): also "
                         "report KL-to-teacher in millinats + top-1 "
                         "agreement. Token ids must match the cache exactly.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from mlx_lm.utils import load_model, load_tokenizer
    mp = pathlib.Path(a.model)
    # VQ artifacts ship their runtime as an in-checkpoint model.py bundle;
    # scoring them is this tool's primary job, so the bundle import is on.
    model, config = load_model(mp, lazy=True, trust_remote_code=True)
    mt = config.get("model_type") or config.get("text_config", {}).get("model_type")
    if mt not in SCORERS:
        raise SystemExit(f"FAIL: no streaming scorer for model_type={mt!r}. "
                         f"Supported: {sorted(SCORERS)}. A generic loop "
                         f"would silently mis-run this family.")
    tok = load_tokenizer(mp)
    ids = tok.encode(open(a.corpus).read())[: a.tokens + 1]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids[: a.tokens]

    logits = SCORERS[mt](model, ids, a)
    tgt = mx.array(ids[1:])
    lse = mx.logsumexp(logits, axis=-1)
    pk = mx.take_along_axis(logits, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
    nll = lse - pk
    ppl = math.exp(float(mx.mean(nll).item()))
    rec = {"model": str(mp), "corpus": a.corpus,
           "tokens": len(ids) - 1, "ppl": round(ppl, 6)}

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
        rec.update(mean_kl_millinats=round(float(mx.mean(kl).item()) * 1000, 4),
                   top1_agreement=round(float(top1.item()), 4),
                   captured_mass=round(float(mass.item()), 4))
    print(json.dumps(rec), flush=True)

    if a.save_topk:
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
        (outd / "meta.json").write_text(json.dumps(
            {"model": str(mp), "corpus": a.corpus, "top_k": a.save_topk,
             "tokens": len(ids)}, indent=1))
        print(f"top-{a.save_topk} cache -> {outd}", flush=True)


if __name__ == "__main__":
    main()
