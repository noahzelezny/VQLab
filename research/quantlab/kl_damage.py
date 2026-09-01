#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Quantization damage as KL divergence from the model's OWN bf16 output.

WHY THIS EXISTS. Perplexity is the project's quality instrument everywhere
else, and it is INVALID on gemma-4-it: the family is RL/distillation
sharpened until raw-text likelihood is meaningless (own greedy output ppl
1.42, plain English ~100, Austen ~700 — reproduced by HF transformers on
unquantized bf16, so it is the model, not the port. See
GEMMA4_PPL_ANOMALY.md). Without a working instrument there is no way to
choose a rung, so the ladder cannot start.

KL fixes this by changing the question. Perplexity asks "how surprised is
the model by this text", which a collapsed distribution cannot answer.
KL asks "how far did quantization move the model from itself":

    damage(rung) = mean_tokens KL( P_bf16(.|ctx) || P_rung(.|ctx) )

The sharpening is IDENTICAL in teacher and student, so it is common-mode
and cancels exactly. No notion of "correct" text is needed, which is the
whole reason ppl failed. And it is continuous and per-token, so it is far
more sensitive per unit of compute than accuracy over discrete items.

TOP-K TRUNCATION IS FAVOURED HERE, NOT MERELY TOLERATED. We store the
teacher's top-k and sum KL over those indices, which underestimates true KL
by the tail mass it omits. That error grows with how DIFFUSE the teacher
is — and gemma-4-it is pathologically CONCENTRATED. The same property that
destroyed perplexity makes this estimate tight. `captured_mass` is reported
every run so the assumption is checked rather than assumed; if it drops,
raise --top-k.

DIRECT FORWARD, NOT LAYER STREAMING. dwq_cache_teacher.py streams blocks
(`blk(a, mask=..., cache=None)`) because the 397B teacher cannot be
resident. gemma-4 cannot be run that way at all — its DecoderLayer returns
a tuple, threads per-layer inputs and shared KV, and alternates
sliding-window masks that a hand-rolled loop gets silently wrong. At
sidecar size the teacher fits, so we call the model and let upstream handle
its own architecture.

CHAT FRAMING IS THE DEFAULT, AND THIS IS A REAL CHOICE. gemma-4-it operates
inside its chat template; raw prose is out-of-distribution for it (scoring
raw made ppl 27, scoring inside a user turn made it 180 — the model behaves
differently there). Quantization damage measured OOD does not necessarily
predict damage in the regime the sidecar actually runs in. So chunks are
wrapped in the model's chat template by default; --raw disables it for
comparison or for base (non-instruct) models.

CACHE FORMAT is byte-compatible with dwq_cache_teacher.py
({indices int32 [N,S,k], logprobs float16 [N,S,k]}, tokens, meta.json) so
the DWQ tooling can consume these caches and vice versa.

    # 1. cache the bf16 teacher once (slow, one pass)
    ./kl_damage.py cache --model <bf16 dir> --out-dir kl_cache_gemma26b \
        --corpus referee/referee_corpus_literary.txt --num-samples 128

    # 2. score each rung against it (teacher NOT resident)
    ./kl_damage.py score --model <rung dir> --cache-dir kl_cache_gemma26b

CALIBRATION (2026-08-17, gemma-4-e2b-it, 16x384 chat-wrapped literary,
top-64, captured_mass 0.969). Our own mlx_lm conversions off the bf16
(see the E-SERIES NOTE below on why not mlx-community's quants).

    rung   size    mean KL (millinats/tok)   top-1 agreement
    bf16   5.2G    -0.002  <- noise floor    100.00%
    q8     4.6G     8.4                       95.69%
    q4     2.5G     635.8                     65.98%
    q2     1.4G     15437.1                    0.28%

Read those as: the self-KL noise floor is ~0.002 millinats (fp16 rounding
of the stored teacher logprobs), so anything above ~0.01 is real signal.
q8 is a mild perturbation; q4 costs a third of the argmax decisions; q2
destroys the model outright — which independently reproduces this lab's
own finding that dense models do not tolerate extreme quant
(EXPERIMENTS.md headline 4), from a completely different instrument.

TOP-1 AGREEMENT is the number to quote to a human. "This rung picks a
different next token than bf16 on 4% of positions" is legible in a way
that nats are not. KL is the sensitive one; agreement is the honest one.

E-SERIES NOTE. gemma-4 e2b/e4b quants ship k_proj/v_proj/k_norm for
KV-shared layers that mlx_lm 0.31.3 does not build, so most of them need
--allow-unmatched (e4b-6bit happens not to). Those tensors are provably
dead — mlx_vlm builds them, mlx_lm drops them, both give ppl 96.62 on
e2b-6bit — so the flag is safe HERE and must not be assumed safe elsewhere.
The calibration table above used our own bf16 conversions regardless, which
is what the methodology rule wants.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


# --------------------------------------------------------------------------
def load_direct(path, allow_unmatched=False):
    """Load whole (not lazy-streamed). See DIRECT FORWARD note above."""
    from mlx_lm.utils import load, load_model, load_tokenizer
    try:
        model, tokenizer, _ = load(path, return_config=True)
        return model, tokenizer
    except ValueError as e:
        if "not in model" not in str(e) or not allow_unmatched:
            # Refuse by default: a LIVE unmatched tensor degrades the model
            # and every number downstream still looks plausible. The gemma-4
            # E-series shared-KV k/v are the known-dead exception (see the
            # E-SERIES NOTE above) — verify before assuming it elsewhere.
            raise
        model, _ = load_model(Path(path), strict=False)
        return model, load_tokenizer(Path(path))


def build_chunks(tokenizer, corpus_paths, num_samples, seq_len, chat, seed):
    """Tokenize corpus into [N, S] int32, optionally chat-wrapped.

    Chat mode wraps each chunk as a user turn so the measurement happens in
    the regime the sidecar actually runs in (see module docstring). The
    chunk is decoded back to text before wrapping so the template's own
    special tokens are applied by the tokenizer, not spliced in by hand.
    """
    text = "\n\n".join(Path(p).read_text(errors="replace") for p in corpus_paths)
    ids = tokenizer.encode(text)
    bos = getattr(tokenizer, "bos_token_id", None)

    rows, i = [], 0
    # Take a FULL seq_len slice and truncate AFTER wrapping. Wrapping only
    # ever adds tokens, so the result is always >= seq_len and the trim
    # below yields exactly seq_len. (Reserving headroom instead left every
    # wrapped row a few tokens short, and the exact-length check silently
    # discarded all of them.)
    raw_len = seq_len
    while len(rows) < num_samples and i + raw_len < len(ids):
        chunk = ids[i:i + raw_len]
        i += raw_len
        if chat:
            body = tokenizer.decode(chunk)
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": body}],
                add_generation_prompt=True, tokenize=False)
            row = tokenizer.encode(wrapped)
        else:
            row = list(chunk)
        if bos is not None and (not row or row[0] != bos):
            row = [bos] + row          # gemma degenerates without BOS
        row = row[:seq_len]
        if len(row) < seq_len:
            continue                   # skip short tails rather than pad
        rows.append(row)

    if len(rows) < num_samples:
        raise SystemExit(
            f"corpus yielded {len(rows)} chunks of {seq_len} tokens, "
            f"need {num_samples}. Use a bigger --corpus or fewer samples.")
    rng = np.random.default_rng(seed)
    arr = np.array(rows[:num_samples], dtype=np.int32)
    rng.shuffle(arr)
    return arr


def forward_logprobs(model, batch):
    """[B,S] tokens -> [B,S,V] full-vocab logprobs (fp32)."""
    logits = model(mx.array(batch)).astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# --------------------------------------------------------------------------
def cmd_cache(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] teacher: {args.model}", flush=True)
    model, tok = load_direct(args.model, args.allow_unmatched)

    print(f"[2/3] corpus: {args.num_samples} x {args.seq_len} "
          f"({'chat-wrapped' if not args.raw else 'raw'})", flush=True)
    arr = build_chunks(tok, args.corpus, args.num_samples, args.seq_len,
                       not args.raw, args.seed)
    mx.save_safetensors(str(out / "tokens.safetensors"),
                        {"tokens": mx.array(arr)})

    print(f"[3/3] teacher forward, top-{args.top_k}", flush=True)
    all_idx, all_lp, mass = [], [], []
    t0 = time.time()
    for s in range(0, len(arr), args.batch_size):
        lp = forward_logprobs(model, arr[s:s + args.batch_size])
        idx = mx.argpartition(-lp, kth=args.top_k - 1, axis=-1)[..., :args.top_k]
        top = mx.take_along_axis(lp, idx, axis=-1)
        m = mx.sum(mx.exp(top), axis=-1)
        mx.eval(idx, top, m)
        all_idx.append(idx.astype(mx.int32))
        all_lp.append(top.astype(mx.float16))
        mass.append(float(mx.mean(m).item()))
        del lp, idx, top
        gc.collect()
        mx.clear_cache()
        print(f"  {s + args.batch_size}/{len(arr)} "
              f"({time.time() - t0:.0f}s, peak "
              f"{mx.get_peak_memory() / 1024**3:.1f}G)", flush=True)

    idx = mx.concatenate(all_idx, axis=0)
    lp = mx.concatenate(all_lp, axis=0)
    mx.save_safetensors(str(out / "teacher_topk.safetensors"),
                        {"indices": idx, "logprobs": lp})
    captured = float(np.mean(mass))
    (out / "meta.json").write_text(json.dumps({
        "teacher": args.model, "corpus": args.corpus,
        "num_samples": args.num_samples, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "seed": args.seed,
        "top_k": args.top_k, "chat_wrapped": not args.raw,
        "shape": list(idx.shape), "captured_mass": round(captured, 6),
        "format": "dwq_cache_teacher-compatible",
    }, indent=1))
    print(f"\ndone -> {out}  indices {idx.shape}")
    print(f"captured_mass {captured:.4f}  "
          f"(top-{args.top_k} holds this fraction of the teacher's "
          f"probability; >0.99 means the truncated KL below is tight)")


# --------------------------------------------------------------------------
def cmd_score(args):
    cache = Path(args.cache_dir)
    meta = json.load(open(cache / "meta.json"))
    toks = np.array(mx.load(str(cache / "tokens.safetensors"))["tokens"])
    t = mx.load(str(cache / "teacher_topk.safetensors"))
    t_idx, t_lp = t["indices"], t["logprobs"]
    B = args.batch_size or meta["batch_size"]

    print(f"cache: {meta['teacher']}")
    print(f"  {meta['num_samples']}x{meta['seq_len']} top-{meta['top_k']}, "
          f"{'chat-wrapped' if meta.get('chat_wrapped') else 'raw'}, "
          f"captured_mass {meta.get('captured_mass')}")
    print(f"student: {args.model}\n", flush=True)

    model, _ = load_direct(args.model, args.allow_unmatched)

    kl_sum = n_tok = agree = 0.0
    t0 = time.time()
    for s in range(0, len(toks), B):
        e = min(s + B, len(toks))
        s_lp_full = forward_logprobs(model, toks[s:e])
        ti = t_idx[s:e]
        tl = t_lp[s:e].astype(mx.float32)
        # student logprobs at the TEACHER's top-k indices
        sl = mx.take_along_axis(s_lp_full, ti, axis=-1)
        p = mx.exp(tl)
        # KL(P_teacher || P_student) restricted to the teacher's top-k
        kl = mx.sum(p * (tl - sl), axis=-1)
        # secondary, directly interpretable: does the student still rank the
        # teacher's argmax first?
        s_top = mx.argmax(s_lp_full, axis=-1)
        t_top = mx.take_along_axis(
            ti, mx.argmax(tl, axis=-1)[..., None], axis=-1)[..., 0]
        ag = (s_top == t_top)
        mx.eval(kl, ag)
        kl_sum += float(mx.sum(kl).item())
        agree += float(mx.sum(ag).item())
        n_tok += kl.size
        del s_lp_full, sl, p, kl, ag
        gc.collect()
        mx.clear_cache()
        print(f"  {e}/{len(toks)} ({time.time() - t0:.0f}s)", flush=True)

    mean_kl = kl_sum / n_tok
    result = {
        "student": args.model,
        "teacher": meta["teacher"],
        "cache_dir": str(cache),
        "mean_kl_nats": round(mean_kl, 6),
        "mean_kl_millinats": round(mean_kl * 1000, 3),
        "top1_agreement": round(agree / n_tok, 6),
        "tokens_scored": int(n_tok),
        "top_k": meta["top_k"],
        "captured_mass": meta.get("captured_mass"),
        "chat_wrapped": meta.get("chat_wrapped"),
    }
    print("\n" + json.dumps(result, indent=1))
    print(f"\nmean KL {mean_kl * 1000:.3f} millinats/token   "
          f"top-1 agreement {agree / n_tok:.2%}")
    print("LOWER KL = less damage. Compare rungs at matched size; the knee "
          "is where KL starts climbing faster than bytes are saved.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="cache bf16 teacher top-k logprobs")
    c.add_argument("--model", required=True, help="bf16 teacher dir")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--corpus", nargs="+",
                   default=["referee/referee_corpus_literary.txt"])
    c.add_argument("--num-samples", type=int, default=128)
    c.add_argument("--seq-len", type=int, default=512)
    c.add_argument("--batch-size", type=int, default=4)
    c.add_argument("--top-k", type=int, default=64)
    c.add_argument("--seed", type=int, default=123)
    c.add_argument("--raw", action="store_true",
                   help="do NOT chat-wrap; measures the OOD regime instead")
    c.add_argument("--allow-unmatched", action="store_true")
    c.set_defaults(func=cmd_cache)

    s = sub.add_parser("score", help="score a rung against a teacher cache")
    s.add_argument("--model", required=True, help="quantized rung dir")
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--out", default=None, help="write result JSON here")
    s.add_argument("--allow-unmatched", action="store_true")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
