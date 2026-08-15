#!/usr/bin/env python
"""Local twin of exo's echo_score — same corpus, same chunked-NLL math.

Exists to arbitrate serving-path bugs: if this and
`score_via_exo.py` disagree on the SAME model, the difference is exo's
loader/serving path (sanitize, chat template, cache), not the weights.

    ~/quantlab/venv/bin/python score_local.py <model_dir> [--raw]

--raw scores the bare corpus (what this does by default is identical);
exo instead wraps the corpus in the chat template via /v1/chat/completions,
so a disagreement that vanishes under --chat implicates the template.
"""
import argparse
import json
import math
import pathlib
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--chat", action="store_true",
                    help="wrap corpus in the chat template first (mimics how "
                         "exo's /v1/chat/completions feeds echo_score)")
    ap.add_argument("--step", type=int, default=1024)
    args = ap.parse_args()

    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, tokenizer, _ = load(args.model, return_config=True, lazy=True)

    corpus = (pathlib.Path(__file__).parent / "referee_corpus.txt").read_text()
    if args.chat:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": corpus}],
            tokenize=False, add_generation_prompt=True)
        toks = mx.array(tokenizer.encode(text))
    else:
        toks = mx.array(tokenizer.encode(corpus))

    n = toks.shape[0]
    from mlx_lm.models.cache import make_prompt_cache
    caches = make_prompt_cache(model)
    total_nll, scored = 0.0, 0
    t0 = time.time()
    for i in range(0, n - 1, args.step):
        chunk = toks[i: min(i + args.step, n - 1)]
        targets = toks[i + 1: i + 1 + chunk.shape[0]]
        logits = model(chunk[None], cache=caches)[0].astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        tgt = mx.take_along_axis(
            logits, targets[:, None].astype(mx.int64), axis=-1)[:, 0]
        chunk_nll = mx.sum(lse - tgt)
        mx.eval(chunk_nll)
        total_nll += float(chunk_nll.item())
        scored += int(chunk.shape[0])
    nll_per_token = total_nll / max(scored, 1)
    print(json.dumps({
        "model": args.model.split("/")[-1],
        "mode": "chat" if args.chat else "raw",
        "total_nll": round(total_nll, 4),
        "tokens_scored": scored,
        "nll_per_token": round(nll_per_token, 6),
        "ppl": round(math.exp(min(nll_per_token, 30.0)), 4),
        "seconds": round(time.time() - t0, 1),
    }, indent=1))


if __name__ == "__main__":
    main()
