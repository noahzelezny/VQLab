#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Single-box streaming referee — no sharding, no distribution, no E23.

E26c: tensor-sharded inference broke DETERMINISTICALLY on the first DWQ'd
397B artifact (uniform logits, ppl 202k) while the identical weights score
KL 0.0201 unsharded — the E23 kernel bug is value-sensitive, so 2-node
scoring can't be trusted per-artifact. This computes the SAME prefix-8192
metric by streaming blocks on one box (flat memory, ~15G): the whole prefix
goes through each block as one causal full-forward, which is mathematically
identical to the chunked+cache referee.

VALIDATE against a known answer before trusting any new number: champion
struct6-tail3x3 = 3.1580 on the 2-node referee.
"""
import argparse
import gc
import json
import math
import pathlib
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--corpus", default=None,
                    help="corpus file (default: referee_corpus.txt, wikitext). "
                         "E30: every number in this arc is wikitext PPL, but "
                         "the actual workload is code — and the structure-bits"
                         "-vs-expert-bits trade was chosen on wikitext alone, "
                         "so the ~1%% edge over the community quant at matched "
                         "size is exactly the claim a domain shift could "
                         "erase. referee_corpus_code.txt is the audit. Never "
                         "compare PPL ACROSS corpora — only models within one.")
    args = ap.parse_args()

    mx.set_cache_limit(8 << 30)

    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, tokenizer, _ = load(args.model, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    n_blocks = len(core.layers)

    text = pathlib.Path(
        args.corpus or (pathlib.Path(__file__).parent
                        / "referee_corpus.txt")).read_text(errors="replace")
    toks = mx.array(tokenizer.encode(text))[: args.max_tokens + 1]
    n = toks.shape[0]
    print(f"corpus: {n} tokens, {n_blocks} blocks, streaming", flush=True)

    results = []
    for r in range(args.trials):
        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters())
        h = core.embed_tokens(toks[: n - 1][None])
        mx.eval(h)
        t0 = time.time()
        for i in range(n_blocks):
            blk = core.layers[i]
            mask = None if blk.is_linear else "causal"
            with mx.stream(mx.cpu):
                mx.eval(blk.parameters())
            blk.eval()
            h = blk(h, mask=mask, cache=None)
            mx.eval(h)
            core.layers[i] = None
            del blk
            gc.collect()
            mx.clear_cache()
        head_norm = core.norm
        lm_head = getattr(model, "lm_head", None) or getattr(
            getattr(model, "language_model", model), "lm_head", None)
        with mx.stream(mx.cpu):
            mx.eval(head_norm.parameters(),
                    lm_head.parameters() if lm_head is not None
                    else core.embed_tokens.parameters())
        total_nll, scored = 0.0, 0
        targets = toks[1:n]
        step = 1024
        hh = head_norm(h)[0]
        for s in range(0, n - 1, step):
            e = min(s + step, n - 1)
            logits = (lm_head(hh[s:e]) if lm_head is not None
                      else core.embed_tokens.as_linear(hh[s:e])
                      ).astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            tgt = mx.take_along_axis(
                logits, targets[s:e][:, None].astype(mx.int64), axis=-1)[:, 0]
            nll = mx.sum(lse - tgt)
            mx.eval(nll)
            total_nll += float(nll.item())
            scored += e - s
            mx.clear_cache()
        nll_per_token = total_nll / scored
        results.append(nll_per_token)
        print(json.dumps({
            "run": r + 1,
            "model": args.model.rstrip("/").split("/")[-1],
            "mode": "streaming-1box",
            "total_nll": round(total_nll, 4),
            "tokens_scored": scored,
            "nll_per_token": round(nll_per_token, 6),
            "ppl": round(math.exp(min(nll_per_token, 30.0)), 4),
            "seconds": round(time.time() - t0, 1),
        }), flush=True)
        if r + 1 < args.trials:
            with mx.stream(mx.cpu):
                model2, _, _ = load(args.model, lazy=True, return_config=True)
            core2 = model2
            for name in ("language_model", "model"):
                while hasattr(core2, name):
                    core2 = getattr(core2, name)
            core.layers = core2.layers
            core.embed_tokens = core2.embed_tokens
            core.norm = core2.norm

    if args.trials > 1:
        ppls = [math.exp(min(x, 30.0)) for x in results]
        spread = (max(ppls) - min(ppls)) / min(ppls)
        print(f"DETERMINISM: {len(ppls)} runs, spread {spread:.2%} "
              f"({'PASS' if spread < 0.001 else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()
