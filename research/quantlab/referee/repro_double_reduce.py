#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Minimal repro for the E22 397B tensor-shard doubling bug.

Observed: scoring the same corpus twice in one process, the second pass
returns nll_per_token EXACTLY 2x the first (8.202491 vs 4.101369). That is
the signature of an all_reduce contribution being added twice, not random
corruption. This strips the scorer to its skeleton: forward ONE fixed chunk
several times (fresh cache each time, like the scorer's trials) and print
logit fingerprints — norm, max, and the logit of a fixed (position, token).
If pass 2 shows ~2x pass 1, the doubling lives in the model forward itself.
Also forwards once with NO cache to isolate the prompt-cache path.
"""
import argparse
import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--passes", type=int, default=4)
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()

    def rprint(*a):
        if rank == 0:
            print(*a, flush=True)

    rprint(f"world_size={world} rank={rank}")

    from mlx_lm.utils import load, sharded_load
    with mx.stream(mx.cpu):
        if world > 1:
            model, tokenizer, _ = sharded_load(args.model, None, group, True)
        else:
            model, tokenizer, _ = load(args.model, return_config=True, lazy=True)

    import pathlib
    text = (pathlib.Path(__file__).parent / "referee_corpus.txt").read_text()
    toks = mx.array(tokenizer.encode(text))[: args.tokens]
    rprint(f"chunk: {toks.shape[0]} tokens")

    from mlx_lm.models.cache import make_prompt_cache

    def fingerprint(logits):
        lf = logits.astype(mx.float32)
        out = {
            "l2": float(mx.sqrt(mx.sum(lf * lf)).item()),
            "max": float(mx.max(lf).item()),
            "probe[10, tok0]": float(lf[10, int(toks[0].item())].item()),
            "last_argmax": int(mx.argmax(lf[-1]).item()),
        }
        return out

    for p in range(args.passes):
        caches = make_prompt_cache(model)
        logits = model(toks[None], cache=caches)[0]
        mx.eval(logits)
        rprint(f"pass {p + 1} (fresh cache): {fingerprint(logits)}")

    logits = model(toks[None])[0]
    mx.eval(logits)
    rprint(f"no-cache pass:      {fingerprint(logits)}")


if __name__ == "__main__":
    main()
