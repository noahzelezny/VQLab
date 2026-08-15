#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Deterministic multi-node perplexity — exo-free, via mlx_lm pipeline sharding.

WHY (2026-08-12): exo's echo_score on a 2-node TENSOR placement returned
4.824 / 3.500 / 22.077 for one unchanged 397B model, and wedged on a fourth
try. Single-node scoring is exact (5.5577 x3, 0.05% vs local mlx_lm), so the
instability is in the multi-node serving path — which means a 123 GB model
could not be honestly benchmarked at all. This scores such models WITHOUT exo:
`pipeline_load` splits the model by LAYER across ranks (each rank holds a
contiguous slice, activations hop rank to rank), so there is no cross-rank
logit gather to get wrong. Rank 0 alone owns the final projection and prints
the score.

Launch across both boxes (run from the M3):

    ~/quantlab/venv/bin/mlx.launch \\
        --hosts noahs-mac-studio.local,nozzlebook-pro.local \\
        --backend ring \\
        ~/quantlab/referee/score_pipeline.py -- --model "<path>"

Both hosts need: the same mlx/mlx_lm versions, the model at the SAME path, and
passwordless ssh. Single-box smoke test (no launcher) also works — it degrades
to world_size 1, which is the case we already trust.

Metric: RAW text, no chat template — the standard anyone means by "wikitext
perplexity". Corpus defaults to the frozen referee slice so numbers line up
with everything else in ~/quantlab, but --data-path takes any text file.

VISION: irrelevant here by construction. mlx_lm loads the language model only
and ignores OptiQ's `optiq/optiq_vision.safetensors` sidecar, which is exactly
right — perplexity is a text measurement. The vision tower is untouched on
disk and still loads under exo for actual multimodal serving; scoring text
through mlx_lm neither needs nor disturbs it.
"""
import argparse
import json
import math
import pathlib
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-path", default=None,
                    help="text file to score (default: the frozen referee corpus)")
    ap.add_argument("--step", type=int, default=1024,
                    help="tokens per forward chunk; must match across runs "
                         "being compared (it sets how much context each "
                         "predicted token gets)")
    ap.add_argument("--trials", type=int, default=1,
                    help="score N times in-process and report the spread — "
                         "the determinism check that caught the exo bug. "
                         "NOT named --repeat: mlx.launch argparse-prefix-"
                         "matches that onto its own --repeat-hosts and "
                         "silently duplicates every host (2 hosts x 3 = "
                         "world_size 6), while your script never sees it.")
    ap.add_argument("--wired-limit", action="store_true",
                    help="mx.set_wired_limit(max_recommended_working_set_size) "
                         "before scoring — what mlx_lm.generate and exo's "
                         "server both do and this script historically did "
                         "not. Probe for the 397B-scale nondeterminism "
                         "(E22): a 62G/rank shard without a wired limit may "
                         "let Metal evict buffers mid-compute.")
    ap.add_argument("--reset-ctx", type=int, default=0,
                    help="reset the prompt cache every N tokens (0 = never). "
                         "E22 workaround: 397B tensor-sharded inference "
                         "corrupts process state once context crosses ~8192; "
                         "capping context below that keeps every chunk on the "
                         "healthy path. Changes the metric (each token sees "
                         "at most N context) — compare only runs with the "
                         "same value.")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="score only the first N corpus tokens (0 = all). "
                         "E23: the healthy, bit-deterministic region at 397B "
                         "scale is the first ~8192 tokens of a fresh process; "
                         "score that slice with one trial per launch.")
    ap.add_argument("--per-chunk", action="store_true",
                    help="print each chunk's nll — E22 debugging: locates "
                         "the exact chunk where two trials diverge")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()

    def rprint(*a):
        if rank == 0:
            print(*a, flush=True)

    rprint(f"world_size={world} (rank {rank}) — "
           f"{'PIPELINE sharded' if world > 1 else 'single node'}")

    from mlx_lm.utils import load, sharded_load
    with mx.stream(mx.cpu):  # E15: lazy-load ops bind to the creating stream
        if world > 1:
            # sharded_load(repo, pipeline_group, tensor_group, ...). Which one
            # a model supports is architecture-specific: Qwen3.5-MoE implements
            # `shard()` (TENSOR parallel) and has no `model.pipeline`, so
            # mlx_lm's own `pipeline_load` — which passes the group as the
            # PIPELINE group — raises "does not support pipelining". Pass the
            # group in the slot the architecture actually implements.
            #
            # NOTE this means we are back on tensor parallelism, the same
            # scheme whose cross-rank logit handling made exo's scores swing
            # 6x. The difference is that here mlx_lm owns the sharding end to
            # end (its own all_reduce inside `shard()`), with no separate
            # serving layer picking which rank's response to return. That is a
            # hypothesis, not a guarantee — which is exactly why --repeat
            # exists: verify determinism before believing any number.
            try:
                from mlx_lm.utils import load_model, _download
                probe, _ = load_model(_download(
                    args.model, allow_patterns=["*.json"]), lazy=True,
                    strict=False)
                pipe_ok = hasattr(probe, "model") and hasattr(probe.model,
                                                              "pipeline")
                del probe
            except Exception:
                pipe_ok = False
            if pipe_ok:
                rprint("sharding: PIPELINE (layer split)")
                model, tokenizer, _ = sharded_load(
                    args.model, group, None, True)
            else:
                rprint("sharding: TENSOR (architecture has no pipeline support)")
                model, tokenizer, _ = sharded_load(
                    args.model, None, group, True)
        else:
            model, tokenizer, _ = load(args.model, return_config=True, lazy=True)

    if args.data_path:
        text = pathlib.Path(args.data_path).read_text()
    else:
        text = (pathlib.Path(__file__).parent / "referee_corpus.txt").read_text()
    toks = mx.array(tokenizer.encode(text))
    if args.max_tokens:
        toks = toks[: args.max_tokens + 1]  # +1: last target token
    n = toks.shape[0]
    rprint(f"corpus: {n} tokens, step={args.step}")

    if args.wired_limit:
        limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(limit)
        rprint(f"wired_limit set: {limit // 2**20} MB")

    from mlx_lm.models.cache import make_prompt_cache

    results = []
    for r in range(args.trials):
        caches = make_prompt_cache(model)
        total_nll, scored = 0.0, 0
        t0 = time.time()
        ctx_used = 0
        for i in range(0, n - 1, args.step):
            if args.reset_ctx and ctx_used + args.step > args.reset_ctx:
                caches = make_prompt_cache(model)
                ctx_used = 0
            chunk = toks[i: min(i + args.step, n - 1)]
            ctx_used += int(chunk.shape[0])
            targets = toks[i + 1: i + 1 + chunk.shape[0]]
            logits = model(chunk[None], cache=caches)[0].astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            tgt = mx.take_along_axis(
                logits, targets[:, None].astype(mx.int64), axis=-1)[:, 0]
            chunk_nll = mx.sum(lse - tgt)
            mx.eval(chunk_nll)
            total_nll += float(chunk_nll.item())
            scored += int(chunk.shape[0])
            if args.per_chunk:
                rprint(f"  trial{r + 1} chunk@{i}: nll={float(chunk_nll.item()):.4f}")
        nll_per_token = total_nll / max(scored, 1)
        results.append(nll_per_token)
        rprint(json.dumps({
            "run": r + 1,
            "model": args.model.rstrip("/").split("/")[-1],
            "world_size": world,
            "total_nll": round(total_nll, 4),
            "tokens_scored": scored,
            "nll_per_token": round(nll_per_token, 6),
            "ppl": round(math.exp(min(nll_per_token, 30.0)), 4),
            "seconds": round(time.time() - t0, 1),
        }))

    if args.trials > 1 and rank == 0:
        ppls = [math.exp(min(x, 30.0)) for x in results]
        spread = (max(ppls) - min(ppls)) / min(ppls)
        print(f"\nDETERMINISM: {len(ppls)} runs, spread {spread:.2%} "
              f"({'PASS — reproducible' if spread < 0.001 else 'FAIL — do NOT trust these numbers'})",
              flush=True)


if __name__ == "__main__":
    main()
