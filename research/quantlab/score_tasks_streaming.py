#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Layer-streaming loglikelihood scorer — task benchmarks for models larger
than RAM.

WHY THIS EXISTS. `referee/score_streaming.py` proved the trick: load ONE
transformer block, push activations through it, free it, move on. Flat ~15 GB
regardless of model size, which is how a 165.6 GiB artifact got scored on a
128 GB box. But it scores exactly one contiguous sequence, so it computes
perplexity and nothing else.

HellaSwag / PIQA / WinoGrande are loglikelihood tasks: thousands of SHORT
independent sequences. The naive port — stream the model once per item —
would re-read 100+ GiB per item and never finish. So the axis is flipped:
every sequence is batched through each layer before that layer is freed, and
the whole task costs ONE pass over the model's bytes.

WHY AN lm-eval LM SUBCLASS. lm-eval collects every request and hands them to
`model.loglikelihood(requests)` in a SINGLE call — exactly the shape one-pass
streaming needs. Subclassing means prompt construction, length normalization
(acc_norm), aggregation and stderr all stay lm-eval's own code. We supply
numbers; we do not reimplement the benchmark.

METHODOLOGY NOTE (this project's standing rule). Never place another
publisher's task numbers beside ours: the same weights (spicyneuron 2.6bit)
score 3.1843 on our referee and 3.852 on their card — a 21% gap that is the
harness, not the model. Run comparators HERE or report ours alone.

VALIDATE BEFORE TRUSTING ANY NUMBER (referee/README.md rule). `--selftest`
scores the wikitext referee corpus through this exact code path and must
reproduce the known ppl for the artifact:
    VQ-2.2bpw 3.1706 | VQ-2.4bpw 2.7655 | VQ-3.1bpw 2.3519
If it does not match to 4 decimals, nothing downstream counts.

    ./score_tasks_streaming.py --model <dir> --selftest
    ./score_tasks_streaming.py --model <dir> \
        --tasks hellaswag,piqa,winogrande --limit 1000 --output-dir results/
"""
import argparse
import gc
import json
import math
import pathlib
import time

import mlx.core as mx

# Positions scored per lm_head chunk. vocab is 248320, so a chunk of 512 in
# fp32 is ~0.5 GB of logits — the largest transient in the whole scorer.
_LOGIT_CHUNK = 512


def _find_core(model):
    """Unwrap language_model/model nesting to the block-bearing module."""
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    return core


def _lm_head(model):
    return getattr(model, "lm_head", None) or getattr(
        getattr(model, "language_model", model), "lm_head", None)


class StreamingLM:
    """lm-eval LM over a layer-streamed forward pass.

    Not constructed until lm_eval is importable; see `_build_lm` for the
    subclassing (done at runtime so this module imports without lm_eval).
    """

    def __init__(self, model_path, batch_seqs=256, verbose=True, direct=False,
                 allow_unmatched=False):
        self.model_path = model_path
        self.direct = direct
        self.allow_unmatched = allow_unmatched
        self.batch_seqs = batch_seqs
        self.verbose = verbose
        self._model = None
        self._tokenizer = None

    # ---- model handling -------------------------------------------------
    def _load(self):
        """Load lazily. Reloaded per pass: streaming DESTROYS the layer list.

        STRICT, AND REFUSE OTHERWISE. mlx_lm's `load()` uses `strict=True`,
        which is the property we want: a tensor the model class does not
        build is a version mismatch, and dropping it degrades the model in a
        way no downstream number reveals.

        THE E-SERIES CASE, MEASURED. gemma-4 e2b/e4b ship
        k_proj/v_proj/k_norm for their KV-shared layers (e2b 140 tensors =
        20 layers x 7, e4b 126 = 18 x 7), which mlx_lm never constructs
        because `has_kv = layer_idx < num_hidden_layers -
        num_kv_shared_layers` (gemma4_text.py:183). Those tensors are
        genuinely DEAD: mlx_vlm 0.5.0 builds them, mlx_lm drops them, and
        both score e2b-6bit at ppl 96.62 — identical to the decimal. So
        --allow-unmatched is safe for this family.

        (An earlier revision of this file blamed those dropped tensors for
        gemma's terrible benchmark scores. That was wrong — the cause is the
        family's collapsed raw-text distribution, proven against HF
        transformers on unquantized bf16. See GEMMA4_PPL_ANOMALY.md.)

        The refusal is still the right default for UNKNOWN checkpoints: an
        unmatched tensor that IS live degrades the model in a way no
        downstream number reveals. Confirm deadness, then pass the flag.
        """
        from mlx_lm.utils import load, load_model, load_tokenizer
        try:
            with mx.stream(mx.cpu):
                model, tokenizer, _ = load(
                    self.model_path, lazy=True, return_config=True)
            return model, tokenizer
        except ValueError as e:
            if "not in model" not in str(e):
                raise
            head = str(e).split("\n")[0]
            n = head.split("Received ")[-1].split(" parameters")[0]
            if not self.allow_unmatched:
                raise SystemExit(
                    f"\nREFUSING TO SCORE: {head}\n"
                    f"  The checkpoint carries {n} tensors this mlx_lm model "
                    f"class does not build. Dropping tensors can silently "
                    f"degrade a model, so this is refused by default.\n"
                    f"  KNOWN-SAFE CASE: the gemma-4 E-series (e2b/e4b) ships "
                    f"k_proj/v_proj/k_norm for KV-shared layers (e2b 140 = "
                    f"20 layers x 7, e4b 126 = 18 x 7). These are genuinely "
                    f"dead: mlx_vlm builds them and mlx_lm does not, and both "
                    f"give ppl 96.62 on e2b-6bit — identical to the decimal. "
                    f"For those, --allow-unmatched is safe.\n"
                    f"  Verify before trusting it on any OTHER checkpoint: an "
                    f"unmatched tensor that IS live degrades the model in a "
                    f"way no downstream number reveals.")
            self._log(f"  NOTE: dropping {n} unmatched tensors "
                      f"(--allow-unmatched). Confirm they are dead for this "
                      f"checkpoint (E-series shared-KV k/v provably are).")
            path = pathlib.Path(self.model_path)
            with mx.stream(mx.cpu):
                model, _ = load_model(path, lazy=True, strict=False)
                tokenizer = load_tokenizer(path)
            return model, tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            _, self._tokenizer = self._load_cached()
        return self._tokenizer

    def _load_cached(self):
        if self._model is None:
            self._model, self._tokenizer = self._load()
        return self._model, self._tokenizer

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ---- direct path: for models that FIT IN RAM ------------------------
    def _forward_direct(self, seqs, spans):
        """Whole-model forward, one padded bucket at a time.

        WHY THIS EXISTS. `_forward_batches` streams layers because the 397B
        artifacts are larger than RAM. It does that by calling each block as
        `blk(h, mask="causal", cache=None) -> h`, which assumes a plain
        decoder stack. gemma-4 breaks every part of that assumption:

          - `DecoderLayer.__call__` RETURNS A TUPLE (h, shared_kv, offset)
            and takes `per_layer_input` and `shared_kv` (gemma4_text.py:333).
          - e4b threads per-layer embedding inputs (PLE) computed from
            `embed_tokens_per_layer` + `per_layer_model_projection`.
          - `num_kv_shared_layers` means layers 24-41 REUSE the KV produced
            by an earlier layer, so it must be carried forward.
          - `layer_types` alternates sliding/full attention with
            `sliding_window: 512`. Handing "causal" to a sliding layer is
            SILENTLY WRONG — it attends outside the window and still returns
            a number.

        Re-deriving all that here would be a second, unverified copy of
        upstream's forward pass, and the mask bug would not announce itself.
        So when the model fits in memory, call the model. mlx_lm's own
        `__call__` handles PLE, shared KV, and the mask pattern correctly by
        construction.

        Streaming stays the path for anything bigger than RAM; this is the
        path for sidecar-sized artifacts.

        MEMORY. Scoring happens INSIDE the bucket loop. Returning logits
        would mean holding [seq, pos, 262144] fp32 for every sequence -- for
        416 sequences of ~60 tokens that is ~26 GB, larger than the model.
        Only the per-sequence (sum logprob, all-greedy) pair survives a
        bucket, so peak stays one bucket's logits.
        """
        model, _ = self._load_cached()
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        buckets = [order[i:i + self.batch_seqs]
                   for i in range(0, len(order), self.batch_seqs)]
        out = [(0.0, True)] * len(seqs)
        t0 = time.time()
        for bi, idx in enumerate(buckets):
            L = max(len(seqs[i]) for i in idx)
            padded = [seqs[i] + [0] * (L - len(seqs[i])) for i in idx]
            logits = model(mx.array(padded)).astype(mx.float32)
            mx.eval(logits)
            for k, i in enumerate(idx):
                ctx_len, n_cont = spans[i]
                start = max(ctx_len - 1, 0)
                end = min(start + n_cont, len(seqs[i]) - 1)
                if end <= start:
                    continue
                # position p predicts token p+1 — same convention as the
                # streamed path (see _loglikelihood_pairs).
                lg = logits[k, start:end]
                tgt = mx.array(seqs[i][start + 1:end + 1])
                lse = mx.logsumexp(lg, axis=-1)
                picked = mx.take_along_axis(
                    lg, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
                lp = picked - lse
                gd = mx.argmax(lg, axis=-1) == tgt
                mx.eval(lp, gd)
                out[i] = (float(mx.sum(lp).item()), bool(mx.all(gd).item()))
            del logits
            gc.collect()
            mx.clear_cache()
            if self.verbose:
                self._log(f"    bucket {bi + 1}/{len(buckets)} "
                          f"({time.time() - t0:.0f}s)")
        return out

    # ---- the core: one streamed pass over many sequences ----------------
    def _forward_batches(self, seqs):
        """Run every sequence through the model, streaming layers.

        seqs: list of list[int] token ids.
        Returns: list of mx.array [len_i, hidden] final-normed hidden states.

        Sequences are length-sorted into padded buckets to cut wasted compute.
        Padding is on the RIGHT: with causal attention a real token never
        attends to a later pad, so scored positions are exact.
        """
        model, _ = self._load()   # fresh model — previous pass freed its layers
        core = _find_core(model)
        n_blocks = len(core.layers)

        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        buckets = [order[i:i + self.batch_seqs]
                   for i in range(0, len(order), self.batch_seqs)]

        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters())

        # Build padded token arrays + embed up front, so the layer loop only
        # ever touches activations.
        states, lengths = [], []
        for idx in buckets:
            L = max(len(seqs[i]) for i in idx)
            padded = [seqs[i] + [0] * (L - len(seqs[i])) for i in idx]
            h = core.embed_tokens(mx.array(padded))
            mx.eval(h)
            states.append(h)
            lengths.append([len(seqs[i]) for i in idx])

        t0 = time.time()
        for b in range(n_blocks):
            blk = core.layers[b]
            mask = None if getattr(blk, "is_linear", False) else "causal"
            with mx.stream(mx.cpu):
                mx.eval(blk.parameters())
            blk.eval()
            for j in range(len(states)):
                states[j] = blk(states[j], mask=mask, cache=None)
                mx.eval(states[j])
            core.layers[b] = None
            del blk
            gc.collect()
            mx.clear_cache()
            if self.verbose and (b + 1) % 10 == 0:
                self._log(f"    block {b + 1}/{n_blocks} "
                          f"({time.time() - t0:.0f}s)")

        with mx.stream(mx.cpu):
            mx.eval(core.norm.parameters())
        out = [None] * len(seqs)
        for j, idx in enumerate(buckets):
            hh = core.norm(states[j])
            mx.eval(hh)
            for k, i in enumerate(idx):
                out[i] = hh[k, :lengths[j][k]]
            states[j] = None
            mx.clear_cache()

        self._head = _lm_head(model)
        self._embed = core.embed_tokens
        with mx.stream(mx.cpu):
            mx.eval(self._head.parameters() if self._head is not None
                    else self._embed.parameters())
        return out

    def _project(self, hidden_rows, targets):
        """logits for stacked [M, hidden] rows -> (sum logprob, all-greedy).

        hidden_rows/targets are already flattened across sequences; chunked so
        the [chunk, 248320] fp32 logit tensor stays ~0.5 GB.
        """
        M = hidden_rows.shape[0]
        logprobs = mx.zeros((M,), dtype=mx.float32)
        greedy = mx.zeros((M,), dtype=mx.bool_)
        parts_lp, parts_g = [], []
        for s in range(0, M, _LOGIT_CHUNK):
            e = min(s + _LOGIT_CHUNK, M)
            rows = hidden_rows[s:e]
            logits = (self._head(rows) if self._head is not None
                      else self._embed.as_linear(rows)).astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1)
            tgt = targets[s:e]
            picked = mx.take_along_axis(
                logits, tgt[:, None].astype(mx.int64), axis=-1)[:, 0]
            parts_lp.append(picked - lse)
            parts_g.append(mx.argmax(logits, axis=-1) == tgt)
            mx.eval(parts_lp[-1], parts_g[-1])
            mx.clear_cache()
        del logprobs, greedy
        return mx.concatenate(parts_lp), mx.concatenate(parts_g)

    # ---- lm-eval interface ---------------------------------------------
    def _loglikelihood_pairs(self, pairs):
        """pairs: list of (context_str, continuation_str) -> [(logprob, greedy)]

        Tokenization follows lm-eval's convention: encode ctx and ctx+cont,
        the continuation is the tail. The scored positions are the ones whose
        PREDICTION targets a continuation token, i.e. hidden[ctx_len-1 : -1].
        """
        tok = self.tokenizer
        # BOS IS LOAD-BEARING ON GEMMA. `tok.encode()` here does NOT prepend
        # the BOS token, and gemma-4 without BOS collapses into degenerate
        # repetition ("The capital of France is the capital of France is...")
        # instead of answering. Scored that way it lands BELOW CHANCE:
        # measured 40.5% acc_norm on hellaswag for 26b-a4b-4bit and 21.2% on
        # litbench (4-choice chance is 25%). Qwen is unaffected, which is why
        # this hid until a second family was scored.
        #
        # Prepend only when the tokenizer declares a BOS and the encoding
        # lacks it, so families that already handle it are untouched.
        bos = getattr(tok, "bos_token_id", None)

        def enc(s):
            ids = tok.encode(s)
            if bos is not None and (not ids or ids[0] != bos):
                ids = [bos] + ids
            return ids

        seqs, spans = [], []
        for ctx, cont in pairs:
            ctx_ids = enc(ctx)
            full_ids = enc(ctx + cont)
            # Guard the degenerate case where the join re-tokenizes the seam.
            if len(full_ids) <= len(ctx_ids):
                full_ids = ctx_ids + tok.encode(cont, add_special_tokens=False)
            n_cont = len(full_ids) - len(ctx_ids)
            if n_cont < 1:            # empty continuation: unscoreable
                n_cont = 1
            seqs.append(full_ids)
            spans.append((len(ctx_ids), n_cont))

        self._log(f"  {len(seqs)} sequences, "
                  f"{sum(len(s) for s in seqs)} tokens")
        if self.direct:
            # Model fits in RAM: use upstream's own forward (handles PLE,
            # shared KV, and sliding-window masks correctly by construction).
            return self._forward_direct(seqs, spans)
        hidden = self._forward_batches(seqs)

        rows, tgts, owner = [], [], []
        for i, (h, (ctx_len, n_cont)) in enumerate(zip(hidden, spans)):
            start = max(ctx_len - 1, 0)
            end = start + n_cont
            end = min(end, h.shape[0])
            if end <= start:
                start, end = 0, min(1, h.shape[0])
            rows.append(h[start:end])
            tgts.extend(seqs[i][start + 1:end + 1])
            owner.extend([i] * (end - start))

        stacked = mx.concatenate(rows, axis=0)
        # targets can run one short if a sequence ends exactly at the boundary
        if len(tgts) < stacked.shape[0]:
            tgts += [0] * (stacked.shape[0] - len(tgts))
        lp, gd = self._project(stacked, mx.array(tgts[:stacked.shape[0]]))
        mx.eval(lp, gd)
        lp, gd = lp.tolist(), gd.tolist()

        out = [[0.0, True] for _ in pairs]
        for k, i in enumerate(owner):
            out[i][0] += lp[k]
            out[i][1] = out[i][1] and bool(gd[k])
        return [(a, b) for a, b in out]


def _build_lm(model_path, batch_seqs, direct=False, allow_unmatched=False):
    """Subclass lm-eval's LM at runtime (keeps this module import-light)."""
    from lm_eval.api.model import LM

    class _LM(LM, StreamingLM):
        def __init__(self):
            LM.__init__(self)
            StreamingLM.__init__(self, model_path, batch_seqs=batch_seqs,
                                 direct=direct,
                                 allow_unmatched=allow_unmatched)

        def loglikelihood(self, requests):
            return self._loglikelihood_pairs([r.args for r in requests])

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError(
                "rolling loglikelihood is the referee's job — use "
                "referee/score_streaming.py for perplexity")

        def generate_until(self, requests):
            raise NotImplementedError(
                "this scorer is loglikelihood-only; the tasks it targets "
                "(hellaswag/piqa/winogrande) never generate")

    return _LM()


def selftest(model_path, batch_seqs, max_tokens=8192, corpus=None):
    """Reproduce the streaming referee's wikitext ppl through THIS code path.

    Same corpus, same prefix length, same math — but routed through the
    batched scorer. A match to 4 decimals is the licence to trust any task
    number this file produces. Known answers: VQ-2.2bpw 3.1706,
    VQ-2.4bpw 2.7655, VQ-3.1bpw 2.3519.
    """
    lm = StreamingLM(model_path, batch_seqs=batch_seqs)
    path = pathlib.Path(corpus or (pathlib.Path(__file__).parent / "referee"
                                   / "referee_corpus.txt"))
    text = path.read_text(errors="replace")
    ids = lm.tokenizer.encode(text)[: max_tokens + 1]
    # One sequence, scored as "predict everything after token 0" — identical
    # to the referee's contiguous full-forward.
    t0 = time.time()
    hidden = lm._forward_batches([ids[:-1]])
    stacked = hidden[0]
    lp, _ = lm._project(stacked, mx.array(ids[1:len(ids)]))
    mx.eval(lp)
    total_nll = -float(mx.sum(lp).item())
    scored = stacked.shape[0]
    npt = total_nll / scored
    print(json.dumps({
        "mode": "selftest-batched-streaming",
        "model": str(model_path).rstrip("/").split("/")[-1],
        "total_nll": round(total_nll, 4),
        "tokens_scored": scored,
        "nll_per_token": round(npt, 6),
        "ppl": round(math.exp(min(npt, 30.0)), 4),
        "seconds": round(time.time() - t0, 1),
    }, indent=1), flush=True)
    print("COMPARE against referee/score_streaming.py for this artifact; "
          "a mismatch invalidates every task number from this file.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="hellaswag,piqa,winogrande")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shots", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-seqs", type=int, default=256,
                    help="sequences per padded bucket in the layer loop")
    ap.add_argument("--direct", action="store_true",
                    help="skip layer streaming and call the model directly. "
                         "REQUIRED for gemma-4 (its DecoderLayer returns a "
                         "tuple, threads per-layer inputs and shared KV, and "
                         "alternates sliding-window masks that the streamed "
                         "path would silently get wrong). Use for any model "
                         "that fits in RAM; streaming is for the rest.")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="load even if the checkpoint carries tensors the "
                         "model class does not build. Investigation only: "
                         "the dropped tensors degrade the model silently.")
    ap.add_argument("--include-path", default=None,
                    help="directory of local lm-eval task YAMLs (e.g. literary/ "
                         "for litbench); stock tasks need no path")
    ap.add_argument("--selftest", action="store_true",
                    help="reproduce the referee wikitext ppl and exit")
    args = ap.parse_args()

    mx.set_cache_limit(8 << 30)

    if args.selftest:
        selftest(args.model, args.batch_seqs)
        return

    import lm_eval

    lm = _build_lm(args.model, args.batch_seqs, direct=args.direct,
                   allow_unmatched=args.allow_unmatched)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # Local task YAMLs (litbench) need a TaskManager pointed at their dir;
    # stock tasks resolve without one, so only build it when asked.
    task_manager = None
    if args.include_path:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager(include_path=args.include_path)
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model=lm, tasks=tasks, limit=args.limit,
        num_fewshot=args.num_shots, log_samples=True, bootstrap_iters=100000,
        task_manager=task_manager,
    )
    elapsed = time.time() - t0

    name = str(args.model).rstrip("/").split("/")[-1]
    summary = {
        "model": name,
        "tasks": tasks,
        "limit": args.limit,
        "num_shots": args.num_shots,
        "seconds": round(elapsed, 1),
        "harness": "lm-eval 0.4.12 via score_tasks_streaming.py "
                   "(layer-streamed loglikelihood)",
        "results": res["results"],
    }
    print(json.dumps(summary, indent=1), flush=True)

    if args.output_dir:
        out = pathlib.Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
        # per-sample records are what the PAIRED bootstrap needs: the models
        # all see the same items, so item difficulty cancels and the paired
        # interval is far tighter than the independent stderr.
        samples = {k: v for k, v in (res.get("samples") or {}).items()}
        if samples:
            (out / f"{name}.samples.json").write_text(json.dumps(samples))
        print(f"wrote {out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
