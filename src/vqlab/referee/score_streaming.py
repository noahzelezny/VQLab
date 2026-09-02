#!/usr/bin/env python3
"""Single-box streaming referee — no sharding, no distribution, no E23.

E26c: tensor-sharded inference broke DETERMINISTICALLY on the first DWQ'd
397B artifact (uniform logits, ppl 202k) while the identical weights score
KL 0.0201 unsharded — the E23 kernel bug is value-sensitive, so 2-node
scoring can't be trusted per-artifact. This computes the SAME prefix-8192
metric by streaming blocks on one box (flat memory, ~15G): the whole prefix
goes through each block as one causal full-forward, which is mathematically
identical to the chunked+cache referee.

VALIDATE against a known answer before trusting any new number: champion
struct6-tail3x3 = 3.1580 on the 2-node referee, and
TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw on wikitext prefix-2048,
where this referee reads 5.9025 against the resident referee's published
5.9003 (scripts/score_ppl_resident.py, which runs the whole model at once).

That 0.0022 (0.037%, 3.9e-4 nats/token) is FULLY ATTRIBUTED and is the
floor for a streaming instrument, not slack to spend:

  - It is NOT the arch adaptation and NOT the streaming machinery. With
    chunk 2048 — where block-major and layer-major traversal coincide,
    because there is only one chunk — this loop reproduces the resident
    referee to four decimals: 5.9158 both, and it stays 5.9158 with lazy
    load, per-block CPU-stream parameter eval, per-block eval and layer
    freeing all switched on. Machinery: value-neutral, measured.
  - It IS evaluation ORDER. A streaming referee must run one block over
    all chunks before it can free that block; the resident referee runs
    one chunk through all blocks. The two are mathematically identical
    and differ in float16 only by non-associativity. Stripping every bit
    of streaming machinery — eager load, no freeing, no per-block eval —
    still reads 5.9025, so the residual is the traversal order itself.
  - The one thing that was NOT noise: loading inside `mx.stream(mx.cpu)`
    moved the number to 5.9247 (0.4%). See the load call in main().

Rule of thumb from the above: cross-instrument agreement on this family is
~0.04%; a margin smaller than that is not a result.

ARCH DISPATCH (2026-09-02). Streaming a block at a time means this referee
reimplements the model's own forward loop, so it is coupled to the block
API — and that API drifted underneath it. It used to assume every decoder
block was `blk(h, mask=..., cache=None)` behind a `blk.is_linear` flag. The
installed qwen4_exp graft (ml-explore PR #1788) renamed the flag to
`layer_type` and takes eight positional arguments, with hyper-connection
tiling before the stack and a gated-residual mixer instead of a final
`norm`. Against that arch the old loop raised — so it FAILED on shipped
artifacts and gated `check-release`.

The fix is a duck-typed plan: probe the block for the shape it presents,
build the prologue/epilogue that shape needs, and stream. Anything that
matches NEITHER shape raises with what was expected and what was found. A
referee that returns a wrong number is worse than one that crashes, so
there is deliberately no "best effort" fallback.
"""
import argparse
import gc
import json
import math
import pathlib
import time

import mlx.core as mx


def _module_globals(obj):
    """The namespace the object's class was defined in.

    A checkpoint that bundles its own model.py is exec'd under a synthetic
    module name that is never registered in sys.modules, so
    `sys.modules[cls.__module__]` is None for exactly the artifacts this
    referee exists to score. The class's own functions close over the real
    module dict. Same trick as smoke.py::_class_ns.
    """
    import sys
    cls = type(obj)
    mod = sys.modules.get(cls.__module__)
    if mod is not None and getattr(mod, "__file__", None):
        return vars(mod)
    for attr in vars(cls).values():
        g = getattr(attr, "__globals__", None)
        if g is not None and g.get("__file__"):
            return g
    return {}


def _need(ns, name, arch):
    fn = ns.get(name)
    if fn is None:
        raise RuntimeError(
            f"streaming referee: the {arch} arch needs `{name}` from the "
            f"module that defines its blocks, and that module does not "
            f"export it. Refusing to score: a substituted mask would "
            f"produce a plausible, wrong number.")
    return fn


def _plan(model, core, ids, chunk):
    """Return (arch, h0, step, head_apply, head_names).

    `step(blk, i, h)` runs decoder block `i` over the whole prefix.
    `head_apply(h)` is everything between the last block and the lm_head.
    `head_names` are the attributes on `core` the head reads, so a
    multi-trial run can restore them.
    """
    blk0 = core.layers[0]

    # ---- shape A: qwen4_exp / PR #1788 — layer_type + 8-arg block ---------
    if hasattr(blk0, "layer_type"):
        ns = _module_globals(blk0)
        create_attention_mask = _need(ns, "create_attention_mask", "layer_type")
        create_ssm_mask = _need(ns, "create_ssm_mask", "layer_type")
        args = core.args
        h = core.embed_tokens(ids)

        # WHY THIS ARCH IS CHUNKED, and the classic one is not.
        #
        # This referee's original claim was that one causal full-forward is
        # "mathematically identical to the chunked+cache referee". That
        # holds for a pure transformer. It does NOT hold here: 36 of the 48
        # blocks are `linear_attention` (gated deltanet), a RECURRENCE, and
        # its chunkwise-parallel update is only associative up to floating
        # point. Measured on the shipped 2.1bpw artifact, wikitext prefix
        # 2048, resident referee, ONLY the chunk size changed:
        #     chunk 512  -> ppl 5.9003   (the published number)
        #     chunk 2048 -> ppl 5.9158   (+0.26%)
        # So "the metric" is not chunk-free for this arch, and a full-
        # forward streaming referee would report a number no other
        # instrument agrees with. It therefore streams BLOCK-MAJOR: each
        # block gets its own cache and sees the prefix in `chunk`-sized
        # pieces in order, then is freed. That is the same SEQUENCE OF CALLS
        # the resident referee makes for that block, with flat memory. Not
        # the same arithmetic order across blocks, though — see the residual
        # 0.037% in the module docstring; the calls match, the float16
        # rounding does not.
        #
        # Per-block masks: the model builds mask/conv_mask/prev_ctx from a
        # REPRESENTATIVE layer's cache (`cache[full_idx[0]]` etc.), which
        # assumes every layer is at the same offset — false in block-major
        # order. Each block's own cache carries the identical offset for its
        # own chunk, so this uses that instead.
        caches = model.make_cache()
        if len(caches) != len(core.layers):
            raise RuntimeError(
                f"streaming referee: make_cache() returned {len(caches)} "
                f"caches for {len(core.layers)} blocks; cannot pair them.")
        ple_layers = list(getattr(core, "ple_layers", []) or [])
        if len(ple_layers) > 1:
            raise RuntimeError(
                "streaming referee: this model has %d PLE layers, and the "
                "arch shares ONE layer's n-gram context cache across all of "
                "them — an assumption block-major streaming breaks. Score "
                "it resident (scripts/score_ppl_resident.py) instead of "
                "getting a quietly wrong number here." % len(ple_layers))
        ctx_len = args.ngram_size - 1
        eos = args.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos

        # hyper-connections: the stack runs on hc copies of the residual.
        h = mx.tile(h, (1, 1, core.hc))
        rope = core.rope
        S = ids.shape[1]

        def step(blk, i, h):
            c = caches[i]
            outs = []
            for s in range(0, S, chunk):
                e = min(s + chunk, S)
                hs = h[:, s:e]
                ids_s = ids[:, s:e]
                if blk.layer_type == "full_attention":
                    mask = create_attention_mask(ids_s[..., None], c)
                    conv_mask = None
                else:
                    mask = None
                    conv_mask = create_ssm_mask(ids_s[..., None], c)
                prev_ctx = None
                if blk.ple is not None:
                    prev = c[3] if c is not None else None
                    prev_ctx = (prev if prev is not None else
                                mx.full((ids.shape[0], ctx_len), eos,
                                        ids.dtype))
                    history = mx.concatenate([prev_ctx, ids_s], axis=1)
                    if c.lengths is not None:
                        ends = mx.clip(c.lengths, 0, ids_s.shape[1])
                        c[3] = mx.take_along_axis(
                            history, ends[:, None] + mx.arange(ctx_len),
                            axis=1)
                    else:
                        c[3] = history[:, -ctx_len:]
                idx_c = c.indexer if (c is not None and
                                      hasattr(c, "indexer")) else None
                o = blk(hs, rope, mask, conv_mask, c, idx_c, ids_s, prev_ctx)
                mx.eval(o)
                outs.append(o)
            caches[i] = None
            return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

        mixer = core.hyper_connection_mixer   # this arch has no final `norm`

        def head_apply(h):
            return mixer(h)

        return "qwen4_exp/layer_type", h, step, head_apply, \
            ["hyper_connection_mixer", "rope"]

    # ---- shape B: the classic mlx-lm block — is_linear + kwargs ----------
    if hasattr(blk0, "is_linear"):
        # Unchanged from the pre-2026-09-02 referee ON PURPOSE: its published
        # numbers (champion struct6-tail3x3 = 3.1580) were measured by this
        # exact full-forward, and for a stack of plain causal blocks the
        # full-forward and the chunked+cache result agree.
        h = core.embed_tokens(ids)
        norm = core.norm

        def step(blk, i, h):
            return blk(h, mask=None if blk.is_linear else "causal", cache=None)

        def head_apply(h):
            return norm(h)

        return "classic/is_linear", h, step, head_apply, ["norm"]

    raise RuntimeError(
        "streaming referee: unrecognised decoder block API.\n"
        f"  block class : {type(blk0).__module__}.{type(blk0).__name__}\n"
        f"  defined in  : {_module_globals(blk0).get('__file__', '?')}\n"
        "  expected    : `.layer_type` (qwen4_exp / PR #1788, 8-arg call) "
        "or `.is_linear` (classic mlx-lm block)\n"
        f"  found attrs : "
        f"{sorted(a for a in vars(blk0) if not a.startswith('_'))}\n"
        "Refusing to score rather than guess a forward signature.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512,
                    help="prefill chunk for archs whose blocks carry a "
                         "recurrent state (qwen4_exp's gated deltanet). The "
                         "metric is NOT chunk-invariant there — 512 is the "
                         "size every published number on this family was "
                         "measured at. Ignored by the classic full-forward "
                         "path, which is chunk-invariant.")
    ap.add_argument("--corpus", default=None,
                    help="corpus file (default: referee_corpus.txt, wikitext). "
                         "Score a code corpus too (scripts/make_code_corpus.py "
                         "builds one) — a margin chosen on one corpus is "
                         "exactly the claim a domain shift can erase. Never "
                         "compare PPL ACROSS corpora — only models within "
                         "one corpus.")
    args = ap.parse_args()

    mx.set_cache_limit(8 << 30)

    from mlx_lm.utils import load
    # NOT `with mx.stream(mx.cpu)`. Running load() on the CPU stream was a
    # measured 0.15% error source: the loader does real work (sanitize,
    # casts, the bundle's own module construction), and doing it in CPU
    # arithmetic instead of Metal changes the weights in their low bits.
    # Measured, block-major chunk-2048, ONLY the load stream changed:
    #     default stream -> 5.9158   (agrees with the resident referee)
    #     mx.cpu stream  -> 5.9247   (agrees with nothing)
    # Per-block parameter eval below stays on the CPU stream — that one was
    # measured to be value-neutral, and it is what keeps memory flat.
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
        ids = toks[: n - 1][None]
        arch, h, step, head_apply, head_names = _plan(model, core, ids,
                                                      args.chunk)
        if r == 0:
            print(f"arch: {arch}", flush=True)
        mx.eval(h)
        t0 = time.time()
        for i in range(n_blocks):
            blk = core.layers[i]
            with mx.stream(mx.cpu):
                mx.eval(blk.parameters())
            blk.eval()
            h = step(blk, i, h)
            mx.eval(h)
            core.layers[i] = None
            del blk
            gc.collect()
            mx.clear_cache()
        lm_head = getattr(model, "lm_head", None) or getattr(
            getattr(model, "language_model", model), "lm_head", None)
        with mx.stream(mx.cpu):
            for hn in head_names:
                hm = getattr(core, hn, None)
                if hasattr(hm, "parameters"):
                    mx.eval(hm.parameters())
            mx.eval(lm_head.parameters() if lm_head is not None
                    else core.embed_tokens.parameters())
        total_nll, scored = 0.0, 0
        targets = toks[1:n]
        step_sz = 1024
        hh = head_apply(h)[0]
        for s in range(0, n - 1, step_sz):
            e = min(s + step_sz, n - 1)
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
            "arch": arch,
            "chunk": args.chunk,
            "total_nll": round(total_nll, 4),
            "tokens_scored": scored,
            "nll_per_token": round(nll_per_token, 6),
            "ppl": round(math.exp(min(nll_per_token, 30.0)), 4),
            "seconds": round(time.time() - t0, 1),
        }), flush=True)
        if r + 1 < args.trials:
            # default stream, for the same reason as the first load
            model2, _, _ = load(args.model, lazy=True, return_config=True)
            core2 = model2
            for name in ("language_model", "model"):
                while hasattr(core2, name):
                    core2 = getattr(core2, name)
            core.layers = core2.layers
            core.embed_tokens = core2.embed_tokens
            for hn in head_names:
                if hasattr(core2, hn):
                    setattr(core, hn, getattr(core2, hn))

    if args.trials > 1:
        ppls = [math.exp(min(x, 30.0)) for x in results]
        spread = (max(ppls) - min(ppls)) / min(ppls)
        print(f"DETERMINISM: {len(ppls)} runs, spread {spread:.2%} "
              f"({'PASS' if spread < 0.001 else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()
