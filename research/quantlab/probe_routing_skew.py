#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Measure REAL expert-routing skew, per layer, over the referee corpus.

WHY: the layer axis of quantization is now mapped and dialed (E25 + the 141 GiB
matched-size sweep), and the next unexplored axis is per-EXPERT precision —
512 experts per layer, currently all forced to one bit-width because they live
in a single batched [512, 4096, 16] tensor. Splitting each layer's experts into
a hot group and a cold group (two SwitchLinear calls, no new kernel) would make
mixed-width shards expressible.

BUT that whole idea rests on an assumption worth falsifying FIRST, cheaply: that
some experts matter much more than others. MoE training uses a load-balancing
auxiliary loss whose entire purpose is to FLATTEN expert usage. If Qwen3.5's
routing is near-uniform on real text, there is no hot group to promote, the
60x512 knobs are all the same knob, and the idea dies here for the price of one
forward pass instead of a weekend of kernel wrangling.

Reads routing straight off SwitchGLU.__call__(x, indices) — the indices ARE the
selections, so no model surgery. Streams blocks and frees them exactly like
referee/score_streaming.py, so it runs in ~15G on one box.

Output per layer: share of routing mass taken by the top 10% / 25% of experts,
the Gini coefficient, and the busiest:quietest ratio. Interpretation:
  gini < 0.10  -> uniform. Load balancing won; per-expert bits buy nothing.
  gini > 0.25  -> real skew. A hot/cold split has something to allocate.
"""
import argparse
import gc
import json
import pathlib

import mlx.core as mx


def gini(counts):
    """0 = every expert used equally, 1 = one expert takes everything."""
    x = sorted(float(c) for c in counts)
    n = len(x)
    tot = sum(x)
    if tot <= 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(x, 1):
        cum += i * v
    return (2 * cum) / (n * tot) - (n + 1) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None, help="write per-layer JSON here")
    ap.add_argument("--corpus", default=None,
                    help="text file to route (default: the referee corpus). "
                         "The point of this flag is the STABILITY question: "
                         "skew alone is useless if the hot set differs per "
                         "domain, because a static bit allocation tuned to one "
                         "corpus overfits it exactly the way E27/E28's trained "
                         "scales overfit tulu. Run wikitext vs code vs prose "
                         "and compare the top-25% sets.")
    args = ap.parse_args()

    mx.set_cache_limit(8 << 30)

    from mlx_lm.utils import load
    from mlx_lm.models import switch_layers

    # Capture the routing decisions without touching the model: wrap the one
    # method that already receives them. `current` is set by the streaming loop
    # below, so counts are attributed to the right layer.
    state = {"layer": -1, "counts": {}, "n_experts": 0}
    orig_call = switch_layers.SwitchGLU.__call__

    def spy(self, x, indices):
        idx = mx.array(indices).reshape(-1)
        n_exp = self.gate_proj.weight.shape[0] if hasattr(
            self.gate_proj, "weight") else int(idx.max().item()) + 1
        state["n_experts"] = max(state["n_experts"], n_exp)
        c = mx.zeros((n_exp,), dtype=mx.int32)
        c = c.at[idx].add(mx.ones(idx.shape, dtype=mx.int32))
        mx.eval(c)
        prev = state["counts"].get(state["layer"])
        state["counts"][state["layer"]] = c if prev is None else prev + c
        return orig_call(self, x, indices)

    switch_layers.SwitchGLU.__call__ = spy

    with mx.stream(mx.cpu):
        model, tokenizer, _ = load(args.model, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    n_blocks = len(core.layers)

    text = pathlib.Path(
        args.corpus or (pathlib.Path(__file__).parent / "referee"
                        / "referee_corpus.txt")).read_text(errors="replace")
    toks = mx.array(tokenizer.encode(text))[: args.max_tokens + 1]
    n = toks.shape[0]
    print(f"corpus: {n} tokens, {n_blocks} blocks", flush=True)

    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    h = core.embed_tokens(toks[: n - 1][None])
    mx.eval(h)
    for i in range(n_blocks):
        state["layer"] = i
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

    rows = []
    for layer in sorted(state["counts"]):
        c = [int(v) for v in state["counts"][layer].tolist()]
        tot = sum(c) or 1
        srt = sorted(c, reverse=True)
        k10 = max(1, len(c) // 10)
        k25 = max(1, len(c) // 4)
        rows.append({
            "layer": layer,
            "experts": len(c),
            "top10pct_share": round(sum(srt[:k10]) / tot, 4),
            "top25pct_share": round(sum(srt[:k25]) / tot, 4),
            "gini": round(gini(c), 4),
            "busiest_over_quietest": (round(srt[0] / srt[-1], 2)
                                      if srt[-1] else None),
            "unused_experts": sum(1 for v in c if v == 0),
            # The identity of the hot set, not just its concentration — this
            # is what a cross-corpus comparison needs.
            "top25pct_ids": sorted(range(len(c)), key=lambda i: -c[i])[:k25],
            "counts": c,
        })

    for r in rows:
        print(json.dumps(r), flush=True)

    if rows:
        g = sum(r["gini"] for r in rows) / len(rows)
        t10 = sum(r["top10pct_share"] for r in rows) / len(rows)
        uniform = 0.10
        print(f"\nMEAN gini {g:.4f} | mean top-10% share {t10:.4f} "
              f"(uniform would be {uniform:.4f})", flush=True)
        print("VERDICT: " + (
            "FLAT — load balancing won; per-expert bit allocation has nothing "
            "to allocate. Idea dies cheap." if g < 0.10 else
            "SKEWED — a hot/cold expert split has real mass to work with."
            if g > 0.25 else
            "MILD — measurable but small; weigh against tail depth "
            "(~0.03 PPL per 0.7 GiB) before building anything."), flush=True)

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
