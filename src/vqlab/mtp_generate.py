"""Generate with MTP speculative drafting, and optionally measure the speedup.

    python -m vqlab.cli mtp-generate --model <artifact> [--sidecar <file>]
        [--prompt ...] [--tokens N] [--benchmark]

The head drafts token t+2 from (trunk hidden at t, embedding of t+1), so each
step verifies one speculative token inside a single 2-token trunk forward:

  accepted -> 2 tokens for 1 trunk forward
  rejected -> the trunk's own t+2 came out of that same forward, so we still
              emit 2 tokens, but roll the caches back and replay the pair so
              the state matches what was actually emitted.

Rollback is O(1). Every qwen4_exp cache slot is REASSIGNED rather than mutated
(cache[0] = ..., cache[1] = state) and mlx arrays are immutable, so keeping
the old references is a free snapshot; attention rolls back through the
supported trim() path, by the offset DELTA rather than a fixed count.

Quality: the trunk verifies every drafted token, so drafting cannot introduce
a token the trunk would not have produced from the same forward. What it does
NOT give is bit-identical output against single-token decoding -- see
--benchmark's chunk control: MLX's chunked and single-token kernels disagree
at genuine near-ties (measured top-2 logit gaps 0.25 and 0.00 against a median
of 3.625), and verification necessarily happens inside a 2-token forward.
"""
import argparse
import importlib
import json
import pathlib
import sys
import time

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mtp_head import SIDECAR_NAME, MTPHead


def snapshot(caches):
    snaps = []
    for c in caches:
        if hasattr(c, "keys"):
            snaps.append(("attn", c.offset, None))
        else:
            snaps.append(("arr", getattr(c, "offset", None), list(c.cache)))
    return snaps


def restore(caches, snaps):
    """Back to exactly where the snapshot was taken. Attention rolls back by
    the offset DELTA: trimming a hardcoded 1 leaves a stale key behind while
    the recurrent caches go back 2, and the streams silently drift."""
    for c, s in zip(caches, snaps):
        if s[0] == "attn":
            n = c.offset - s[1]
            if n > 0:
                c.trim(n)
        else:
            c.cache = list(s[2])
            if s[1] is not None:
                c.offset = s[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", default=None,
                    help=f"quantized MTP sidecar (default: {SIDECAR_NAME} in "
                         "the artifact dir, if present)")
    ap.add_argument("--prompt", default="Explain why vector quantization "
                    "compresses neural network weights better than scalar "
                    "rounding.")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--benchmark", action="store_true",
                    help="also run plain greedy decoding and report the "
                         "speedup plus the chunked/single-token control")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from mlx_lm.utils import load
    model, tok = load(a.model, lazy=False, trust_remote_code=True)
    core = model.model
    arch = importlib.import_module(type(core).__module__)
    tie = model.args.text.tie_word_embeddings

    side = pathlib.Path(a.sidecar) if a.sidecar else \
        pathlib.Path(a.model) / SIDECAR_NAME
    if not side.exists():
        raise SystemExit(f"no MTP sidecar at {side}; build one with "
                         f"`vqlab mtp-pack` (the head is optional -- without "
                         f"it this model decodes normally)")
    before = mx.get_active_memory()
    head = MTPHead.from_sidecar(model, arch, side)
    mx.clear_cache()
    print(f"MTP head resident: "
          f"{(mx.get_active_memory() - before) / 2**30:.2f} GiB", flush=True)

    grab = {}

    class _Spy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            grab["h"] = x
            return self.inner(x)

    core.hyper_connection_mixer = _Spy(core.hyper_connection_mixer)

    def logits_of(out):
        return core.embed_tokens.as_linear(out) if tie else model.lm_head(out)

    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True)
    prompt = mx.array([ids])

    def speculative(n_tokens=None, timed=True):
        n_tokens = a.tokens if n_tokens is None else n_tokens
        cache = model.make_cache()
        mcache = arch._AttnCache()
        lg = model(prompt, cache=cache)
        h_last = grab["h"][:, -1:]
        t1 = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(t1, h_last)
        out, acc, steps = [], 0, 0
        t0 = time.perf_counter()
        while len(out) < n_tokens:
            steps += 1
            msnap = snapshot([mcache])
            d2 = mx.argmax(head.draft_logits(h_last, t1[None], mcache)[:, -1],
                           axis=-1)
            mx.eval(d2)
            snap = snapshot(cache)
            lg2 = model(mx.concatenate([t1, d2])[None], cache=cache)
            true_t2 = mx.argmax(lg2[:, 0], axis=-1)
            mx.eval(true_t2)
            if bool((true_t2 == d2).item()):
                acc += 1
                out.extend([int(t1.item()), int(d2.item())])
            else:
                restore(cache, snap)
                restore([mcache], msnap)
                out.extend([int(t1.item()), int(true_t2.item())])
                lg2 = model(mx.concatenate([t1, true_t2])[None], cache=cache)
            h_last = grab["h"][:, -1:]
            t1 = mx.argmax(lg2[:, 1], axis=-1)
            mx.eval(t1, h_last)
        return out[: n_tokens], time.perf_counter() - t0, acc / steps, steps

    speculative(n_tokens=8)          # warm the kernels
    spec_out, spec_s, acc, steps = speculative()
    print(f"\n{tok.decode(spec_out)}\n", flush=True)
    print(f"speculative: {a.tokens} tok in {spec_s:.2f}s = "
          f"{a.tokens / spec_s:.2f} tok/s  ({steps} steps, "
          f"acceptance {acc:.3f})", flush=True)
    rec = {"model": a.model, "sidecar": str(side), "tokens": a.tokens,
           "speculative_tok_s": round(a.tokens / spec_s, 2),
           "acceptance": round(acc, 4), "steps": steps}

    if a.benchmark:
        # warm the single-token path too, for the same reason
        c0 = model.make_cache()
        lg0 = model(prompt, cache=c0)
        t0w = mx.argmax(lg0[:, -1], axis=-1)
        for _ in range(8):
            lg0 = model(t0w[None], cache=c0)
            t0w = mx.argmax(lg0[:, -1], axis=-1)
        mx.eval(t0w)

        cache = model.make_cache()
        lg = model(prompt, cache=cache)
        t = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(t)
        base = []
        t0 = time.perf_counter()
        for _ in range(a.tokens):
            base.append(int(t.item()))
            lg = model(t[None], cache=cache)
            t = mx.argmax(lg[:, -1], axis=-1)
            mx.eval(t)
        base_s = time.perf_counter() - t0
        print(f"baseline:    {a.tokens} tok in {base_s:.2f}s = "
              f"{a.tokens / base_s:.2f} tok/s", flush=True)
        print(f"SPEEDUP: {base_s / spec_s:.2f}x", flush=True)

        # Does the trunk reproduce its OWN greedy choices when the same tokens
        # are fed as one chunk? This separates runtime numerics (which every
        # correct speculative implementation inherits) from a rollback bug.
        c2 = model.make_cache()
        full = mx.array([list(ids) + base[:-1]])
        lgf = model(full, cache=c2)
        pred = mx.argmax(lgf[0, len(ids) - 1:], axis=-1)
        mx.eval(pred)
        bad = [i for i, (p, b) in enumerate(zip(
            [int(x) for x in pred.tolist()], base)) if p != b]
        row = lgf[0, len(ids) - 1:].astype(mx.float32)
        top2 = mx.sort(row, axis=-1)[:, -2:]
        gaps = top2[:, 1] - top2[:, 0]
        mx.eval(gaps)
        print(f"chunk control: chunked vs single-token greedy disagree at "
              f"{len(bad)}/{len(base)} positions"
              + (f"; top-2 logit gaps there "
                 f"{[round(float(gaps[i].item()), 3) for i in bad]} vs median "
                 f"{float(mx.median(gaps).item()):.3f}" if bad else ""),
              flush=True)
        rec.update({"baseline_tok_s": round(a.tokens / base_s, 2),
                    "speedup": round(base_s / spec_s, 3),
                    "outputs_identical": base == spec_out,
                    "chunk_control_disagreements": len(bad)})

    print(json.dumps(rec), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
