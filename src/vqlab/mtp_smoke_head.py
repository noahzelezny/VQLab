"""Head-alone smoke probe: does a packed sidecar load and forward, and what
does one draft step cost at T=1 vs T=2?

    python -m vqlab.cli mtp-smoke-head --model <artifact> --sidecar <s.st>
        [--steps 20]

This is NOT an acceptance probe -- it never runs the trunk, so it says
nothing about whether the head drafts the right tokens (that is
`mtp-probe35` / `mtp-accept`, both of which need the whole trunk resident).
What it does answer, on a box far too small to hold the trunk:

  * the sidecar reloads into the registered head class, with the stored
    quantization recipe replayed and every parameter slot filled;
  * a forward runs at T=1 and at T=2 without shape errors -- T=2 is the
    real speculative case (one row per COMMITTED token advances the head's
    cache two positions per step), and it is the path where a missing
    attention mask or a mis-shaped cache shows up;
  * what a draft step costs, which sets the ceiling on any speedup: if the
    head's own forward is a large fraction of the trunk's, speculation
    cannot pay for itself no matter how good acceptance is.

The trunk is loaded LAZILY and only `embed_tokens` is ever touched, so this
costs the sidecar plus one embedding table, not the model.
"""
import argparse
import pathlib
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vqlab.mtp import registry


def _time(fn, *, warmup, iters):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3      # ms/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="artifact dir (lazy; only embed_tokens is read)")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--family", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()

    from mlx_lm.utils import load
    model, _ = load(a.model, lazy=True)
    spec = registry.resolve(model, a.family)
    arch = spec.arch_module(model)
    head = spec.head_cls().from_sidecar(model, arch, a.sidecar)
    print(f"family {spec.name} -> {spec.head}", flush=True)
    print(f"  sidecar {pathlib.Path(a.sidecar).stat().st_size / 2**30:.2f} "
          f"GiB on disk, resident after load "
          f"{mx.get_active_memory() / 2**30:.2f} GiB", flush=True)

    D = head.D
    vocab = model.language_model.args.vocab_size if hasattr(
        model, "language_model") else model.args.vocab_size
    make_cache = getattr(head, "make_draft_cache", None)

    for T in (1, 2):
        # One cache object, not a list: the head owns a single block, and
        # the loop passes it the same way (loop.py's `dcache`).
        cache = (make_cache() if make_cache is not None
                 else spec.make_draft_cache(arch))
        h = mx.random.normal((1, T, D)).astype(mx.bfloat16)
        ids = mx.random.randint(0, vocab, (1, T))
        ms = _time(lambda: head._trunk(h, ids, cache),
                   warmup=a.warmup, iters=a.steps)
        print(f"  T={T}: {ms:8.2f} ms/forward  ({ms / T:6.2f} ms/position)",
              flush=True)

    print(f"  peak resident {mx.get_peak_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
