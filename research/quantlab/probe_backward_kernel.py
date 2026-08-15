#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E26 diagnosis: WHICH op in the 397B tail backward exceeds the Metal
watchdog? Times one DecoderLayer's forward and backward in isolation, at two
sequence lengths, single box, no distribution. If backward time is
~seq-independent, the long op is the weight-sized quantized-matmul VJP
(dL/dscales traverses the full expert tensor); if it scales with seq, it's
the GDN/attention path. Run on the M4 (the rank that times out).
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=57)
    args = ap.parse_args()

    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, _, _ = load(args.model, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    blk = core.layers[args.layer]
    print(f"layer {args.layer}: {'linear' if blk.is_linear else 'full'}-attn",
          flush=True)
    with mx.stream(mx.cpu):
        mx.eval(blk.parameters())
    print(f"materialized {mx.get_active_memory() / 1024**3:.1f}G", flush=True)

    def unfreeze(_, m):
        if (hasattr(m, "bits") and hasattr(m, "group_size")
                and getattr(m, "mode", None) == "affine" and m.bits < 8):
            m.unfreeze(keys=["scales", "biases"], recurse=False)
    blk.freeze()
    blk.apply_to_modules(unfreeze)
    blk.train()

    for S in (128, 512):
        x = mx.random.normal((1, S, 4096)).astype(mx.bfloat16)
        mask = None if blk.is_linear else "causal"

        # forward only
        t0 = time.time()
        y = blk(x, mask=mask, cache=None)
        mx.eval(y)
        t_fwd = time.time() - t0

        params = blk.trainable_parameters()

        def loss_fn(p, xin):
            blk.update(p)
            return blk(xin, mask=mask, cache=None).astype(mx.float32).sum()

        t0 = time.time()
        _, grads = mx.value_and_grad(loss_fn)(params, x)
        mx.eval(grads)
        t_bwd = time.time() - t0
        print(f"seq {S}: fwd {t_fwd:.2f}s  fwd+bwd {t_bwd:.2f}s", flush=True)

    # separate timing: the shared-expert vs switch-expert grads
    print("done", flush=True)


if __name__ == "__main__":
    main()
