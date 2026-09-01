#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""VQ-PF1 probe #2: time ONE REAL transformer block at the scorer's bucket
shape, VQ artifact vs spicyneuron affine artifact.

probe_vq_prefill_regime.py already showed the isolated VQ expert kernel is
at PARITY with gather_qmm at the scorer's shape (177 rows/expert), so the
~9x seen end-to-end is NOT in the expert kernel. This probe reproduces the
scorer's inner loop -- `blk(state, mask="causal")` over a [batch_seqs, L]
padded bucket -- on a real block, and reports per-bucket wall time plus a
breakdown of first-call vs steady-state.

  ./probe_block_prefill.py --model <artifact dir> [--layer 3] [--buckets 4]
"""
import argparse
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--batch-seqs", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=36,
                    help="padded bucket length; hellaswag/piqa avg is ~36")
    ap.add_argument("--buckets", type=int, default=4)
    args = ap.parse_args()

    mx.set_cache_limit(8 << 30)
    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, _, _ = load(args.model, lazy=True, return_config=True)

    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    blk = core.layers[args.layer]

    t0 = time.perf_counter()
    with mx.stream(mx.cpu):
        mx.eval(blk.parameters())
    blk.eval()
    t_load = time.perf_counter() - t0

    B, L = args.batch_seqs, args.seq_len
    # NOT embed_tokens.weight.shape[-1]: quantized embeddings are packed
    # uint32, so that reports 768 for a 4096-wide model.
    hidden = blk.input_layernorm.weight.shape[-1]
    print(f"model  {args.model.rstrip('/').split('/')[-1]}")
    print(f"  layer {args.layer}  bucket [{B}, {L}] = {B*L} tokens  "
          f"hidden={hidden}")
    print(f"  block weight materialize: {t_load:.2f} s")

    times = []
    for i in range(args.buckets):
        h = mx.random.normal((B, L, hidden)).astype(mx.float16)
        mx.eval(h)
        mx.synchronize()
        t0 = time.perf_counter()
        out = blk(h, mask="causal", cache=None)
        mx.eval(out)
        mx.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    bucket {i}: {dt*1e3:9.1f} ms")
        del h, out
        mx.clear_cache()

    steady = times[1:] or times
    per_bucket = sum(steady) / len(steady)
    print(f"  steady-state per bucket: {per_bucket*1e3:.1f} ms")
    print(f"  => 32 buckets x 60 blocks = "
          f"{per_bucket*32*60/60:.1f} s/block-equivalent... "
          f"per block: {per_bucket*32:.1f} s, "
          f"full sweep: {per_bucket*32*60/3600:.2f} h")


if __name__ == "__main__":
    main()
