#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E26: cache the STUDENT's frozen-prefix activations for solo tail training.

The first (n_layers - tail) layers are frozen during tail-DWQ, so their
outputs are constant across training — recomputing them every step is what
made distributed training necessary (and the Metal watchdog killed that).
Stream them ONCE over the same batches as the teacher-target cache and save
each batch's layer-K activations. Training then needs only tail + norm +
head on one box.

MUST match the target cache exactly: same data pipeline, seed, batch order,
and the student's tokenizer (= teacher's for the 397B pair, verified E26).
"""
import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tail-layers", type=int, default=6)
    ap.add_argument("--data-path", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--max-seq-length", type=int, default=513)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    from mlx_lm.quant.dwq import load_data
    from mlx_lm.tuner.trainer import iterate_batches
    from mlx_lm.utils import load

    with mx.stream(mx.cpu):
        model, tokenizer, _ = load(args.student, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    n_blocks = len(core.layers)
    cut = n_blocks - args.tail_layers
    print(f"prefix = layers [0, {cut}) of {n_blocks}", flush=True)

    train_data, valid_data = load_data(
        tokenizer, args.data_path, args.num_samples, args.max_seq_length)

    out = Path(args.out_dir)

    def stream_split(data, split):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        batches = []
        lengths_all = []
        for batch, lengths in iterate_batches(
                data, args.batch_size, args.max_seq_length, seed=args.seed):
            batches.append(mx.array(batch[:, :-1]))
            lengths_all.append(np.array(lengths))
        print(f"{split}: {len(batches)} batches", flush=True)

        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters())
        acts = [core.embed_tokens(b) for b in batches]
        mx.eval(acts)

        for i in range(cut):
            blk = core.layers[i]
            mask = None if blk.is_linear else "causal"
            with mx.stream(mx.cpu):
                mx.eval(blk.parameters())
            blk.eval()
            t0 = time.time()
            nxt = []
            for a in acts:
                y = blk(a, mask=mask, cache=None)
                mx.eval(y)
                nxt.append(y)
            acts = nxt
            core.layers[i] = None
            del blk
            gc.collect()
            mx.clear_cache()
            print(f"  {split} block {i}/{cut - 1}: {time.time() - t0:.1f}s",
                  flush=True)

        for i, (a, ln) in enumerate(zip(acts, lengths_all)):
            mx.save_safetensors(str(d / f"{i:010d}.safetensors"),
                                {"acts": a.astype(mx.bfloat16),
                                 "lengths": mx.array(ln)})
        print(f"{split}: wrote {len(acts)} act files", flush=True)

    stream_split(valid_data, "valid")
    with mx.stream(mx.cpu):
        model2, _, _ = load(args.student, lazy=True, return_config=True)
    core2 = model2
    for name in ("language_model", "model"):
        while hasattr(core2, name):
            core2 = getattr(core2, name)
    # stream_split closes over `core`; repoint its consumed pieces at the
    # fresh load (valid pass set the prefix layers to None).
    core.layers = core2.layers
    core.embed_tokens = core2.embed_tokens
    stream_split(train_data, "train")

    (out / "meta.json").write_text(json.dumps({
        "student": args.student, "tail_layers": args.tail_layers,
        "cut": cut, "num_samples": args.num_samples, "seed": args.seed,
    }, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
