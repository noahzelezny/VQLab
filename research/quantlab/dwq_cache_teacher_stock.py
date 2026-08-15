#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Teacher targets for mlx_lm's stock DWQ trainer — computed by STREAMING.

mlx_lm.quant.dwq's own `compute_dwq_targets` runs the teacher resident
(`model(batch)`), which is impossible for the 751G bf16 397B on 96G/128G
boxes. This produces byte-compatible target files by streaming the teacher
block-by-block instead (E18 flat-memory machinery): identical data pipeline
(load_data + iterate_batches, same seed), identical file format
(<target-dir>/{train,valid}/NNNNNNNNNN.safetensors with top-1024 {logits,
indices}), so the stock trainer's `--target-dir` path consumes them as if
compute_dwq_targets had written them.

The ONLY divergence from stock: batches for ALL blocks must be fixed up
front (every block sees every batch before the next block loads), so this
holds every batch's activations at once. 2048x512 tokens x 4096 dims x bf16
= 8.6G — fine. Run on the M3 with the Thunderbay teacher.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def find_core(model):
    m = model
    for name in ("language_model", "model"):
        while hasattr(m, name):
            m = getattr(m, name)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--data-path", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--max-seq-length", type=int, default=513,
                    help="stock dwq default is 1025; batches train on "
                         "batch[:, :-1] so 513 -> 512-token sequences")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path override — MUST be the STUDENT's "
                         "tokenizer. Distillation targets are only valid on "
                         "the student's token sequences; if student and "
                         "teacher tokenizers differ (t2.4 does), the "
                         "length-sorted batch order diverges and every "
                         "target misaligns. Verify with an encode-compare "
                         "before trusting a cache.")
    args = ap.parse_args()

    # Stock data pipeline, stock seeding — order must match the trainer.
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    from mlx_lm.quant.dwq import load_data
    from mlx_lm.tuner.trainer import iterate_batches
    from mlx_lm.utils import load, load_tokenizer

    num_samples = args.num_samples
    tokenizer = load_tokenizer(args.tokenizer or args.teacher)
    train_data, valid_data = load_data(
        tokenizer, args.data_path, num_samples, args.max_seq_length)

    print(f"[1/2] teacher (lazy, cpu stream): {args.teacher}", flush=True)
    with mx.stream(mx.cpu):
        teacher, _, _ = load(args.teacher, return_config=True, lazy=True)
    core = find_core(teacher)
    n_blocks = len(core.layers)

    target_dir = Path(args.target_dir)

    def stream_split(data, split):
        out = target_dir / split
        out.mkdir(parents=True, exist_ok=True)
        # Fix the batches up front (same iterate order as the trainer).
        batches = []
        for batch, _len in iterate_batches(
                data, args.batch_size, args.max_seq_length, seed=args.seed):
            batches.append(mx.array(batch[:, :-1]))
        print(f"[2/2] {split}: {len(batches)} batches, streaming "
              f"{n_blocks} blocks", flush=True)

        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters())
        acts = [core.embed_tokens(b) for b in batches]
        mx.eval(acts)

        # NOTE: layers are consumed (set to None) — one split per load.
        for i in range(n_blocks):
            blk = core.layers[i]
            mask = None if blk.is_linear else "causal"
            with mx.stream(mx.cpu):
                mx.eval(blk.parameters())
            blk.eval()
            t0 = time.time()
            acts = [
                (lambda y: (mx.eval(y), y)[1])(blk(a, mask=mask, cache=None))
                for a in acts
            ]
            core.layers[i] = None
            del blk
            gc.collect()
            mx.clear_cache()
            print(f"  {split} block {i}/{n_blocks - 1}: "
                  f"{time.time() - t0:.1f}s "
                  f"(peak {mx.get_peak_memory() / 1024**3:.1f}G)", flush=True)

        head_norm = core.norm
        lm_head = getattr(teacher, "lm_head", None)
        with mx.stream(mx.cpu):
            mx.eval(head_norm.parameters(),
                    lm_head.parameters() if lm_head is not None
                    else core.embed_tokens.parameters())
        for i, a in enumerate(acts):
            h = head_norm(a)
            logits = (lm_head(h) if lm_head is not None
                      else core.embed_tokens.as_linear(h))
            # Stock format: top-1024 by argpartition on raw logits, same dtype
            idx = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]
            top = mx.take_along_axis(logits, idx, axis=-1)
            mx.eval(idx, top)
            mx.save_safetensors(str(out / f"{i:010d}.safetensors"),
                                {"logits": top, "indices": idx})
            del h, logits
            mx.clear_cache()
        print(f"{split}: wrote {len(acts)} target files", flush=True)

    # valid first (small, fast smoke of the whole path), then reload for train
    stream_split(valid_data, "valid")
    print("reloading teacher for train split", flush=True)
    del teacher, core
    gc.collect()
    mx.clear_cache()
    with mx.stream(mx.cpu):
        teacher, _, _ = load(args.teacher, return_config=True, lazy=True)
    core = find_core(teacher)
    stream_split(train_data, "train")

    (target_dir / "meta.json").write_text(json.dumps({
        "teacher": args.teacher, "data_path": args.data_path,
        "num_samples": num_samples, "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size, "seed": args.seed,
        "format": "mlx_lm.quant.dwq compute_dwq_targets-compatible",
    }, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
