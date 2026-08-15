#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Cache the bf16 teacher's top-k logits for end-to-end DWQ at 397B.

WHY: E16 proved sequence-level DWQ works (35B: 8.474->8.345 at same size),
but it needs teacher logits and the 751G bf16 teacher cannot be resident on
any box (96G + 128G). Block-wise DWQ (E18/E20) is disproven. This decouples
the two: stream the teacher ONCE, block by block (flat memory, the E18
machinery), over the fixed calibration corpus, then project through the
final norm + lm_head and save top-k logprobs + indices per position. The
student then trains end-to-end against the cache with NO teacher in memory.

Cache size is trivial: 512 samples x 512 tokens x k=64 x 8B ~ 1.1 GiB.

Corpus: same default as every DWQ run here (tulu-3 mixture) — NEVER the
referee wikitext-test slice (that is the exam, not the study guide).

Output dir gets: teacher_topk.safetensors {indices int32 [N,S,k],
logprobs float16 [N,S,k]}, tokens.safetensors {tokens int32 [N,S]},
meta.json (corpus/seed/shapes/teacher path).
"""
import argparse
import gc
import json
import math
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


def lazy_load(path):
    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, tokenizer, config = load(path, return_config=True, lazy=True)
    return model, tokenizer, config


def build_batches(tokenizer, data_path, num_samples, seq_len, batch_size, seed):
    from mlx_lm.quant.dwq import load_data
    need_tokens = num_samples * seq_len
    train, _valid = load_data(
        tokenizer, data_path,
        num_samples=max(64, math.ceil(need_tokens / 120)),
        max_seq_length=seq_len + 1,
    )
    stream = []
    for tokens, _off in train:
        stream.extend(tokens)
        if len(stream) >= need_tokens:
            break
    if len(stream) < need_tokens:
        raise RuntimeError(f"corpus too small: {len(stream)} < {need_tokens}")
    arr = np.array(stream[:need_tokens], dtype=np.int32).reshape(
        num_samples, seq_len)
    return arr, [
        mx.array(arr[i: i + batch_size])
        for i in range(0, num_samples, batch_size)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-path", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--top-k", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] teacher (lazy, cpu stream): {args.teacher}", flush=True)
    teacher, tokenizer, _ = lazy_load(args.teacher)
    core = find_core(teacher)
    n_blocks = len(core.layers)

    print(f"[2/3] corpus: {args.num_samples} x {args.seq_len} "
          f"from {args.data_path}", flush=True)
    arr, batches = build_batches(tokenizer, args.data_path, args.num_samples,
                                 args.seq_len, args.batch_size, args.seed)
    mx.save_safetensors(str(out / "tokens.safetensors"),
                        {"tokens": mx.array(arr)})

    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    acts = [core.embed_tokens(b) for b in batches]
    mx.eval(acts)

    print(f"[3/3] streaming {n_blocks} blocks", flush=True)
    for i in range(n_blocks):
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
        print(f"  block {i}/{n_blocks - 1} done in {time.time() - t0:.1f}s "
              f"(peak {mx.get_peak_memory() / 1024**3:.1f}G)", flush=True)

    print("final: norm + lm_head -> top-k", flush=True)
    head_norm = core.norm
    lm_head = getattr(teacher, "lm_head", None)
    with mx.stream(mx.cpu):
        mx.eval(head_norm.parameters(),
                lm_head.parameters() if lm_head is not None
                else core.embed_tokens.parameters())

    all_idx, all_lp = [], []
    for a in acts:
        h = head_norm(a)
        logits = (lm_head(h) if lm_head is not None
                  else core.embed_tokens.as_linear(h)).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # top-k via argpartition
        idx = mx.argpartition(-logprobs, kth=args.top_k - 1,
                              axis=-1)[..., : args.top_k]
        lp = mx.take_along_axis(logprobs, idx, axis=-1)
        mx.eval(idx, lp)
        all_idx.append(idx.astype(mx.int32))
        all_lp.append(lp.astype(mx.float16))
        del h, logits, logprobs
        mx.clear_cache()

    idx = mx.concatenate(all_idx, axis=0)
    lp = mx.concatenate(all_lp, axis=0)
    mx.save_safetensors(str(out / "teacher_topk.safetensors"),
                        {"indices": idx, "logprobs": lp})
    (out / "meta.json").write_text(json.dumps({
        "teacher": args.teacher, "data_path": args.data_path,
        "num_samples": args.num_samples, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "seed": args.seed,
        "top_k": args.top_k, "shape": list(idx.shape),
    }, indent=1))
    print(f"done -> {out} (indices {idx.shape}, logprobs {lp.shape})",
          flush=True)


if __name__ == "__main__":
    main()
