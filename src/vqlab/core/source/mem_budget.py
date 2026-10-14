"""Per-block eager-eval budget, shared by the streamed probes.

`mx.eval(blk.parameters())` before a layer's forward is how the streamed
scorers bind their weight reads to the CPU stream (quantlab IV.1). It assumes
a layer fits. qwen4_exp breaks that assumption hard: its PLE n-gram tables
live in LAYER 1, and in bf16 they are 96 GiB across 128 shards of
[2500012, 160], against 4.8 GiB for an ordinary layer.

Asking MLX for 100 GiB on a 96 GiB box does not raise. It SPINS inside
eval_impl's memory-limit check -- get_memory_limit / get_active_memory under
a mutex, plainly visible in a `sample` of the process -- so the job holds ~5%
CPU, reads from disk at full speed, prints nothing, and looks exactly like a
slow layer (F115).

A per-TENSOR guard catches none of this: every shard is only 0.75 GiB. The
budget has to be per BLOCK.
"""
import gc
import time

import mlx.core as mx
import mlx.utils

from vqlab.util.fs import GIB


def eval_params_budgeted(module, budget_gb):
    """Eval a block's weights up to a total budget; leave the rest lazy.

    Largest tensors are skipped first, so the cheap weights still get the
    eager treatment. Skipping an eval costs only peak-memory bookkeeping:
    the read ops were created under the CPU stream at load time, which is
    what binds the stream, so what is left lazy does NOT migrate to the GPU
    stream.

    Returns (evaluated_bytes, skipped_bytes).
    """
    budget = int(budget_gb * GIB)
    flat = [(n, a) for n, a in mlx.utils.tree_flatten(module.parameters())
            if isinstance(a, mx.array)]
    flat.sort(key=lambda t: t[1].nbytes)          # cheapest first
    keep, keep_bytes, skip_bytes = [], 0, 0
    for _, arr in flat:
        if keep_bytes + arr.nbytes > budget:
            skip_bytes += arr.nbytes
            continue
        keep.append(arr)
        keep_bytes += arr.nbytes
    if keep:
        mx.eval(keep)
    return keep_bytes, skip_bytes


def stream_layers(core, caches, h, lazy_over_gb, forward):
    """The streamed scorers' layer walk: one block materialised at a time.

    For each layer i: budgeted eval of its weights under the CPU stream,
    `parts = forward(blk, caches[i], h)` (the family's chunk loop, one output
    per chunk), concatenate, eval, then drop the block and its cache so its
    memory is returned before the next one loads. Returns the final h.
    """
    n = len(core.layers)
    for i in range(n):
        blk = core.layers[i]
        with mx.stream(mx.cpu):
            _, skipped = eval_params_budgeted(blk, lazy_over_gb)
        if skipped:
            print(f"  layer {i}: {skipped / GIB:.1f} GiB left lazy",
                  flush=True)
        t0 = time.time()
        parts = forward(blk, caches[i], h)
        h = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        mx.eval(h)
        # The model's layer list (and any property over it) holds the only
        # persistent reference to the block; dropping the entry frees it.
        core.layers[i] = None
        caches[i] = None
        del blk, parts
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1} {time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/GIB:.1f}G)", flush=True)
    return h
