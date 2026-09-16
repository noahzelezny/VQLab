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
import mlx.core as mx
import mlx.utils


def eval_params_budgeted(module, budget_gb):
    """Eval a block's weights up to a total budget; leave the rest lazy.

    Largest tensors are skipped first, so the cheap weights still get the
    eager treatment. Skipping an eval costs only peak-memory bookkeeping:
    the read ops were created under the CPU stream at load time, which is
    what binds the stream, so what is left lazy does NOT migrate to the GPU
    stream.

    Returns (evaluated_bytes, skipped_bytes).
    """
    budget = int(budget_gb * 1024 ** 3)
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
