# Memory cheat sheet — MLX local models

Peak too high? Try these, in order. Each one is measured to cost ~nothing.

1. **Cap the buffer cache** — `mx.set_cache_limit(4 << 30)`.
   Freed buffers otherwise pile up invisibly (not in "active memory").
   Biggest single win. Zero speed cost measured at 26k-token prefill.

2. **Chunk prefill** — 2048-token chunks, `mx.eval(cache_state)` +
   `mx.clear_cache()` between chunks. Bounds the transient to one chunk.

3. **Break up long lazy graphs** — one `mx.eval` mid-graph caps the
   high-water mark. (Per-layer weight decode, long pipelines.)

4. **Set a tripwire** — `mx.set_memory_limit(N)`. Doesn't fix anything;
   turns a machine freeze into a stack trace naming the allocation.

5. **Don't trust "Peak memory" logs** — they count freed transients.
   Measure RSS from outside: `while :; do ps -o rss= -p PID; sleep 0.2; done`.

6. **MoE: weight size ≠ resident.** mmap loads expert pages on first
   touch; routing concentrates. Measure your workload — ours ran a
   94 GiB MoE in 36 GiB resident for chat. Budget weight size only for
   worst case. (Prewarming stacks touch everything; that's optional.)

7. **fp32 out of nowhere?** One fp16 tensor + bf16 activations promotes
   the forward to fp32 (2x memory, some kernels won't launch). Print
   activation dtypes per layer.

VQLab bundles ship 1-3 as defaults (`VQLAB_CACHE_LIMIT_GB` /
`VQLAB_PREFILL_CHUNK` to override). Numbers and methodology: model cards.
Unexplained peak? Open a discussion — the last several were bugs, and we
fixed them.
