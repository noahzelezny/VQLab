# Taming memory peaks in local MLX models — a field guide

Draft for Noah's review. Intended home: VQLab repo top-level doc, linked
once from each model card ("Memory behavior: see the playbook"), and
postable as an HF community article. Voice: practitioner-to-practitioner.

---

Local models get abandoned in the first hour for one reason more than
any other: the memory graph. You download a 12 GB model, send it a long
prompt, watch "peak memory" hit 38 GB, and conclude your machine can't
run it. Ours could all along. The peak was never the model — it was
allocator defaults, and every one of them is fixable with a setting.

We've spent the last months shipping vector-quantized builds of large
models for Apple Silicon, and every time a memory number looked
disqualifying, scrutiny found a knob, not a wall. This is the list, so
you don't need our stubbornness.

## The three numbers people conflate

- **Resident (RSS)** — pages your process actually holds. The only
  number your other models compete with in a multi-agent setup.
- **Peak resident** — RSS at its worst moment. What you actually budget.
- **"Peak memory" counters** (`mx.get_peak_memory()`, mlx-lm's log
  line) — a cumulative high-water mark of *allocations*, counting
  transients that were freed microseconds later. On any model that
  decodes or dequantizes weights on the fly, this reads several times
  RSS. It is not a RAM requirement. Measure RSS from outside the
  process (`ps -o rss -p <pid>` at 5 Hz beats any in-process counter).

## The knobs, in order of impact

1. **Cap the buffer-reuse cache: `mx.set_cache_limit(4 << 30)`.**
   MLX parks every freed buffer in a reuse cache and returns nothing to
   the OS. During prefill, per-layer transients park there by the tens
   of gigabytes — invisible to the "active memory" counter, very
   visible to your machine. A 4-8 GiB cap returns pages immediately.
   Measured cost on our builds, 26k-token prefill: **zero** wall-time
   change. (VQLab bundles set this by default since 2026-09;
   `VQLAB_CACHE_LIMIT_GB` overrides.)

2. **Chunk your prefill, and evaluate per chunk.** One lazy graph over
   a long prompt holds every layer's intermediates at once. 2048-token
   chunks with an `mx.eval` + `mx.clear_cache()` between them bound the
   transient to one chunk's worth. On a ~100 GB-class model this took
   the prefill transient from "tens of GB" to under 3 GB, flat with
   context length.

3. **Bound graph depth generally.** The prefill trick generalizes: any
   long lazy chain (per-layer weight decode included) accumulates until
   evaluated. One well-placed `mx.eval` mid-graph costs nothing
   measurable and caps the high-water mark. (Our dense builds do this
   internally now: 2048-token prefill peak went from 2.0x resident to
   1.17x, prefill slightly *faster*.)

4. **Set a hard memory limit as a tripwire: `mx.set_memory_limit(...)`.**
   Not a fix — a diagnostic. A runaway allocation then raises with a
   stack trace naming the culprit instead of freezing your machine.

5. **Know your MoE's real residency.** With mmap'd weights, expert
   pages load on first touch. Routing concentrates: our 94 GiB-weights
   MoE holds ~36 GiB resident under varied chat and ~58 GiB through a
   2k-token prefill. Weight size is the guarantee; measured residency
   for your workload is often far less. (Serving stacks that prewarm
   touch everything — that's a choice, not a necessity.)

6. **Watch dtype promotion.** One fp16 tensor meeting bf16 activations
   promotes the whole forward to fp32 — double memory, and on some
   attention geometries a kernel that won't launch at all. If your
   memory doubled for no reason, print activation dtypes layer by layer.

## What "usable for multi-agent" means concretely

Peak ≈ resident + a small bounded transient. On our current builds:
dense 27B peaks at exactly its 12.5-15.5 GiB file sizes; the 80B MoE
peaks at what its routing touches plus ~3 GiB. Three models on one
128 GB box stopped being aspirational the day the transients were
capped — nothing about the models changed.

Every number above is externally measured and reproducible; methodology
on each model card. If you find a peak we haven't explained, open a
discussion — an unexplained peak is a bug report, and the last several
turned out to be exactly that.
