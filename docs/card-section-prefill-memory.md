# Draft: "Long prompts without OOM" card section

Canonical text below; per-repo tailoring notes at the end. Nothing
published — for Noah's review.

---

## Long prompts and memory (read this if you're near the fit line)

If the model loads and generates fine but a **long prompt** freezes the
machine or kills the process partway through, you are not out of memory
for the *model* — you are out of memory for the **prefill transient**.
Peak memory during prompt processing scales with the prefill chunk size
(activations for the whole chunk are alive at once), and on hybrid
architectures with recurrent/linear-attention layers (this family), the
transient additionally scales with chunk size × per-layer recurrent
state. None of that shows up at load time or during short-prompt tests,
which is why it surprises people. Things we learned serving these
artifacts at the fit line:

- **Lower the prefill chunk size.** `mlx-lm`'s Python API takes
  `prefill_step_size` (default 2048); on a tight fit, 512 shrinks the
  prefill peak to roughly a quarter for a few percent of prefill wall
  time:

  ```python
  from mlx_lm import load, generate
  model, tokenizer = load("TheDrainFlorist/<this-repo>")
  generate(model, tokenizer, prompt=long_prompt, prefill_step_size=512)
  ```

  There is no CLI flag for it as of mlx-lm 0.31.9.

- **If you run your own serving loop, evaluate each chunk.** Call
  `mx.eval([c.state for c in cache])` (and `mx.clear_cache()`) after
  every prefill chunk, as upstream `mlx_lm` does. Without it, MLX's lazy
  graph can accumulate the *entire prompt's* computation before
  evaluating — we measured a +56 GiB transient on a 6.9k-token prompt
  this way. The chunk size only bounds the peak if each chunk is
  actually forced.

- **Set a memory limit so failure is a traceback, not a frozen Mac.**
  `mx.set_memory_limit(<bytes>)` a few GiB under physical RAM. It won't
  prevent the OOM, but it converts a hard system freeze into a Python
  exception that tells you which allocation failed.

- **KV cache quantization (`kv_bits=8`) is a real lever but test it on
  your workload first.** It roughly halves KV memory, and on some
  Qwen-family MoE checkpoints we have measured it producing gibberish.
  We don't ship a blanket recommendation; if you try it, check output
  quality before trusting long runs.

- **On an exo cluster, use the
  [`vq-serving`](https://github.com/noahzelezny/exo/tree/vq-serving)
  branch** (supersedes `vq-codebook-replicate`; carries the same
  codebook-replicate guard). It ships the per-chunk eval fix above,
  per-model prefill chunk-size defaults, and two env knobs:
  `EXO_PREFILL_STEP_SIZE` (chunk size override) and
  `EXO_MLX_MEM_LIMIT_GB` (per-box memory ceiling). With these, a
  96+128 GB pair prefills 9k-token prompts on artifacts within ~15 GiB
  of the pair's total serving headroom.

---

## Per-repo tailoring notes (not part of the published text)

- **397B ladder (2.2 / 2.4 / 2.6 / 3.1bpw)**: full section verbatim.
  The 2.2/2.4 cards say "128 GB Mac, tight/roomy" — those users are
  exactly the audience. Also update the existing Hardware section's
  branch pointer `vq-codebook-replicate` → `vq-serving` on all four.
- **Qwen3.8-Flash-Next (2.1/3.2/4.4/5.5bpw)**: full section (hybrid
  arch, recurrent layers — the chunk×state sentence applies literally).
- **Qwen3.6-35B / Qwen3.8-27B / gemma rungs**: short version — drop the
  exo bullet and the recurrent-state clause; these fit comfortably on
  most target machines, keep only the chunk-size + memory-limit tips.
- **GLM-5.3 cards**: unpublished, on hold pending the soak test; when
  they go out they get the strong version plus the measured 512-chunk
  requirement.
