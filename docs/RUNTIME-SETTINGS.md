# Runtime settings — what makes these artifacts runnable, in one place

*2026-09-18. Consolidates the settings knowledge that was scattered across
vq_switch.py comments, FINDINGS-LOG, RUNTIME-SHIP-PLAN, MEMORY-PLAYBOOK and
several env files. Written as the seed of a per-artifact RESOLVER: everything
below should be decided from an artifact's own config.json and the box's
memory, not remembered by an operator.*

**The premise.** A VQ artifact that a downloader cannot load is worth nothing,
and the defaults in stock mlx-lm and exo are not tuned for these shapes. Every
number here was paid for; the failure mode it prevents is "the model is
basically unrunnable," not "the model is 5% slow."

---

## 1. The three layers, and which are already solved

| layer | ships where | status |
|---|---|---|
| **VQ kernels** (`vq_switch`, 4523 lines) | **vendored into each artifact** as `model.py`, pointed to by `config.json: model_file` | SOLVED. Absent from every env; correct as downloaded by design. |
| **architecture model files** (`qwen4_exp`, `qwen3_5`(+`_moe`), `gemma4_text`, `glm5_next`) | grafted into `site-packages/mlx_lm/models/` | **UNSOLVED.** Unversioned, unpinned, per-env, and DRIFTED -- see §5. |
| **settings** (33 env knobs, prefill chunk, cache limit) | operator's shell | PARTLY solved: good defaults are baked, but the two that prevent OOM are not resolved per artifact. |

`vqlab serve` already covers the serving surface (mlx-lm's OpenAI server with
VQLab decode underneath). What it does not own is the environment.

## 2. The two knobs that decide runnable vs not

Everything else on this page is performance. These two are capability.

### `VQ_DECODE_CHUNK` -- experts decoded to dense fp16 per prefill chunk

**THIS IS THE MEMORY KNOB, NOT THE KV CACHE.** Measured 2026-08-15 on a 128 GB
M4 Max running the 110.8 GiB 397B: prefill grew **3.35 MB/token** where the KV
cache theory predicts 0.059 -- a 57x gap owned entirely by these buffers.

    transient = chunk * out * in * 2 bytes
      chunk=128 -> 1.0 GiB (down_proj) / 2.0 GiB (gate_up)
      chunk= 16 -> 0.12 GiB            / 0.25 GiB

On a box where the model nearly fills RAM, this caps your context length.
It is also FASTER small -- steady-state ms/bucket, M4, weights resident:

    chunk |  2.2bpw  2.4bpw  3.1bpw
      16  |   677.2   651.9   663.6
      32  |   716.1   683.0   691.9
      64  |   791.4   756.3   775.7
     128  |   984.1   943.2   947.3   <- the OLD default, 1.37x SLOWER

The knee is identical across K128/K256/K2048, so codebook size does not move
it. Default 32, auto-sized down from free memory at import.

### `VQLAB_PREFILL_CHUNK` -- prompt chunk width (default 2048)

Bounds the prefill transient for the trunk. mlx-lm's server does not expose
it, so this env var is the operator's only handle, and in `serve.py` it
deliberately WINS over the caller's value. Token-identical at every width
(`tests/test_mtp_prefill.py` gates this) -- it is purely a memory knob.

### `VQLAB_CACHE_LIMIT_GB` -- buffer cache cap (default 4.0)

Freed MLX buffers pile up invisibly; they do not appear in "active memory."
Biggest single win in MEMORY-PLAYBOOK, zero measured speed cost at 26k-token
prefill. `=0` disables.

## 3. Performance knobs with a measured basis

| knob | default | basis |
|---|---|---|
| `VQ_MOE_GEMMSEG_CBDEV` | `auto` | F124: device beats threadgroup by **20.9%** on prefill at d4-K2048. ~447 fleet modules ride on the selector. |
| `VQ_MOE_GEMMSEG_RTILE` | `32` | F25/F33: 64 is SLOWER everywhere (0.75-0.97x), confirmed on exo and local. **Do not set 64.** |
| `VQ_GEMMSEG_OTILE64` | `1` | F54 arm 1: +5.1-6.6% prefill, bit-exact. |
| `VQ_GEMMSEG_PH2V`, `VQ_D4_WALK` | `1` | F56: stack reaches +11.9% over shipped. |
| `VQ_GEMMSEG_BF16IO`, `VQ_DECODE_BF16IO` | `0` | **Numerics-active** (F103/F105), family-local, up to +0.97% ppl. v1.5 = both off; see RUNTIME-SHIP-PLAN. |
| `VQ_D8_SIMDSUM` | `0` | Built and measured (1.16-1.20x gate/up) but NOT bit-identical. Off pending a policy call; F126 argues the end-to-end payoff (+1.4-1.8%) is below the instrument's noise floor. |
| `VQ_GEMMSEG_PIPE` | `0` | Arm 1.5 measured NEGATIVE (-1.8-2%). Leave off. |

Full list: 33 env knobs in `vq_switch.py`. Most should never be touched; they
exist so a finding stays reproducible.

## 4. Traps that have each cost a run

* **Env-file ordering.** F33: `exo-env.sh` is sourced BEFORE `ring-env.sh`,
  which assigns `RTILE=32` unconditionally. A knob set in the former is
  silently overwritten -- an RTILE=64 "experiment" benchmarked 32 twice and
  was reported as "no difference." **A resolver must own the final value, not
  hope an env file wins.**
* **Do not ship a setting as an env var the user must set.** RUNTIME-SHIP-PLAN
  is explicit: the artifact must be correct as downloaded. v1.5 is a flag
  DEFAULT flipped in the bundle, not an instruction in a README.
* **"Peak memory" logs count freed transients.** Measure RSS from outside.
  MLX Metal allocations do not appear in RSS either (F115: `ps` said 11.7 GiB
  while the process held ~60), so neither number alone is trustworthy.
* **MoE weight size != resident.** mmap pages experts in on first touch and
  routing concentrates; a 94 GiB MoE ran chat in 36 GiB resident. Budget full
  weight size for worst case only.
* **fp32 out of nowhere.** One fp16 tensor against bf16 activations promotes
  the forward to fp32 -- 2x memory, and some kernels will not launch.

## 5. The unsolved layer: architecture files drift between boxes

Measured 2026-09-18 across this lab's two envs (both mlx-lm 0.31.3):

    file              qwen4exp venv    exo env      artifacts affected
    qwen4_exp           1136 lines     1138 lines   4  (cosmetic shim only)
    qwen3_5              574            535         11 (+ _moe subclasses it)
    gemma4_text          688            675         2
    qwen3_5_moe           52             52         -
    glm5_next          MISSING        MISSING       3  (load in neither)
    vq_switch          MISSING        MISSING       -  (correct: in the artifact)

* `qwen4_exp`: predicate-arity compat shim. Arithmetic-neutral, safe.
* `qwen3_5`: QK-norm rewritten to `mx.fast.rms_norm`. **Algebraically
  identical** (`inv_scale = D**-0.5` makes the forms equal) but NOT
  bit-identical: max rel diff 9.3e-07 fp32, **1.2e-02 bf16**. Also
  `PipelineMixin` present in one env and absent in the other -- the env named
  `exo` is the one WITHOUT pipeline-parallel support.
* `gemma4_text`: one env drops `k_proj`/`v_proj`/`k_norm` for KV-shared
  layers, the other loads them. Different parameter sets; not verified as
  dead weight.

**Consequence under F87 (one harness):** a number measured in one env and
compared against the other on those 11 artifacts is a harness violation. 1e-2
in bf16 is well above the ~4e-4 scale this repo treats as kernel-equivalence,
and F120 showed a load-path change on this family moving Flash-3.2 prose KL
156.7034 -> 155.1233.

**Fix:** version the four architecture files as real source with a pinned
mlx-lm, and install them from that source rather than by hand. That is the
substance of a runtime package; the endpoint is a thin shell over it.

## 6. What a resolver should decide, per artifact

Input: the artifact's `config.json` (authority order rule 1) + box memory.

1. `model_file` -> the bundled VQ runtime. Already automatic.
2. `model_type` -> which architecture file must be present, **at which
   pinned version**. Currently unchecked; §5 is the result.
3. artifact bytes vs `max_recommended_working_set_size` -> `VQ_DECODE_CHUNK`
   and `VQLAB_PREFILL_CHUNK`. Partly automatic (chunk auto-sizes at import);
   the prompt width is not.
4. per-module `d`/`K` -> `CBDEV` arm. Already automatic (F124 validated).
5. repo weights v1/v2 -> v1.5 vs v2 flag set, per RUNTIME-SHIP-PLAN.

Items 2 and 3 are the gap. They are also exactly the two that produce
"unrunnable" rather than "slow."
