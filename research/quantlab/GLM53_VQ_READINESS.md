# GLM-5.3-Flash — VQ pipeline readiness survey

Date: 2026-08-28. Status: **survey only — no GPU work performed.** All tensor
facts below are MEASURED from the safetensors index + per-shard JSON headers
(struct+json over the files, per FINDINGS rule; no mx.load, no eval). Runtime
module names are PROVISIONAL until an mlx-lm `glm5_next` class exists — marked
inline. Purpose: the fit launches the day mlx-lm gains the class.

Source checkpoint (MEASURED):
`/Volumes/Thunderbay SSD/Mlx_Models/hub/models--zai-org--GLM-5.3-Flash-BF16/snapshots/f12e0fe1f6b2ea274c11a569582edfd99d993c5e/`
120 shards, 38,770 tensors, 598.51 GiB, dtypes BF16 (38,479) + F32 (291).
`architectures: [Glm5NextForConditionalGeneration]`, `model_type: glm5_next`
(text sub-config `glm5_next_text`, vision `glm5_next_vision`). MIT license.

**Correction to the working brief:** the config declares **288 routed experts
+ 1 shared expert** (`n_routed_experts: 288`, `n_shared_experts: 1`), not
256+1. 12,384 expert tensors per projection = 43 MoE layers × 288. Total
params MEASURED from headers: 321.3 G ("320B" checks out).

---

## 1. Tensor shape survey (MEASURED from headers)

### Param mass by bucket

| bucket | GiB | Gparam | share |
|---|---|---|---|
| routed experts | 582.52 | 312.74 | 97.3% |
| attention (KDA + DSA + indexer) | 11.55 | 6.20 | 1.9% |
| embeddings + lm_head | 2.36 | 1.27 | 0.4% |
| vision tower (`model.visual.*`) | 1.05 | 0.56 | 0.2% |
| dense MLP (layers 0–2) | 0.94 | 0.50 | 0.2% |
| other (hc, norms, router, MTP glue) | 0.10 | 0.05 | 0.0% |
| **total** | **598.51** | **321.32** | |

97.3% of mass in routed experts — the most VQ-favorable ratio of any family
yet (397B was ~94%). Protected budget (everything non-expert) is ~16 GiB bf16.

### Layer structure (MEASURED)

- 45 main layers + 1 MTP layer (index 45: `eh_proj` [4096,8192], `enorm`,
  `hnorm`, `shared_head.norm`). 46× `input_layernorm` confirms.
- Layers 0–2: dense MLP (gate/up [12288,4096], down [4096,12288]).
- Layers 3–45 (42 main + the MTP layer 45; MEASURED from the index): MoE
  (43 layers with experts + `mlp.gate` router
  [288,4096] bf16 + F32 `e_score_correction_bias` [288] + shared_experts).
- Attention: 34 KDA layers (`self_attn.{q,k,v}_proj` [8192,4096],
  `{q,k,v}_conv1d` [8192,1,4], `f_a/f_b/g_a/g_b_proj`, F32 `A_log`/`dt_bias`)
  and 12 DSA/MLA layers (11 main per `full_attn_layers` + the MTP layer;
  `q_a/q_b`, `kv_a_proj_with_mqa`, `kv_b_proj` [32768,512], `indexer.*`).
  `o_proj` is [4096,8192] on KDA layers, [4096,16384] on DSA layers.

### Expert stack layout — the headline finding

**Experts are UNFUSED, per-expert 2D tensors.** No `gate_up_proj` anywhere
(zero keys). Three separate tensors per expert:

    model.language_model.layers.{li}.mlp.experts.{e}.gate_proj.weight  [2048, 4096]
    model.language_model.layers.{li}.mlp.experts.{e}.up_proj.weight    [2048, 4096]
    model.language_model.layers.{li}.mlp.experts.{e}.down_proj.weight  [4096, 2048]

This is DeepSeek-style HF layout — unlike qwen3_5/qwen4_exp's fused
[E, 2I, H] stack, and unlike anything the MoE fitter has read before: there
is **no [E, out, in] stack in the checkpoint at all**. The consumer must
either stack 288 tensors per (layer, proj) at read time, or fit from an
MLX-format converted source whose sanitize has already stacked them
(the qwen3_5_mlx pattern). See §3.

Shared experts are the same shapes, one per MoE layer
(`mlp.shared_experts.{gate,up,down}_proj.weight`) — protected budget, as on
qwen4_exp.

### Last-dim divisibility (MEASURED)

Subvector axis (input dim): gate/up = 4096, down = 2048. Both divide by
d=2, 4, and 8 cleanly (4096/4=1024, 2048/4=512; /8 = 512, 256). Group-64
scales also divide both. **No gemma-style down_proj blocking problem.**
Dense-MLP dims 12288/4096 and vision dims (1024, 4096, 10240) also all
divide by 4 and 8, should any of that ever be targeted.

### Per-layer-embedding analog: none

No ngram/PLE-style lookup tables exist. The closest structural novelty is
the **hyper-connections (mhc) parameters**: per layer `hc_attn_fn` /
`hc_ffn_fn` [24, 16384] bf16 plus F32 `hc_{attn,ffn}_base` [24] and
`hc_{attn,ffn}_scale` [3] (`hc_mult: 4`, `hc_sinkhorn_iters: 20`). Total hc
mass ≈ 68 MiB across all 45 layers — negligible; protect at 8-bit or bf16.

---

## 2. Draft FAMILY registry entry (PROVISIONAL — do not commit to VQLab yet)

Module naming below is inferred from **mlx-vlm's shipped glm5_next**
(see §4): its decoder uses `DeepseekV32MoE`, whose routed experts live in
`self.switch_mlp = SwitchGLU(...)`, shared experts in `shared_experts`,
router in `gate`; its sanitize stacks per-expert HF tensors into
`...mlp.switch_mlp.{proj}.weight` [E, out, in]. An eventual mlx-lm class
will almost certainly reuse its own deepseek_v32/glm_moe_dsa MoE (same
`switch_mlp` convention) — PROVISIONAL until that file exists.

```python
"glm5_next": {
    # GLM-5.3-Flash (320B, 8-of-288 + 1 shared). PROVISIONAL 2026-08-28:
    # target_substr matches mlx-vlm's glm5_next (DeepseekV32MoE.switch_mlp,
    # SwitchGLU) and mlx-lm's own deepseek_v32 convention; re-verify against
    # the actual mlx-lm glm5_next class before first fit.
    # HF source is UNFUSED per-expert 2D: experts.{e}.{gate,up,down}_proj
    # ([2048,4096]/[4096,2048]) — no gate_up stack, no [E,out,in] stack.
    # {e} in the template below is the expert index (0..287): the fitter
    # must stack per-expert tensors along a new E axis at read time, a
    # source layout no current family exercises. Alternative: fit from an
    # MLX-format conversion (sanitize pre-stacks -> switch_mlp.{key}.weight,
    # the qwen3_5_mlx pattern) once a runtime can produce one.
    # MoE layers 3-45 MEASURED (42 main + MTP layer 45, which carries its
    # own 288 experts and IS in the 43 count; layers 0-2 dense).
    # Protected: router (mlp.gate, bf16 per E7), shared_experts, all
    # attention (KDA conv1d [8192,1,4] is 3-D — exclude), hc_* (~68 MiB),
    # dense layers 0-2, embeddings, vision.
    "target_substr": "switch_mlp",
    "src_key": "model.language_model.layers.{li}.mlp.experts.{e}.{key}.weight",
    "proj": {"gate_proj": ("gate_proj", None),
             "up_proj": ("up_proj", None),
             "down_proj": ("down_proj", None)},
},
```

MTP caveat: layer 45 (MTP) carries a full 288-expert stack (MEASURED —
expert layer indices are exactly 3–45). Whether the eventual runtime loads
it (mlx-vlm's sanitize drops `mtp.`-substring keys, but these keys are
plain `layers.45.*`) is **UNKNOWN** until the class exists. If MTP is
dropped, 1/43 of expert mass (~13.5 GiB bf16) leaves the fit scope.

Size ESTIMATE (analytic, not a fit): experts at d4/K256 (2.0 bpw codes +
group-64 fp16 scales ≈ 2.25 bpw effective) ≈ 82 GiB + protected ~16 GiB
bf16 (or ~8.5 GiB at q8) → **~90–98 GiB artifact class** — M4-residentable,
same territory as the 397B rungs. ESTIMATE; price properly with the size
model once a struct base exists.

---

## 3. What breaks current tooling

1. **`stream_convert` — BLOCKED on runtime.** It calls
   `mlx_lm.utils.load_model`; no `glm5_next` class in mlx-lm main as of
   2026-08-28 (checked the models tree — parents all present:
   `glm_moe_dsa.py`, `deepseek_v32.py`, `kimi_linear.py`, `gated_delta.py`).
   Nothing else to fix in the script itself: its predicate keys off
   `target_substr` and `mlp.gate`, both of which carry over. The
   `ngram_embedding` branch is inert here (no such keys). One review item:
   the struct-base marker is `bits: 2` group 64 — fine, expert in-dims
   2048/4096 divide 64.
2. **`fit_ple` — NOT APPLICABLE.** GLM-5.3-Flash has no lookup-table-like
   tensors (§1). The hc tensors are tiny and dense-shaped; they go in the
   protected budget, not through any fitter. Skip this stage entirely.
3. **Pack alignment — CLEAN.** `vq_pack` blocks 32 codes word-aligned;
   rows/d: down 2048/d ∈ {1024, 512, 256}, gate/up 4096/d ∈ {2048, 1024,
   512} for d ∈ {2,4,8} — every row is a whole number of 32-code blocks
   (min 256 ≥ 32, all multiples). No padding pathology; no change needed.
4. **`graft_vision` — WORKS WITH FLAGS, two traps.** Vision prefix is
   `model.visual.*` (333-tensor-analog: 297 tensors, 1.05 GiB) — already in
   the script's default `--prefixes`. Traps: (a) `patch_embed.proj.weight`
   is 5-D [1024, 3, 2, 14, 14] — if the artifact ends up in mlx
   channels-last layout, `--permute-conv5` is required, and exactly which
   layout the eventual runtime wants is UNKNOWN until it exists; (b) if the
   artifact is written in mlx layout (`language_model.model.*` /
   `visual.*`-style prefixes), the graft needs `--dest-prefix` plus the
   identity-probe rename check — run the rename against the runtime's own
   converted index (the flag renames, it does not check), the same
   discipline as the Qwen3.8-27B graft. Also copy `vision_config` +
   image/video token ids (defaults now handle `vision_config,
   image_token_id`; GLM adds `video_token_id` and start/end ids — pass them
   in `--copy-config-keys`).
5. **New for this family — expert stacking at fit time.** No existing
   FAMILY entry reads unfused per-expert keys (qwen3_5/qwen4_exp slice a
   fused stack; the _mlx variants read pre-stacked). Either (a) teach the
   fitter's reader to gather `experts.{0..287}.{proj}` into [288, out, in]
   per (layer, proj) — 288 header-offset reads per tensor, cheap and
   streaming-friendly — or (b) fit from an MLX-format conversion, which
   re-blocks on the runtime. (a) keeps the fit independent of the runtime
   and matches the "fit from bf16 HF source" convention; recommend (a).
6. **Router dtype note:** `moe_router_dtype: float32`, sigmoid scoring,
   `noaux_tc` top-k with F32 correction bias — router stays bf16/f32
   protected (E7 rule), no action, just don't let a blanket quant predicate
   catch `mlp.gate` or `e_score_correction_bias`.
7. **probe_init_sweep (FINDINGS I.13) applies before any fit** — the ++
   seeding penalty profile is a first-pass job for any new family. Runnable
   the moment tensors can be read (it needs no runtime), so it can go
   BEFORE the mlx-lm class lands if we implement §3.5(a) reading first.

## 4. Watch item: runtime state (checked 2026-08-28)

- **mlx-lm main: no `glm5_next`.** MEASURED against the GitHub models tree
  today. No open glm5_next PR found (the qwen4_exp analog was PR #1788; no
  equivalent yet). Relevant in-tree parents all present: `glm_moe_dsa`,
  `deepseek_v32`, `kimi_linear`, `gated_delta`. Related: issue #879 (GLM-5
  glm_moe_dsa support, closed via PR #867 "Add GLM5", merged Feb 2026) —
  the DSA/MLA half already runs in mlx-lm; the missing pieces for
  glm5_next are the KDA mix + hyper-connections wiring + config plumbing.
- **mlx-vlm main: `glm5_next` EXISTS** (`mlx_vlm/models/glm5_next/`;
  a search-result summary says it landed ~2026-08-26 — date PROVISIONAL,
  existence verified directly). Its language side imports deepseek_v32's
  MoE (`switch_mlp` / `SwitchGLU`), remaps `hc_attn_*`→`attn_hc.*`,
  `hc_ffn_*`→`ffn_hc.*`, and concatenates the KDA q/k/v conv1d weights.
  **Consequence: the fit may not actually need to wait for mlx-lm** — the
  scoring/smoke path could run under mlx-vlm, and community MLX conversions
  already exist on HF (orcarouter/GLM-5.3-Flash-MLX,
  pipenetwork/GLM-5.3-Flash-MLX-4bit — unexamined, PROVISIONAL as
  evidence only that conversion is possible). Decision for launch day:
  target runtime = whichever of mlx-lm-glm5_next / mlx-vlm resolves, and
  per FINDINGS III.13, instrument the import and name the resolved path in
  every runtime-dependent claim.
- Re-check cadence: watch ml-explore/mlx-lm PRs/issues for "glm5_next";
  the mlx-vlm implementation is the reference to diff any mlx-lm class
  against (module names could differ — that's why §2 stays PROVISIONAL).

## Launch-day checklist (in order)

1. Confirm mlx-lm (or chosen runtime) module tree: does the MoE land on
   `mlp.switch_mlp`? Fix §2's entry, drop PROVISIONAL.
2. Implement per-expert stacking read (§3.5a) if not already done.
3. `probe_init_sweep.py` on the new family (FINDINGS I.13) — first fit-shaped
   job, before any real fit.
4. Measure the seed-noise floor at the chosen geometry before any
   comparison (FINDINGS III.12).
5. stream_convert `--struct --family glm5_next` → struct base; price the
   rung with real numbers before fitting.
6. After any artifact: graft with §3.4's flags, then GENERATE ONE TOKEN
   through the shipping fused path (FINDINGS III.11) — d4 dense/MoE kernel
   ceilings (IV: d4 safe to K2048 threadgroup, device-codebook beyond)
   apply unchanged.
