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
5. **Expert stacking at fit time — BUILT 2026-08-28 (VQLab 2a6b8ba),
   option (a) below.** `src/vqlab/expert_src.py` is the single loader for
   every source layout; the glm5_next FAMILY entry exists (target_substr
   still PROVISIONAL); vq_397b_codes / verify_artifact / probe_init_sweep
   all route through it, and probe_init_sweep gains `--family`. CPU-only
   selftest gates both directions; loader verified byte-identical against
   direct reads of the real checkpoint (L3 gate_proj, L45 down_proj), and
   288 experts confirmed from the index on layers 3/20/45. GPU `vqlab
   selftest` NOT yet run (boxes busy — contention rule); run it once a box
   is idle, before the first real fit. Original analysis: no existing
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

---

## Addendum 2026-08-28 (late): community-conversion + upstream follow-up

Requested by the paper/arc session. All read-only (Hub file reads + GitHub);
community repos treated as UNTRUSTED — inspected on the Hub only, nothing
downloaded, installed, or executed.

### Module naming now confirmed at the CHECKPOINT level

`pipenetwork/GLM-5.3-Flash-MLX-4bit` (17 shards, ~178 GB, no bundled python
— it relies on mlx-vlm's in-tree class) has, per its own
`model.safetensors.index.json` and `config.json` (read raw on the Hub):

- routed experts: `language_model.model.layers.{li}.mlp.switch_mlp.{gate,up,down}_proj.weight`
- shared expert: `...mlp.shared_experts.{...}_proj.weight`; router:
  `...mlp.gate.weight` + `.e_score_correction_bias`
- hyper-connections renamed to modules: `...attn_hc.{base,fn,scale}`,
  `...ffn_hc.{...}` (the bf16 checkpoint's flat `hc_attn_*`/`hc_ffn_*`)
- quantization overrides: indexer modules at 8-bit, rest 4-bit g64 — and
  NO expert/gate keys in the override map (router quantized? not visible in
  the summary — check on a full read before using this artifact as a
  comparator; per III.3 it must pass check_comparator regardless)
- vision keys and layer-45 (MTP) keys: NOT VISIBLE in the fetched excerpt
  (the index summary covered layers 0–18) — **UNKNOWN, not absent**; the
  open mlx-vlm PR #2044 ("MTP support for GLM-5.3-Flash") implies MTP is
  currently NOT loaded by the shipped class.

So the draft family entry's `target_substr: "switch_mlp"` is now confirmed
against a real shipped conversion's tensor names, not just mlx-vlm source
reading. It stays PROVISIONAL only with respect to a future mlx-lm class
(which could name modules differently). `orcarouter/GLM-5.3-Flash-MLX`
(1.14 TB incl. bf16 + 2/3/4/6-bit subdirs) likewise ships no python and
targets mlx-vlm.

### Upstream state, pinned with links (MEASURED from GitHub)

- mlx-vlm glm5_next landed **2026-08-26**, PR #2030, commit `fa27a9a`
  (github.com/Blaizzy/mlx-vlm/pull/2030) — the earlier "~08-26 PROVISIONAL"
  date is now confirmed.
- Active mlx-vlm glm5_next PRs as of 08-28/29: #2044 (MTP support, open),
  #2074 (Dflash2), #2086/#2087 (indexer fixes), #2091 (chat template) —
  the class is under active repair; pin the mlx-vlm commit in any scoring
  claim (III.13's instrument-the-import rule).
- mlx-lm: still NO glm5_next class or PR (models tree re-checked 08-28).
  **Runtime timeline reading: the mlx-vlm path is here today; mlx-lm is
  not in flight anywhere we can see.** Planning consequence: assume
  scoring/smoke via mlx-vlm; treat an mlx-lm class as an upside surprise,
  not a scheduled event.

### MTP consequence for the fit

If scoring runs under today's mlx-vlm class (MTP not loaded, pending
PR #2044), layer 45's 288-expert stack (~13.5 GiB bf16, 1/43 of expert
mass) is dead weight in the artifact for that runtime. Options at struct-
base time: fit it anyway (future-proof, costs fit time), or exclude layer
45 from --vq-layers and graft it bf16/affine later. Decide when the
scoring runtime is chosen; no action now.

---

## Addendum 2026-08-29: PROVISIONAL scorer design — glm5_next in stream_score

Requested by the paper session. Everything here is read from mlx-vlm main's
`glm5_next/language.py` via raw fetch (summarized, not line-verified) —
**PROVISIONAL throughout; line-verify against the pinned mlx-vlm commit
before writing the scorer.** No code changed; stream_score's SCORERS
registry and hard-refusal on unknown model_type stay as they are until a
scorer exists.

### What qwen4_exp's scorer threads per layer (for contrast, MEASURED from
our own stream_score.py)

`(h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx)` — the ids/
prev_ctx pair feeds each layer's n-gram PLE lookup (the reason a
hidden-state-only loop silently starves qwen4_exp); h is tiled x hc BEFORE
the stack and resolved by a GLOBAL hyper_connection_mixer after it; no
final norm.

### What glm5_next's scorer needs (PROVISIONAL, from mlx-vlm main)

Structurally SIMPLER than qwen4_exp on three axes, different on two:

1. **Layer signature is clean:** `layer(x, mask=None, cache=None)` — no
   ids/prev_ctx threading (no PLE), no separate idx_cache argument. The
   streaming loop's shape is the standard embed → per-layer
   eval/run/free → head.
2. **hc lives INSIDE the layer** (attn_hc/ffn_hc modules per layer, expand
   + reduce within the block), not as a global pre-tile + post-mixer. The
   model broadcasts h to (B, S, hc_mult=4, D) once before the stack and
   takes `h.mean(axis=2)` after — the streaming loop must reproduce both
   bookends. Activation cost: hc_mult x the hidden state (4 x 4096 x seq),
   still trivial next to a resident layer.
3. **Final norm EXISTS** (`model.norm` RMSNorm after the mean) — opposite
   of qwen4_exp; forgetting it is the classic silent-wrong-answer.
4. **Per-layer cache is heterogeneous:** KDA layers want a 2-slot cache
   (conv state + gated-delta recurrent state) and an ssm mask from
   `create_ssm_mask`; DSA layers want a KV cache pair whose second slot
   the indexer uses for pooling/top-k. For a SINGLE full-sequence teacher
   pass (our scoring mode) cache can likely be None/fresh per layer as in
   score_qwen4_exp — but whether the DSA indexer path tolerates cache=None
   at prefill is UNVERIFIED and is the first thing to test.
5. **Loader is the real fork.** stream_score loads via
   `mlx_lm.utils.load_model`; a glm5_next artifact under today's runtime
   needs mlx-vlm's loader, and the text stack sits at
   `model.language_model.model.layers` (one level deeper than qwen4_exp's
   `model.model.layers`). Options: (a) a second load path in stream_score
   keyed off the SCORERS entry (scorer declares its loader), or (b) load
   the LanguageModel directly and skip the VLM wrapper. Either way the
   hard-refusal design holds: glm5_next simply stays refused until its
   entry lands with its loader named.

### Unknowns to resolve before writing it (each is one cheap check once a
runtime is importable)

- DSA indexer with cache=None at full-sequence prefill (see 4).
- Whether sinkhorn iterations run at load/sanitize time or per forward
  (not visible in the fetched summary; `hc_sinkhorn_iters: 20` in config).
- MTP layer 45: today's mlx-vlm class doesn't load MTP (PR #2044 open) —
  the scorer must count on 45 layers, not 46, under that runtime.
- Layer-streaming memory: one GLM MoE layer resident is ~13.6 GiB bf16
  (288 experts x 47.2 MB + overhead) — fine on either box; ESTIMATE.

---

## Addendum 2026-08-29: runtime abstraction BUILT; serving story settled

Noah's ruling executed (VQLab d8059fe): provision for either runtime,
never fork mlx-lm.

- **`runtime_load.load_for_family()`** dispatches mlx_lm vs mlx_vlm on a
  per-family registry field. glm5_next carries `runtime: "mlx_vlm"`,
  `model_type: "glm5_next"`; when mlx-lm lands the class, ONE field flips
  and no call site changes. stream_convert, stream_score and smoke all
  route loads through it and print the III.13 resolved-runtime line.
- **Serving story: SETTLED (source read).** mlx-vlm's utils.py honours the
  in-checkpoint `model_file` bundle — the same mechanism as mlx_lm — so
  bundled-runtime GLM artifacts serve under either runtime. (Read from
  mlx-vlm main 08-29; line-verify at the pinned commit before shipping a
  bundle claim, per III.13.)
- **stream_convert** under an mlx_vlm runtime keeps vision modules bf16 —
  the tower rides the whole pipeline in the struct base, so
  **graft_vision is NOT needed for glm5_next** (supersedes §3.4's flags
  discussion for this family; the traps there still apply to any family
  that DOES graft). PROVISIONAL until a struct base is built and its
  tower verified non-zero.
- **stream_score** has a glm5_next scorer implementing the design note
  (hc bookends, per-layer-type masks resolved from the LOADED module,
  final norm). It is **UNVALIDATED and gated**: refuses without
  `--allow-unvalidated`, stamps `"unvalidated": true` into its record so
  the number cannot silently enter a ladder. Validation standard (rule 5):
  reproduce a direct full-model forward to all printed decimals; that run
  also answers the DSA-indexer-with-fresh-cache unknown.
- **smoke** dispatches the same way; the mlx_vlm generate branch is
  PROVISIONAL text-only.

Still pending: GPU `vqlab selftest` of the expert_src rewiring (watcher
armed for the M3 queue going quiet; the running d2/K1024 fit is itself the
first real-GPU traffic through the rewired fused-path reader). No mlx_vlm
import has ever executed in this repo's venvs — every mlx_vlm code path is
PROVISIONAL until one does.

### Pack-alignment check, 2026-08-29 (MEASURED through the packer itself)

Per the paper session's request, checked not assumed: `vq_pack` requires
`NSUB % 32 == 0` (BLOCK=32, asserted in `words_per_row`). GLM expert
widths run through the real function:

    d2: gate/up NSUB 2048, down 1024 — both %32==0
    d4: gate/up NSUB 1024, down  512 — both %32==0
    d8: gate/up NSUB  512, down  256 — both %32==0

Every geometry packs cleanly; no qwen4_exp-style unpacked ride for any
glm5_next projection. (The row-aligned kernel variant remains a separate
VQLab roadmap item for qwen4_exp's d8 down_proj, queued after the Qwen
ladder.)
