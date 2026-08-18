# STATE — resume point (2026-08-18 ~11:00)

Written so work continues without this session's context. Everything below
is either committed or reproducible from committed scripts.

## LIVE JOBS

- **M4 (nozzlebook-pro.local)**: gemma K=2048 VQ fit running.
  `~/qlab/vqfit_k2048.log`, out ->
  `/Volumes/Thunderbay SSD/Exo Models/gemma26b-rungs/vq-K2048-d4`.
  Was past L09 at relerr 0.1875 (vs K256's 0.3142). ~30 min total.
  When done: `add_model_file.py --artifact <out> --k 2048 --dim 4`, then
  `kl_damage.py score --model <out> --cache-dir <kl_cache_gemma26b>`.
- **M3**: idle. claude-code-ingest service running (backlog drained, steady
  state, 0% cpu — leave it alone).
- M4 venv is `~/qlab-venv` (python3.12, mlx-lm 0.31.3, mlx 0.32.0 — exact
  parity with M3). Scripts live in `~/qlab/`. `timeout` does NOT exist on
  M4; `setsid` does not exist on macOS. Use `nohup ... & disown`.

## HEADLINE RESULTS (all committed, tables in CRUSH_RESULTS.md)

**gemma-4-26b-a4b — the win.** VQ K=256 d=4, 8.4G (9.5G with vision
grafted). Chat-native litbench (generative+cyclic): **79.81%, exactly tying
mlx-community's 15G 4bit**, at 63% of the size. Nothing below 15G exists
upstream. This is the publishable artifact.
  - `vq-K256-d4` (8.4G text-only) / `vq-K256-d4-sighted` (9.5G, vision
    grafted, text path bit-identical: KL 3363.109 / 42.65%).
  - NO AUDIO exists in 26b-a4b (0 tensors vs e4b's 752) — not a drop-in
    sidecar replacement; it trades audio for literary/text quality.

**Qwen3.8-27B — nothing to add.** Uniform wins outright; q4 at 14G is free
(0.996x). OptiQ calibrated LOSES to uniform (1.179x vs 1.116x at 2G larger),
attention floor loses harder (1.621x). Three mixed-precision attempts all
lost. See E40/E42.

**Qwen3.6-35B-A3B — fit works, quality does NOT clear the bar.**
  | artifact | size | ppl vs bf16 | agreement |
  |---|---|---|---|
  | mlx-community 8bit | 35G | 0.999x | 96.18% |
  | mlx-community 4bit | 19G | 1.041x | 85.61% |
  | our VQ K=256 | 10G | 1.141x | 79.50% |
  | our affine base | 11G | 1.224x | 75.99% |
  Noah's judgement: 4bit "hits the shelf", 8bit is the only usable one — and
  the numbers agree (8bit essentially lossless). So the bar is ~96% well
  under 35G, NOT "beat 4bit". VQ beats its own affine base at matched size
  (the pattern that held on 397B + gemma), but K=256 is not enough here.
  NOT publishable as-is. Next lever: larger K + packing (below).

## THE SIZING FACT I GOT WRONG (don't repeat it)

Codes round up to **uint16 for ANY K > 256** (`vq_397b_codes.py:84`), so
K=2048 and K=8192 cost IDENTICALLY unpacked (both report 4.25 bpw stored).
The real lever is **packing after the fit**, which compresses to true
bit-width. Recomputed for gemma:

    K=2048  gate/up packed@3.00bpw + down unpacked@4.25 -> ~12.3G
    K=8192  ...@3.50 -> ~13.25G      K=32768 ...@4.00 -> ~14.20G

(4-bit envelope for gemma is ~14.19G text-only.)

**PACKING BLOCKER, must fix before packing gemma:** `vq_pack.py:42`
ASSERTS on `NSUB % 32 != 0` — it does not skip gracefully. gemma's
`down_proj` has NSUB=176 (moe_intermediate 704 / d4), so the packer WILL
crash. Fix per the 397B session's read: in `pack_artifact.py`, skip the
tensor when `nsub % 32 != 0` — leave it in out_data and write NO `vq_meta`
entry (absent `pack_bits` is exactly what signals unpacked). `add_model_file.py`
needs no change (it decides per-tensor from `codes.dtype`). Mixed
packed/unpacked in one artifact is supported by construction.
Qwen3.6-35B has moe_intermediate 512 -> NSUB 128, packs cleanly, no issue.

## KNOWN FAILURE

gemma K=8192 fit CRASHED: Metal GPU timeout in k-means sampling at L5/30
(`RuntimeError: [METAL] Command buffer execution failed ... kIOGPUCommandBufferCallbackErrorTimeout`).
Nothing salvaged. K=2048 retry uses `--expert-chunk 16` and stays on the
threadgroup-resident kernel path (K<=2048) rather than the `vq_fused_d4_bigk`
device-memory fallback. If K>2048 is wanted later, expect to tune
expert-chunk/sample down further.

## INSTRUMENTS (all validated, see E39/E41/E42)

- `kl_damage.py` — KL to the model's OWN bf16. THE gate for gemma (ppl is
  invalid on gemma-4, proven vs HF transformers). Caches:
  `kl_cache_gemma26b` (chat-wrapped literary), `kl_cache_qwen38`,
  `kl_cache_qwen36` (both --raw wikitext).
- `litbench_chat.py --generative --cyclic` — the ONLY valid cross-model
  form. Single-token mode penalises reasoners (had 26b at 37.5%, below its
  own 8-bit quant). Generative + cyclic are decision-grade.
- `kl_ppl_calibrate.py` — ppl AND KL together, for models where ppl works.
- Agreement metric FLOOR is ~82% / ~400 mnats (E41): two near-lossless
  artifacts disagree 17.7%. Read damage against that, not against zero.
  Floor is setup-specific — re-measure if corpus/cache changes.

## FAMILY TABLE (vq_397b_codes.py)

- `qwen3_5` (default) — HF-format fused `gate_up_proj`, ships the 397B.
  VERIFIED byte-identical to the old hardcoded literals; do not touch.
- `gemma4` — MLX-format, pre-split, `language_model.model.*`, no fusion.
- `qwen3_5_mlx` — NEW. Same qwen3_5_moe arch but from an mlx-community
  MLX-format bf16: `language_model.model.layers.{li}.mlp.switch_mlp.{key}.weight`,
  no `.experts.` segment, no fusion. Use for Qwen3.6-35B-A3B-bf16.

## NEXT STEPS (in order of value)

1. Score the K=2048 gemma fit when it lands. If it beats K256's 42.65%
   agreement materially, it is the better publish candidate at ~12.3G packed.
2. Implement the `pack_artifact.py` nsub%32 skip, then pack. Packing is a
   safe re-runnable final pass (round-trip verified per tensor).
3. Qwen3.6-35B: retry with larger K (it packs cleanly, no blocker) to chase
   the ~96% 8bit bar. K=256 at 79.50% is not enough.
4. Publish decision: gemma VQ is the only artifact currently clearing its
   bar.
