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

---

## K=2048 ROUND (08-18, both machines)

Larger codebook is the lever that worked on BOTH families. relerr 0.31 -> 0.187
in each case; the fit improvement converted to real quality in each case.

**gemma-4-26b-a4b** (M4, 2653s, 13.7 GiB unpacked)
  | rung | size | KL (mnats) | agree |
  |---|---|---|---|
  | struct8-e8 (affine) | 25G | 441 | 79.95% (ceiling) |
  | VQ K2048 d4 | 13.7G unpacked | 1856 | 56.56% |
  | VQ K256 d4 | 8.4G | 3363 | 42.65% |
  Recovers ~1/3 of the K256 -> 8bit gap. litbench (generative+cyclic) still
  running on M4 — that is the instrument the 15G community 4bit was measured
  on (79.81%), so it is what settles the "beat 4bit at 4bit size" target.

**Qwen3.6-35B-A3B** (M3, 3097s, 17.6 GiB unpacked -> 13.0 GiB packed)
  | rung | size | agree |
  |---|---|---|
  | mlx-community 8bit | 35G | 96.18% |
  | **VQ K2048 d4 PACKED** | **13.0G** | **87.33%** |
  | mlx-community 4bit | 19G | 85.61% |
  | VQ K256 d4 | 10G | 79.50% |
  BEATS community 4bit at 68% of its size. Still short of the 96.18% 8bit
  bar, which was the stated goal — publishable as "better than the 4bit,
  smaller than the 4bit", NOT as "8bit quality".

**Packing verified end-to-end.** pack_artifact.py on the real Qwen artifact:
120/120 packed, 17.6 -> 13.0 GiB (0.734x), and the packed model scores
85.535 mnats / 87.33% — IDENTICAL to unpacked. Pure representation change,
confirmed on a 13G artifact rather than only on the synthetic test.

**gemma packed size will NOT hit the analytic target.** down_proj is NSUB=176
(not %32) and is exactly 1/3 of all code elements (down is [hidden,176],
gate/up are [704,704] — equal element counts). So 1/3 of codes stay uint16:
effective 12.7 bits, not 11; stored ~3.42 bpw, not 2.75. Estimate ~11.5G
text-only / ~12.6G sighted — still inside the 15G 4bit budget, thinner
margin than first quoted. Reaching that last third needs a block size
dividing 176 (16 works but changes word alignment) — real size on the table
if 12.6G ever needs to be 11G.

**gemma K2048 PACKED (08-18): 13.7 -> 11.5 GiB (0.838x).** 60/90 packed,
30 down_projs (NSUB=176) copied through uint16 as designed — real size hits
the 11.5G projection exactly. Text-only; sighted (+vision graft ~1.1G)
projects ~12.6G, still under the 15G community 4bit it beats by 6.73
litbench points. KL identity check PASSED: packed scores 1856.250 mnats / 56.56% — identical to unpacked to three decimals, same as Qwen. Both packed artifacts verified pure representation changes.

---

## EVENING ROUND 2 (08-18) — tail ladder, instruments, verification

**THE PUBLISH SET (all verified sizes, all measured):**
| artifact | size | headline |
|---|---|---|
| gemma vq-K256-d4-sighted | 9.43G | litbench ties 15G community 4bit; replaces 19G e4b |
| gemma vq-K2048-d4-sighted | 12.53G | litbench 86.54% = bf16 ceiling (84.62) |
| qwen36 vq-K2048-d4-packed | 13.0G | ppl 1.029x; beats 19G 4bit (1.041x) |
| qwen36 vq-tail20-d2k2048-packed | 18.1G | ppl 1.007x vs 8bit 0.999x @ 35G — the 32GB artifact |

Full numbers: E45. Failures + fixes: E44. K story: E43. Domain scan:
CRUSH_RESULTS (uniform damage, d=2-gemma falsified).

**INSTRUMENTS (all committed):**
- winrate_bench.py — blind paired literary win-rate, dual-order judging,
  VERDICT-line parsing, enable_thinking=False generation. Judge:
  Qwen3.8-27B q4.
- verify_artifact.py — decode-from-artifact relerr vs bf16, packed or not.
  RUN WITH --threshold 0.35 BEFORE ANY HF UPLOAD.
- vq_397b_codes.py — now has --tail-from/--tail-geom AND --relerr-abort
  refit/abort gate (kmeans is unseeded; fits are non-deterministic, E44).

**IN FLIGHT when this was written:**
- M3: prose gens (bf16+K2048 done, K256 running) -> auto-judge bf16-vs-K2048
  -> queued verify_all of the 4 publish artifacts (logs_verify_all.log)
- M4: re-judge of thinking gens -> queued tail30 shard-2 repair
  (~/qlab/repair_tail30.log on M4)
- still unassigned: judge bf16-vs-K256 prose (fire on first free machine)

**DECISIONS RESOLVED TONIGHT:** K=8192 dead (K ladder exhausted, E45 F1).
d=2-gemma dead (domain scan). tail30 pending repair, not blocking publish.
gemma publish gate = the two win-rate verdicts.

---

## LATE-NIGHT ROUND (08-18) — d=2 changes everything; READ E46

**THE RULE THAT MATTERS MOST:** Qwen decisions are made on **PPL** (it is
valid there); gemma decisions need the **BLIND WIN-RATE** (winrate_bench),
because gemma has no valid ppl and KL over-reports MoE routing damage.
Top-1 agreement is SECONDARY everywhere. Two wrong calls tonight came from
reading Qwen off agreement (E46).

**QWEN — tail30 achieves bf16 PARITY:**
| rung | packed | ppl vs bf16 | agree |
|---|---|---|---|
| mlx-community 8bit | 35G | 0.999x | 96.18% |
| **vq-tail30-d2k2048-packed** | **20.7G** | **1.000x** | 90.30% |
| vq-tail20-d2k2048-packed | 18.1G | 1.007x | 89.77% |
| vq-K2048-d4-packed | 13.0G | 1.029x | 87.33% |
| mlx-community 4bit | 19G | 1.041x | 85.61% |
tail30 = the 32GB accessibility artifact. NOTE it needed a shard-2 repair
(E44) — the broken version read 160 mnats / 83.79%.

**GEMMA — the d2 ladder (blind judging still REQUIRED before claims):**
| artifact | sighted | KL | agree | fit relerr |
|---|---|---|---|---|
| struct8-e8 ceiling | 25G | 441 | 79.95% | — |
| vq-K512-d2 | ~16G? | pending | pending | 0.0589 |
| **vq-K256-d2-sighted** | **14.75G** | 950 | 68.27% | 0.0873 |
| vq-K2048-d4-sighted | 12.53G | 1856 | 56.56% | 0.1877 |
| vq-K256-d4-sighted | 9.43G | 3363 | 42.65% | 0.3136 |

**d=2 KERNEL NOW EXISTS** (4b2d016, vq_switch.py). Before it, d=2 artifacts
emitted pure `<pad>` on decode while scoring perfectly on teacher-forced
instruments. d=2 now runs 51.0 tok/s, FASTER than d=4's 47.2. Unsupported
(dim, pack_bits) now RAISES instead of silently using another dim's kernel.

**BLIND WIN-RATE (settled, E44):** Sonnet, blind, key withheld:
bf16 beat vq-K2048-d4 36-20 (p=0.044, mostly weak confidence); beat
vq-K256-d4 34-12 (p=0.0016). Local Qwen judge agreed on the small one
(13-2, p=0.007). CONTROL: bf16 vs itself = 20/20 tie, so the instrument is
calibrated. The d2 gemmas MUST get the same treatment.

**IN FLIGHT:** M3 qwen flat-d2-K256 (~18.8G projected, uint8 = no packing;
if it matches tail30's 1.000x it wins on size AND simplicity). M4 gemma
d2-K512 pack+KL.

**NEXT:** score/ppl the flat-d2 qwen; pack+graft+score d2-K512 gemma;
winrate generations for both d2 gemmas -> judging chip; verify_artifact
--threshold 0.35 on anything that ships.

---

## OVERNIGHT ROUND 3 (08-19) — THE d=2 HEADLINE IS RETRACTED

**READ E46's BRACKET AND E47 BEFORE TRUSTING ANY d2 CLAIM.**

**Matched-bpw bracket (gemma, same base/source/fitter/cache):**
| geometry | bpw | agree | vs d4 line |
|---|---|---|---|
| d4 K256 | 2.25 | 42.65% | (anchor) |
| d2 K32 | 2.50 | 48.84% | -0.77 |
| d4 K2048 | 2.75 | 56.56% | (anchor) |
| d2 K64 | 3.00 | 57.68% | **-5.84** |
| d2 K256 | 4.00 | 68.27% | no d4 comparator |
| d2 K512 | 4.75 | 72.72% | no d4 comparator |

At MATCHED BYTES d=4 with a big codebook WINS on gemma, and d2's deficit
WIDENS with bpw. The 397B session pre-registered 63.52% for d2-K64 before it
existed; it came in at 57.68%. What survives for d=2: it keeps climbing where
the d4 K-ladder is known to flatten (untested above K2048 on gemma), it fits
8x cheaper, and at K<=256 it needs no packing and decodes FASTER than d4.

**IN FLIGHT (all verified before believed):**
- M3: gemma d2-K1024 fit -> then pack/graft/prose chain -> then LEADS:
  gemma d4-K8192 (3.50 bpw, the decisive d4-saturation test) and qwen
  tail30-d2k512 (parity below 20G?).
- M4: gemma d4-K4096 (3.25 bpw, MEASURED point beside d2-K64) and d4-K512
  (2.50 bpw, target-1 candidate: d4 line predicts ~49.6% vs K256's 42.65%
  for +0.8G).

**M4 IS INTERMITTENTLY WRONG (E47, A/B proven).** Everything it fits is
verified on M3. Use `verify_artifact.py --outlier 3.0`, NOT --threshold —
an absolute bar is geometry-specific and cries wolf.

**STILL OWED:** blind win-rate judging for the d2 gemmas (prose regen queued;
old gens were pre-kernel <pad>). No gemma quality claim is real without it.

---

## OVERNIGHT ROUND 4 (08-19) — BOTH QUALITY TARGETS IMPROVED

**QWEN QUALITY — new champion, strictly dominant (E49):**
| rung | packed | ppl vs bf16 | agree |
|---|---|---|---|
| mlx-community 8bit | 35G | 0.999x | 96.18% |
| **vq-tail30-d2k512-packed** | **17.9G** | **0.991x** | **90.75%** |
| vq-tail30-d2k2048-packed | 20.7G | 1.000x | 90.30% |
| vq-tail20-d2k2048-packed | 18.1G | 1.007x | 89.77% |
| mlx-community 4bit | 19G | 1.041x | 85.61% |
A CHEAPER tail beat a richer one on every axis while being 2.8G smaller.
0.99-1.00x = "at parity" (mild quant reduces referee ppl slightly; seen at
E40 too), NOT "beats the teacher".

**GEMMA QUALITY — d2 ladder, still climbing:**
| artifact | sighted | KL | agree |
|---|---|---|---|
| struct8-e8 ceiling | 25G | 441 | 79.95% |
| vq-K1024-d2-packed | 17.41G | 609 | 75.90% |
| vq-K512-d2-packed | 16.08G | 744 | 72.72% |
| vq-K256-d2 | 14.75G | 950 | 68.27% |
| vq-K2048-d4 (was shipping) | 12.53G | 1856 | 56.56% |
ALL still need BLIND JUDGING before any quality claim (KL over-reports MoE
damage). Prose generation running.

**PACKED d=2 RUNS THROUGH PREFILL, NOT FUSED.** The fused packed kernel is
d4-shaped and returns NaN at d=2; prefill is D-generic (verified 2.6e-4 vs a
numpy vq_pack.unpack reference). vq_switch routes packed-d2 to prefill:
correct but ~8.4 tok/s vs 25.4 unpacked. Chip task_d993902d queued for a
fused packed-d2 kernel. NOTE: generate prose/benchmarks from the UNPACKED
artifact — identical weights, 3x faster.

**K8192 WAS NEVER A TIMEOUT.** k-means chunked its one-hot at a fixed 2M
rows regardless of K -> 2e6*k*4 bytes = 65.5 GB at k=8192, over Metal's
62.6 GB cap. Fixed (chunk scales with k). d4-K8192 requeued; both sessions
pre-registered 59.92% as the decision boundary (E48).

**M4 STILL FAILING:** another command-buffer timeout on gemma d4-K4096.
That measured d4 point is still missing; requeue on M3.

**OPERATIONAL NOTE (learned the hard way, 08-19).** Do NOT chain background
jobs as N scripts each `pgrep`-waiting on the previous BY NAME. Renaming or
killing one breaks every downstream wait condition and they all stampede the
GPU simultaneously (happened twice tonight; the second time three fits and a
generation ran at once and all stalled at 0 progress). Use ONE sequential
script — a single process cannot race itself. See scratchpad/QUEUE.sh.

---

## MORNING HANDOFF (08-19) — WHAT TO DO FIRST

**TWO CHIPS ARE WAITING FOR A CLICK. Both are blocking real conclusions:**
1. `task_7ae8af6c` **Blind-judge d2 gemma prose vs bf16** — 120 anonymized
   pairs, key withheld. THIS IS THE GATE on every gemma quality claim; KL
   over-reports MoE damage and litbench saturates, so nothing about the d2
   gemmas is settled until this runs. Decode with:
   `./score_blind_verdict.py --verdict winrate/claude_verdict_d2K512.json --tag d2K512`
   (and `--tag d2K1024`). It prints the decoded win/loss, an exact sign test,
   AND the judge's raw positional split as an instrument check.
2. `task_d993902d` **Packed d=2 fused kernel** — the two best gemma artifacts
   currently decode at ~8.4 tok/s through the prefill fallback instead of
   ~50. Correctness is fine; only speed is blocked.

**SCOUT WAS NOT STARTED.** Noah asked for the dispatcher only once the
exploration was exhausted. It was not — the d2 ladder was still climbing and
the cheaper-tail lead was still paying at the time of writing. Start it with
`python scripts/scout_services.py list` (from /Users/noahzelezny/Documents/
AgenicAI) to find the dispatcher's service name, then `start <name>`.

**BEST ARTIFACTS AS OF THIS WRITING** (all verified; gemma unjudged):
| target | artifact | size | evidence |
|---|---|---|---|
| gemma small | vq-K256-d4-sighted | 9.43G | litbench ties 15G 4bit; bf16 beat it 34-12 blind |
| gemma quality | vq-K1024-d2-packed | 17.41G | 75.90% agree vs 79.95% ceiling — UNJUDGED |
| qwen small | vq-K2048-d4-packed | 13.0G | ppl 1.029x vs 4bit's 1.041x @ 19G |
| qwen quality | **vq-tail30-d2k512-packed** | **17.9G** | **ppl 0.991x** — dominates the 20.7G rung |

**IN THE QUEUE when this was written** (scratchpad/QUEUE.sh, one sequential
process, logs_QUEUE.log): gemma d4-K8192 (decisive, boundary 59.92%), gemma
d2-K2048 (top of ladder), qwen tail30-d2k256 (cheaper tail again), gemma
d4-K512 (target-1 candidate).

**M4 IS DOWN FOR FITS.** 4 failures overnight (3 corrupt artifacts + repeated
command-buffer timeouts) plus A/B-proven wrong compute (E47). Everything ran
on M3. Do not trust an M4-fitted artifact without verifying it on M3.

---

## FINAL OVERNIGHT STATE (08-19) — ladders essentially exhausted

**GEMMA — d2 ladder run to its knee. Gains are shrinking toward the ceiling:**
| artifact | packed bpw | sighted | agree | delta |
|---|---|---|---|---|
| (8-bit ceiling) | — | 25G | 79.95% | — |
| **vq-K2048-d2-packed-sighted** | 5.75 | **18.74G** | **77.89%** | +1.99 |
| vq-K1024-d2-packed | 5.25 | 17.41G | 75.90% | +3.18 |
| vq-K512-d2-packed | 4.75 | 16.08G | 72.72% | +4.45 |
| vq-K256-d2 | 4.25 | 14.75G | 68.27% | — |
Next rung (d2-K4096, 6.25 bpw, ~20G) would gain maybe ~1 point and exceed
the qwen build's size. NOT worth it.

**GEMMA d4 SATURATES and cannot go higher:** K256 42.65 -> K512 45.04 ->
K2048 56.56 -> K8192 61.32 (3.50 bpw). Every +0.25 bpw costs a K DOUBLING at
d=4, so 5.75 bpw would need K=2^22. See E50.

**QWEN — tail knee FOUND at K512 (cheaper won once, lost twice):**
| rung | packed | ppl | agree |
|---|---|---|---|
| **vq-tail30-d2k512-packed** | **17.9G** | **0.991x** | 90.75% |
| vq-tail30-d2k256-packed | 16.5G | 1.002x | 89.92% |
| vq-tail30-d2k2048-packed | 20.7G | 1.000x | 90.30% |

**WHAT REMAINS — blind judging, nothing else.** Prose generated for
d2-K512, d2-K1024, d2-K2048 vs bf16; blind pairs built with keys withheld.
Chip task_7ae8af6c covers K512/K1024 — ADD d2K2048 (blind_pairs_d2K2048.json)
when running it. Decode every verdict with:
    ./score_blind_verdict.py --verdict winrate/claude_verdict_<tag>.json --tag <tag>

**PEER SESSION ENDED** (socket gone). Their result is in E47.3: all three
published 397B artifacts verified CLEAN, 513 tensors. A final message to
them about the bpw correction (E50) was undeliverable — it matters to them
because their pre-registered boundary was computed from my wrong numbers.
