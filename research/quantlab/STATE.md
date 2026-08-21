# BOX SCHEDULE (08-21 09:50) — this session manages M3 + coordinates M4

## ROLES
- THIS session: M3 schedule, all gate chains, all scoring, coordination
  with the M4 session. Does NOT own paper/ or webgui/ (Noah's own sessions,
  briefs in handoff/). Card fixes ARE ours.
- M4 session (uds:/tmp/cc-socks/39597.sock): fits only, hands over
  ungated. Never gates its own artifacts (E47).

## M3 QUEUE (serial, GPU-bound)
| when | job | verdict |
|---|---|---|
| ->10:40 | E89 d8 fit (155/171 tensors, shard 22/27) | — |
| 10:40-12:00 | E89 inline gate chain | **d8 verdict ~12:00** |
| 12:00-12:45 | E92 chain (flatk256-refit-packed, 111G, on share) | **K256 refit** |
| 12:45-13:30 | E93 chain (flatk512-packed, 122G, on share) | **new ~122G rung** |
| as they land | score M4 artifacts (kl_damage, minutes each) | E94, E95 |
| ~14:00 | flagship 2nd referee pass (restores reproduced-twice claim) | — |
| 14:30-16:00 | **exo 2-box smoke of the 3bpw** (needs BOTH boxes quiet) | throughput |
| after chains clear | land the 2 lazy-load patches (pack_artifact from
  logs/pack_artifact_cpuread.patch + graft_vision twin); card editing pass | |

## M4 QUEUE (fits only)
1. E94 35B K8192 refresh — CORRECTED invocation (vq_397b_codes
   --family qwen3_5_mlx --base rotlab-35B-qwen36-e2, layers 0-39, k 8192
   d4). NOTE: out dir reads 0B at 09:50 — confirm it launched.
2. E95 dense 27B — fit_dense_vq.py --family qwen3_8 --dim 4 --k 256.
3. d2-K64 + d2-K128 refits (repair E82's corrupt arm; law 10 second point).
4. Conditional on E94 gain: refresh d4-K2048/K4096.
HARD: M4 quiet by 14:30 for the exo smoke.

## BLOCKERS ON THE RELEASE (must clear before upload)
- Card says "corrected k-means implementation (fixed 2026-08-18)" but E94
  registers the mechanism as UNIDENTIFIED. Soften or identify. OURS to fix.
- check_card_placeholders.sh must pass on all cards (G still has
  __PREDECESSOR_REVISION__ by design until upload time).
- Cards C and F go stale on rename — edit in the same session as the upload.
- 6 more record inconsistencies flagged by a fresh review: see the bottom of
  paper/OUTLINE.md. Noah has the list; reconcile when the queue is quiet.

# MORNING BOARD (08-21 03:30 — night results are IN)

## THE HEADLINE: E91 resolved — BOTH mechanisms real, exactly additive
- **New best 397B ever: flat-K2048-refit = 2.3410 / 2.5963 @ 143.65 GiB**
  (post-graft). Fitter effect −0.0109 (geometry matched vs shipped 3.1);
  harvest cost +0.0042 for −3.72 GiB (0.0011 ppl/GiB; bodies
  block-identical EXCEPT L10-11, 4 tensors of fit noise inside the 0.0042).
  NOTE: the two contrasts share endpoint E91 — they are a decomposition by
  construction, NOT a measured additivity (that claim was struck, see E91
  correction). Each contrast stands alone.
- E80 rung (2.3452 @ 139.93) = best-per-GiB. Both beat shipped 3.1 and
  spicy 3.5bit. NOTHING SWAPPED — Noah decides on waking.
- Size model: 6 hits 1 in-band 0 misses; points now STAMPED pre/post-graft
  (tower = exactly 912,020,960 bytes, measured).

## TODAY'S QUEUE (in order)
1. E89 d8 fit RESUMED 03:18 (~107 tensors left, lands early-mid afternoon;
   its inline chain follows automatically; watcher armed on logs_e89.log).
2. Gate E92 (flatk256-refit-packed, on share, M4-packed via CPUREAD copy)
   — registered: beats shipped 2.4's 2.7655/2.6383 if fitter effect holds
   at K256. Graft must grow dir by EXACTLY 912,020,960 bytes (E92 ->
   ~111.62, size test confirmation).
3. Gate E93 (flatk512-packed) — the missing rung; bar = interpolation
   112.0/2.7655 <-> 143.7/2.3519 at 122.31; graft growth must be exact too.
4. AFTER E89's chain fully clears: land the two lazy-load patches
   (pack_artifact from peer's parked diff at logs/pack_artifact_cpuread.patch,
   + graft_vision twin) — the last mapped mines, both known-bad-testable.
5. THEN: Noah's decisions — swap/publish the refreshed ladder, 35B refresh
   pass (audit vq_35b kmeans parity first), e4b rebuild, collection post.

## APPROVED / DIRECTED BY NOAH (08-21 morning)
- **UPLOAD CHECKLIST (hard gates, in order)**: (1) check_card_placeholders.sh
  on EVERY card in the bundle — a human reader's eye supplies meaning for
  __TOKENS__, so review alone cannot catch them; (2) predecessor revision
  hash recorded and substituted at upload time; (3) Noah reads the full set;
  (4) exo 2-box smoke passes; (5) cards C/F/G + chart upload TOGETHER.
- **Flagship swap APPROVED**: flat-K2048-refit ships as "3bpw" @ 143.65.
  Staging with peer. Smoke = EXO 2-BOX on our fork (artifact > any single
  box); card MUST note upstream PR #2268 stalled -> our codebook-replicate
  fork required for sharding. Old 3.1bpw repo MOVED not deleted.
- **Repo name: MoEMash.** License direction: dual (noncommercial free /
  commercial paid). Paper framing: narrow and thorough — (1) data-free VQ
  beats calibrated affine at matched bytes in the 2-3.5bpw MoE-expert
  regime; (2) flat nodes are provably the peaks (4.7:1 shallow:body ratio
  closes the counter-design); harvest prices sizes between nodes.
- **Future dense target: Qwen3.8-27B** (true dense; source + kl_cache_qwen38
  ALREADY ON DISK). Directive: start FLAT — no tail complications.
- **Future: webGUI** (exo-style) for the repo.
Overnight incidents (all resolved, see night logs): pack_artifact disease
fired on M4 exactly as pre-flagged (my scheduling miss); vision-tower units
mismatch briefly looked like a size-model bias (peer measured it dead).

# OVERNIGHT BOARD (08-20 22:20) — read this first tomorrow

## FOUR UNGATED ARTIFACTS EXPECTED BY MORNING — gate serially on M3, read
## each against its registration BEFORE looking at the next:
- **E89** (M3, fitting now): flat d8-K16384 397B — rate-twin of shipped
  flat-K128. Grid in E89 amendments: ~3.1706 = rate-twin equivalence
  extends (clean close); 3.10-3.15 = dimension pays; <3.05 = memorization
  check FIRST. Opponent numbers: 3.1706/2.6988 (re-scored today).
- **E91** (M4): flat-K2048 shard-reuse refit — THE MECHANISM DECIDER.
  vs E80 rung 2.3452/2.5969 isolates harvest; vs shipped 3.1 2.3519/2.5987
  isolates fitter vintage. Predicted packed 143.7 (size model test #5).
- **E92** (M4 queue): flat-K256 refit. Predicted ~111.6 (#6). Beats
  shipped 2.7655/2.6383 if fitter effect is real.
- **E93** (M4 queue): flat-K512, the missing rung. Predicted ~122.6 (#7).
  Bar = interpolation between 112.0/2.7655 and 143.7/2.3519 at its size.
M4 queue log: grep '@@@' on M4's queue log for the night's outcome. A dead
fit = resume manually (same invocation, do NOT delete output). Peer staged
the fixed fitter (fc7c7e6) to install between E91 and E92 — verify md5 in
their log.

## TODAY'S SETTLED RESULTS (all in EXPERIMENTS E73-E93)
- **E80: the 3.1-class harvest rung WON — 139.93 GiB, 2.3452/2.5969, beats
  the shipped 3.1 on both corpora.** All gates green. Size model 4/4.
  Mechanism honestly UNRESOLVED (fitter-vintage confound) until E91.
- **E90: u8view SHIPPED to published 2.4bpw** — +25-32% prefill (35B-
  measured, 397B unclaimed on card), decode token-identical, referee exact.
  check_bundle gate born (known-bad tested both directions).
- E79: E71's swap conclusion was a proxy-score error; ladder is monotone;
  cheap-shallow = size-targeting tool (~2x byte-efficiency), 4/4 pricing.
- E87: d4 beats d2 ~12% KL at matched 2.00 bpw (clean pair; E82's 3.3x was
  a corrupt-arm artifact, E85). E84: one-instrument 35B ladder; qwen3.6
  8->4bit cliff 10.5x; d4-K8192 (56.4) beats mlx 4-bit (78.6) smaller.
- e4b: dtype confound (E76), repo PRIVATE, republish plan agreed.
- FINDINGS.md exists — laws/retractions/rules; cite the commit when citing
  a law. New gates: check_scripts_sync, check_comparator, check_bundle,
  resume completeness, fitter resume-metadata fix (fc7c7e6).

## DON'T RELEARN
- Never edit scripts a running chain hasn't invoked; fits RESUME (never
  rm output); stored bytes are not sizes (pack first); score nothing
  without an outlier gate on a trusted box; comparison rows name artifact+
  instrument+commit; M3 is 96 GB — no resident loads >90 GiB (streaming
  referee is fine); cold/warm SMB load differs 5.5x.

# COMPACTION HANDOFF (08-20 ~13:20) — read this first

**Before proposing ANY experiment: read FINDINGS.md** — settled laws,
retractions, and instrument rules. If an idea re-tests a law or re-opens a
retraction without new evidence, drop it.

## IN FLIGHT
- M3: run_rung21_m3.sh -> logs_rung21.log / logs_live_rung21.log —
  K64 shallow / d4k128 body, abort 0.60. Launched 12:56, due ~14:15,
  then its inline gate chain (verify, pack, graft, check_vision,
  check_release, referee both corpora) ~45 min after that.
  PREDICTED SIZE 99.0 GiB (see size model below). Quality TBD.
- M4 (peer, uds:/tmp/cc-socks/39597.sock): 3.1-class (K512 shallow /
  d4k2048 body). Run 3, healthy past the run-2 death point after the
  cpu-stream fix. Due ~15:00. E47: verify FROM M3 before believing it.
  MY REGISTERED SIZE BET: 140.3 +/- 0.5 GiB.

## TODAY'S RESULTS (measured, all in EXPERIMENTS.md E73-E76)
- **2.2-class rung LOST** (E74): K32/K128, 97.2 GiB / 2.069 bpw honest,
  prose 3.2730 vs shipped 2.2's 3.1706 (+3.2%), code 2.7055 vs 2.6988.
  E72's "2.2-class wins biggest" FALSIFIED as stated. All gates green.
- **Cheap-shallow is HARVEST, not reallocation** (E74 addendum, peer
  correction, verified from configs): shipped 2.4 = flat K256 @112.0G,
  shipped 3.1 = flat K2048 @144.0G, shipped 2.2 = flat K128 @100.9G.
  Every cheap-shallow build HOLDS the body and harvests shallow bits.
- **SIZE MODEL (validated twice out-of-sample)**: new = base - 1.87 GiB
  x (shallow bits harvested). Shallow 1.87 GiB/bit, body 8.81 GiB/bit
  (measured by differencing shard headers). flat K128 predicted 100.93
  vs 100.9; cheap-shallow 2.3 predicted 108.3 vs 107.9.
- **FLOOR framing supersedes "low-bit lever"**: shallow tolerates
  harvesting to a floor; 2 bits off a rich body worked, 2 bits off an
  already-cheap body did not. Running rung tests 1 bit. Dose-response at
  constant body: 0 bits = 3.1706, 1 bit = TBD, 2 bits = 3.2730.
- **e4b deficit is DTYPE PROMOTION** (E76): VQEmbedding emits fp16 into a
  bf16 model (vq_dense.py:185, the one path that doesn't cast to x.dtype).
  Real gaps: prefill -11% (not the published -21%), decode -17% (not -8%).
  Published prefill came from a 21-token prompt, +/-8% scatter, n=1 draws.
  Casting to bf16 recovers all speed but FAILS the KL gate (9.021 vs
  7.451) — the accidental fp32 path IS the quality win. **E69's "VQ beats
  affine for embeddings" is CONFOUNDED; do not repeat it until the fair
  test is run.**

## PUBLISHED STATE
- gemma-4-e4b-it-VQ-PLE: **SET PRIVATE 08-20 ~13:15** at Noah's direction
  (card oversold decode, mechanism unproven). Republish plan agreed:
  (1) ship C1 fused gather (bit-exact, +6% decode -> real gap -12%),
  (2) KL identity must reproduce 7.451/95.70, (3) run the FAIR E69 test
  (give the 8-bit the same fp32 path — settles the mechanism), (4) rewrite
  card with long-context prefill, prompt length stated, n>=3 with scatter.
  Do this AFTER the 397B work.
- gemma-4-26b-a4b-it-VQ-6.2bpw: **PUBLIC, CLEAN, unaffected** — verified
  it uses only VQLinear (90 modules, no VQEmbedding), and every VQLinear
  path casts to x.dtype. The dtype bug cannot touch it.

## NEW GATES THIS SESSION (both known-bad tested first, per E70 rule)
- check_scripts_sync.sh (69a7041): chain scripts must md5-match repo HEAD.
  Motivated by M4 running a stale fitter -> GPU timeout.
- check_comparator.py (2d754ad): comparator must hold the teacher's full
  core tensor set. A comparator that loads short scores WORSE and flatters
  us. Known-bad: mlx-community e4b-8bit FAILs (54 extra sites).

## STANDING RULES REAFFIRMED TODAY
- Never edit a script a running chain has not yet invoked (Python reads at
  invocation, not chain start). Cost us the overnight run once already.
- "Crashed at the write/save step" = where deferred lazy work gets PAID,
  not where the bug is. Two instances now (verify, fitter).
- cpu-stream fix: `with mx.stream(mx.cpu):` must wrap op CREATION (load AND
  slice), not just the eval. Now in verify_artifact.py and vq_397b_codes.py
  (9a08166). NOT yet in graft_vision.py / pack_artifact.py — do that sweep
  when no chain is running.
- Report nothing as measured that was predicted. Pre-register before fits.

## !! E79 CORRECTION (08-20 14:45) — READ BEFORE ANY SWAP !!
The "cheap-shallow 2.3 beats shipped 2.4" result is FALSE. E71 used the
bf16-scales PROXY score (2.8197) in the shipped-2.4 column. Real artifacts,
today's instrument: shipped 2.4 = 2.7655/2.6383, cheap-shallow 2.3 =
2.7790/2.6479. **The shipped 2.4 wins BOTH corpora.** Cheap-shallow is
-4.1 GiB and ~8% slower prefill, i.e. a size play with a measured quality
cost. THE SWAP MUST NOT PROCEED as designed. The ladder is monotone; there
never was an anomaly. See E79.

## DECISIONS PENDING (Noah's)
- ~~Swap ALL THREE 397B rungs to cheap-shallow~~ RETRACTED, see E79 above.
- (historic) Swap ALL THREE 397B rungs to cheap-shallow if ladder wins ("we can swap
  all the models" — approved contingent on numbers; he has NOT posted
  publicly yet, deliberately). The proven middle rung: cheapshallow 2.30bpw
  107.9G, ppl 2.779/2.6479, decode wash, prefill -8%, peak -3.8G vs
  shipped 2.4 (E71). Old rungs get MOVED not deleted. Honest-bpw names
  (bytes*8/403.4e9): ladder would be ~2.1 / 2.30 / ~3.0.

## PUBLISHED (see PUBLISHED STATE above — e4b is now PRIVATE)
- HF collection "gemma-4 VQ (Apple Silicon)":
  gemma-4-26b-a4b-it-VQ-6.2bpw (18.74G) PUBLIC + gemma-4-e4b-it-VQ-PLE
  (7.39G) **PRIVATE since 08-20 13:15 — do not treat as live**.
  Cards: charts embedded AFTER frontmatter, runtime = M4 Max, env knob
  documented as VQ_DECODE_CHUNK (SCOUT_ prefix stripped from all public
  artifacts; internal runtime keeps legacy alias). 26B small build RETIRED.

## KEY FACTS THIS SESSION KEEPS RELEARNING
- All three shipped 397B rungs = same struct6-tail3x3 base, differ ONLY in
  flat codebook K128/K256/K2048. Base was REBUILT (122G) and lives again.
- Fit chain does NOT propagate tokenizer files (copy pair from shipped
  2.4bpw; check_release gates it). graft_vision needs --prefixes
  model.visual on this family.
- verify_artifact on 397B: create src load+slice UNDER `with
  mx.stream(mx.cpu)` (stream binds at op creation) or the Metal watchdog
  kills it on disk stalls.
- pack only non-byte-aligned widths (bits%8==0 saves nothing, costs 37%).
- Every new gate: acceptance-test against KNOWN-BAD input first.
- E-numbers through E80. gemma cheap-shallow = E60 (+0.94, UNDER bar).
- E79: E71's swap conclusion RETRACTED (proxy-score comparator). The
  ladder is monotone in size; cheap-shallow buys arbitrary-size targeting
  (~2x byte-efficiency of flat steps), never a quality win at a flat rung's
  own size.

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
Chip **task_8af591c9** covers all three (K512/K1024/K2048); the earlier
two-artifact chip was withdrawn and replaced. Decode every verdict with:
    ./score_blind_verdict.py --verdict winrate/claude_verdict_<tag>.json --tag <tag>

**PEER SESSION ENDED** (socket gone). Their result is in E47.3: all three
published 397B artifacts verified CLEAN, 513 tensors. A final message to
them about the bpw correction (E50) was undeliverable — it matters to them
because their pre-registered boundary was computed from my wrong numbers.

**SCOUT: nightly-dispatcher started** once the last prose job cleared the GPU
(services list also shows `overnight-runner`, not started — Noah asked for
the dispatcher). Verify with:
    cd /Users/noahzelezny/Documents/AgenicAI && \
      .venv/bin/python scripts/scout_services.py status nightly-dispatcher

---

# ===== COMPACTION HANDOFF (08-19 midday) =====


## 0. PUBLISH HOLD (08-19 afternoon) — gemma-small verdict is NOT decision-grade yet

Noah caught it: e4b-8bit (84.62%) scores ABOVE its own bf16 teacher (82.69%)
on litbench — noise announcing itself. At n=104 the 4.8pt "falsification"
gap is ~1.3 SE. Publish of MODEL_CARD_GEMMA_SMALL is HELD pending E56
(pre-registered protocol + readings in EXPERIMENTS.md). New facts:
e4b-8bit is 8.38 GiB vs gemma-small 9.43 GiB — the incumbent is SMALLER,
so gemma-small must WIN outright; a tie keeps e4b-8bit.
Queued: gemma_small_verdict.sh (litbench per_item reruns incl. cyclic
26b-bf16, paired McNemar, domain gens, constraint pass-rates).
M4 runs the 26b-vs-e4b bf16 faceoff control (~/qlab/bf16_faceoff.log).
Instruments: paired_litbench.py, check_constraints.py,
winrate/prompts_domains.json (make_domain_prompts.py).
Also learned: verify_artifact --outlier cries wolf on MIXED-geometry
artifacts (arm3's d4K2048 tail reads 3.2x the d2K512 median at a healthy
0.188) — read the gate per-geometry-region on tail/head builds.


## 0b. PUBLISHED (08-20 midday) — gemma collection live; 397B swap awaiting Noah

- Collection: https://huggingface.co/collections/TheDrainFlorist/gemma-4-vq-apple-silicon-6a873d1717a70d83dbee7f02
  - gemma-4-26b-a4b-it-VQ-6.2bpw (18.74G, blind-indistinguishable from bf16;
    card carries sizing guidance: small bracket -> e4b-8bit)
  - gemma-4-e4b-it-VQ-PLE (7.39G; KL 7.451 vs incumbent 8.149; the one
    surface where data-free VQ beats calibrated affine outright)
  - 26B small build RETIRED unpublished (card marked; E56 hold resolved by
    replacement, not by softening)
- 397B cheap-shallow: ALL gates green (verify 171/171, vision 333/333,
  check_release, referee reproduced exactly). Swap picture complete (E71):
  prose ppl -1.4% / code tie / decode wash / prefill -8% / -4G disk /
  -3.8G peak. NOAH'S CALL pending.
- Gates added this cycle: check_release.py (required files + tokenizer
  FUNCTION), check_vision site-counting, byte-aligned pack skip; house
  rule: every gate acceptance-tests against a KNOWN-BAD input first.

## 1. A PUBLISHED CLAIM WAS FALSIFIED — read this first

**e4b-8bit scores 84.62%; our gemma-small scores 79.81%.** Same instrument
(litbench generative + cyclic, n=104, 0 unparsed), e4b's positional lean is
MILDER than ours (35/21/26/22 vs 50/17/20/17). So the "26B in the e4b-8bit
sidecar slot" premise FAILS on literary work. MODEL_CARD_GEMMA_SMALL.md now
says so and tells readers to keep e4b-8bit for that use.

**Related instrument hazard:** the `26b bf16 | 48G | 84.62%` row in
CRUSH_RESULTS' generative table is **NON-cyclic**, sitting among cyclic
rows. There is NO cyclic litbench for 26b bf16. Do not compare it against
cyclic numbers.

## 2. MEASURED gemma ladder — packed AND sighted, all apples-to-apples

| build | geom | K | bpw | sighted | KL | agree |
|---|---|---|---|---|---|---|
| struct8-e8 (8-bit ref) | affine | — | 8 | 24.98G | 441 | 79.95% |
| **vq-K2048-d2** | d2 | 2048 | 5.75 | **18.74G** | 537 | 77.89% |
| vq-K1024-d2 | d2 | 1024 | 5.25 | 17.41G | 609 | 75.90% |
| vq-K512-d2 | d2 | 512 | 4.75 | 16.08G | 744 | 72.72% |
| vq-K256-d2 | d2 | 256 | 4.25 | 14.75G | 950 | 68.27% |
| vq-K8192-d4 | d4 | 8192 | 3.50* | 13.42G | 1426 | 61.32% |
| vq-K2048-d4 | d4 | 2048 | 3.00* | 12.53G | 1856 | 56.56% |
| vq-K64-d2 | d2 | 64 | 3.25 | 12.09G | 1779 | 57.68% |
| vq-K512-d4 | d4 | 512 | 2.50* | 11.65G | 2969 | 45.04% |
| vq-K32-d2 | d2 | 32 | 2.75 | 10.76G | 2570 | 48.84% |
| **vq-K256-d4** | d4 | 256 | 2.25 | **9.43G** | 3363 | 42.65% |
*nominal; d4 K>256 pays the stranded-third penalty (E54), effective is higher.

Sizes match the one-parameter model to <0.1 GiB. Packing verified as a pure
representation change on the newly packed rungs (d2-K32: 2569.541 both ways).

## 3. IN FLIGHT

- **M4**: flat d2-K512 (qwen) FIT DONE — *not yet verified*. Policy: verify
  on M3 (`verify_artifact.py --outlier 3.0`) before believing any number.
  Then add_model_file -> pack -> kl_ppl_calibrate.
- **M3 queue** (serialized, logs_*.log): verify_packed (running), then
  qwen_k8192.sh, then arm3_headup.sh.

## 4. THE ARM EXPERIMENT — predictions are pre-registered, do not rationalize

My "tail30" builds are **HEAD-DOWN** builds: `--tail-from 10 --tail-geom X`
puts the CHEAP geometry on layers 0-9. I named them for the end I was
promoting; the 397B session caught it, and the wrong name had already put a
wrong mechanism into BOTH sets of notes.

| arm | L0-9 | L10-29 | L30-39 | size | ppl |
|---|---|---|---|---|---|
| 1 flat | d2-K512 | d2-K512 | d2-K512 | ~19.6G | fit done, unscored |
| 2 head-DOWN | **d4-K2048** | d2-K512 | d2-K512 | 17.88G | **0.991x** |
| 3 head-UP | d2-K512 | d2-K512 | **d4-K2048** | ~17.9G | queued |

Arms 2 and 3 differ ONLY in which end is cheap. Readings agreed in advance:
arm2>arm3 = shallow layers are cheap (and head-down SHRINKS artifacts);
arm3>arm2 = deep promotion was the active ingredient (E49 survives);
arm2~arm3 = position irrelevant, E49 dies as a schedule claim.
BOTH sessions predict arm2 > arm3; this session adds a magnitude (arm3
0.005-0.015 worse, ~0.996-1.006x). See E55.

## 5. THE DURABLE RESULT (E55) — outlives the schedule question

**Do not rank VQ builds by relerr across different allocations.** Two
measured inversions on Qwen3.6: flat d2-K256 has the BEST fit error (0.0834)
and WORST ppl (1.016x); both head-down builds fit worse and score better.
Both sessions reasoned from that proxy to a wrong conclusion within hours.
Use ppl where valid, blind judging where not.

## 6. UNCHANGED, VERIFIED, PUBLISHABLE

| target | artifact | size | evidence |
|---|---|---|---|
| gemma quality | vq-K2048-d2-packed-sighted | 18.74G | indistinguishable from bf16 blind (11-23, 26 tie, p=.058) |
| gemma small | vq-K256-d4-sighted | 9.43G | ties 15G 4bit; LOSES to e4b-8bit (see §1) |
| qwen quality | vq-tail30-d2k512-packed | 17.88G | ppl 0.991x vs 8bit 0.999x @ 35G |
| qwen small | vq-K2048-d4-packed | 12.96G | ppl 1.029x vs 4bit 1.041x @ 19G |

Four model cards written (MODEL_CARD_{GEMMA,QWEN}_{QUALITY,SMALL}.md).
Scout nightly-dispatcher running. M4 is unreliable (E47) — verify its output
on M3 always. Stage commits BY PATH, never -A; no Co-Authored-By trailer
(AGENTS.md Commits 1-3, 5).
