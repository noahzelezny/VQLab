# Flash-Next (Qwen3.8-Flash-Next, qwen4_exp) — family arc ledger

Arc opened 2026-08-27. Prior entries for this arc live in paper/LEDGER.md
(2026-08-27 through 2026-08-28, commits 03194f53..96f83dc4); this file is
authoritative from here forward.

## 2026-08-28 — state at ledger creation; overnight ladder run

MODEL: 180B total / 10-of-512 active MoE + 51.2B ngram PLE (one bank, 128
shards, [~2.5M, 160]) + MTP head + vision tower. BF16 teacher and FP-side
facts in paper/LEDGER.md entries.

INSTRUMENTS (all VQLab, all public): prose = pinned WikiText; code =
public mlx corpus (canonical since fd5336c); stream_score (layer-streamed
ppl + KL-to-cached-teacher, validated to all printed decimals); teacher
top-64 cache at Exo Models/flashnext_teacher_topk_prose.

AFFINE LADDER (2048 tok): q3 3.649bpw/75GiB 12.8502 prose; q4 4.649/96
6.4534, KL 293.9; q5 5.649/116 5.2434, 91.7; q6 6.649/137 4.9155(-4.9%,
slice artifact, KL 52.8 ranks it truly); q8 8.649/178 5.1968, 27.1; bf16
5.1662 / 1.9015 code. Family is ~20x more quant-sensitive than the 27B at
matched rungs (best rung 27 mnats vs 1.25).

VQ RUNG 1 — d4/K2048, 66.5 GiB (~3.1 bpw), THE 96GB TIER. Experts mean
relerr 0.1875 (144/144, zero refits); PLE 0.1813 (128/128); PLE codes
row-packed 55 B/row. Scores: prose 5.2911 (+2.4%), code 1.9384 (+1.9%),
KL 146.6 mnats, top-1 86.6%. Beats q4 on every column at 30 GiB less;
within 0.9% of q5 at 49.5 GiB less. ALL SIX GATES PASS (incl. cross-box
verify on M4, worst tensor 0.1912; vision 333/333 grafted). 16.6 tok/s
decode resident on the M3, 71.8 GB peak.

RUNG SIZING RULE (Noah): build to the box, not the bpw — headroom ratio
~0.69 of RAM (66.5:96 ≈ 101:128). The d8/K16384 (~54 GiB) was cut mid-fit
as an awkward tier (partial fit kept, 27 tensors, resumable); geometry
lesson from its abort: relerr thresholds are GEOMETRY-DEPENDENT (d8
normal ~0.32-0.36 per the 397B's 0.3156 mean; 0.35 default is
d4-calibrated — documented in the fitter, e0a9a4a).

OVERNIGHT QUEUE (self-driving, in value order):
1. d2/K256 fits -> auto-assembly -> gates -> scores (~102 GiB, 128GB tier,
   the q4-slot quality rung; uint8 codes = no pack step)
2. d8/K4096 fits (~48 GiB, 64GB tier; PLE rows 20x12=30 B aligned)
3. d2/K1024 fits (~124 GiB, diminishing-returns tier; PLE 100 B/row,
   needs pack-ple at assembly) — killable if morning has better uses

GLM-5.3 readiness survey spun off as a separate session (task_378d8069);
its deliverable lands as GLM53_VQ_READINESS.md and belongs in
research/glm53-flash/ when it arrives.

## 2026-08-29 (night) — rung 2 assembled; a decode bug caught by the identity gate; table cells filled

RUNG 2 (d2/K256): fits mean relerr experts 0.0837 / PLE ~0.0795 (best
geometry yet). Assembled 115.4 GiB -> PLE pack to 8-bit rows (80 B/row,
round-trip-verified) -> 92.4 GiB (~4.3 bpw, 0.72 of 128 GB — on the
sizing rule). Pre-pack scores (M3 streamer): prose 5.3825 (+4.2%), code
1.9033 (+0.1%), KL 59.04, top-1 91.85% — q6-class fidelity 45 GiB under
q6. Cross-box verify (M4): 144/144 from artifact bytes, means
0.0834-0.0840. Gates: check-release/check-bundle PASS; smoke PASS on M4.

DECODE BUG, caught by the packed-path identity check: post-pack streamed
scores read NaN — VQPLEEmbedding's unpack HARDCODED 11-bit strides, so
8-bit K256 rows were read misaligned (indices past the codebook; junk
gather). SMOKE PASSED on the same broken decode — the sharpest proof yet
that generation is a weak gate and the identity re-score is not optional.
Codes on disk always correct (pack-time round trip). Fixed: stride+mask
derived from codebook size (VQLab commit "unpack stride from codebook
size"); bundle regenerated; re-score running — rung 2 is QUARANTINED
until the packed path reproduces prose 5.382537 exactly.

LITERARY COLUMN (M4 sweep, 2048 tok) + q3 KL — table now dense:
  q3 19.4794 | q4 9.0975 | q5 7.8895 | q6 7.7097 | q8 7.6695
  teacher 7.6643 | VQ d4/K2048 7.8077 (+1.9%)
  q3 prose KL: 1083.35 mnats, top-1 61.9% (completes the KL column;
  affine at 3.65 bpw is not merely worse, it is a different model)

## 2026-08-29 (night, cont.) — rung 2 QUARANTINE LIFTED: packed path exact

Fixed decode reproduces every printed digit: prose 5.382537, KL 59.0428,
top-1 0.9185, code 1.903324 — bit-equivalent to pre-pack. Literary
7.7112 (+0.6% vs teacher; q6-level). RUNG 2 IS DONE: 92.4 GiB, ~4.3 bpw,
all gates, three corpora, KL — the 128 GB-tier quality rung.

Two shipped rungs now:
  d4/K2048  66.5 GiB  KL 146.6  top-1 86.6%  (96 GB tier)
  d2/K256   92.4 GiB  KL  59.0  top-1 91.9%  (128 GB tier)

## 2026-08-29 — rung 3 (d8/K4096) assembled and scored: the 64 GB floor, decision pending

43.8 GiB (~2.0 bpw whole; PLE rows packed to 30 B). All gates PASS incl.
smoke; splice succeeded after splice_ple was pinned to CPU (4th watchdog
instance; first M4 attempt corrupted the packed dir mid-splice — rebuilt
clean from intact fit dirs). d8 down_proj (NSUB=80) rides UNPACKED
pending the padded-tail GPU acceptance (below).

Scores (2048 tok): prose 7.0390 (+36%), code 2.2605 (+19%), literary
10.5480 (+38%), KL 556.10 mnats, top-1 74.3%. Reading: keeps the VQ
quality-per-byte line above affine (half of q3's 1083 mnats at 58% of
its size) but is a FLOOR rung in absolutes — worse than q4 at less than
half q4's bytes. Precedent for shipping honest floors: 397B VQ-2.2bpw
(~+31% prose) is the family's most-downloaded artifact. Ship / mixed-
allocation retry / hold — Noah's call.

Also this morning (walk window): padded-tail pack format landed (VQLab
e4d5a6a + 1992ff6) — unaligned NSUB packs via zero-padded tail block,
ceil-WPR in all 7 packed kernels, aligned output byte-identical to HEAD
(verified), NSUB=80/12-bit round-trip exact; gated behind
--pack-unaligned until bundle_accept's new unaligned cases pass on
metal. GLM arc (peer session): mlx-vlm venv staged, scorer line-verified
against INSTALLED code (3 fixes), glm5_next confirmed in the pip
release; GPU-window checklist agreed (selftest, bundle_accept, rule-5).

## 2026-08-29 — 64GB-rung lever: diagnostic hybrid queued, prediction registered

Plan (Noah-approved): (1) DIAG hybrid — d8/K4096 experts + d4/K2048 PLE
(~57 GiB, instrumental only, never ships) — one streamed KL isolates
whether experts or PLE tables drive rung 3's 556 mnats. (2) Apply the
lever where diagnosis points: experts -> tail-weight-pow p~2 with
--tail-weight-from ~11 (E102's scarce-centroid regime is exactly
d8/K4096; E106/E110 shallow-layer guard); PLE -> weighted PLE fit or
byte reallocation.

PREDICTION, registered before the run (Noah): the experts, not the PLE,
carry the damage — the hybrid's KL stays high (~500). If instead KL
collapses toward ~200, the tables were the bleeder and the hunch is
wrong. Arithmetic note either way: expert relerr 0.4156 vs PLE 0.4094 at
this geometry are nearly equal, so relerr alone cannot adjudicate —
which is the point of measuring at the output.

## 2026-08-29 — DIAGNOSTIC VERDICT: experts carry the damage (prediction confirmed)

Hybrid (rough d8/K4096 experts + good d4/K2048 PLE, 58.7 GiB,
instrumental): KL 544.28 vs rung 3's 556.10, top-1 75.1% vs 74.3%,
prose 6.9739 vs 7.0390. A 2.3x PLE reconstruction improvement moved
~2% of the divergence: THE EXPERTS ARE THE BLEEDER. Noah's registered
prediction confirmed. Corollary worth bits: PLE tables are nearly
damage-free at 2.0 bpw — low rungs can push them cheaper and spend on
experts. Also: equal relerr (0.416 vs 0.409) produced wildly unequal
output damage — §4.3's law again, measured at component level.

Lever fires: d8/K4096 expert refit with --tail-weight-pow 2
--tail-weight-from 11 (E102 scarce-centroid regime; E106/E110 shallow
guard). Same bytes; also repacking down_proj via --pack-unaligned
(criterion met this morning) -> target ~41.3 GiB.

## 2026-08-29 ~10:45 — compaction anchor: what is in flight

M3: weighted expert refit d8/K4096 (--tail-weight-pow 2 --tail-weight-from
11) -> Exo Models/qwen4exp_vq_fit_d8k4096_tw, ~2h. EXPECTATION SET: its
relerr will read WORSE than 0.4156 by design (weighted objective trades
mean for tail); judge by KL only. On completion: assemble via the standard
chain, pack experts WITH --pack-unaligned (criterion met), PLE from the
existing d8/K4096 PLE fit (diagnostic proved PLE near-damage-free), target
~41.3 GiB, score, compare KL vs 556.
M4: d2/K1024 assembly+scores -> fills TABLE.md's last row (~124 GiB rung,
no smoke possible on owned boxes — 192GB-class artifact).
Then: 64 GB ship/hold call (Noah's), release block (cards/HF/VQLab tag),
DIAG hybrid dir is deletable after the refit scores land.
GLM: fully validated stack, ready for teacher pass + affine ladder + fits
whenever GPU frees; peer session had VQLab commit permission denials —
surfaced to Noah, unresolved.

## 2026-08-29 — d2/K1024 rung lands (M4)

Assembled + scored clean on the M4: packed 144 expert tensors 174.6 ->
132.4 GiB, PLE row-packed at row_bytes=100 -> 111.6 GiB final (better
than the ~124 estimate). check-bundle PASS; verify + smoke still owed
(192GB-class — no owned box can smoke it; same caveat as recorded for
the class).

Scores: prose 5.2449 / code 1.8975 / literary 7.6358 / KL 34.14 mnats /
top-1 94.1%. Sits between q6 (52.76 @ 137 GiB) and q8 (27.06 @ 178 GiB):
beats q6 on every column at 25 GiB less, and gets within 1.26x of q8's KL
at 66 GiB less. Literary reads below bf16 — slice artifact, noted in
TABLE.md. Ladder table now complete; only the weighted d8 refit remains
in flight.

## 2026-08-29 — weighted refit scored: THE LEVER FAILS AT THIS RUNG

d8/K4096 tail-weighted refit (--tail-weight-pow 2 --tail-weight-from 11),
assembled with --pack-unaligned (first production use), smoke PASS:

  prose 7.2576 (was 7.0390)   KL 581.30 (was 556.10)   top-1 73.3% (74.3%)
  code  2.2162 (was 2.2605)   literary 10.9853 (was 10.5480)   43.7 GiB

Verdict: WORSE on KL, top-1, prose, literary; only code improved slightly.
The E102 scarce-centroid lever does not transfer to this family/geometry —
at d8/K4096 on Flash-Next experts, tail emphasis buys the tail less than
it costs the body. Negative result recorded; the BASELINE d8/K4096
(43.8 GiB, KL 556.10) remains the 64GB-tier candidate. Refit fit dir
(90.2 GiB) + tw artifact (44 GiB) + DIAG hybrid (58.7 GiB) now cleanup
candidates — Noah's call.

Silver lining: this run took --pack-unaligned through a full model load
for the first time and caught two stale floor-WPR defects the GPU kernel
acceptance could not see (bundle shim allocation; lossy WPR->NSUB in
input_dims/from_weights). Fixed in VQLab ad6918c, smoke-verified;
value-identical for all aligned artifacts. It also established that the
original d8/K4096 pack had left down_proj UNPACKED (aligned packer
skipped NSUB=80); an unaligned repack of the baseline would shave ~0.1
GiB — not material.

## 2026-08-29 — reallocation play launched: d8/K16384 experts + d8/K256 PLE

Noah: "the best possible configuration for a 64gb tier model... might be
the thing to do." The diagnostic's corollary, applied: move bytes FROM the
near-damage-free PLE TO the experts. tw artifacts deleted (lever failed).

M3: resume d8/K16384 expert fit (27/144 tensors already on disk from the
mis-aborted run; --relerr-abort 0.45, d8-calibrated). M4: fresh PLE fit
at d8/K256 (16x smaller K than the 12-bit fit -> fast; geometry otherwise
matches the K4096 manifest: group 32, iters 12, seed 1234).

Target: ~45.5 GiB (experts 12->14 bits, PLE 12->8 bits). Bet: KL
meaningfully below the baseline's 556.10. Assembly will use
--pack-unaligned end-to-end (load path fixed in VQLab ad6918c).

## 2026-08-29 — reallocation WINS: the 64GB rung is d8/K16384 + K256 PLE

Assembled (--pack-unaligned end-to-end), all gates + smoke PASS, 43.7 GiB
— byte-neutral with the d8/K4096 baseline. Scores:

  KL 419.88 (was 556.10, -24%)   top-1 77.8% (74.3%)
  prose 6.0216 (7.0390)   code 2.1018 (2.2605)   literary 9.1096 (10.5480)

Every column improved at the same bytes. The diagnostic's corollary holds
at full strength: PLE relerr 0.409 -> 0.571 (K4096 -> K256) cost nothing
measurable, expert relerr 0.416 -> 0.353 (K4096 -> K16384) bought 136
mnats. Fit economics: 27/144 expert tensors reused from the mis-aborted
K16384 run; K256 PLE fit took ~9 min on the M4 (vs ~80 at K4096).

TABLE.md 64GB row replaced (update-in-place rule). The d8/K4096 baseline
artifact (43.8 GiB) and its fit dirs are now superseded — cleanup is
Noah's call. This artifact is the 64GB ship/hold candidate.

## 2026-08-29 — layer-leverage probe + L0/L1 mixed splice

New VQLab instrument (`vqlab layer-leverage`): interleaved teacher/student
streamed pass, per-layer local damage + trajectory drift. Findings on the
64GB rung: allocation is NOT flat — L1 is a monster (local 0.310, 2.4x any
other layer; the traj jump across it is 0.204 of the final 0.481), L8-11
near-free (~0.04), warm late band L28-39 (~0.11-0.14).

Splice probe (mixed-K artifacts load natively; per-module vq_modules):
d2/K256 experts into L0+L1 only, one-shard surgery + hardlinks ->
qwen4exp_vq_packed_mixL01, 45.0 GiB. Gates + smoke PASS.
KL 390.09 (was 419.88), top-1 78.8% (77.8%), prose 5.9033 (6.0216).
Probe ranking confirmed CAUSAL. Rate: ~23 mnats/GiB.

q4 (294) at 64GB does not fall out of this lever alone: full late band
+10 GiB -> ~56 GiB (breaks headroom). Candidate next: top-4 late layers
(L36/L35/L39/L31, +3.3 GiB -> ~48 GiB), plausibly KL ~330-350.

## 2026-08-29 — late-4 splice + PLE floor found

mixL01p4 (d2/K256 experts in L0,L1,L31,L35,L36,L39): KL 361.51, top-1
80.3%, prose 5.8328, 48.3 GiB. Cumulative from the flat rung: 419.88 ->
361.51 (-14%) for +4.6 GiB. Gates + smoke PASS throughout.

PLE probe #2: K256 -> K16 tables (1.0 -> 0.5 bpw rows, fit relerr 0.78,
~2 min on M4). KL 361.51 -> 387.50 (+26 mnats) at -3.0 GiB. THE PLE HAS A
FLOOR between 8-bit and 4-bit rows: K256 is free, K16 is not. Exchange
rate note: late-band expert bits buy ~8.7 mnats/GiB and K16 PLE bits lose
~8.7 mnats/GiB — a wash, so at this margin PLE-vs-expert reallocation
moves ALONG the frontier. The 64GB-tier candidates are 48.3 GiB @ 361.5
(best quality) vs 43.7 @ 419.9 (max headroom); ship pick is Noah's.

## 2026-08-29 — quiet-layer scoop FAILS: sensitivity is not linear

Downgraded the probe's 10 quietest layers (L3,8-11,13,16,23,45,46; local
0.040-0.065) from d8/K16384 to d8/K256 in the best mix (fit relerr 0.575,
scatter mini-fit via new --vq-layers comma lists, 157s resume). Gates +
smoke PASS. KL 460.27 — worse than the FLAT rung (419.88), destroying the
late-band gains (361.51). ~-3 GiB was not worth +99 mnats.

Law-shaped takeaway: layer-leverage local_rel is measured AT the current
damage level and does NOT extrapolate — quiet at relerr 0.35 is not quiet
at 0.575. Upgrading hot layers (validated causal) and downgrading cold
ones (refuted) are NOT symmetric operations. mixL01p4q artifact + quiet
fit dir are cleanup candidates. The 64GB frontier stands: #2 mixL01
(45.0 GiB, KL 390.09) best-in-tier; #3 mixL01p4 (48.3, 361.51) is
96GB-territory by headroom.

## 2026-08-29 — leverage map is FAMILIAL

Probed the 3.1 rung (d4/K2048) with layer-leverage: top-10 hot set and
quietest-10 set are IDENTICAL to the 2.0 rung's, layer for layer (hot:
L0,1,31-33,35-39; quiet: L3,8-11,13,16,23,45,46); Pearson r=0.905 across
all 48 layers; L1 dominates both. The damage map is a property of the
MODEL, not the quantization geometry — one probe per architecture serves
the whole ladder. (M4 note: the probe Metal-timeouts on the M4 over SMB;
M3 runs it clean. Instrument needs its memory pass before GLM.)

#2 (mixL01, 45.0 GiB) SHIPS as the 64GB rung — Noah's call. Card scores
complete: prose 5.9033 / code 2.0762 / literary 8.9450 / KL 390.09 /
top-1 78.8%. Launched: hot-6 splice (L0,1,31,35,36,39 <- d2/K256) into
the 66.5 GiB 3.1 rung -> qwen4exp_vq_packed_31mix6, ~69 GiB expected.

## 2026-08-29/30 — hot-6 splice transfers to the 3.1 rung

qwen4exp_vq_packed_31mix6 (66.5-GiB rung + d2/K256 experts in the familial
hot-6): 69.4 GiB, KL 123.46 (was 146.61, -16%), top-1 87.0%, prose 5.2114
— below q5 affine (5.2434 @ 116 GiB) and 0.045 off the teacher. No new
probe or fit needed: familial hot set + existing donors, ~30 min. Gates +
smoke PASS after re-bundle (the old rung carried a pre-fix 1422-line
bundle; check-bundle caught it — release block must re-bundle the shipped
66.5/92.4/111.6 artifacts to the current runtime).

## 2026-08-30 ~00:15 — overnight queue (Noah back ~09:00)

92.4 mix landed earlier: qwen4exp_vq_packed_92mix6, 94.1 GiB, KL 50.33
(was 59.04), top-1 92.8%, prose 5.2229 — beats q6 affine at 43 GiB less.
Gates PASS; smoke on the M4 (94 GiB > M3's preflight bar).

M3 chain (overnight_queue.sh -> stage2): d2/K4096 hot-6 mini-fit (running,
relerr ~0.021) -> 111.6 mix (pack-at-splice 12-bit, ~112.9 GiB target)
-> streamed KL -> GLM teacher prose pass (598.5 GiB, ppl + top-64 cache ->
glm53_teacher_topk_prose) -> GLM teacher code/lit anchors -> GLM q4 affine
convert. M4: GLM struct base (stream-convert --struct, glm5vlm venv built
locally tonight; stream_convert also needs mlx-lm — installed).
Peer quantlab-20 tasked (msg f68acc01): fix layer_leverage memory
accumulation + M4 SMB Metal timeout before any GLM-teacher probe.

## 2026-08-30 ~01:00 — 111.6 mix: the lever's endpoint

qwen4exp_vq_packed_111mix6 (hot-6 <- fresh d2/K4096 mini-fit, relerr
0.0205, 32 min; 12-bit pack-at-splice): 114 GiB, KL 32.69 (was 34.14),
top-1 93.7% (94.1%), prose 5.2539 (5.2449). A WASH for +2.6 GiB — the
hot-6 lever's curve completes: -24%, -16%, -16%, -15%, ~0% up the ladder.
Where the base geometry already fits its hot layers well, richer donors
buy nothing. Original d2/K1024 rung stays the shipped artifact; 111mix6 +
hot6 fit dir are cleanup candidates (post-arc, per the don't-delete rule).

92.4 mix smoke PASSED on the M4 — qwen4exp_vq_packed_92mix6 fully gated.

GLM night ops: struct base relaunched CPU-PINNED on the M4 after the
watchdog killed the GPU attempt (~37s/shard, ~75 min); teacher chain
relaunched on the M3 after installing mlx-lm into the M3 glm5vlm venv
(stream_score imports it for load_tokenizer; only the M4 venv had it).
Peer's layer_leverage fix (VQLab 55864f9) reviewed + CPU-verified;
probe_glm5_next now registered — GLM probe is code-complete.

## 2026-08-30 — RELEASED

All four rungs live on HF under TheDrainFlorist, collection
qwen38-flash-next-vq-data-free-apple-silicon-6a94513559d3614812c3d9bd:
VQ-2.1bpw (45.0, KL 390.1) / VQ-3.2bpw (69.4, 123.5) / VQ-4.4bpw (94.1,
50.3) / VQ-5.5bpw (111.6, 34.1). Cards carry qwen-community-1.0 rider,
three-curve chart, leverage-mix section, gates record; no negative-
existence claims (v4 lesson). VQLab pushed public through 4d1497d (32
commits). Local artifacts KEPT on disk for 397B/GLM benchmarking (Noah).

## 2026-08-30 — MTP accounting (Noah's question)

The teacher's MTP head (31 tensors, 4.86 GiB bf16 — draft embedding tap,
hyper-connection mixer, one transformer layer) is NOT in any shipped
artifact, affine or VQ: PR #1788's qwen4_exp class does not implement MTP
and sanitizes mtp.* away, so no stage ever saw it. Zero quality impact —
MTP is a speculative-decode speed mechanism no MLX runtime implements.
If one appears, the recipe is a graft_vision-style mtp graft (+4.9 GiB
bf16 / ~+1.2 quantized). Cards updated to state text is decoded by the
main model without the MTP head.

## 2026-08-30 — 8-bit MTP graft shipped to all four rungs

The teacher's MTP head (a full 512-expert MoE draft layer, 4.86 GiB bf16)
now rides in every release artifact as model-mtp-graft.safetensors,
8-bit g64 (2.58 GiB; the 3D expert tensors quantize fine — first gate
missed them, widened to ndim 2-3). Disk-only: current runtimes sanitize
mtp.* away, resident sizes unchanged; config carries quantization entries
+ an mtp_graft marker for any future MTP decoder. Correction recorded:
the earlier +1.25 GiB estimate was 4-bit arithmetic; 8-bit is +2.58.
Cards (HF + exo) updated; uploads deduped by xet (one 2.77 GB transfer).

## 2026-08-30 — post-graft smoke gate (retroactive, and a process note)

The MTP grafts were PUSHED BEFORE SMOKING — a violation of the gate rule
caught by Noah ("shouldn't we have tested before publishing?"). Smoke run
retroactively on the M4: 2.1bpw PASS through the shipping runtime;
remaining three share the identical graft/config structure and inherit
the verdict per Noah's scoping. Rule restated: NOTHING reaches HF without
a smoke, including metadata-only and rider-file changes.

## 2026-08-30 — MTP graft PULLED (Noah)

The MTP graft episode is reverted end-to-end: graft files deleted from all
four HF repos, config markers and mtp quantization entries removed (HF +
local), cards restored to "MTP head not included", exo cards restored,
post-removal smoke PASS (2.1bpw, M4). One bf16 graft copy parked locally
(Exo Models/mtp_graft_bf16_2.1.safetensors.parked) for the offline work.

Process verdict, in Noah's words: shipping it was rushing to publish an
unbuilt feature. MTP returns to HF only as a WORKING, MEASURED feature:
module in the bundle (8-bit resident, ~+2.6 GiB), acceptance probe, then
the draft-verify decode loop, each gated. Until then the cards claim only
what runs.

## 2026-08-30 ~evening — compaction anchor #2

RELEASED + stable: four Flash-Next VQ rungs live on HF (collection
qwen38-flash-next-vector-quantized-mlx, notes normalized); MTP graft
pulled everywhere (Noah); VQLab pushed through 4d1497d; exo cards done
(incl. refreshed 4bit card); artifacts renamed to exo model_id names
with symlinks back; local artifacts KEPT for benchmarking.

MTP (offline, nothing published): goal is a WORKING measured feature
(module -> acceptance probe -> decode loop, gated). Probe harness built
(vqlab mtp-probe + scratchpad/mtp_dense_test.py). KEY FINDING: harness
scores 0.0 acceptance on BOTH qwen4_exp 2.1bpw AND the dense 27B teacher
(whose head MTPLX provably makes work) -> ONE bug in shared harness
logic, not in hc wiring. Verified clean: weight loading (checksummed,
all slots matched), head (1.000 argmax vs model()), capture (norm-hook
on real forward), norm styles/concat orders/orientations/shifts all
swept. MTPLX (~/Documents/AgenicAI/MTPLX, Apache-2.0) mirrors my op
order exactly. NEXT STEP (Noah go pending): install + RUN MTPLX on the
27B, dump its mtp_forward intermediates, diff junction-by-junction.

GLM: d8/K16384 fit PARKED at 77/129 tensors (resumable shard-skip,
12/19 shards banked; relaunch cmd in scratchpad/glm_fit_d8k16384 log
header / ledger 2026-08-30). Ladder q3/q4/q6 + teacher + struct base
done. Rungs approved: ~100 (in progress), 120, 140.

Also: stale pre-compaction Monitor watchers killed; stale overnight
/loop stopped. Vision tower stays bf16 until a vision quality gate
exists (Noah agreed).

## 2026-08-30 — MTP SOLVED (root cause: RMSNorm gain convention)

The 0.0-acceptance bug is found and fixed. It was never the hc wiring,
the concat order, the capture, or the h/e pairing -- all of which I had
verified correct. It was the +1.0 zero-centered RMSNorm convention, and
the two architectures resolve it in OPPOSITE places, which is why one
diagnosis could not cover both:

  qwen3_5 (dense 27B): gains stored delta, applied by a conventional
    nn.RMSNorm. mlx-lm's trunk sanitize adds 1.0 -- but mtp.* tensors
    load outside sanitize, so the hand-loaded sidecar kept raw gains.
    Evidence: trunk L0 input_layernorm mean -0.0334 on disk vs +0.9666
    in the live model. Shifting the sidecar: 0.0000 -> 0.7285.
  qwen4_exp (Flash-Next): gains also stored delta, but applied by the
    arch's OWN zero-centered RMSNorm (y = norm(x) * (1 + weight)), which
    adds the 1.0 itself. My hand-rolled rms() did plain `n * w` and
    dropped it. Using arch.RMSNorm: 0.0000 -> 0.6992. Pre-shifting the
    stored weights here would DOUBLE-count -- the MTPLX-style heuristic
    gate was one signal from firing on this checkpoint, so mtp_probe now
    never mutates gains and always applies them via the arch's classes.

Ablations (dense 27B, 512 positions): pre_norm/e-h 0.7285, post_norm/e-h
0.7188 (near-tie, matching MTPLX exposing hidden_variant as a knob);
h-e concat 0.0000/0.0020 (dead). Also fixed a vacuous sweep: the probe
permuted input AND weight together, so its eh/he "variants" were the
same computation. qwen4_exp has separate fc_embedding/fc_hidden tensors
([2560,2560] each), so concat order is not ambiguous at all there.

MEASURED (Flash-Next 2.1bpw, the worst rung, 512 positions, teacher-
forced greedy acceptance vs the main model's own next-token choice):
  wide hidden norm grouped (group_size=2560, per stream)  0.6992
  wide hidden norm flat over the 10240 row                0.6562
Grouped wins, as the arch predicts -- every other 10240-wide norm in
qwen4_exp is RMSNorm(hc_dim, group_size=hidden_size).

MTP head precision (same probe, acceptance vs main):
  bf16   4.86 GiB   0.6992
  8-bit  2.73 GiB   0.6953
  6-bit  2.13 GiB   0.6992
Differences are 0-2 positions in 512, i.e. inside noise at this sample
size: 8-bit and 6-bit are both indistinguishable from bf16. Note the
head's precision cannot affect OUTPUT quality at all -- the main model
verifies every drafted token, so a worse draft costs a rejection (speed),
never a wrong token. Recommend 8-bit (saves 2.13 GiB/rung); 6-bit needs
a larger corpus before claiming the extra 0.6 GiB.

STILL NOT SHIPPED, per the smoke-first rule. Acceptance is a proxy only.
Remaining before anything reaches HF: MTP module in the bundle -> real
draft/verify decode loop -> measured tok/s speedup -> smoke on-device.
Artifacts: parked_mtp_graft_bf16.safetensors + q8sim/q6sim variants.

## 2026-08-30 (cont) — MTP decode loop built and measured

Noah's call: the head ships OPTIONAL ("as long as its optional i think
that's fine"), 6-bit. Precision settled at 4096 positions (2.1bpw):
bf16 2888/4096 = 0.7051, 8-bit 2890 = 0.7056, 6-bit 2894 = 0.7065.
The whole spread is 6 hits and 6-bit comes out nominally HIGHEST, which
is itself proof the differences are noise (a quantized head cannot beat
its own bf16). Binomial SE at n=4096 is +/-0.7pp vs a 0.14pp spread.
6-bit head = 2.13 GiB. Head precision cannot affect output quality at
all -- the trunk verifies every drafted token, so a worse draft costs a
rejection (speed), never a wrong token.

Optionality is mechanically real: mlx-lm discovers weights by globbing
`model*.safetensors` (utils.py:349) and does NOT consult the index, so a
file named `mtp-head-q6.safetensors` is invisible to the stock loader and
loads only when our bundled model.py asks for it. The vision tower proves
the other half -- model-vision-graft.safetensors DOES match the glob and
is read every load, but sanitize drops visual.* before materialization,
which is why resident is 44.96 GiB. Vision tower is only 0.84 GiB bf16.
NOTE: optional means optional RESIDENCY, not optional download -- exo and
most HF clients pull whole repos, so the gate belongs in the runtime.

Headroom, 45 GiB rung, everything on: 44.96 trunk + 0.84 vision + 2.13
head + 0.75 KV@32k = 48.68 GiB. KV is cheap here: only 12 of 48 layers
are full-attention, 2 kv heads, head_dim 256 => 24 KiB/token (0.75 GiB
@32k, 3.0 GiB @128k); the 36 linear layers hold ~56 MiB of constant
state. Stock 64GB limit is ~48 GiB (75%) -- Gemini corroborates, but
NEITHER of us measured it on a stock box, and this machine is not stock
(iogpu.wired_limit_mb = 86016 = 88% of 96 GiB). It is the one assumed
number in the headroom table.

DECODE LOOP (scratchpad/mtp_decode.py) -- built, correct, measured.
Rollback is O(1): every qwen4_exp cache slot is REASSIGNED, not mutated
(cache[0] = ..., cache[1] = state), and mlx arrays are immutable, so
keeping the old references is a free snapshot; attention uses trim() by
the offset DELTA (trimming a hardcoded 1 left a stale key while the
recurrent caches rolled back 2).

MEASURED (2.1bpw, 96 tokens, 6-bit head): baseline 15.6 tok/s ->
speculative 18.8 tok/s = 1.20x, acceptance 0.708 over 48 steps.

Bit-identical output is NOT achievable and demanding it is a false gate.
Chunk control: the trunk disagrees with its OWN single-token greedy
stream at 2/96 positions when the same tokens are fed as a chunk, and
the top-2 logit gaps there are 0.25 and exactly 0.00 against a median
gap of 3.625 -- literal ties, numerics not meaning. Verification always
happens inside a 2-token forward, so every correct speculative
implementation on this runtime inherits it. Gate is now: divergence must
be confined to near-ties.

INVALID: the 3.2bpw run (0.34 tok/s speculative vs 4.71 baseline) is a
memory-pressure artifact, not a result -- 69.4 trunk + 2.13 head + the
2-token forward's larger buffers + live snapshot references thrashed
against the 84 GiB wired limit (1.76 GB swap touched). Discarded, not
relaunched. A clean large-rung number needs a box with real headroom.

OPEN: 1.20x is well under the ~1.55x ceiling implied by 0.708 acceptance
(the 2-token forward is not free, and the head forward adds cost). The
MTP head's own KV cache is also only approximately aligned -- it advances
one position per step while two tokens commit, which costs acceptance but
never correctness. Both are speedup headroom, not blockers.

STILL NOT SHIPPED. Noah decides whether 1.20x justifies 2.13 GiB.

### Correction (same day): the 3.2bpw slowdown, and what the sim files measure

I hypothesized the VQ decode kernel had a seq=1 fast path that speculative
verification would miss. Noah asked the right question -- "wouldn't that have
affected the 45gb artifact in the same way?" -- and the microbenchmark
(scratchpad/seqcost.py) REFUTES the hypothesis outright:

  2.1bpw: seq=1 61.09 ms, seq=2 49.15 ms, seq=4 58.29 ms
  3.2bpw: seq=1 60.83 ms, seq=2 48.30 ms, seq=4 56.23 ms

Per-forward cost is identical across artifacts, and seq=2 is CHEAPER than
seq=1. No kernel cliff. Speculative verification is not the problem.

The real cause is a measurement error of mine: the q8sim/q6sim grafts are
quantize->DEQUANTIZE simulations, all 4.86 GiB bf16 on disk (verified). They
measure the weight ERROR faithfully, so every acceptance number stands --
but every decode run carried a 4.86 GiB bf16 head, never 2.13 GiB. So:
  2.1bpw: 45.0 + 4.86 ~= 50 GiB, room to spare -> genuine 1.20x
  3.2bpw: 69.4 + 4.86 ~= 74 GiB against an 84 GiB wired limit -> thrashing
Both "slow" 3.2bpw runs were memory, one via swap (Safari open, Noah
confirmed yellow pressure) and one via buffer-cache thrash with swap flat.

Consequences: the 2.13 GiB figure in the headroom table is what a REAL
quantized module would cost and remains unbuilt/unverified. A clean
large-rung speedup number needs that module, or a box with real headroom.
The MTP head forward is also expensive in absolute terms (~44 ms inferred
on the 2.1bpw run, against a 48-61 ms trunk forward) precisely because it
is bf16 with its own 512-expert MoE -- quantizing it for real should cut
that materially and raise the speedup above 1.20x.

## 2026-08-30 (cont) — REAL 6-bit MTP module: 1.56-1.67x measured

Noah: "can we actually do the q6 since that should be faster anyway?"
Yes -- and it roughly doubles the win. nn.quantize over the head's block
and mixer at 6 bits / group 32, router left full precision (mirroring
qwen4_exp's own quant_predicate). NOTE the predicate must gate on
hasattr(mod, "to_quantized"): nn.quantize hands it EVERY submodule and
raises on RMSNorm otherwise.

MEASURED (96 tokens, greedy, real quantized head):

  rung     head       baseline    speculative   speedup   acceptance
  2.1bpw   6-bit      16.07 t/s   26.87 t/s     1.67x     0.708
  2.1bpw   bf16       15.63 t/s   18.81 t/s     1.20x     0.708
  3.2bpw   6-bit      15.19 t/s   23.71 t/s     1.56x     0.625
  3.2bpw   bf16       15.95 t/s    6.14 t/s     (thrash)  0.625

Head resident measured via mx.get_active_memory() delta: 2.12 GiB, which
confirms the 2.13 GiB projection in the headroom table exactly. That
number is no longer an assumption.

Why bf16 was so much worse: acceptance is IDENTICAL (0.708 / 0.625), so
the head drafts just as well either way -- the bf16 version simply costs
~44 ms per draft forward gathering 512 bf16 experts, against a 48-61 ms
trunk forward. Quantizing the head removes most of the draft cost. The
bf16 3.2bpw row is a thrash artifact (74 GiB against an 84 GiB limit) and
is listed only to mark it as explained, not as a speed result.

1.67x sits slightly ABOVE the naive ceiling implied by 0.708 acceptance
(~1.55x) because a seq=2 trunk forward is cheaper than seq=1 (49 vs 61
ms, measured identically on both artifacts -- see the refuted kernel
hypothesis above), so verification is better than free per token.

REMAINING BEFORE SHIP: the head is currently quantized at runtime from
the bf16 graft. Shipping needs a pre-quantized 6-bit sidecar written to
disk (~2.1 GiB) named OUTSIDE the `model*.safetensors` glob, plus the
runtime gate in the bundled model.py, plus an on-device smoke. Nothing
published yet; Noah's call.

## 2026-08-30 (cont) — MTP sidecar built, smoked, staged on disk

Noah: "lets do it. It looks like it works and is helpful, lets make it an
option!" Built as VQLab commands (VQLab e6eb09a):
  vqlab mtp-pack      bf16 graft -> quantized drafting sidecar
  vqlab mtp-generate  draft/verify loop, --benchmark for the numbers
  src/vqlab/mtp_head.py  the wiring, in ONE place

Sidecar: mtp-head-q6.safetensors, 71 tensors, 2.14 GiB on disk, 6-bit /
group 32, router left full precision. Built lazily -- no trunk weights are
materialized, so packing costs ~0 GiB. One file serves all four rungs
(verified it loads against the 3.2/4.4/5.5 configs: fa_idx 3, hc 4,
D 2560, tie False everywhere); the head depends on the config, not on the
trunk's quantization. Copied into all four artifact dirs.

OPTIONALITY PROVEN, not asserted -- this is the claim I got wrong before:
  sidecar sitting in the 2.1bpw artifact dir              yes
  present in the 138 files mlx-lm's model*.safetensors    NO
    glob discovers (utils.py:349, never reads the index)
  resident with the sidecar on disk       44.96 GiB, byte-identical
                                          to the pre-MTP measurement
  stock generate() still works            "Paris.\nThe capital of France
                                          is Paris.\nIs"

SMOKE PASSED end to end from the sidecar loaded off disk (2.1bpw):
  head resident 2.12 GiB
  speculative 25.54 tok/s vs baseline 15.72 tok/s = 1.62x
  acceptance 0.667 (prompt differs from the 0.708 run)
  chunk control 2/96, top-2 gaps 0.125/0.125 vs median 2.500

A warmup trap worth remembering: the first sidecar run measured 1.16x with
the SAME weights, purely because Metal compiles the 2-token trunk path and
the quantized head kernels on first use and mtp-generate timed the
speculative path first (mtp_decode.py had run baseline first and warmed
them). Both paths are now warmed before timing. Any future A/B on this
runtime needs the same care.

NOT YET UPLOADED. Remaining: HF upload of the sidecar to 4 repos (8.6 GiB)
+ model card sections + exo card note. Card text goes to Noah first --
public claims, and the wired-limit figure for stock 64GB boxes is still
the one number neither of us has measured.

## 2026-08-30 (cont) — Noah's four questions, answered by measurement

"Is it actually 1.6x?" -- fair challenge; the single prompt used until now
drove the model into a degenerate repetitive stretch, which is unusually
easy to draft. Swept six prompt types, 128 tokens each, 2.1bpw, sidecar
loaded off disk (scratchpad/mtp_bench_prompts.py):

  prompt      acceptance   baseline   speculative   speedup
  factual        0.656     13.53      21.29         1.57x
  code           0.703     13.72      24.68         1.80x
  prose          0.578     15.05      24.27         1.61x
  reasoning      0.797     14.66      24.86         1.70x
  technical      0.641     14.48      22.63         1.56x
  chat           0.812     14.25      25.09         1.76x

  speedup    min 1.56x  MEDIAN 1.65x  max 1.80x
  acceptance min 0.578  median 0.679  max 0.812

The suspicion was backwards: the degenerate "technical" prompt is the
SLOWEST row, not the fastest. 1.6x is fair and slightly conservative;
"1.56-1.80x, median 1.65x" is the honest card claim. Prose is the hardest
to draft (0.578) and still returns 1.61x, because a rejected draft is not
wasted -- the trunk's own next token comes out of the same forward.

"How do you configure whether or not to use the MTP?" -- there is NO
in-model switch, and the card must not imply one. mlx-lm's generate() has
no MTP hook, so the head is a separate CODE PATH, not a config flag:
stock load/generate ignores the sidecar entirely (proven: resident
44.96 GiB with the file present), and drafting happens only when someone
runs `vqlab mtp-generate`. Configuration == which command you run.

"Will it still work with exo?" -- yes, and better than expected: exo
builds its download allow-list from the weight_map in index.json
(download/huggingface_utils.py:100, get_allow_patterns), and its bare
"*.safetensors" fallback fires only for repos with NO index. Ours have
one, and the sidecar is not in it. So exo does not even DOWNLOAD the
sidecar: no wasted bandwidth, no resident cost, model behaves exactly as
today. Exo users simply get no drafting. (Note exo's vision path globs a
broader "*.safetensors" -- vision.py:361,424 -- but that is the vision
component loader, not the text weight download.)

"Mention MTP might not work on 64GB without raising the wired limit" --
agreed, goes in the card.

## 2026-08-30 (cont) — the usability wall: DO NOT SHIP the sidecar alone

Noah: "but that still makes someone running it difficult." Correct, and it
is the decisive objection. Stating it plainly:

  There is NO configuration-only path to a working MTP feature.
  mlx-lm's generate() has no MTP hook, so no arrangement of files --
  sidecar in-repo, sidecar in its own repo, standalone runner script --
  makes a stock user's decoding faster. Every arrangement still ends in
  "download a thing, then run our script instead of your normal tool."

Shipping the sidecar + a script would be the SAME mistake as the pulled
8-bit graft, one layer up: a feature that technically exists and that
essentially nobody can practically use. The correct read of "make it an
option" is an option people can actually take.

Real usability requires code where users already are. Two venues:

  1. MTPLX (github.com/youssofal/MTPLX, Apache-2.0, "Powered by MTPLX"
     attribution clause). Purpose-built for MLX MTP spec-decode, has a
     user-facing CLI, and is organized as ONE PATCH MODULE PER FAMILY:
     deepseek_mtp_patch, glm_mtp_patch, hy_v3, mimo, nemotron_h,
     qwen3_5, step3p5. A qwen4_exp_mtp_patch.py is a known-shape
     contribution, and we now have every junction characterized and
     measured, so it is bounded work rather than research.
  2. Upstream mlx-lm qwen4_exp: keep mtp.* through sanitize, expose the
     head, and hook generation. Broader reach, slower, not our timing.

RECOMMENDATION: do not upload the sidecar as a standalone-script
curiosity. Either write the MTPLX qwen4_exp patch (then the card can say
"install MTPLX, point it at this model" -- a real instruction), or park
the head and record the finding. The research stands either way: the head
works, 1.56-1.80x median 1.65x, 2.12 GiB, optionality proven.

Sidecars are staged in the four local artifact dirs; NOTHING uploaded.
If we park, they should come back out of those dirs to avoid an
accidental future upload.

## 2026-08-30 (cont) — AMENDS the "DO NOT SHIP" entry above

Two corrections to my own reasoning, both prompted by Noah.

1. I OVERSTATED the wall. MTPLX is NOT necessary and never was: every
   number today came from our own `vqlab mtp-generate`, which implements
   the draft/verify loop directly against mlx-lm. MTPLX was a reference
   for the wiring only; after reading it I never ran it. And VQLab is a
   real package (pyproject [project.scripts] vqlab = vqlab.cli:main,
   Apache-2.0, deps just mlx/mlx-lm/numpy/safetensors), so "pip install
   vqlab" is the SAME size of ask as "pip install mtplx". I was treating
   "needs our tool" as disqualifying while proposing a different tool with
   identical friction. This is also unlike the pulled 8-bit graft, which
   was never executed by anyone; this is built, measured across six
   prompt types, and smoke-tested from the sidecar on disk.

2. But the entry-point limit is REAL and structural, not a packaging
   oversight. Stock mlx-lm cannot run this, for two independent reasons:

   a. The MTP head cannot pose as a draft_model. mlx-lm's speculative
      path calls _step(draft_model, draft_cache, y) (generate.py:593) --
      the draft model gets token ids and its own cache, never the trunk's
      hidden state, which is exactly what the head consumes.
   b. Stock mlx-lm cannot do speculative decoding on this architecture AT
      ALL, with any draft model: the path raises ValueError unless
      can_trim_prompt_cache passes, and for Flash-Next it is False --
      measured, _LayerCache.is_trimmable() == False for the 36
      linear-attention layers (only _AttnCache is trimmable).

   Our loop works precisely because it rolls recurrent caches back by
   REFERENCE SNAPSHOT rather than trimming, which is available only
   because qwen4_exp reassigns cache slots instead of mutating them. That
   is a real contribution, not just glue: it is the only working
   speculative decoding for this hybrid architecture in MLX.

So the answer to "is there no way to run it except MTPLX or vqlab": yes,
that is the complete list, and (b) means even a generic draft model is
refused. Shipping with "pip install vqlab; vqlab mtp-generate" is an
honest, usable story; MTPLX/upstream would only broaden reach. The card
must state plainly that drafting is NOT available through stock mlx-lm,
exo, or LM Studio.

## 2026-08-30 — COMPACTION ANCHOR #3 (MTP arc)

STATE: MTP is solved, built, measured, and staged. NOTHING UPLOADED.

Solved: root cause was the zero-centered RMSNorm convention (qwen3_5
stores delta and mlx-lm's sanitize adds +1.0; qwen4_exp's own RMSNorm
applies y = norm(x) * (1 + weight) itself). Acceptance 0.0 -> working.

Measured (2.1bpw unless noted, real 6-bit head, sidecar off disk):
  head 2.12 GiB resident (confirms the 2.13 projection exactly)
  speedup 1.56-1.80x, MEDIAN 1.65x across six prompt types
  acceptance 0.578-0.812, median 0.679
  3.2bpw rung: 1.56x
  optionality PROVEN: sidecar in the artifact dir, resident still
    44.96 GiB (byte-identical), stock generate() unaffected

Built (VQLab e6eb09a): src/vqlab/mtp_head.py (wiring in one place),
`vqlab mtp-pack`, `vqlab mtp-generate`. Sidecar mtp-head-q6.safetensors,
2.14 GiB, 71 tensors, staged in ALL FOUR local artifact dirs. If we park,
REMOVE THEM so they can't be swept into a future upload.

ENTRY-POINT LIMIT (structural, measured, not a packaging problem):
  - mlx-lm's speculative path REQUIRES can_trim_prompt_cache; for this
    model it is False (_LayerCache.is_trimmable() False on the 36 linear
    layers). Stock mlx-lm refuses speculative decoding on this arch with
    ANY draft model.
  - The head needs the trunk hidden state; mlx-lm's draft_model interface
    (generate.py:593) passes only token ids + its own cache.
  So: VQLab today, or MTPLX/upstream later. That is the complete list.

MTPLX assessed (vetted, not asserted): v2.10.1, PyPI + Homebrew tap +
native Mac app + mtplx.com, CI, Apache-2.0, single author Youssof
Altoukhi. Purpose-built for MTP on Apple Silicon. Runs an OpenAI-
compatible server (/v1/chat/completions, /v1/models) and its CLI takes
--model as a path -> real integration story. Uses exact rejection
sampling with residual correction (Leviathan/Chen), so it is
distribution-preserving at temperature -- technically BETTER than our
greedy-only loop. Claims 1.6x M4 mini / 2.24x M5 Max (their numbers,
unverified by us; consistent with our 1.65x). Risks: bus factor 1; a PR
lands on his roadmap not ours; NOTICE adds an attribution requirement
beyond stock Apache-2.0 -- "Powered by MTPLX" must appear IN-PRODUCT, a
README/marketing page explicitly does not satisfy it.

DEMAND IS REAL: HF discussion #1 "Is there MTP for this model?" by
miabchdave, open, on TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw (the
only discussion across all four repos). Notably the LARGEST rung, where
2.12 GiB costs ~2 mnats (the 94.1->111.6 stretch is ~0.9 mnats/GiB) vs
the steep curve at 45 GiB (~11 mnats/GiB and steepening below).

AWAITING NOAH (nothing done unilaterally):
  1. Post a reply to discussion #1? Draft written in-conversation; its
     key ask is "what are you running it under?", which decides whether
     the MTPLX patch is worth it.
  2. Upload the sidecar to the 5.5bpw repo only (recommended) vs all
     four vs a separate head repo.
  3. MTPLX: open an issue asking whether qwen4_exp support is wanted,
     BEFORE writing a patch.
  4. Noah's "MTP fast variant" idea (smaller trunk + head at constant
     footprint) -- sound, but belongs on a 128GB rung, not the 45.

GLM: still parked at 12/19 shards, resumable by shard-skip. Untouched
all day.

CORRECTIONS I MADE TODAY (kept, so they are not re-derived): the VQ
kernel hypothesis was REFUTED by Noah's question (seq=2 is cheaper than
seq=1, identically on both artifacts); the q6sim/q8sim grafts are
dequantized bf16 and never cost 2.13 GiB; my "DO NOT SHIP / MTPLX is
necessary" call was overstated and is amended above; and two timing
results were artifacts (memory pressure with Safari open, and cold Metal
kernels -- 1.16x cold vs 1.62x warm on identical weights).

--------------------------------------------------------------------
2026-08-30 (later): MTP PARKED -- Noah's decision.

Rationale (Noah's, and it holds): no runtime can drive the head today
-- the limit is structural in mlx-lm's speculative entry points
(_draft_generate never passes trunk hidden state; the path refuses
non-trimmable caches, and 36 linear-attention layers are not trimmable)
-- and anyone with the wherewithal to build a working MTP runtime env
can pull Qwen's own head and wire it themselves. Shipping an inert
2.14 GiB shard serves no one. VQLab-as-serving-endpoint was considered
and rejected: a research tool standing in for mlx-lm's whole serving
surface (sampling, streaming, templates, concurrency) is a tall order
for zero current users. An mlx-lm fork was likewise rejected as an
install path (runtime replacement + permanent rebase burden).

Actions taken:
  - Removed staged mtp-head-q6.safetensors from ALL FOUR artifact dirs
    on /Volumes/Thunderbay SSD/Exo Models (so no future upload can
    sweep them). KEPT: top-level mtp-head-q6.safetensors (2.14 GiB) and
    parked_mtp_graft_bf16.safetensors (4.86 GiB) -- the head is
    rebuildable in minutes via `vqlab mtp-pack` if runtimes catch up.
  - Nothing uploaded to HF. No MTPLX issue/PR. No mlx-lm fork.

Still pending Noah's approval: a reply to HF discussion #1 explaining
the situation plainly (head exists and measures 1.5-1.8x; no runtime
support anywhere yet; VQ release quality is independent of MTP). Draft
is in-conversation. Revisit trigger: mlx-lm grows MTP-draft support
upstream, or a credible runtime asks for the sidecar.

The code stays: vqlab mtp-{probe,pack,generate} are merged in VQLab
(through e6eb09a) and are the reproducible evidence for every number
above.

2026-08-30: Replied to HF discussion #1 (5.5bpw repo) with Noah's
approved text (comment 6a94b118334fc0cd5c373a4b): head works, 1.5-1.8x
measured, no runtime support anywhere; rungs were sized to machine
calibers (64/96/128 GB) without the ~2 GB q6 head; will revisit if
mlx-lm adds support. MTP arc now fully closed pending that trigger.

## 2026-09-02 — Two suspected MTP obstacles measured; both null

exo stage-0 A/B (M3, 2.1bpw, sequential engine, warm, 378-token greedy)
landed at 1.28x vs the 1.56-1.80x banked at 128 tokens, and the gap drew
two theories. Both died under measurement:

1. "exo overhead": direct-runtime A/B on the same box, same lengths —
   baseline 17.2 vs exo 16.9, MTP 22.1 vs exo 22.8, acceptance equal
   (0.772). exo adds ~nothing; the loop just IS ~1.3x at this length on
   this text.
2. "head-cache decay, window it": a 2048-token run with per-128-token
   segment stats shows rate tracking SEGMENT ACCEPTANCE, not length —
   segments 1408-1920 (head cache at max) ran 25.9-26.5 tok/s at
   acceptance 1.0, FASTER than the opening. The 128->378 "decay"
   (1.38x -> 1.28x) was the text's acceptance profile. Windowing stays
   unbuilt until someone measures a real decay at 8k+ (MTPLX's collapse
   was at 34k). Head KV is ~1KB/token — memory is a non-issue at agent
   lengths.

Also re-confirmed from the shipped model.py's own commentary: the
packed-d4 SIMD twin was measured 0.94-1.07x and deliberately not
shipped; the 2.1bpw already dispatches its best-known route. Absolute
decode (17.4 vs stock-3bit's 27.0 on the same box/prompt) is the kernel
research frontier, not a missed flag — and stock 3bit is 30 GiB larger.

Context numbers for the eventual card: exo + MTP serves 2.1bpw at
22.8 tok/s where stock affine 3bit (75G, no MTP) serves 27.0 — within
17% at 60% of the size.

## 2026-09-02 — the decode gap is NOT in the VQ kernels (M3 Ultra)

Opened to close the 17.4 vs 27.0 tok/s gap on the 2.1bpw against stock
affine 3-bit, on the premise (M1_KERNEL_PLAN.md, "Decode") that VQ reads
FEWER bytes per token so the roofline says it should WIN. It measures the
other way, and the premise is true only of the expert tensors, which turn
out to be a tenth of the per-token read.

MEASUREMENT SETUP. One driver for both artifacts (scratchpad/bench.py):
same box, same prompt, chat template applied, `tok.eos_token_ids = set()`
so BOTH sides generate exactly 377 timed tokens, first-token-to-last
timing, two 24-token warmups before any timed run (the Metal-JIT trap from
the 08-30 entry). Never both models resident at once. A CAUTION for anyone
repeating this: with the RAW prompt the stock 3-bit repo emits EOS as its
FIRST token and the run silently measures nothing (0 tokens), while the VQ
one generates normally -- clearing the eos set is what makes the two
comparable at all.

Attribution is by STUB ABLATION: replace a module class's `__call__` with a
shape-correct `mx.zeros` return, so its work leaves the step while the token
count is held fixed. The generated text changes; the timing share is real.

    ms/token            stock 3-bit    VQ 2.1bpw    delta
    TOTAL                    37.05        56.76    +19.71
      MoE experts             3.09         9.84     +6.75   (34% of gap)
      n-gram embedding        2.11         3.54     +1.43   ( 7% of gap)
      everything else        30.10        41.14    +11.04   (56% of gap)
    tok/s                    26.99        17.62

Both baselines reproduce the reported figures exactly (26.99 / 27.00 on
repeat; 17.62 / 17.59, and 17.43 under the raw-prompt driver).

SO THE MAJORITY OF THE GAP IS IN CODE THAT CONTAINS NO VQ AT ALL. "Everything
else" is the same architecture on both sides -- linear attention, hyper-
connections, dense projections, lm_head. The only difference is what the
build recipe quantized them to: this artifact keeps every non-expert tensor
at 8-bit / group 64, the stock comparator is 3-bit / group 32. Per-token
weight bytes, straight from the safetensors headers (scratchpad/bytes.py):

                       VQ 2.1bpw   stock 3-bit
    dense (per token)    5.878 GiB     2.111 GiB    2.78x MORE
    experts (10/512)     0.646 GiB     0.961 GiB    0.67x
    TOTAL per token      6.525 GiB     3.073 GiB    2.12x MORE

The plan's roofline argument holds for the expert tensors and is swamped by
everything around them. 3.77 GiB of extra dense traffic against 11.04 ms of
extra time is 366 GB/s -- a plausible achieved streaming rate on this box,
so the "everything else" gap is bandwidth on weights the VQ work never
touches. NOTE the isolated-matvec bench does NOT show this (8-bit and 3-bit
measure the same ~30 us at N=1 on every projection shape except lm_head,
2.10x): a single weight benched in a loop is cache-resident, so that bench
reports a launch floor, not the streaming cost. It misled me for an hour;
the ablation on the resident model is the honest instrument.

    => The single largest available decode win on this rung is a BUILD
       RECIPE change -- quantize the non-expert tensors lower -- not a
       kernel change. Not actioned here: it is a new artifact and a
       quality re-referee, not a kernel edit.

WHICH KERNELS THE 2.1bpw ACTUALLY DISPATCHES. Worth stating because I got it
wrong first and benched an unrepresentative layer for an hour. The artifact
is MIXED GEOMETRY (config.json vq_modules, 144 expert modules):

    layers 0-1    d2 / K256  unpacked uint8      6 modules   ( 4%)
    layers 2-47   d8 / K16384 14-bit packed    138 modules   (96%)
                    gate/up NGRP=40 -> vq_fused_packed14_d8_simd  (92)
                    down    NGRP=10 -> vq_fused_packed14_d8       (46)

So the hot path is the packed-d8 simdgroup kernel from the 09-01 entry,
already shipped. Layer 1 is the d2 outlier -- do not bench it and call it
the model.

WHERE THE d8 KERNEL'S TIME GOES (real artifact codes, 48 dispatches per eval
so the ~265 us eval round-trip is amortised; run-to-run variance on this
harness is ~4%, which bounds every claim below):

    gate/up  N=10   66.7 us   (4.1 MB of codes+scales ->  61 GB/s)
    down     N=10   49.1 us   (4.4 MB                  ->  97 GB/s)
    projected expert cost/token, 46 layers x 3:  8.33 ms
      (the stub ablation says 9.84 ms/token; the balance is the 144 Python
       calls and their broadcast/reshape, so the microbench IS predictive
       here at decode N -- unlike above N=20, per the 09-01 note)

Stock gather_qmm on the same shape reaches ~290 GB/s in situ. Ours is at
7-12% of the machine's 819 GB/s. Two candidate causes were tested and BOTH
ARE REFUTED:

  1. CODEBOOK RESIDENCY. Held the code stream and the arithmetic fixed and
     shrank only the codebook's DISTINCT hot set by tiling a K-entry table
     across the same 256 KiB address range (scratchpad/split.py):

         hot set    256 KiB   64 KiB   16 KiB   4 KiB
         gate/up      69.2     66.6     64.5     63.6  us
         down         49.5     47.8     49.2     48.8  us

     8% from a 64x smaller hot set. L2 residency holds exactly as the M1f
     entry claimed; the gather footprint is not what costs.

  2. CODEBOOK LOAD WIDTH. The two adjacent half4 loads per code (cb4[2c],
     cb4[2c+1]) were folded into one 16-byte uint4 load with as_type --
     bit-identical, strictly fewer loads. Measured 8.11 ms/token against
     8.41 ms for the SAME code on a second run of the unmodified kernel:
     the effect is inside run-to-run variance. NOT BANKED, reverted.

What is left, and it is the concrete next lever: the kernel issues roughly
four device loads per code (one or two packed words through the generic
VQ_CODE macro, plus the two codebook loads) and sustains ~123 G loads/s,
i.e. about one per core per cycle -- it is LOAD-ISSUE bound. The fix with
precedent is the one _SRC_DENSE_PACKED_D2 already uses: buffer a lane's
packed words in REGISTERS once and extract its codes from registers,
which measured 82 vs 169 us there. It does not port mechanically to d8/14-bit
because a lane's 8 codes span 112 bits at a runtime bit offset, and a
runtime index into a register array spills; the offset is compile-time only
after specialising on (j0 & 31)/SPG parity (sh0 is 0 or 16 for BITS=14,
SPG=8). Bounded but fiddly, and NOT ATTEMPTED here -- a subtle bit-extraction
kernel is not something to rush against a bit-identity gate.

SMALL-N SCALING (asked for by the GLM-5.3 MTP arc, whose T=2 trunk costs
1.50x a T=1 and which suspected the VQ switch kernels of paying full kernel
latency per (token, expert) pair). Real artifact codes, both geometries,
against gather_qmm at 3-bit on the same shapes (scratchpad/nscale.py):

    Flash-Next d8/K16384/14-bit, top_k=10, N=10 -> N=20 (T=1 -> T=2)
        gate  68.3 -> 110.5 us  = 1.62x     gather_qmm 1.36x
        down  58.4 ->  79.0 us  = 1.35x     gather_qmm 1.48x

    GLM-5.3 d4 packed, top_k=8, N=8 -> N=16 (T=1 -> T=2)
        K512  down  106.2 -> 173.7 = 1.64x  gather_qmm 1.71x
        K512  gate  103.6 -> 173.5 = 1.67x  gather_qmm 1.47x
        K2048 down  115.7 -> 200.7 = 1.73x  gather_qmm 1.75x
        K2048 gate  128.2 -> 213.7 = 1.67x  gather_qmm 1.47x

    => THE HYPOTHESIS IS REFUTED. Doubling the pairs costs 1.35-1.73x, not
       2x, on every shape, and mlx's OWN gather_qmm scales 1.36-1.75x on the
       same shapes -- statistically indistinguishable. The VQ kernels are not
       worse than native at multi-token scaling, so a T=2 trunk costing 1.50x
       is what this MoE shape does under ANY quantization and is not a VQ
       defect. On GLM that leaves the DSA indexer L>1 path as the suspect
       still standing.
       Both kernel families are strongly LATENCY-bound at small N: per-pair
       cost falls 8.8x (VQ) and 20x (gather_qmm) from N=1 to N=80, so VQ's
       disadvantage GROWS with N (1.7x at N=1, 2.8x at N=10, 3.9x at N=80) --
       it saturates earlier. That is the same load-issue ceiling as above.

TWO KERNEL CHANGES MADE, both BIT-IDENTICAL on live weights, NEITHER
measurable end-to-end on this rung:

  a. _SRC_FUSED_D2_U32. The d=2 analogue of the d4 U8-VIEW dispatch: an
     unpacked uint8 code row is byte-for-byte a little-endian uint32 stream,
     so codes are reinterpreted with mx.view (zero copy) and the kernel reads
     ONE word per four codes instead of four uchar loads. The existing loop
     already steps four subvectors per q, so a q is exactly one word.
     Measured on real 2.1bpw layer-1 codes:

         tensor      N   NGRP   uchar us   u32 us   speedup
         gate_proj  10     40       57.8     39.3     1.47x
         gate_proj  20     40       78.3     51.4     1.52x
         up_proj    10     40       69.5     37.4     1.86x
         up_proj    20     40       69.9     48.4     1.44x
         down_proj  10     10       57.6     39.2     1.47x
         down_proj  20     10       72.1     47.8     1.51x

     Bit-identical on every row (mx.array_equal on real codes, and on the
     LIVE 45 GiB artifact for both d2 geometries). Worth 1.47-1.86x on the
     d2 path -- but this rung has only 6 d2 modules of 144, so end-to-end it
     is ~0.2 ms/token, inside noise. It is the WHOLE expert path on the
     gemma d2 rungs, which is why it is kept.

  b. VQPLEEmbedding: `_unpack` IS THE IDENTITY FUNCTION AT BITS=8 and was
     being executed anyway. At K<=256 a "packed" row is one byte per code, so
     bit0 = arange(nsub)*8 gives window byte i, shift 0, mask 0xFF -- the
     three-byte-window extraction reproduces its own input. It cost a
     concatenate, three mx.take gathers and six elementwise ops per shard per
     token, on the 16 n-gram shards a token touches (~160 tiny dispatches).
     Skipped now. The group scales are also applied by broadcast over a
     [.., ngrp, G] view instead of materialising mx.repeat's full-width copy
     -- same products, same rounding, one fewer temporary. Bit-identical on a
     live shard; 480 -> 388 us per shard-call in isolation.

END-TO-END, both changes bound into the loaded 45 GiB artifact (the bundle
embeds its own vq_switch, so `custom_model._fused` and
`VQPLEEmbedding.__call__` are rebound to the repo's; all FOUR live expert
geometries gated bit-identical first, and the generated text is unchanged):

    before (bundle)      17.41 tok/s   57.43 ms/tok
    + d2 uint32 kernel   17.19 tok/s
    + PLE unpack/repeat  17.19 tok/s

i.e. NO END-TO-END WIN on this rung -- the changes are real and measured in
isolation, and they land on 4% of the modules and on a path worth 1.4 ms.
Recorded as such rather than dressed up.

ALSO MEASURED AND DELIBERATELY NOT SHIPPED: _SRC_FUSED_D2_SIMD, the
simdgroup-per-row d2 twin (source kept in vq_switch.py with the numbers, so
the negative is not re-derived). 0.80-1.09x the thread-per-row kernel on real
codes. It does not win because the layout exists to hide DEPENDENT DEVICE
round-trips, and at d2/K256 the codebook is 1 KB in THREADGROUP memory --
there is no device round-trip to hide, so the two barriers per block, the
32-step shuffle and 31 idle lanes at the write are pure overhead. Not
occupancy either: it launches 204,800 threads against 6,400 for the same
time. The d2 bottleneck was the WIDTH OF THE CODE LOAD, which (a) fixes.

WHAT I DID NOT GET TO
  - the register-buffered packed-code fetch for d8/14-bit (the live lever)
  - the NGRP=10 down_proj layout (46 modules run thread-per-row because the
    NGRP>=32 gate excludes them; a multi-row-per-simdgroup variant that packs
    3 rows of 10 groups into one simdgroup would fill 30/32 lanes and is
    bit-exact by construction, since each row keeps its own ascending-group
    fma chain). Not written.
  - the build-recipe change that owns 56% of the gap: non-expert tensors at
    8-bit. Needs a new artifact and a quality re-referee.
  - re-bundling. Kernel changes here are in-repo only; artifacts embed their
    own kernel source and were touched READ-ONLY throughout.

## 2026-09-02 — d8 REGISTER-BUFFERING ARC: the lever failed, the threadgroup shape paid

Follow-up to the packed-d8 profiling arc. Target was the register-buffered
packed-d8 kernel; what shipped is a threadgroup-shape fix found while
isolating why the register buffer lost. Everything below is measured on REAL
Flash-Next 2.1bpw expert tensors read straight out of the artifact
(`model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.{codes,codebook,
vq_scales}`, expert axis sliced to E=64 — no model was loaded; another agent
had the box).

NOTE ON THE INHERITED RECORD: the profiling arc's ledger entry is NOT in this
tree (the last entry here is 2026-08-30, MTP). Its diagnosis was carried in
by hand and is reproduced faithfully in the kernel comments; if that entry
lives on another branch it should be merged so this one has its antecedent.

### Method (the box was contended, and that changed the methodology)

Another agent ran builds and model loads throughout. One-shot numbers were
worthless, so: every variant JIT-warmed and run once before timing; variants
timed INTERLEAVED inside each rep (A,B,A,B,...) so a contention burst hits
both arms; 50 back-to-back dispatches per timed window; 13-21 reps; headline
is MIN over reps (contention can only make a window slower) with the median
alongside, and a claimed win must hold on BOTH.

TWO MEASUREMENT TRAPS were hit and are worth recording, because each produced
a confident wrong number first:
  * building 50 lazy dispatches and calling `mx.eval` on only the LAST array
    leaves the other 49 dead and never executed. That measures submit
    overhead: every shape and every N pinned at ~14 us and reported 2.5 TB/s.
    `mx.eval(list)` is required.
  * serialising with `mx.eval` per dispatch measures ~380 us of sync, not the
    kernel.
With the harness fixed, gate_proj at E=64/N=10 reproduced the prior arc's
68.4 us/dispatch (measured 65.5), which is what confirmed the harness.

### 1. REGISTER BUFFERING: BUILT, BIT-IDENTICAL, 1.66x SLOWER. Negative banked.

Built as `_SRC_FUSED_PACKED_D8_SIMD_RB`. Bit-identical to the shipped kernel
on real gate_proj tensors and on synthetic codes at every geometry tested,
verified before any timing.

Real gate_proj, N=10, min/med us:  regbuf 106.3/119.4 vs VQ_CODE 64.0/68.4.
At N=20: 205.3/236.8 vs 105.9/113.2. Arms do not overlap.

TWO forms were built, because the first failure had an obvious suspect:
  (a) compile-time specialisation on the phase p = g & (NPH-1), an if/else
      chain of NPH bodies with every word index and shift a literal — the
      form the profiling arc prescribed. 1.9x slower.
  (b) branchless: load a constexpr NW-word window from a runtime word offset
      (pointer arithmetic, no register indexing), then FUNNEL-SHIFT it down
      by the runtime bit offset so every extraction index is a literal.
      Built because (a)'s phase varies WITHIN a simdgroup — lane L owns group
      b*32+L, so consecutive lanes have consecutive phases and all NPH bodies
      execute serially. 1.75x slower — i.e. essentially identical to (a).

So divergence is NOT the cause, and that is the finding. The isolating
experiment, base kernel only, real gate_proj N=10:

    base (runtime SPG, rolled loop)              60.3 us
    base + compile-time SPG, still rolled        64.2 us   (free)
    base + compile-time SPG + unroll(full)       94.2 us   (1.56x SLOWER)
    register-buffered (needs the unroll)        106.3 us

WHY IT CANNOT BE RESCUED AT d=8. The unroll is not incidental to register
buffering, it is a precondition: the register array must be indexed by
compile-time values or it spills to memory, which is strictly worse than the
device reads it replaces. So any register-buffered d8 kernel pays the ~1.5x
unroll penalty, and the prize it is playing for is bounded above by ~25% —
the profiling arc's own count is ~4 device loads per code, of which 2 are
codebook and only 1-2 are the code read. Paying 1.5x for at most 1.33x is a
losing trade, and no amount of bit-extraction cleverness changes the ratio.

This is the structural difference from `_SRC_DENSE_PACKED_D2` (82 vs 169 us),
where the precedent came from: at d=2/G=64 a lane's scale group is EXACTLY
one 32-code pack block, so its 32 codes amortise the unroll over 4x more
work AND start at bit 0 with no phase at all. At d=8 a lane owns only
SPG = G/8 = 8 codes. The precedent does not transfer, and now we know the
reason is the work-per-unrolled-body ratio, not the bit arithmetic.

KEPT IN TREE, correct and tested, dispatched only under `VQ_D8_REGBUF=1`, so
the negative is reproducible without a rebuild.

### 2. THE WIN: 8 rows/threadgroup, not 32. 1.21-1.28x, bit-identical.

Found while sweeping occupancy to explain the unroll penalty. The packed-d8
simd kernel took `_EXPERT_ROWS_TG = 32` from the DENSE kernels, where that
value was swept (2026-08-19) and where it is correct. It was never swept
here. Re-swept, real tensors, min us/dispatch:

  gate L2  N=10   r2 69.2  r4 51.9  r8 49.9  r16 54.7  r32 61.9
  gate L2  N=20   r2 111.5 r4 81.7  r8 82.2  r16 93.0  r32 103.4
  up   L2  N=10   r2 66.3  r4 49.0  r8 52.1  r16 56.8  r32 60.9
  up   L2  N=20   r2 109.2 r4 78.9  r8 80.7  r16 89.1  r32 106.5

A clear INTERIOR optimum: r2 starves the machine of threadgroups, r32 starves
each thread of registers (the same pressure the unroll experiment exposed),
4 and 8 are indistinguishable. Held on a second and third layer (L20, L35)
and on a 21-rep re-run. 8 chosen over 4 for x-tile amortisation at equal
measured cost. Through the REAL dispatcher, min/med us, r32 -> r8:

  gate N=1    24.3/28.2  -> 22.9/28.0    1.06x / 1.01x   (dispatch-bound)
  gate N=5    42.8/45.2  -> 36.8/40.5    1.16x / 1.12x
  gate N=10   59.2/72.2  -> 49.1/55.3    1.21x / 1.31x
  gate N=20   99.3/103.2 -> 77.8/80.4    1.28x / 1.28x
  up   N=5    39.1/41.8  -> 33.5/35.7    1.17x / 1.17x
  up   N=10   59.7/61.7  -> 48.0/49.2    1.24x / 1.25x
  up   N=20   97.1/101.2 -> 77.3/78.7    1.26x / 1.29x

Code-stream bandwidth, N=10: 61 -> 73 GB/s of the M3 Ultra's 819. N=20:
74 -> 93. N=1 is flat across the entire sweep (23.4-24.5 us), so nothing
regresses at seq=1 decode. The kernel SOURCE is untouched — only the
threadgroup shape — so bit-identity is by construction, and is asserted
(r4/r8/r32 against r32 on the 397B gate/up geometry) rather than argued. The
DENSE constant is deliberately left at 32; only the packed-d8 expert path
moves, via its own `_EXPERT_ROWS_TG_D8_PACKED`.

### 3. STRETCH TARGET REASSESSED: the NGRP=10 down_proj is not the laggard.

The prior arc flagged the NGRP=10 down_proj shape (46 modules) as untouched,
because the NGRP>=32 gate declines it from the simd layout. Measured rather
than assumed, real down_proj [E,2560,42] / IN=640:

  tgx sweep of the thread-per-row kernel it actually dispatches (N=10):
    tg32 56.3  tg64 51.3  tg128 46.7  tg256 45.6  tg512 44.9  tg1024 48.8
  The shipped tgx=256 is already at the optimum; tg512's 1.5% is inside
  variance. There is no free threadgroup-shape win here.

  And per code byte it is the FASTEST shape measured, not the slowest:
    down_proj N=10  44.9 us   95.8 GB/s codes      (gate at r8: 71.8)
    down_proj N=20  70.9 us  121.3 GB/s codes      (gate at r8: 88.3)
  i.e. ABOVE the 61-97 GB/s band the profiling arc reported. Its short IN
  (640) means one thread walks only 80 codes, so the thread-per-row latency
  chain that motivated the simd layout is 4x shorter here — the layout it is
  declined from is the layout it does not need. A multi-row-per-simdgroup
  kernel for NGRP<32 was NOT built: the measurement says the headroom is not
  there. Reopen only if a shape with NGRP<32 AND long IN appears.

### What remains

  * END-TO-END TOK/S IS NOT MEASURED. Explicitly out of scope this session
    (no model may be loaded — shared box). The 1.21-1.28x is per-dispatch on
    isolated real shapes; the prior arc's own record shows the microbench is
    NOT predictive above decode-sized N (the simd layout's 1.35x microbench
    vs -13.6% end-to-end at seq=3), so the rows=8 change MUST be validated
    end-to-end on the 397B and Flash-Next rungs before any published claim.
    Note the two are not the same kind of change — rows/TG is bit-identical
    and shape-only where the simd layout was a different reduction — but the
    gate that caught that regression (`_EXPERT_SIMD_MAX_N = 20`) was set by
    end-to-end measurement and its interaction with rows=8 is unmeasured.
  * The unpacked d8 simd and d4 devcb simd kernels still use
    `_EXPERT_ROWS_TG = 32` and were NOT swept — the 2.1bpw artifact is
    packed, so there were no real tensors to sweep them on. Same inheritance,
    same suspicion, no measurement. Cheap to do on an unpacked artifact.
  * The profiling arc's ledger entry should be merged into this file.

## 2026-09-02 — rows=8 validated END-TO-END (+3.6% on the dense-heavy rung)

A-B-A on the shipped 2.1bpw vs an APFS clone re-bundled with the merged
runtime (3 warmed 377-token runs per side, shipped run twice bracketing):
shipped 17.39/17.40/17.41 then 16.99/17.09/17.25; rows8 17.91/17.94/17.95.
Medians 17.32 -> 17.94 tok/s (+3.6%), rows8 strictly above both shipped
brackets. Matches the stub-ablation arithmetic (experts are 9.8 of
58 ms/token here; a ~1.25x expert kernel predicts ~+4%) — the microbench
did NOT invert end-to-end this time. Expected to matter far more on the
397B, where the expert share of each token dominates; unmeasured there.

Bycatch: the first A/B attempt caught a bundle-template regression — the
Sep-1 dual-runtime config coercion fired for mlx_lm arches and broke
qwen4_exp re-bundles under mlx-lm 0.31.9 (fixed same day, hasattr
TextConfig guard; published artifacts unaffected — they carry their
original bundles).

## 2026-09-02 — rows=8 on the 397B over TCP tensor: NULL (interconnect-dominated)

Controlled A/B on the cluster (fresh instance per arm, warm discard, six
300-token runs each, same prompt): baseline median 17.06 tok/s
(16.93-17.20), rows8 median 17.13 (16.48-17.75). +0.4%, arms overlap —
no effect. Reading: TCP tensor pays per-layer all-reduces every token;
at ~58 ms/token that sync term dominates and hides per-dispatch expert
kernel gains entirely (consistent with tensor ~= pipeline speeds on this
pair). The single-box +3.6% stands; the cluster lever is the
interconnect (RDMA/TB5), not the kernel. rows8 bundle left deployed on
both boxes' 3.1bpw (bit-identical outputs; .pre-rows8 backups on M3).
Methodology note: an OOM-contaminated first attempt produced a
declining 15.6->8.4 series — fresh-instance-per-arm was required for a
0.27 tok/s baseline spread.

## 2026-09-02 — the 397B 2x2: interconnect is the whole story (+19% RDMA)

TB5 cable connected; first jaccl/RDMA run on this pair. Six 300-token
runs per cell, fresh instance per arm, medians:

                 old bundle   rows=8
  TCP  tensor      17.06       17.13   (+0.4%, null)
  RDMA tensor      20.26       20.41   (+0.7%, sub-noise)

RDMA buys +19% and collapses variance (spread 0.33 vs up to 1.3 on
TCP). rows=8 is sub-noise on the 397B in BOTH interconnects — the
Flash-Next +3.6% does not transfer; this model's per-rank expert
shapes don't live where the rows sweep bites. rows=8 left deployed
(bit-identical, mildly positive trend, one bundle version fleetwide).

## 2026-09-02 — post-kernel retest: rows=8 and MTP multiply (24.4 tok/s)

Re-bundled the local 2.1bpw with rows=8 and re-ran the direct A/B (same
prompt/lengths as the morning's numbers): baseline 17.2-17.5 -> 18.0-18.2
(+3.9%, matches the A-B-A), MTP 22.1-22.8 -> 24.37/24.45 (+8%),
acceptance identical (0.772). MTP gains double the baseline's because
every speculative step is a T=2 forward and rows=8's win grows with N —
the two optimizations multiply rather than add. Net stack: 24.4 tok/s
with drafting, within 10% of stock affine 3-bit (27.0) at 60% of its
size; +40% over the pre-kernel baseline. GLM measured the same day:
rows=8 does NOT move its T=2 ratio (see glm53 ledger) — indexer remains
its sole gate. Published HF repos still carry the original bundles;
republish is a separate, gated decision.

## 2026-09-02 — matched-bytes 397B cluster comparison: the VQ tax measured at scale

Same boxes, same RDMA tensor topology, same 300-token protocol:

  spicyneuron 2.6bit affine (121 GiB)   29.8/29.8/30.2 clean runs
                                        (two colder runs 15.8/21.0 —
                                        first-touch; median-of-clean ~29.9)
  VQ 2.6bpw (122.3 GiB)                 20.3 median (19.86-20.39, tight)
  VQ 3.1bpw (154 GiB)                   20.4 median

Affine decodes ~1.45x faster at matched bytes — the third independent
measurement of the ~1.5x VQ decode tax (Flash-Next single-box: 27.0 vs
17.4; GLM T=2 scaling parity aside). Also: VQ 2.6 and 3.1 decode at the
SAME speed despite 32 GiB size difference — cluster decode is dispatch/
sync-bound, not weight-byte-bound, for the VQ builds. The quality side
is unchanged (our 2.6bpw beats their 2.6bit on both corpora, published);
cards get the honest split: they win tok/s, we win quality-per-byte,
and the tax is the kernel arc's target number.

## 2026-09-02 — DISPATCH-REDUCTION ARC: the launch floor is ~5 us, not 14. Hypothesis REFUTED, both cluster fingerprints dissolve.

Third kernel arc of the day, opened on two independent fingerprints that both
smelled like a per-DISPATCH latency floor rather than a bandwidth wall:

  1. the packed-d8 kernel sits at 61-97 GB/s of the M3 Ultra's 819 with the
     absolute floor unexplained (register-buffering refuted, occupancy banked);
  2. on the cluster, VQ 2.6bpw (122 GiB) and VQ 3.1bpw decode at the SAME speed
     (20.3 vs 20.4 tok/s, RDMA tensor) while affine 2.6bit at matched bytes runs
     1.45x faster — "speed not scaling with bytes".

A decode step dispatches ~144 VQ kernels (48 layers x gate/up/down). At the ~14
us/dispatch an earlier arc reported that is ~2 ms/token and worth chasing.

EVERY ARM BELOW IS NEGATIVE. Nothing shipped. Both fingerprints have simpler
explanations that are measured, not argued.

Instrument: `scripts/bench_dispatch_floor.py` (kept in tree so the negative is
reproducible). Real layer-20 2.1bpw tensors (d8/K16384 14-bit packed — the 96%
geometry), expert axis sliced to E=64, ~1 GiB resident. NO MODEL LOADED (both
boxes ran a validation queue). Arc-standard methodology: JIT-warm, interleaved
A,B,A,B arms, min over 15 reps with median alongside, `mx.eval` on the FULL
output list.

### 1. THE LAUNCH FLOOR IS ~5 us DEPENDENT / ~3.5 us INDEPENDENT

Trivial kernel (one store), chain of M, us per dispatch, min/median:

     grid      M   dependent   independent
        1    144    4.36/5.41    3.89/4.68
        1    288    4.75/5.21    3.31/4.18
     1024    288    5.01/5.24    3.72/4.28
    65536    288    6.63/8.07    3.33/4.11

The dependent number (~5 us) is the honest one — decode IS a dependency chain.
It is flat in grid size across 1 -> 65536 threads, i.e. it is launch cost.

WHERE THE "14 us" CAME FROM. It is the same quantity measured at small M, where
the ~140-200 us eval round-trip has not amortised: M=8 reports 20 us/launch,
M=48 reports 5.2, M=288 reports 2.6. The prior arc's 14 us was a submit-overhead
artifact of the dead-work trap it documented, not a per-launch cost. Correcting
it is what kills this arc's premise.

### 2. MULTIPLY IT OUT: the launch budget is ~1% of the token

    dispatches/token   at 5.0 us   share of 56.8 ms/tok   share of 19.7 ms gap
      144 (VQ only)      0.72 ms          1.3%                  3.7%
      304 (+ ~160 PLE)   1.52 ms          2.7%                  7.7%
      600 (everything)   3.00 ms          5.3%                 15.2%

Even at a deliberately generous 600 dispatches per token, pure launch overhead
cannot account for the 19.7 ms gap to stock affine 3-bit. THE DISPATCH
HYPOTHESIS IS REFUTED on this box.

Corroborating, from the same run: if decode were dispatch-bound, per-dispatch
cost would be flat in N. It is not — real gate_proj at M=48 runs 11.1 us at N=1
and 37.2 us at N=10, a 3.4x. The work is the cost.

WHERE THE FLOOR ACTUALLY IS. Decomposing one real gate_proj dispatch at N=10
(45.1 us dependent-chain cost at M=48):

    ~5 us   launch                      (arm A)
    ~2 us   kernel fixed cost above launch — the OUT sweep intercepts at 7.7 us
            at OUT=8 and rises linearly to 36.6 at OUT=640
    ~5 us   wave-drain / serialisation premium on real work: the dependent
            arm costs 1.27x the independent one at M=48 (49.2 vs 38.7 us),
            and only ~5 of that ~10 us delta is launch
    ~33 us  ACTUAL WORK — the load-issue bound the d8 arc already located

So ~16% of a dispatch is fixed cost and ~84% is the kernel doing its job badly.
The frontier is still the 61-97 GB/s load-issue ceiling, unchanged.

### 3. CANDIDATE (a) — FUSE gate_proj + up_proj: NULL AT ITS OWN UPPER BOUND

Chain of 46 real layer-shaped MLP steps (gate | up | SwiGLU | down), dependent
layer to layer. The fused arm concatenates codes and scales along OUT into one
OUT=1280 dispatch — exactly the geometry a real fusion would build at load
time — but reuses gate's codebook for BOTH halves, so it is charged nothing for
the two-codebook selection a real fusion needs. Upper bound, min ms (median):

       N    unfused        fused        saving
       1    2.815(2.920)   2.809(2.903)   +0.2% [med +0.6%]
       5    4.188(4.279)   4.249(4.359)   -1.4% [med -1.9%]
      10    5.794(5.890)   5.873(6.011)   -1.3% [med -2.1%]
      20    9.201(9.320)   9.269(9.358)   -0.7% [med -0.4%]

Zero to slightly negative at every N, including the decode point N=10.

WHY, AND THIS IS THE STRUCTURAL POINT: gate and up are INDEPENDENT of each
other, so they already overlap. The MLP's dependency chain is depth-2 per layer
(gate‖up, then down), not depth-3. Fusing them removes a LAUNCH but not a
SERIALISATION STEP, and per §2 a launch is worth 5 us against a 5.8 ms chain.
The 46 launches saved are worth 0.23 ms in principle and measure as nothing.
NOT BUILT — and note the real version would also have to solve the two
codebooks (gate and up codebooks are NOT identical in this artifact; verified
with mx.array_equal), i.e. a kernel change and a load-time repack, for a prize
its own upper bound says is zero.

### 4. CANDIDATE (b) — mx.compile / graph-level batching: NULL

Same 46 layer-steps, top_k=10, one token's expert path. Three arms over the
same work, min ms (median):

    raw _fused calls                 5.837 (5.939)
    VQSwitchLinear.__call__          6.903 (7.114)   +1.066 ms
    the same under mx.compile        6.803 (6.927)   +0.966 ms

mx.compile buys 1.4%, inside this harness's variance. MLX does NOT coalesce the
VQ call sequence and cannot: `mx.fast.metal_kernel` dispatches are opaque to
the compiler, so there is nothing for it to fuse across. Measured rather than
assumed, as the arc brief asked.

BYCATCH WORTH KEEPING: the module glue costs 1.07 ms/token (18% on top of the
kernels) — the broadcast / reshape / astype dance in `VQSwitchLinear.__call__`,
on 144 modules. That is a real, un-chased number, larger than the entire launch
budget for the VQ path, and it is NOT dispatch (compile does not touch it).
Nobody has tried to remove it. It is the one live lever this arc turned up.

### 5. CANDIDATE (c) — M1_KERNEL_PLAN.md dispatch fusion: nothing to test

The plan has no dispatch-fusion item beyond the per-module dispatch it already
describes; its only "launch" mention is the note that M=1 microbench numbers are
launch-overhead jitter — which §1 now quantifies.

### 6. THE CLUSTER FINGERPRINT DISSOLVES: it was never a latency floor

"VQ 2.6 and 3.1 decode at the SAME speed despite 32 GiB size difference" needs
no floor to explain it. An MoE decode step reads all the dense weights but only
top-8 of 512 experts, so TOTAL size is a poor predictor of decode cost.
Per-token reads, from safetensors headers only (`scripts/per_token_bytes.py`,
no model load):

                        dense    experts/token   PER-TOKEN    total
    VQ 2.6bpw          7.320 GiB   1.797 GiB      9.116 GiB  122.3 GiB
    VQ 3.1bpw          7.322 GiB   2.131 GiB      9.453 GiB  143.7 GiB
    affine 2.6bit      7.838 GiB   1.761 GiB      9.599 GiB  120.6 GiB

    VQ3.1 / VQ2.6 per-token = 1.037   (measured decode 20.4/20.3 = 1.005)
    affine / VQ2.6 per-token = 1.053  (measured decode 29.9/20.3 = 1.47)

The two VQ rungs differ by 21 GiB on disk but only 3.7% PER TOKEN, and decode
at 0.5% apart. Size-invariance is arithmetic, not a floor. Meanwhile affine
reads 5.3% MORE bytes per token and still runs 1.45x faster — so the ~1.5x VQ
tax is an ACHIEVED-BANDWIDTH deficit at matched traffic, which is exactly the
61-97 vs ~290 GB/s the d8 arc measured. Same wall, third sighting, no new
mechanism.

(Note: the header sum makes the 3.1bpw 143.7 GiB where the matched-bytes entry
above says 154; the ratio is unaffected, but the 154 should be re-sourced.)

### VERDICT AND WHAT IS LEFT

A third measured negative, and the cleanest of the three: the arc's premise was
built on a mis-measured constant, and correcting the constant removed the
question. The VQ decode tax is NOT per-dispatch. It is the packed-d8 kernel's
load-issue ceiling, which now has all three of this session's independent
measurements pointing at it and no competing explanation left standing.

  NOTHING SHIPPED — no kernel or dispatcher change survived, because none was
  built past the measurement that killed it. `scripts/bench_dispatch_floor.py`
  and `scripts/per_token_bytes.py` are committed as the reproducible evidence.
  No bit-identity gate was run, because no kernel changed; the fusion arm is a
  cost-only upper bound and is labelled as such in the script.

  PENDING GATE: none for this arc — there is no change to validate end-to-end.
  The rows=8 end-to-end gate from the prior arc is already closed (+3.6%).

  THE LIVE LEVERS, in the order the measurements rank them:
    1. the packed-d8 load-issue ceiling (~84% of a dispatch; unbroken after
       three attempts: register buffering, load width, codebook residency);
    2. the build-recipe change — non-expert tensors below 8-bit — which the
       stub ablation says owns 56% of the Flash-Next gap and which no kernel
       work can reach;
    3. the 1.07 ms/token of module glue found in §4, unchased, cheap to try.
