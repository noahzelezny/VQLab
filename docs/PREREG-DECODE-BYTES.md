# Pre-registration — decode time vs ACTIVE BYTES on Flash (F130 timing arm)

Registered 2026-09-18, before any timing run, box currently contended (exo
serving Scout), so the arms below are queued for an idle window.

## Background

`vqlab active-bytes` (new) walks the safetensors headers and bills each
tensor by how a decode step READS it: dense every token, routed top-k of
n_exp, or gathered a few rows. It never loads a tensor and never touches the
GPU, so the census itself is contention-immune.

The census says Flash-2.1 reads **5.284 GB/token** and Flash-4.4 reads
**5.939 GB/token** — the expert stack is only 12.1% / 21.8% of the traffic,
and the 2.1 -> 4.4 step is **+12.4% bytes**, not the ~2x the bpw label
suggests.

## Predictions

* **P1.** Flash-4.4 decode is SLOWER than Flash-2.1 by 8-17% on the same box,
  same harness, same session (centre 12.4%, the byte ratio). Specifically NOT
  ~2x slower, and specifically not equal.
* **P2.** Effective bandwidth (active bytes / measured ms) lands within
  +/-10% of 97 GB/s on BOTH Flash rungs and on 35B-3.4 — i.e. decode time on
  this family is proportional to active bytes at a fixed rate, and the
  "unexplained 2.3x" of F22/F23 is entirely the denominator.
* **P3.** Re-measuring Flash-2.1 decode on the CURRENT (v2, 2026-09-17)
  artifact reproduces F48's 54.31 ms/tok within 5%. The census is taken on
  today's bytes; F48's time predates the v1->v2 rebuild, and the pairing in
  F130 is provisional until this arm runs.
* **P4 (the mechanism arm).** Deleting GatedDeltaNet (42% of Flash-2.1's
  bytes) removes MORE decode time than deleting the whole VQ expert module
  (12.1%). F48 measured the VQ module at 6.54 ms of 54.31; P4 predicts the
  GDN arm exceeds that, and lands in 15-30 ms.

Falsification is recorded as falsification. P1 failing high (>=1.8x) would
mean bpw does drive decode after all and the census is measuring the wrong
thing; P2 failing would mean the fixed-cost story of F22 survives with a
corrected denominator.

## Arms, when the box is idle

1. `vqlab active-bytes` on both rungs (done, deterministic).
2. Decode timing, one process per arm, alternating, n>=3, RATIO quoted never
   absolutes (the ~100 GiB decode instrument is bimodal, rule III).
3. The GDN-deletion arm needs a stub that does NOT assume KVCache: Flash's
   linear layers carry `ArraysCache(size=2 or 4)` and its full-attention
   layers `QSAKVCache`. This is the fix F48's crashing arm needed.

---

# Addendum — the 35B rung curve (registered 2026-09-18 21:22, before the run)

Flash-4.4 (97 GB) does not fit the M3's 103 GB beside ~34 GB of resident apps,
so the 2.1-vs-4.4 pair cannot be run here. The 35B family substitutes and is
BETTER: four rungs, all resident-safe, giving a CURVE instead of a pair.

`vqlab active-bytes`, measured:

| rung | on disk | active GB/tok | expert GB/tok | expert share |
|---|---|---|---|---|
| 3.4 | 14G | 1.796 | 0.378 | 21.0% |
| 3.8 | 16G | 1.859 | 0.441 | 23.7% |
| 4.6 | 19G | 1.961 | 0.543 | 27.7% |
| 5.4 | 22G | 2.079 | 0.661 | 31.8% |

Resident spans +57%; ACTIVE bytes span only +15.8%. That gap is the whole
thesis of F130 and this is its direct test.

## Predictions

Basis: tonight's Flash arms make the VQ expert path byte-proportional at
time/bytes = 0.96 and 105.7 GB/s effective, while the dense trunk is not
(GDN 0.37, attention 0.54). The trunk is IDENTICAL across these four rungs --
only expert bytes move -- so the proportional component should dominate the
differences.

* **P5.** Decode ms/tok rises MONOTONICALLY 3.4 -> 3.8 -> 4.6 -> 5.4.
* **P6.** 5.4 is **+10% to +20%** slower than 3.4 (centre +14.5%, from
  +0.283 GB of expert bytes at 105.7 GB/s = +2.70 ms on a ~18.6 ms step).
  Explicitly NOT ~0% ("bpw does not affect speed") and explicitly NOT faster
  at the low rung by the ~37% that resident-size intuition suggests.
* **P7.** Per adjacent pair, the measured delta matches
  (delta expert bytes)/105.7 GB/s within +/-40%.

P6 failing LOW (<5%) would mean even the expert path is not byte-bound on this
family and F130's mechanism is family-local, not general. P6 failing HIGH
(>25%) would mean something beyond bytes scales with rung -- kernel geometry
per K, the first place to look.

Instrument: `vqlab decode-ladder --arm baseline`, one process per rung,
interleaved with a repeat of 3.4 last as the drift check, on the idle M3.
RATIOS only (rule III).


---

# OPEN, carried forward (2026-09-18, end of session)

* **F135 — RUN, see findings.** Result: VQ reaches 259-352 GB/s against
  affine's 574 on the same runtime; the smaller rung is the slower one and
  ships the geometry Metal rule IV names as failing on the cap. Original
  registration kept below for the record.
* ~~**F135 — VQ vs affine at a known byte ratio, NOT RUN.**~~ The 27B DENSE pair:
  VQ-3.9 moves 11.750 GB/tok against affine-8bit's 27.229 (2.32x), and the VQ
  arm carries the LIGHTER trunk (4-bit vs 8-bit), so the confound that
  inflates the Flash parity number runs the other way. If VQ is not ~2.3x
  faster at decode, effective bandwidth is the story. Script staged at
  `vqlab-scratch/f135_vq_vs_affine.sh`. NOTE: qwen3_5 is dense, a different
  runtime from the MoE fused path (Metal rule IV) -- it says nothing directly
  about Flash's kernels.
* **F134b — RUN. All five prefill arms provably distinct under the widened
  all-positions checksum (baseline 31529082, hc 2609625, gdn 116305447, vq
  26766899, sharedexp 22897323). F134's vq prefill result is now proven by
  the channel, not only by timing.**
* ~~**F134b — prefill checksum re-verify, NOT RUN.**~~ The prefill arms' timings
  stand on the re-typing assert but were never proven by the checksum channel,
  because the single-token version collided. Staged at
  `vqlab-scratch/f134b_verify.sh`.
* **The 4-bit trunk build.** F134 measured the SPEED case (4-bit ~10% faster
  than 8-bit at batch 1, neutral at prefill, half the bytes). The QUALITY case
  is unmeasured: geo-build a 4-bit-trunk Flash-2.1 and run `kl-ladder`,
  paired, three corpora at 12288, |t|>2 (the F118 gate). Do NOT assume 6-bit
  is the safe middle -- it is not byte-aligned and law I.9 measured -8% for
  bit-extraction at that width.
* **Is the flat 8-bit Flash/GLM trunk deliberate?** Undocumented in the card
  and anomalous against `stream_convert.py`'s own `--bits 4 --protect-bits 8`
  default and against the 397B/35B/27B configs. Noah to confirm before any
  refit treats it as a gap rather than a decision.
* **A matched-BYTE, different-GEOMETRY twin** to settle law I.9 properly
  (F132 challenged it on confounded evidence -- bytes and geometry co-varied).


## The I.9 twin — why it still needs a build (2026-09-18)

The E87/E88 rate twins would have served directly: matched PACKED size 13.83
GiB, same source, differing only in geometry (d4-K256 / d2-K16 / d8-K65536 at
2.00 bpw). They are **NOT_FOUND on every Thunderbay root** -- not retained.

The fit archive cannot assemble a substitute either. Computing bits/weight
over every stored geometry finds exactly one exact rate match:

    d8-K16384  14 bits / 8 dims = 1.75 b/w   92 modules   cb 262144 B  DEVICE
    d4-K128     7 bits / 4 dims = 1.75 b/w   24 modules   cb   1024 B  threadgroup

Matched rate, opposite kernel paths -- the ideal pair. **Module overlap: 0.**

So the twin is a SUBSET BUILD: take N modules that already have d8-K16384
fits, fit those same N at d4-K128 (K<=256 is the cheap fitting regime), then
geo-build two artifacts byte-identical everywhere except those N. F135 gives
this a sharper target than law hygiene: it would isolate whether the
threadgroup cap is the cause of the 1.6-2.2x efficiency spread, or merely
correlated with it.
