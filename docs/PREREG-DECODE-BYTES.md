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
