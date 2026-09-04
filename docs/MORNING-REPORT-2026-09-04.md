# Morning report — Friday 2026-09-04

Everything below was measured overnight; invocations and TSVs in the
research dirs. Nothing was published. The cluster is stock-config, idle,
zombie-free; both boxes healthy.

## 1. exo stage-1 MTP — the release claim (family-split, honest)

| rung | stock 300/2000 tok | MTP 300/2000 | acceptance | verdict |
|---|---|---|---|---|
| GLM 2.7 (fits 1 box) | 19.5 / 5.7 | 22.3 / 11.0 | 0.887/0.701 | +14% -> 1.9x |
| **GLM 3.1 (cluster-only)** | 6.9 / 6.2 | **18.5 / 17.4** | 0.727/0.642 | **2.7-2.8x** |
| 397B 2.6 (cluster-only) | 23.7 / 23.5 | 23.7 / 23.4 | 0.853/0.851 | parity |

The claim: **"On GLM cluster rungs, stage-1 MTP delivers up to 2.8x
long-generation decode. 397B stock does not degrade with context and MTP
is parity there."** Mechanism consistent with the overhead diagnosis
(GLM's wide-state snapshots are the prime per-step suspect; 397B has no
such state). All over TB4 TCP — TB5/RDMA re-bench queued as upside.
Caveat that ships with it: drafting currently requires EXO_NO_BATCH=1;
the 2-3 day batch-depth-1 dispatch retires it post-release.

## 2. 397B v2 sweep — COMPLETE, verdict positive

57/57 single-layer candidates measured (~104 s each; the 10-30 min
planning band was pessimistic by 10x). 26/57 layers help (GLM was
26/42). A late band (L37-L55) dominates; L47 alone is +0.0224 prose.

| build @ bytes | prose ppl | d vs base |
|---|---|---|
| base (2.4bpw, 111.8 GiB) | 2.7624 | — |
| **BEST6 {40,42,44,45,47,48} @ 112.74 GiB** | **2.6957** | **+0.0667** |
| CTRL6 (bottom-6) @ 112.74 GiB | 2.7762 | −0.0138 (worse than base) |

Effects compose essentially additively (sum of singles 0.0670 vs
measured 0.0667). The control LOSES — targeting is demonstrated, the GLM
methodology transfers. The spliced BEST6 artifact exists on disk.
**Remaining before a v2 ships:** code-corpus score, 30k-context peak
check (budget: 112.74 + the rung's ~5.9 GiB context overhead = ~118.6 vs
119.2 usable — fits, barely; trunk+head+30k does NOT), full gate, card,
Noah's naming call (v2 vs 2.5bpw), and the head-reservation decision.

## 3. Flash sweep — COMPLETE, shipped mix VINDICATED

48/48 layers swept + byte-matched verdict at 45.78 GiB:
shipped hot-2 {0,1} KL 390.09 beats measured-singles BEST2 {0,2}
(395.59) and the control {33,34} (429.59, worse than flat). {0,1} is
super-additive, {0,2} sub-additive — the probe's top picks compose
better than greedy singles. The shipped 2.1 card stands, now backed by
direct measurement. (Probe RANKING is still noisy: it rated L33 hot and
promoting L33 hurts — GLM's failure mode reproduced in the tail, not the
head.) No Flash v2 warranted at this size point.

## 4. Corrections staged (need your eyes, then docs-class publishes)

- **GLM 2.7 card**: shimmed single-box numbers (plain 20.3 flat; mtp
  20.0@1000/17.1@2000, acc ~0.75), 'no single-box speedup' strengthened,
  the false 'serve includes the fix' sentence now true (5e937f9).
- LEDGER entries for the shim A/B (with the two standing rules: >=1000
  tokens for single-box mtp A/Bs; shim moves acceptance 0.827->0.75 via
  ULP-compounding at near-ties).

## 5. Incidents, honestly

- **The OOM that made you quit Scout was mine**: instance swaps orphan
  exo runners that pin GPU memory invisibly; four pinned ~34 GiB. Now
  cleaned on every bounce (in the driver), documented in memory, and the
  night ran clean after.
- A bench-printer shell bug ate the stock-arm stdout for one arm; the
  exo logs carried the MTP numbers and the 2.6 stock arm was re-run
  clean. GLM 3.1 numbers came from the MTP-loop logs.
- One verdict control was launched with 3 guessed layers; caught and
  relaunched with the measured bottom-6 before it scored.

## 6. Friday go/no-go list (your calls)

1. exo MTP release: push mtp-stage1 (or merge to vq-serving), claim per
   §1, flip GLM 3.1/3.6 card MTP caveats to the measured 2.7-2.8x; 397B
   cards keep 'parity' language. GO/NO-GO.
2. GLM 2.7 card correction publish (docs-class, M4). GO/NO-GO.
3. 397B v2: finish validation (code score + 30k check + gate) today and
   ship as the week's capstone, or park for the weekend. Also: name, and
   trunk-only vs head-reservation stance.
4. Flash: add a card line citing the measured vindication? Also the
   staged mtp-head-q6.safetensors sitting in the shipped 2.1 artifact
   dir (one --all from shipping unmeasured) — bless with caveat or
   delete the staged copy.
5. Batch-depth-1 MTP dispatch (retires EXO_NO_BATCH): schedule next
   week.
