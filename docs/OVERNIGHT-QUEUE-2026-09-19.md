# Overnight working state — 2026-09-19 (written before compaction)

Noah is asleep. Scout's services are deliberately DOWN (text-embedder was the
146 W GPU contender) so the box is free for measurement. Do NOT start them
without asking; he reversed the earlier restore request on purpose.

## In flight right now

* `f139_ss_d2.sh` — rung 4.8 arms (the 4.5 rung is DONE, see below).
* `f139_smokes_after.sh` — queued behind it; re-runs the four smokes my broken
  `grep -E "->|PASS|FAIL"` swallowed (needs `grep -E --`).
* Logs: `/Volumes/Thunderbay SSD/vqlab-scratch/f139_*.log`.

## Recorded this session (all committed + pushed, branch perf/decode-byte-census)

| F | one line |
|---|---|
| F130 | byte census; the decode 2.3x is the DENOMINATOR, not a fixed cost. CORRECTED by F131's mechanism. |
| F131 | hyper-connections are 44.4% of Flash decode for 12.9% of bytes; deleting 71% of bytes removes 35% of time |
| F132 | 35B rung curve +40.6% for +15.8% bytes; law I.9 CHALLENGED (bytes and geometry co-vary — my confound) |
| F133 | mx.compile on hc is DEAD (+0.69%, a real small regression); quantized matmul is already one fused kernel |
| F134 | the regime INVERSION: hc owns decode (44%->9%), VQ+GDN own prefill (11%->30%, 15%->28%) |
| F135 | VQ at 45% of affine's byte efficiency (259/352 vs 574 GB/s). **Its cap mechanism is CORRECTED IN PLACE — read the banner.** |
| F136 | VQ_DENSE_SS = 1.146x decode, NOT bit-exact; D4_WALK 1.4%; DEVX 1.351x; VQ linears ~54.5% of the step. 4.8 rung UNCERTIFIED (drift check failed) |
| F137 | **the KL gate never reaches the DECODE kernels on dense artifacts** — decode-path numerics ship untested |
| F138 | SS is faster AND more accurate on 3.9 (prose -0.605, t=-4.31); the "8 ULP" was DISAGREEMENT, not error |

## Results that are VOID — do not cite them

* The FIRST F137 run (both arms identical to 3 decimals): the SS kernels were
  never reached. Per-pos arrays moved to `f137_perpos_VOID/`.
* The FIRST hc-micro dtype matrix: a foreign GPU job at 146 W; bf16 swung
  72 -> 403 -> 1116 us.
* The FIRST 4.5 arms in F139: `VQ_DENSE_FUSED_MAX_N` covers PACKED codes only;
  `_fused_max_n` returns `_DENSE_FUSED_MAX_N_PLAIN` FIRST for unpacked. 4.5 is
  unpacked, so it stayed on `_decode_matmul`. Needs BOTH overrides.
* F136's 4.8 decode contrast (closing drift baseline 211.5% spread).

## Standing caveats on everything SS

1. Scoring runs at **N=512 with the fused gate FORCED** — not the shipped
   decode regime. Clean version needs a chunk<=32 teacher cache.
2. **The bf16 27B teacher is GONE** (searched SSD+HDD exhaustively). Noah's
   call: just re-download from HF when needed; do NOT build archival guards.
3. Everything is the **DENSE** runtime. 17 of 20 artifacts are MoE and
   UNTESTED.

## F139 result so far (rung 4.5, d2-K256 UNPACKED) — paired

    prose   54.062 -> 53.200   -0.863  -1.60%  sem 0.9164  t -0.94   null
    code    20.616 -> 20.757   +0.141  +0.68%  sem 0.0703  t +2.01   ADVERSE
    lit    287.096 -> 287.082  -0.014  -0.00%  sem 0.3665  t -0.04   null

So SS is **geometry-dependent**, not uniformly good: d4-packed favours it,
d2-unpacked is neutral-to-slightly-negative. Noah's threshold: he does not
care about <0.5 mnat moves, and every adverse number is +0.141. Apply P4
(magnitude governs) SYMMETRICALLY — it was registered before the run and must
not be used only to dismiss unfavourable results.

## The queue, by value

1. **PLE-swap on the 4.4.** Cheapest "smarter" lever: pure symlinks, zero
   fitting. 4.4 ships PLE at 80 B/row (28.8 GB of its 101.4 GB resident); the
   2.1 ships 20, the 3.2 ships 55. At 20 B/row the 4.4 lands ~80 GB and FITS
   the 103 GB M3 — a far more capable rung on one box. F119 priced ple21 on
   the 3.2 at +24/+19/+15 mnats (expensive) and ple44 at ~free, but **nobody
   has run it on the 4.4**. Instrument: `vqlab ple-swap`, then `kl-ladder`.
2. **The 4.5 vs 4.8 split.** 4.8 is WORSE on prose (59.677 vs 54.062) and
   BETTER on code (16.431 vs 20.616) despite being the higher rate. Both have
   per-position arrays on identical positions, so `vqlab kl-pair` compares
   them directly — nearly free.
3. **Does the MoE KL gate reach ITS decode kernels?** F137 is dense-only and
   most of the fleet is MoE. Code read (`VQ_EXPERT_SIMD_MAX_N`, the gemmseg
   early return) plus one probe.
4. **Flash trunk 8-bit -> 4-bit.** Biggest unshipped speed+size lever: ~10%
   faster at batch 1, 43% fewer bytes/token, spends NO expert budget. Flash
   and GLM-Flash sit at flat 8-bit while 397B/35B/27B run 4/6, against
   `stream_convert`'s own `--bits 4` default. `geo-build` into SCRATCH, then
   `kl-ladder`.
5. **The remaining ~1.9x** on VQ vs affine. Real Metal work on the
   reduction/MAC; code fetch is eliminated (1.4%) and dispatch is noise on
   dense.

## Rules I work under while he sleeps

* Measurement only on shipped artifacts. **Nothing irreversible**: no publish,
  no default flip, no writes into `Exo Models/`. Builds go to scratch.
* Pre-register every arm; grade it honestly; a falsified prediction stays
  falsified.
* Every run self-checks: bracketing drift baselines, `gpu_power` gate before
  AND after, checksums with the CORRECT expected direction (deletion must
  differ, replacement must match), and a channel proving the arm took.
* If a result needs judgement I would normally bring to him, STOP and queue it.
