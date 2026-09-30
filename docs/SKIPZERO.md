# vq-skipzero — dead rows cost nothing, so don't store or load them

**Status (2026-09-29): a runtime feature.** `runtime/vq_switch.py` serves
compact rows natively (`VQSwitchLinear(row_table=...)`, `#if SZ` in the walk
decode and gemmseg2 kernels); a bundle opts in with `vq_skipzero` in its
config. Pack/convert tools: `src/vqlab/experimental/skipzero/`. Findings:
F173, F175–F178, F183–F184.

## The finding

The Qwen3.5-397B-A17B **teacher** has output rows whose weights are ~1e-29 —
zero in all but name. 12.43% of every 397B rung's 64-element weight groups sit
in them (`vqlab zero-groups`), concentrated in the early layers (L0 58%, L1 55%,
L2-L4 32-45%), and 97.8% of those groups fill *entire* rows `[expert, out_row]`
(whole dead experts are only 0.5%). Every VQ rung spends full code bytes on
them. They encode nothing.

It is a property of that teacher, not of our quantizer:

| family | rungs scanned | dead-group share |
|---|---|---|
| Qwen3.5-397B-A17B | 2.2 / 2.4 / 2.6 / 3.1 | **12.43%** on every rung (10.2-15.4 GiB) |
| Qwen3.6-35B-A3B | 3.4 / 3.8 / 4.6 / 5.4 | 0.75% (0.08-0.14 GiB, ~all L0) |
| Qwen3.8-Flash-Next, GLM-5.3-Flash, Qwen3.8-27B, gemma-4 | all | 0.00% |

## Two stages

**Stage 1 — on disk** (`sz-pack` / `sz-check`). Drop fully dead rows' codes and
scales, keep a bit-packed row mask, expand at load with a shim appended to the
artifact's own `model.py`. Dead rows become exact zeros (the original held
~1e-5 x codebook).

**Stage 2 — resident** (`sz-resident` / `sz-bitexact`). Keep the compact rows in
memory with an `[E, OUT]` row table (-1 = dead). Forked decode and prefill
kernels look the row up; a dead row writes 0 and reads no code or scale bytes.
Forked per runtime vintage by exact text patch (the 35B's `walk` vintage; the
397B 2.4's older `u8` vintage), refusing anything else; `selftest` fails if a
`runtime/` edit stops a fork applying. The shipped runtime is untouched.

## Measured

| | result |
|---|---|
| **Quality, stage 1** — 397B 2.4 full (F175), paired KL, 3 corpora x 12288 | SAME everywhere: prose -0.16 (t=-0.3), code -0.89 (t=-1.5), lit -0.43 (t=-0.5) mnats |
| Quality, stage 1 — 397B layers 0-4 only (F173) | SAME everywhere |
| Quality, stage 1 — 35B 3.4 (F176) | **bit-identical**: paired delta exactly 0.000 (path verified exercised) |
| **Correctness, stage 2** — 35B (F177) and full 397B on the M4 (F178) | **byte-equal to stage 1** on every check: all switch modules at N=1/8/4096/4097 (decode + prefill kernels), in-model outputs, logits, a 9216-token prefill, 32 greedy tokens |
| Synthetic stress, stage 2 | 24/24 byte-equal, both vintages, up to ~60% dead rows and N=9000 |
| **Disk**, 397B 2.4 | 114.2 -> 101.9 GiB (**-12.3 GiB, -10.8%**) |
| **Resident memory**, 397B 2.4 on the M4 | 107.96 -> 96.02 GiB active (**-11.93 GiB, -11.1%**); load 60 s -> 47 s |
| Knurlogic M3+M4 pipeline (F178 addendum) | resident and stage 1 give identical 32 greedy tokens + top-20 logprobs; M4 rank -2.25 GiB |

Against the ORIGINAL rung, stage-1 zeroing lets later positions drift (logits
T512 max 7.97, greedy tokens diverge after a while) — perturbations compounding
over 60 layers, which the KL gate measured as no quality change (F175).

## What it buys

Each 397B rung takes roughly the memory of the rung below it, at its own
quality: 2.6 -> ~113 GiB (where 2.4 is today), 2.4 -> ~102 (2.2's size),
3.1 -> ~133. The 2.2 lands at ~96 GiB, still over what a 96 GiB box can hold
with a runtime. Only the 397B benefits.

## Reproduce

```
vqlab zero-groups <artifact>                         # price it (headers + scales, CPU)
vqlab sz-pack <artifact> --out <scratch>/<name>      # stage 1 (--dry-run prices, --layers A-B)
vqlab sz-check <packed> <artifact>
vqlab sz-resident <packed> --out <scratch>/<name>_resident    # stage 2
vqlab sz-bitexact <resident> <artifact> --ref2 <packed> --out <dir>   # gate = vs <packed>
vqlab sz-bitexact --synthetic --vintage both --out <dir>
```
The 397B full-model gate needs a machine that holds it resident (128 GB).

## Open

1. **Shipped as a runtime feature** (2026-09-29): 576/576 synthetic arms and
   the real 35B 3.4 byte-equal to the expanded form. 397B 2.4: peak memory
   119.87 -> 107.06 GB at the same decode speed, KL per-position arrays
   byte-equal to stage 1; 397B 2.6: KL unchanged, peak 117.67 GB. Older
   bundles report STALE in `check-bundle` (same output, missing the gains).
2. **The other 397B rungs** (2.2 / 3.1): same pack, same gates, at re-release.
3. **Skip the down-projections' swap**: their row tables cost more than their
   dead rows save (-0.12 GiB net). Output is identical either way (stage 2 is
   byte-equal, whichever modules are swapped).
4. A non-leader Knurlogic rank exposes no status endpoint, so per-rank memory
   on the M3 rank was not read.
