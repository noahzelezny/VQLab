# MTP validation on the 397B-2.2 cluster (2026-09-07 evening)

Unblocked by two things landing together: the pipeline `recv_like` dtype fix
(exo 574a7bd7), which is what had been killing the 397B at warmup, and the
MTP session's rebuilt sidecar with BF16 norm gains.

## Sidecar verified before measuring

`mtp-head-q6.safetensors`, all three paths (M3 Thunderbay, M3 ~/.exo, M4):

    {BF16: 30, U32: 10}   zero F32   5.412 GiB   .fp32norms-bak retained

The MTP session established that the F32 was dtype-only and the VALUES were
always correct — the graft stores deltas, the sidecar stores absolute gains
(1 + delta), and `from_sidecar` correctly does not re-shift. Their proof was
a structurally identical head (`mtp-397b-e3q8`) scoring 0.9082 acceptance;
norms running at delta+2.0 could not do that. So this is a regression check
against a known-good number, not a first measurement.

## Acceptance — 12 prompts, 128 tokens each, 1536 tokens total

| metric | value |
|---|---|
| acceptance, token-weighted | **0.9141** |
| mean / min / max per prompt | 0.9141 / 0.8590 / 0.9530 |
| decode | 29.4 tok/s median (26.7-30.7) |

Histogram: {0.86: 1, 0.89: 3, 0.91: 2, 0.92: 3, 0.95: 3}.

Lands on the MTP session's 0.9082 (shipped trunk) and 0.9121 (new trunk).
The rebuilt BF16 sidecar performs identically — no regression from the cast.

A first smoke on a 4-token "hi" reported `acceptance 1.000`. That is a
degenerate case, not a result, and is why the real measurement uses twelve
substantive prompts at 128 tokens. Worth remembering before anyone quotes a
1.000 from a short probe.

## End-to-end speedup — same 12 prompts, MTP on vs off

Steady state, 11 prompts (prompt 0 dropped from both arms; the MTP arm pays
a one-time 9.0 s warm cost there):

| arm | mean | median | range |
|---|---|---|---|
| MTP ON | 5.060 s | 5.030 s | 4.84-5.41 |
| MTP OFF | 6.046 s | 6.040 s | 5.98-6.13 |

**1.195x mean / 1.201x median.** Ranges do not overlap — the slowest MTP
prompt (5.41 s) beats the fastest non-MTP prompt (5.98 s).

The OFF arm is much tighter (2.5% spread) than the ON arm (11%), which is
expected: acceptance varies per prompt, so the speedup does too.

### Caveat that limits what this number means

`EXO_MTP` does not only toggle drafting — as of exo ff1c6f39 it also selects
the sequential engine when drafting would engage. So this A/B is
(MTP + sequential engine) vs (no MTP + default engine), NOT MTP in
isolation. The direction is solid and the ranges are clean, but attributing
all 1.20x to speculative decoding would overstate it. Separating the two
needs a third arm holding the engine fixed.

Also: both arms ran with `VQ_MOE_FUSED_GEMM_D8=1`, i.e. on top of the newly
promoted fused d8 prefill path, not the legacy one.

## Configuration measured

Both nodes: `EXO_MTP=1`, `VQ_MOE_FUSED_GEMM_D8=1`, `EXO_DTYPE_PROBE=1`.
397B-2.2 pipeline-sharded across M3+M4. MTP stage 1 engaged on both ranks.

## Open

- The published sidecars carry the same F32 norms the MTP session found on
  disk (397B 7, Flash-2.1 2, GLM-3.6 1 by HF header survey). That is a
  published defect, and republishing is Noah's gated call — it should go out
  as a family-wide update, not piecemeal, and alongside these numbers.
- A third arm (engine fixed, drafting toggled) would separate the speculative
  gain from the engine change.
