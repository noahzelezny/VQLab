# vq_wdec fusion kernel — measured perf (2026-09-05)

Kernel: commit f74ce4f (`vq_wdec_packed{BITS}_d{D}` — packed codes → fp16
weight tile in one dispatch, replacing `_unpack_rows` + `_decode` in
`vq_dense._decode_matmul._tile`'s packed arm). Bit-exactness ship gate
(`tests/test_vq_wdec.py`, uint16 bit-pattern compares) PASSES.

Design doc: `../logs/reviews/dense-fusion-design.md` §4.3 (T11–T14), §6.
Machine: M3 Ultra 96 GB, exo idle. Interpreter: repo `.venv`.
Artifact: `TheDrainFlorist--Qwen3.8-27B-VQ-4.8bpw` (K512 / d2 / bits9 / G64,
packed), local SSD copy under `~/.exo/models/`.

## T11 — prefill peak, real packed linears, N=2048

`scripts/bench_dense_prefill_peak.py`, arms = `VQ_DENSE_DECODE_FUSE` 0 vs 1
(separate processes; import-time flag), depth 16 vs 48 linears
(`--layers 8` / `24`). `peak − resident` = transient. Bit-identity across
tile arms asserted by the bench in every run (all YES).

| depth | arm | tile 0 transient | tile 512 | tile 64 | tile 16 | wall (tile 0) |
|---|---|---|---|---|---|---|
| 16 linears | FUSE=0 | 0.728 G | 0.748 G | 0.476 G | 0.248 G | 0.461 s |
| 16 linears | FUSE=1 | **0.271 G** | 0.291 G | 0.242 G | 0.189 G | 0.385 s |
| 48 linears | FUSE=0 | 0.728 G | 0.748 G | 0.476 G | 0.248 G | 1.286 s |
| 48 linears | FUSE=1 | **0.271 G** | 0.291 G | 0.242 G | 0.189 G | 0.852 s |

- **Flat in depth confirmed** in both arms: transients are byte-identical at
  16 vs 48 linears (0.728 G / 0.271 G), so the win is per-layer-bounded, not
  a graph-depth artifact (R6 closed).
- Transient 0.728 G → 0.271 G untiled (−63%). The doc's 658 → 178 MB target
  was per-widest-layer; measured whole-chain live-set transient lands in the
  same class (0.271 G ≈ one w buffer + GEMM operands).
- Wall at 48 linears: 1.286 → 0.852 s (1.51×) end-to-end including GEMM.

## T12 — packed decode flat (decode only, real tensors)

Old chain (`_unpack_rows` + `_decode`) vs `wdec_decode`, same process,
interleaved, min of 9 rounds (sweep table below): **~9.2 ms → 0.65–0.73 ms**
per full-module decode on both mlp shapes (gate 17408×5120,
down 5120×17408). Target was "toward the 1.65 ms unpacked class" — measured
**~13–14×**, past the target. (9.2 ms baseline reproduces the ledger's
~9.5 ms flat.)

## T13 — VQ_WDEC_ROWS_TG sweep

Interleaved A/B in one process (`_WDEC_ROWS_TG` monkeypatched per trial),
9 rounds, real tensors, ms:

| rows_tg | gate min | gate med | down min | down med |
|---|---|---|---|---|
| 2 | 0.676 | 0.916 | 0.694 | 0.808 |
| 4 | 0.662 | 0.718 | 0.731 | 0.919 |
| 8 | 0.649 | 0.875 | 0.686 | 0.733 |
| 16 | 0.706 | 0.758 | 0.687 | 1.120 |
| 32 | 0.679 | 0.735 | 0.667 | 0.783 |
| old | 9.191 | 9.323 | 9.013 | 9.258 |

Flat within noise (min spread 0.649–0.731 across all settings, both shapes).
The store-bound kernel is bandwidth-limited, not occupancy-limited at these
shapes. **Shipped default 4 stands; no change.**

## T14 — CB_TG A/B

**N/A by construction**: no threadgroup-codebook variant was implemented —
the shipped kernel has zero threadgroup allocation (device-LUT only), which
is the design's structural defense against the E124/E134 cap-gate class.
Nothing to remove before ship.

## Verdict

All perf claims now measured, all at or past target. Remaining gated step
(Noah's call, NOT taken here): ship `model.py` to artifacts through the
vqlab publish gate.
