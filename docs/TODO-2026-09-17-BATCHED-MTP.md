# VQLab TODO — after the batched-MTP measurements (2026-09-17)

Context: the exo fork's batch engine now drafts (exo `mtp-stage1`, commits
1e140d4d / 2a2a422e). The engine picks drafting vs plain steps by MEASURED
cost per token, so every item below is about the kernels the engine sits on,
not the engine. Numbers are aggregate tok/s on one box, 500-token greedy
generations unless noted.

    Flash-Next 4.4bpw (M4)   plain  19.8 / 33.2 / 44.8 / 50.7  at 1/2/3/4 rows
                             draft  26.4 / 34.0 / 42.0 / 47.8
    Flash-Next 2.1bpw (M3)   plain  17.8 / 25.3 / 34.4 /  --
                             draft  24.5 / 32.7 / 37.5 / 45.8
    397B 2.2bpw (M4)         draft  22.2 / 26.7 / 23.6 / 18.3   (selector took
                             plain steps above 1 row; batching itself collapses)

The deciding number for MTP on any rung is now `cost(2-wide forward) /
cost(1-wide forward)`, not acceptance. It is ~1.5 on Flash-Next 4.4bpw, and
on the 397B the d8 kernels price a 2-wide forward like two tokens (the
file's own bench: 53.4 / 80.1 / 95.4 / 128.8 ms for 1/2/3/4 tokens).

## P0 — the 397B d8 decode kernel does not amortize tokens

1. **Token-sharing d8 decode kernel.** Today every (row, token) thread
   re-fetches the expert's code row and does its own device-codebook gathers;
   tokens routed to the same expert share nothing. Sort pairs by expert, fetch
   a code row once, dot it against all x rows for that expert. Bench at
   N = 10 / 20 / 30 / 40 pairs on real 397B routing (not synthetic indices —
   routing overlap at top-10 of 512 decides how much there is to share).
   Gate: whole-forward ms at seq 1..4, the way `_EXPERT_SIMD_MAX_N`'s comment
   measured it, not the isolated microbench (it was "NOT predictive").
2. **Measure N = 21..29 and re-set `_EXPERT_SIMD_MAX_N`.** The simdgroup
   kernel is gated to 20 pairs = exactly 2 tokens at top-10; a 2-row batch
   with drafting (4 tokens/forward) or a 3-row batch falls to the thread-
   per-row kernel. The comment says 21..29 is "deliberately unmeasured".
   Measure it; if the sharing kernel above lands, re-sweep the cutoff.
3. **Codebook residency for d8.** K=16384 x d8 x fp16 = 256 KB, so the
   codebook lives in device memory and every gather is a random device
   read. Measure a threadgroup-resident variant at iso-bytes (smaller K with
   the bits spent elsewhere, or a 2-level codebook) against the current
   devx+ss path. This is the F82-F87 Flash-geometry question applied to d8.

## P1 — ship what is already measured

4. **Publish the v2 runtime** (docs/V2-RUNTIME.md): +11.9% prefill / +3.5%
   decode over shipped, 1-ULP, repo default but not on the Hub. Runtime
   travels in the bundle, so it is a re-bundle + publish of 20 repos.
5. **wdec kernel**: uncommitted in vqlab, bit-exact. Commit -> bench -> ship
   gate.
6. **Fused VQ-GEMM prefill**: 1.43x measured, default off. Promote or write
   down why not.
7. **art_flash_r1**: beats shipped at iso-bytes (F82-F87). Ship candidate.

## P2 — benches the batched engine needs

8. **`vqlab mtp-bench --rows N`**: mirror exo's `mtp/batch_loop.py` step
   (2-wide verify over B rows, whole-batch replay on any rejection) so a
   kernel change is scored against the step the engine actually runs.
   Report per rung: 1-wide ms, 2-wide ms, their ratio, acceptance, and the
   break-even rows from `1 + (1 - a**B)` vs the ratio.
9. **Per-rung cost-ratio table** in FINDINGS: cost(2-wide)/cost(1-wide) at
   B = 1..4 for every shipped rung. This is the one number that says
   whether MTP pays on that rung; today it is known for two of ~20.
10. **Batched-path overhead**: exo's plain steps through the batched cache
    path run ~7% slower than mlx-lm's own batch generator on Flash-Next
    (31.0 vs 33.2 at 2 rows before drafting engaged). Profile the left-
    padding mask + `_BatchAttnCache` indexer path; that 7% is paid whether
    or not drafting is on.

## P3 — expert streaming (the "edge" question)

11. **Bytes-per-token audit for streaming.** 397B: 10 experts x 3 proj x
    (1024 x 4096) x 60 layers = 7.5B expert weights per token; at 2.2 bpw
    that is ~2.1 GB/token vs ~7.5 GB at 8-bit. An M4 Max SSD reads ~5-7
    GB/s, so pure streaming caps near 3 tok/s at 2.2 bpw and under 1 at
    8-bit. Measure real routing locality on traced prompts (how many
    experts per layer repeat across consecutive tokens) to size an LRU of
    hot experts in RAM; that locality, not raw bandwidth, is what the
    streaming repos live on.
12. **mmap the expert code tensors** (codes are the streamable part; the
    codebook, scales, attention and shared experts stay resident) and
    measure a rung one size up from what fits: the 397B 3.1bpw on the M4,
    or the 2.2bpw on the M3's 96 GB.
