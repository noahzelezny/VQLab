# Morning report — 2026-09-10 (overnight parity round)

**The answer you asked for: parity to affine is reachable, and we now know
exactly where it lives.** Full data: FINDINGS-LOG F49; swarm output in
scratchpad/swarm6*/ideas.json (+270 KB salvage, unextracted).

## The one-table version

| | affine 35B-8bit | our 35B-3.4bpw |
|---|---|---|
| prefill | 2622 tok/s | ~2040 (78%) |
| expert path | 1.26 s | 1.94 s (1.54x) |
| ...kernel body | — | 1.37 s (**within ~9% of affine's WHOLE path**) |
| ...wrapper (dtype/host/dispatch) | — | 0.57 s (**the gap**) |

The Metal kernel is essentially parity-class. The gap is the Python/dtype
wrapper we fully attributed in F41-F43. And **dense is already there**:
27B-VQ-4.8 vs 27B-8bit = 98.3% end-to-end, quantized path 1.05x, at 60% of
the bits.

## The path to parity (in order, with ceilings)

1. **bf16-I/O kernel variant** — kills the ~0.20 s dtype round-trip. Swarm7
   produced a 3-proposal design family (T/TO split templating); F46 already
   measured +3.6% at decode. Biggest single step.
2. **Finish the host work** — memo shipped (+1.9%); the tile-builder
   vectorization + upload cache is the remaining ~0.1-0.15 s.
3. **xt leading-dimension padding** — swarm7's one "viable" kernel verdict,
   lifted from reading mlx's quantized.h (first round ever grounded on the
   competitor's source). Target: the last ~8% of kernel body.
4. Together: ~93-97% of affine at 40-60% of the memory.

## Do FIRST, though: the buffer-leak reproduction

Both VQ serving runners crashed `[metal::malloc] Resource limit (499000)`
under sustained long-output decode. Stock models don't. If the VQ decode
path leaks Metal buffer allocations, that is a RELIABILITY defect in what we
shipped yesterday — reproduce locally (watch active buffer count over a long
decode), fix, and it may warrant a patch push. Speed work waits behind this.

## Swarm-round accounting (honest)

- 15 structured proposals + 12 refutation verdicts (2 viable, 4 flawed, rest
  uncertain) + ~270 KB salvage awaiting extraction (needs a serving model).
- The harness fought us all night; three real fixes landed: bounded OVERTIME
  in sub_call (committed, mutation-verified), the 16k output cap (root cause
  of the "phantom tool calls" epidemic — that error string misdiagnoses
  truncation and should be split), step clock 3600 s.
- The 40k output budget I set then CRASHED the VQ runners (see above) — my
  doing, documented; per-workload budgets needed.
- 27B-4.8 as a swarm driver perseverates on failed tool calls (looped
  identical searches in the exo log); Flash-4.4 drove fine. Not a quant
  claim — an observed behavior difference, one night, two models.

## Cluster state at handoff

No instances loaded (both dropped by the runner crash; nothing re-placed —
settle-first rule). All swarm processes exited. Affine 35B-8bit now local on
the SSD hub cache. M4 deletions last night (DeepSeek-V4-Flash, spicyneuron
2.6bit) remain SSD-restorable.
