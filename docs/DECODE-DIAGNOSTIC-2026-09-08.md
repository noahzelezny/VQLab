# The decode diagnostic: the collective hypothesis is FALSIFIED (2026-09-08)

Four Haiku workers, four independent frames, all four converged on one
recommendation: run the DEBUG `step overhead` line before building anything.
It was the right call, and it killed the idea they converged on.

## The run

397B-2.2, 2-node ring, exo master launched with `-v`, one 2000-token
generation (22.7 tok/s). 31 samples of
`engines/mlx/generator/batch_generate.py:482`, one per 64 steps:

    overhead   first5  0.064 ms    last5  0.074 ms    max 0.23
    next       first5 41.28  ms    last5 44.87  ms    max 47.96

**`overhead` is 0.17% of step time.** `next=` — time inside
`_mlx_gen.next()`, i.e. mlx-lm's own batch step — is the other 99.83%.

## What that kills

`BatchGenerator.step()` calls `agree_on_tasks()` every token
(`worker/runner/llm_inference/batch_generator.py:405`), and both all_gathers
in `mx_all_gather_tasks` (`engines/mlx/utils_mlx.py:1089`, `1105`) pass no
`stream=` and force a sync via `.tolist()`. All of that lives in `overhead`.

It is real, it is still worth tidying, and **it cannot be the problem**:
deleting it entirely would buy **0.17%**. The per-token collective on the
default GPU stream is NOT the decode cost. Neither is anything else in exo's
response loop.

Four workers agreeing was a strong signal about WHERE TO LOOK. It was not
evidence about the answer, and the measurement they all recommended is the
only reason we know that in one run instead of after a day of kernel work.

## The collapse did not reproduce either

DECODE-LOOP-OVERHEAD.md (2026-09-03) measured the batch engine going
51.3 -> 175.4 ms/token between 300 and 2000 tokens on GLM-2.7bpw — a 3.4x
collapse. Today, on 397B-2.2 over the same span:

    next grows 41.28 -> 44.87 ms = 1.087x

**No collapse.** Either it is GLM-2.7-specific, or something fixed it between
09-03 and now. UNRESOLVED, and worth one GLM-2.7 run to settle since the
whole Q2 half of that document rests on it.

## Where decode's cost actually is

Everything outside the model forward is now excluded, by measurement:

| suspect | verdict | evidence |
|---|---|---|
| weight bandwidth | NO | 1-5% of 819 GB/s across four artifacts |
| the 2-node ring | NO | 1.11x, same model, box vs ring |
| exo's response loop | NO | 0.17% of step time (this run) |
| per-token collectives | NO | same 0.17% |
| **inside `_mlx_gen.next()`** | **99.83%** | this run |

44 ms/token is spent inside mlx-lm's forward + sample. That is where the next
probe goes: per-layer dispatch count, kernel launch overhead, and the VQ
runtime's own template instantiation (its source notes 4.5-7 us HOST time per
call). The cross-architecture fact still needs explaining — Flash-2.1 and
35B-A3B move nearly identical bytes per token (0.58 vs 0.54 GB) and differ
2.3x — and it now has to be explained INSIDE the forward.

## Method note

This is the second time today that a cheap controlled measurement overturned
a confident inference (the first: the ring "costing ~2.1x" on decode, measured
at 1.11x). Both times the wrong answer was reached by reasoning from
uncontrolled comparisons, and both times the right answer cost under an hour.
