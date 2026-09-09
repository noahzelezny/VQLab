# Decode: four workers, one plan (2026-09-08)

Every swarm round of this arc targeted PREFILL, because the mission said
"decode is NOT the target" — carried over from the 87%-of-affine goal. Noah
asked why nobody was on decode. The night's own measurements say that was the
wrong priority:

| | headroom |
|---|---|
| prefill | 13%, against a strong affine baseline |
| decode | running at **1-5% of memory bandwidth** |

    model            layers  active GB/tok  ms/tok  effective GB/s
    35B-A3B (1 box)     40       0.54        23.9      22.5
    Flash-2.1 (1 box)   48       0.58        54.1      10.7
    397B-2.2 (ring)     60       1.93        48.1      40.2
    397B-3.1 (ring)     60       2.72        62.5      43.6
    M3 Ultra peak                                     819

Flash and 35B move nearly IDENTICAL bytes per token and differ 2.3x. Within
the 397B family, time tracks bytes (1.43x size -> 1.30x time); across
architectures it does not. Same model on 1 box vs the 2-node ring is only
1.11x, so the interconnect is NOT it. That is the signature of a large FIXED
per-token cost.

## Four Haiku workers, four frames, one answer

Frames: per-token collectives / the length collapse / host+dispatch overhead /
measurement-only. They did not see each other. **All four opened with the same
recommendation**, and it is the sentence exo's own DECODE-LOOP-OVERHEAD.md
(2026-09-03) ends on without finishing: *"The single log line that finishes
the job already exists in the tree and needs one read-only run."*

They found it. `src/exo/worker/engines/mlx/generator/batch_generate.py:482`,
every 64 steps at DEBUG:

    step overhead: {overhead}ms (next={next}ms total={total}ms)

`next=` is time inside mlx-lm's batch step; `overhead` is exo's own response
loop. Run 2000 tokens and read the two columns:

* `next=` flat, `overhead` grows  -> exo's per-step work. The collective fix
  below applies.
* `next=` grows                   -> inside mlx-lm's batch arithmetic. Both
                                     suspects below are wrong.

**That run costs one generation and no code change. Nothing should be built
before it.**

## The fix it would license — VERIFIED anchors

`BatchGenerator.step()` opens (`worker/runner/llm_inference/batch_generator.py:405`):

    if not self._queue:
        self.agree_on_tasks()

`_queue` is empty throughout steady decode, so this fires EVERY token. It
reaches `mx_all_gather_tasks` (`engines/mlx/utils_mlx.py:1069`), where BOTH
all_gathers — line 1089 and line 1105 — pass **no `stream=`** and each ends in
`.tolist()`, forcing a sync. `mx_any` (1014) and `mx_barrier` pass
`stream=mx.default_stream(mx.Device(mx.cpu))`. So the model runs on mlx-lm's
`generation_stream` while two syncing collectives per token land on the
DEFAULT GPU stream.

`SequentialGenerator` in the SAME file already solves this for the other
engine: `check_for_cancel_every = 50` (line 101), throttling its analogous
call to ~0.03/token (lines 274-285). BatchGenerator has no throttle.

Two changes, both small: add the `stream=` kwarg to both all_gathers, and
throttle the call the way the sibling generator already does.

## What the workers got WRONG — the reason to verify before quoting

Same pattern as the local swarms: sound direction, unreliable specifics.

| claim | verdict |
|---|---|
| `step overhead` log at batch_generate.py:482 | REAL |
| both all_gathers lack `stream=` (1089, 1105) | REAL |
| agree_on_tasks at batch_generator.py:405 | REAL |
| SequentialGenerator already throttles | REAL (check_for_cancel_every=50) |
| growing `mx.array(self.tokens[e])` per step, opt_batch_gen.py:75 | REAL — but already FALSIFIED as the cause (15 us at N=2000) |
| "edit scripts/exo_model_cards/GLM--GLM-2.7bpw.toml" | **NO SUCH FILE** — zero GLM cards exist |
| "clear_cache every 256 / 512 tokens" | **WRONG** — batch_generate.py:494 is inside `close()`, not a cadence |
| "y.item() at generate.py:469" | **WRONG LINE** — .item() is at 511/565/632 |

Three fabricated specifics out of eight checked. The convergence is the
signal; the line numbers are not evidence until grepped.

## Method note

This is the first thing all night where four independent workers agreed on a
NEXT MEASUREMENT rather than a design. Prefill rounds 1-3 produced ~24 design
proposals; the only win of the arc (1.76x) came from the one proposal that
asked to A/B two configurations. The measurement-only frame was added for
that reason and it converged with the other three.
