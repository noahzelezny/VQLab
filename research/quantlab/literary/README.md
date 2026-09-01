# litbench — discriminative literary understanding

104 hand-authored multiple-choice items testing metaphor, irony, subtext,
narrative causality, tonal register, symbol/motif, unreliable narration, and
understatement. 13 items per category, 4 choices each.

## Why discriminative and not generation+judge

The sidecar's literary job is content-making and description writing, and the
tempting instrument is "generate prose, have a judge score it." That needs a
judge model, sampling, and variance control, and it cannot run on the existing
harness at all.

Understanding is the cheap leading indicator, and it fits the harness we
already trust: `output_type: multiple_choice` is pure loglikelihood, so it
runs through `score_tasks_streaming.py` unchanged, is deterministic, is
resumable, and `analyze_task_bench.py`'s McNemar paired testing applies
without modification. No judge variance, no sampling noise.

**This does not measure generation quality.** It measures whether the model
can tell an ironic reading from a literal one. Those correlate; they are not
the same thing. Treat a litbench win as necessary, not sufficient.

## Why perplexity would not have worked

Predicting literary text well and understanding irony are different
quantities. A model can score excellent PPL on a novel by modelling its
surface statistics. The referee corpora stay the instrument for
*compression*; litbench is the instrument for *comprehension*.

## Running it

```bash
./run_literary_bench.sh
```

Or directly:

```bash
./venv/bin/python score_tasks_streaming.py --model <dir> --tasks litbench --include-path literary --output-dir results_literary
```

`--include-path` was added to the scorer for this task; stock tasks still
resolve without it. Analyze with
`./analyze_task_bench.py --dir results_literary --tasks litbench`.

## Build step

The shards (`items_*.jsonl`) are authored with the correct answer **first** —
the only honest way to write distractors is to write the answer and then
attack it. `build_litbench.py` merges, validates, and deterministically
shuffles (seed 20260817) into `litbench.jsonl`, which is the file lm-eval
reads. Always rebuild after editing a shard; the runner does it for you.

Validation is not decoration. It has already caught two real defects in these
items: a stray empty fifth choice, and a double space at the context/choice
seam that would have re-tokenized every continuation.

## Known limitations — read before citing a number

**1. Residual length bias.** Correct answers skew long: length-rank 3.23/4
where 2.50 is unbiased, mean +3.5 characters over distractors. Worst in
`tone_register` (rank 3.38, +7.1 chars) and `symbol_motif` (3.54, +4.6).

Consequence: **`acc_norm` is the metric of record**, and
`analyze_task_bench.py` is configured that way. Byte-length normalization is
what stops the bias from being scored as literary understanding. Report `acc`
alongside it, never instead of it. Reducing the bias at source — trimming
gold answers, padding distractors — is the highest-value next edit.

**2. n=104 limits what it can resolve.** Units below are **accuracy points
(pp)** — the gap in percent-correct that a paired McNemar test can tell apart
from noise. Nothing to do with perplexity.

Resolving power is `1.96 × √(d/n)`, where `d` is how often the two models
disagree per item. `d` differs sharply by regime:

| comparison | discordance | n=104 | n=400 |
|---|---|---|---|
| two different models | ~30% | 10.5pp | 5.4pp |
| two quants of one model | ~5% | **4.3pp** | 2.2pp |

The ~5% figure is measured, not assumed: this project's own 397B task-bench
runs show 44/1000 and 55/1000 discordant pairs (`analyze_task_bench.py`, the
`disc W/L` column). Near-identical models agree on almost everything, so the
quant case is the *sensitive* regime, not the hard one.

Practical consequence: at n=104 litbench comfortably separates **models**
(the bf16 shootout, where gaps should be large), and resolves quant deltas
down to ~4.3pp. To reach the 2pp deltas that mattered on the 397B ladder it
needs roughly **480 items** — the build script prints this figure each run.

**3. Single-author items.** Every passage and distractor was written in one
pass by one author. Systematic blind spots are likely and unmeasured.

## Provenance

All passages are original, written for this benchmark. No copyrighted text is
reproduced.
