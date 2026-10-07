# Measurement runbook: where decode and prefill time goes

The order to run the instruments in, what each answers, and what it cannot
see. Written for the session that runs them on the Mac (a person, Claude
desktop, Scout, Chuck); every command is a `vqlab` CLI entry, and the
GPU-free ones are also on the MCP `run` allowlist.

The rule behind the order: **partition before you optimize.** Every kernel
idea in KernelProposals.md is a guess about where the waste is until one of
these has pointed at it.

## 0. Before any timing

* Pin and smoke the artifact (`vqlab pin`, `vqlab smoke --max-tokens 8`);
  record `stat -f %m <artifact>/model.py`. AGENTS.md, "Two agents, one
  artifact root".
* Quiet box. The timed tools refuse a busy Mac (Rule III); do not override
  it to get a number tonight.
* State the machine and its peak bandwidth: M4 Max 546 GB/s, M3 Ultra
  819 GB/s. Every bandwidth figure below is a fraction of that.

## 1. Decode: which stage is below the roofline?

```
vqlab bench decode-timeline --art <pin> --json-out dt.json [--context 64] [--reps 3]
vqlab bench stage-bandwidth <pin> --timeline dt.json --peak-gbs 546
```

`decode-timeline` partitions one decode token into stages that sum to the
whole (cumulative prefixes, never per-op evals: F133). `stage-bandwidth`
joins that to the weight bytes each stage reads (billed as `active-bytes`
bills them: dense, routed top-k, gathered rows) and prints, per stage kind
and per stage, GB/s, the roofline floor, and **excess ms: the time above
the floor**. Rank by excess. A stage with many bytes near its floor has
nothing to give; a stage with few bytes and many ms is overhead (launches,
small-op chains, sync); a big-byte stage far below the roofline is the
kernel to look at.

Read the timeline's verdict and noise floor before any single stage; only
aggregates are trustworthy below the noise floor.

Not covered: KV-cache reads (weights only; fine at context 64, understated
at long context). **decode-timeline's prefix forward is written for the
qwen3_5 hybrid trunk** (`is_linear`, `fa_idx`, `create_ssm_mask`); on
another family it fails loudly. Making it generic the way prefill-timeline
is (truncate the trunk's `layers`, stub the last MLP) is the first job if
the artifact under study is GLM.

## 2. Serving: where does a served request's time go?

On the Mac, with the model served by a Knurlogic that carries request
spans (`usage.knurlogic.timing.spans_s`, Knurlogic's
`engine/runtime/spans.py`):

```
vqlab bench serve-timeline --url http://127.0.0.1:<port> --model <name> \
    --prompt-tokens 2048 --gen-tokens 64 --n 3 --json-out st.json
```

Knurlogic partitions each request itself (HTTP build, queue, tokenize,
memory admission, prompt-cache lookup, and the gap / forward / host time of
its prefill and decode steps), so the buckets sum to the whole by
construction. The tool sends fresh prompts (no prompt-cache hits), a
discarded same-length warm-up first, and prints the median of each bucket
with min/max. It also prints the **engine-only prefill rate**:
prompt tokens / `prefill_forward`.

**This is the F188 check.** F188 compared a served prefill (~97 tok/s, 2734
tokens) with a plain forward (~245 tok/s, 2048 tokens) at unequal lengths.
Run serve-timeline at 2048, and a plain forward at 2048 with the same
chunk (`vqlab bench decode-ladder --mode prefill --prefill-tokens 2048`, or
`prefill-timeline --tokens 2048`):

* engine-only rate ~ plain forward, but TTFT far worse: the gap is in the
  serving buckets, and the table names which.
* engine-only rate itself far below the plain forward: the gap is inside
  the executor's step (chunk size, kernel path, the sequential GDN/KDA
  scan), and `prefill-timeline` on the same chunk is the next instrument.
* neither: F188's gap was the length mismatch; record it as such.

## 3. Prefill: which stage?

```
vqlab bench prefill-timeline --art <pin> --tokens 2048 --json-out pt.json
```

The same partition for one prefill, any architecture: attention / linear /
MLP half of every layer. Prefill is compute-bound, so stage-bandwidth's GB/s
does not apply here; rank by ms.

## 4. Kernel: why is THIS kernel slow?

```
MTL_CAPTURE_ENABLED=1 vqlab bench gpu-capture --art <pin> --out step.gputrace
open step.gputrace
```

One decode step, recorded for Xcode's Metal debugger. Performance ->
Timeline shows dispatch order and the gaps between kernels; Performance ->
Counters gives per-kernel duration, occupancy, memory bandwidth and the
ALU / memory limiter percentages. A kernel with a high memory limiter and
low bandwidth is doing scattered access (the codebook gather); low
occupancy means registers or threadgroup memory cap the threads in flight.
Only for the kernels steps 1-3 pointed at.

## What each instrument cannot see

| instrument | blind to |
|---|---|
| decode-timeline | overlap between stages (it reports drift); families other than qwen3_5 |
| stage-bandwidth | KV-cache bytes; anything the timeline did not time |
| serve-timeline | the client side and the network; time two requests share is charged to both |
| prefill-timeline | the serving layer |
| gpu-capture | anything outside the captured step(s); counters are Xcode's, not the lab's |
