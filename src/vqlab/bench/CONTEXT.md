# bench/ — Bench: how fast is it, and where does the time go?

## Inputs
An artifact (pinned), an idle box.

## Process
`vqlab decode-timeline` (stages that sum to the whole), `prefill-timeline` (the same partition for one prefill, any architecture), `decode-ladder` (deletion arms + checksums), `active-bytes` (the roofline denominator), `stage-bandwidth` (the two joined: GB/s and excess ms per decode stage), `serve-timeline` (a Knurlogic-served request's own partition, n runs), `gpu-capture` (one decode step as an Xcode .gputrace, for the per-kernel WHY), `prefill-bench`, `coverage`, `host-attrib`, `hc-micro`.

Library: `timeline_common.py` (`trunk()`: the module holding layers + embed_tokens; `layer_kind()`: (lin|attn, moe|dense)), shared by the two timelines; `power.py` (`gpu_watts()`: GPU package watts from macmon, -1 without it). `score/kernel_truth.py` exports `parse_arm` / `module_attr` for decode-ladder.

## Outputs
Printed timings and ratios.

## Rules that bite
- n>=3 with scatter, one process per arm, never on a contended box.
- Quote a RATIO between arms from one session, never an absolute (the decode instrument is bimodal).
- Never quote an effective bandwidth without `active-bytes` (F130).

- `speed-pair-knurlogic`: the same ratio rule when Knurlogic places the model (one Mac, or a pipeline split for models too big for one box). Arms are Knurlogic model NAMES; it refuses a run whose served name is not the arm (identity collapse: Knurlogic does not hash model.py) or whose runtime is not the bundled model.py.
- `speed_pair_knurlogic` re-exports its HTTP/job helpers from `core/knurlogic_client.py`; `box_quiet` reads the lease via `core/gpu_lease.py`; `mtp_bench` takes its CLI plumbing from `mtp/cli_common.py`. bench never imports agents or ship.

## Files
- `active_bytes.py`: `active-bytes`: weight bytes read per decode token, by component
- `box_quiet.py`: is this Mac quiet enough to time something? (lease, load, GPU power)
- `decode_ladder.py`: `decode-ladder`: per-component decode deletion arms with output checksums
- `decode_timeline.py`: `decode-timeline`: every stage of one decode token, timed, summing to the whole
- `gdn_chunk_truth.py`: `gdn-chunk-truth`: chunked vs sequential gated-delta scan against fp64
- `gpu_capture.py`: `gpu-capture`: one decode step's Metal work as an Xcode .gputrace
- `hc_micro.py`: `hc-micro`: the hyper-connection chain in isolation, dtype x batch
- `host_attrib.py`: `host-attrib`: separate the unattributed host cost of a decode step
- `kernel_coverage.py`: `coverage`: which kernel path each module of an artifact takes
- `lut_bench.py`: `lut-bench`: what the codebook LUT read costs the decode expert kernel
- `mtp_accept.py`: `mtp-accept`: paired draft-acceptance comparison across prompts
- `mtp_arms.py`: `build mtp-arms`: one pinned, servable arm per draft head, for a head A/B
- `mtp_bench.py`: `mtp-bench`: MTP drafting speedup, acceptance and numerics control
- `power.py`: `gpu_watts()`: GPU package power from macmon
- `prefill_bench.py`: `prefill-bench`: prefill time on one artifact under both codebook arms
- `prefill_timeline.py`: `prefill-timeline`: every stage of one prefill, timed, summing to the whole
- `serve_timeline.py`: `serve-timeline`: where a Knurlogic-served request's time goes
- `speed_pair.py`: `speed-pair`: decode and prefill speed of two artifacts, as a ratio
- `speed_pair_knurlogic.py`: `speed-pair-knurlogic`: the same ratio for models served by Knurlogic
- `stage_bandwidth.py`: `stage-bandwidth`: achieved memory bandwidth of every decode stage
- `timeline_common.py`: what the two timelines share: `trunk()`, `layer_kind()`
