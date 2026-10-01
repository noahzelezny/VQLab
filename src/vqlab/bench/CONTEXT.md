# bench/ — Bench: how fast is it, and where does the time go?

## Inputs
An artifact (pinned), an idle box.

## Process
`vqlab decode-timeline` (stages that sum to the whole), `prefill-timeline` (the same partition for one prefill, any architecture), `decode-ladder` (deletion arms + checksums), `active-bytes` (the roofline denominator), `prefill-bench`, `coverage`, `host-attrib`, `hc-micro`.

## Outputs
Printed timings and ratios.

## Rules that bite
- n>=3 with scatter, one process per arm, never on a contended box.
- Quote a RATIO between arms from one session, never an absolute (the decode instrument is bimodal).
- Never quote an effective bandwidth without `active-bytes` (F130).

- `speed-pair-knurlogic`: the same ratio rule when Knurlogic places the model (one Mac, or a pipeline split for models too big for one box). Arms are Knurlogic model NAMES; it refuses a run whose served name is not the arm (identity collapse: Knurlogic does not hash model.py) or whose runtime is not the bundled model.py.
