# bench/ — Bench: how fast is it, and where does the time go?

## Inputs
An artifact (pinned), an idle box.

## Process
`vqlab decode-timeline` (stages that sum to the whole), `decode-ladder` (deletion arms + checksums), `active-bytes` (the roofline denominator), `prefill-bench`, `coverage`, `host-attrib`, `hc-micro`.

## Outputs
Printed timings and ratios.

## Rules that bite
- n>=3 with scatter, one process per arm, never on a contended box.
- Quote a RATIO between arms from one session, never an absolute (the decode instrument is bimodal).
- Never quote an effective bandwidth without `active-bytes` (F130).
