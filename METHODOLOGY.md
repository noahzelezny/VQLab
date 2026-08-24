# METHODOLOGY — the measurement discipline that makes the numbers believable

This project's results survived ~130 logged experiments in which **no wrong
number ever announced itself**. Every wrong number produced along the way — a
proxy score in a headline table, a corrupt artifact manufacturing a 25x
effect, a units mismatch impersonating a bias — was plausible, internally
consistent, and caught only by pre-registration, a cheap measurement, or a
second reader. The rules below are the distillation. They are mechanical, and
they are part of the tool: several of them are enforced by shipped gates.

If you use VQLab to produce a quant and publish a number from it, we ask
you to follow the same rules. They cost minutes; skipping any one of them
produced at least one false result in the lab record.

## 1. Gate before you believe

**Run the outlier gate on the assembled artifact before any score from it is
believed** (`vqlab verify --outlier 3.0`), and run it on a *different box*
than the one that fit it, when you have one. A corrupt artifact scores
plausibly and silently; the fitter's own log structurally cannot see it — it
reports what it *computed*, not what reached disk. The lab saw write-time and
deferred-read corruptions whose fit logs were perfectly clean.

The gate is relative (catches collapsed tensors). It does NOT catch a
uniformly worse fit, and mean relative error is a bulk statistic that is
blind — or at low K anti-correlated — with output damage. Never tune a fitter
by reconstruction error.

## 2. Packed bytes are the only size

Stored bytes include whole-byte padding and are not a size. An unpacked
d2/K512 dense artifact reads 21.6 GiB against its true 14.6. **Quote packed
bytes measured on disk, or analytic bpw — never stored bytes.** And a row's
size and quality must come from the *same artifact*: a size read from one
directory and a KL from another is not a result.

After packing, the index's `metadata.total_size` must be *recomputed from the
packed shards*, never copied from the unpacked index (the packer does this;
do not bypass it). Derive a size from the bytes, never from a field that says
what the bytes are.

## 3. Noise floors before margins

Unless you seed them, two fits of identical geometry produce different
artifacts — every measurement in the paper is a single unseeded draw.
Before believing a small margin at a geometry, **measure the seed-noise floor
at that geometry** (n≥3 fits, same recipe, score all). Measured examples:
dense-27B d2/K256 spans 2.085 mnats KL / 0.0447 ppl across three draws.
Third-decimal perplexity differences between single-draw artifacts are not
interpretable; KL separations of 5+ mnats and top-1 separations of ~1 pp are.

A floor belongs to the geometry it was measured at. **Do not inherit it**
across K or d — and when you do inherit one anyway, say so in the reading.

**Check whether your fitter seeds before you measure a floor.** VQLab's
dense fitter seeds by default (`--seed 1234`); pass `--seed -1` (or
distinct seeds) for floor work. The MoE fitter does not seed at all.

Know the signature of getting this wrong, because it does not look broken.
Seeded fits are near-identical but **not bitwise identical**: with the RNG
pinned, a measured 0.0100% of codes still differ (13,350 of 133,693,440 at
d2/K256, max codebook delta 4.9e-04) and shard hashes differ — a second
nondeterminism source survives the seed, unidentified, with Metal reduction
order the untested candidate. So a floor accidentally measured with a fixed
seed comes back **small, nonzero, and entirely plausible** rather than
zero. That is more dangerous than zero: a zero floor is self-evidently
broken and gets caught on sight, while a suspiciously tiny floor reads as a
real measurement and certifies every third-decimal margin in sight. **If a
floor comes back far tighter than you expected, suspect a pinned seed
before congratulating your fits on their consistency.**

**And check an instrument's description against the instrument itself, not
against what it was built to achieve.** Two descriptions in this project's
own documents survived review while being wrong — a manifest called a
"content hash" that hashes a 1 MiB prefix, and fitters described as
"seeded" when every published artifact was an unseeded draw. Both read as
obviously true, which is precisely why nobody resolved them. Read the
source of the tool you are describing.

## 4. Generate one token before calling anything releasable

Every byte-level gate reads bytes; none runs the model. An artifact in this
lab passed the outlier gate, release checks, vision check AND the scoring
referee — then raised on its first real forward pass, because the referee
scores through the reference decode path while serving uses the fused kernel.
A rung can *score* normally and be unable to *serve*. **Smoke-generate
through the exact fused path the artifact ships with** before release. Note
this needs the whole model resident: check artifact bytes against box RAM
first (`preflight_ram`).

## 5. Test the copy that ships

Which runtime executes — the artifact's bundled `model.py`, the venv's
site-packages, or your repo checkout — depends on the loader and the
environment, and both directions have produced wrong conclusions. **Never
assume; instrument the import (`mod.__file__`) and name the resolved path in
any runtime-dependent claim.** Acceptance harnesses must import the bundled
runtime *lifted from the artifact* as the unit under test (`vqlab
bundle-accept`), not whatever a module path resolves to.

## 6. Comparators are instruments too

A comparison row must name the artifact AND the instrument that produced it.
A number older than the artifact it faces gets re-measured, not cited. A
comparator that loads short scores worse and flatters you — structurally
check comparators (`check_comparator`) before their row is believed. One
instrument per model family; no cross-instrument rows. Report BOTH KL and
ppl where available: the two output metrics have been observed to rank a
pair oppositely, and a winner declared on one metric alone is not a winner.

## 7. Pre-register, then read

Write down the prediction and the reading grid *before* the number exists —
including what each outcome will mean. A falsified prediction is recorded as
falsified, never reframed. This is the single cheapest control in the list
and it retired more wrong claims than any gate.

## 8. Provenance before attribution

Before designing an experiment to explain a difference between two runs,
check the *inputs* both runs consumed — base artifact, source, config, tool —
by content, not just mtime (metadata answers "was this touched," never "did
this change"). Two silent in-place overwrites in the lab record each cost
days of algorithmic hypothesizing that a directory listing would have ended.
Concretely:

- Stamp every gated artifact with an external manifest (`vqlab manifest
  write`), stored *outside* the artifact, and `manifest check` before
  citing any number from it. **Know what the stamp proves:** it records
  per-shard bytes, mtime, and a sha256 of the first 1 MiB — an *identity*
  stamp that answers "are these the shards I stamped," not a full-content
  hash. A mid-shard change that preserves byte count and mtime would pass
  it. For a publish-time check, or any conclusion that rests on content
  being unchanged, compute full sha256 over each shard (and compare
  against the remote object hash when publishing).
- A refit must never aim `--out` at a scored artifact's path: fits resume,
  so it would emit a repaired-looking artifact containing the suspect bytes,
  and even a clean refit destroys the evidence for the published number.
- Record the resolved interpreter stack (python version, mlx version, the
  actual `.so` that loaded) per fit. The lab spent days on a score gap whose
  best-supported axis turned out to be the interpreter stack, not the
  algorithm.

## 9. Speed numbers

n≥3 with scatter, prompt length stated, one process per arm, never on a
contended box, never on a model larger than RAM. Quote a **ratio between
arms measured in the same session, never an absolute** — decode throughput
at ~100 GiB residency was measured to be bimodal on the same box, same
artifact, back to back. Load time is path-dependent (local vs network) and
belongs in no benchmark table.

## 9b. Contention is not only a timing hazard

The obvious reason not to share a box mid-experiment is that timings become
meaningless. There is a second reason, and it is unmeasured rather than
ruled out: **contention can perturb GPU reduction order, and reduction order
is a known source of numerical nondeterminism here.** With the RNG fully
pinned, two fits still differ at ~0.0100% of codes from that residual alone;
whether it is large enough to move an output metric has never been measured.
So a fit that ran under contention may differ from one that did not by a
term nobody can currently bound.

Practical rule: **run compute one experiment at a time on a box, and record
whether anything else was running.** If an arm ran contended and its twin did
not, either re-run the contended arm clean or report the comparison as
carrying an unbounded contamination term. Do not reason that "contention only
affects timing" — that is the incomplete version of this rule, and it is the
one most people carry.

## 10. Every gate must fail before it is trusted

A new gate must FAIL on a known-bad input and PASS on a known-good one
before its pass means anything. A gate that silently skips its checks is the
same failure mode — check that the gate actually ran what it claims (one
acceptance suite here printed "n/a" for every check on its first run and
looked green).

## MLX-specific engineering rules (each cost ≥1 run to learn)

- **Threadgroup ceiling:** fused kernels that cache the codebook in
  threadgroup memory obey `K * dim * 2 < 32768` strictly (d4 safe to K2048,
  d2 to K4096; `xs[]` shares the budget). Over the cap, Metal reports
  `XPC_ERROR_CONNECTION_INTERRUPTED` — it is an over-allocation, not a
  compiler fault. Larger codebooks must use device-memory kernels (the
  shipped d8 and post-E135 d4 paths). Compute `K*dim*2` first.
- **Lazy-read hazard:** any script that `mx.load`s and later saves without
  an eval in between can materialize reads inside a GPU command buffer,
  where under memory pressure they can return zeros. Load under
  `with mx.stream(mx.cpu):` with `mx.eval` *inside* the block, assert no
  tensor is all-zero, then re-read what you wrote.
- Fits resume: never `rm -rf` a fit output dir, never write in place, and
  never edit a script a running chain has not yet invoked.
- Dense and MoE are different runtimes; a smoke on one path says nothing
  about the other. A dense artifact's bundle must carry BOTH `vq_switch.py`
  and `vq_dense.py` — the dense fused path calls `_dense_fused`, which lives
  in vq_switch. Measured: a bundle carrying vq_dense alone raises
  ModuleNotFoundError on a stock mlx-lm — it scores fine and cannot serve.
  `check-bundle` gates this.
- Healthy relative error scales with K (K2048-class ~0.19, K256 ~0.31,
  K128 ~0.46). Set `--relerr-abort` per geometry.
- Never register a duration derived from a probe: probes time the cheap
  half. A duration is measured only from a completed run of the same shape.
