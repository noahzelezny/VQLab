# PROCESS — what to do with a NEW model family, before fitting anything

Written 2026-08-21 after E107-E109. The lab spent weeks on a fitter regression
whose cause was a seeding choice interacting with layer depth, and every step
of that could have been front-loaded into a two-hour characterisation pass. Do
that pass first. It is cheap, it is mechanical, and it tells you which knobs
matter for THIS family before any codebook is fit.

Related: FINDINGS.md holds the laws; EXPERIMENTS.md holds the chronology. This
file is the recipe.

---

## 0. Establish the instrument BEFORE the artifact

- Build a teacher cache (`kl_damage.py cache`) and confirm it is deterministic.
  A family whose raw scores are non-deterministic is a family you cannot rank
  quants in — find that out now, not after four fits. [gemma-4-e4b, FINDINGS II]
- Identify the comparator set and record HOW each was produced. A locally
  converted 4-bit and a downloaded 4-bit are different instruments; name which.
- Verify `verify_artifact.py` can parse the family's tensor layout. Adding a
  family entry is minutes; discovering the gate cannot read your artifacts
  after you have twenty of them is a day.

## 1. Profile the weight geometry (no fitting)

For a sample of layers spanning depth x every projection:

- **Distribution shape per layer.** Fraction of weight energy in the top 1% of
  |w|, kurtosis, fraction of near-zero weights. Shallow layers are routinely a
  different animal from body layers and the difference is measurable in
  minutes.
- **Where the bytes are.** GiB per bit by layer band. On the 397B the body
  (L10-56) is 8.81 GiB/bit against the shallow band's 1.87 — so a per-tensor
  effect that only hits the body dominates the artifact, and one that only hits
  shallow layers barely registers. Know this ratio before you reason about any
  per-tensor result.

## 2. Run the INIT SWEEP (`probe_init_sweep.py`) — the pass this file exists for

Layers x all three projections, K at the low end of your intended range, init
as the only variable, evaluated on HELD-OUT experts, >=2 seeds, verdict
requiring both tail buckets to move by more than the run-to-run spread.

It answers, per family, in ~30 minutes:

- Does k-means++ seeding help or hurt, and **where**? On the 397B it is
  uniformly better below L15 and sells the tail on 18 of 24 body tensors
  (E109). That flip is invisible to mean relerr (delta -0.00033 across the
  body), so no gate we own would ever surface it.
- Is the answer depth-structured, projection-structured, or neither? Sample
  enough of both axes to tell them apart — three tensors is not enough, and
  sampling across an unknown boundary reads as "unreliable effect" (E108).
- **Which K regime is safe.** The penalty shrinks or reverses as K rises. If
  the target bpw forces low K, seeding choice is a quality lever; if K is
  large, it is free.

## 3. Only then fit

Carry into the fit: which init, which K band, and whether shallow and body
layers want different treatment. If the sweep says the family is depth-split,
a flat recipe is leaving quality on the table and a per-band recipe is the
first thing to try.

---

## Writing a new fitter

**Start from the guard list, not from a working fit.** `fit_dense_vq.py` was
generalised from `fit_e4b_vq.py` by copying its structure — and silently did
not inherit `--relerr-abort`, which `vq_397b_codes.py` had carried for weeks.
Consequence, measured 2026-08-21: the dense 27B shipped one tensor whose
codebook, codes and scales were all exactly zero against a healthy source. The
fitter computed relerr 1.0000, printed it, and carried on. It was invisible
until the outlier gate learned to read dense artifacts hours later.

Minimum guard list for any new fitter:
- `--relerr-abort` with a sane default, and a refit-then-fail path
- per-tensor relerr printed AND checked (printing is not checking)
- code dtype chosen from K, never hardcoded (uint8 at K<=256, else uint16)
- output written to a NEW directory, never in place (fits resume)
- index `metadata.total_size` computed from what was written, never copied

## Box policy

**M4 has produced four known tensor-collapse incidents; M3 has produced zero.**
08-15: qwen tail30 (5 tensors, one at relerr 1.0000), gemma d2-K512 (4 tensors
0.54-0.99, with a HEALTHY fit log), qwen d2-K64 (3 tensors >0.5). 08-21: the
dense 27B L60 up_proj at relerr 1.0000. The gemma case is the important one —
the fit log read clean while the WRITTEN artifact was corrupt — so the abort is
not sufficient and the post-hoc gate is the real control.

Standing policy, unchanged since 08-15: **every M4-fitted artifact is verified
on M3 before any number from it is believed**, and a repair refit goes to M3.

**The abort does NOT make verification optional, and the two failure shapes are
different.** `--relerr-abort` reads the fitter's COMPUTED relerr, so it catches
a compute-time collapse (E95: the fitter printed 1.0000 itself) and CANNOT
catch a write-time corruption (gemma d2-K512: fit log reported worst 0.0611
while the WRITTEN artifact held 4 tensors at 0.54-0.99). **Only the post-hoc
outlier gate, run on a different box against the bytes on disk, covers both.**
Distinguishing them is diagnostic: if the fit log shows the bad value, the
failure is in compute; if the log is clean and the artifact is not, the failure
is between memory and disk.

**UPDATED 08-21: "between memory and disk" has a named third shape, and it is
not a write at all — it is a DEFERRED READ.** MLX `mx.load` is lazy; a value
left unevaluated until a later save is materialized inside a GPU command
buffer, and under memory pressure that read can return ZEROS. Signature: the
producing log is clean, the consumer writes all-zero tensors, an eager re-read
of the source is fine, and a rebuild from identical inputs succeeds — the
transience reads as a storage fault and is not one. Seen twice on 08-21, on
two different code paths and both transports (build_dense_vq.py splice, local
SSD; fit_dense_vq.py source read, SMB), so the transport is incidental.
**Fix at the read, do not build workarounds around the gate:** load under
`with mx.stream(mx.cpu):` with `mx.eval` INSIDE the block (creation-binding
alone is measured-insufficient), then assert no tensor is all-zero, then
re-read what you wrote. Keep the post-hoc gate as defence in depth — it is no
longer the only catcher, but it is still the only one that runs on a
different box against the bytes on disk.

## Standing gates, once artifacts exist

1. `verify_artifact.py --outlier 3.0` — relative, catches collapsed tensors.
   It does NOT catch a uniformly worse fit, and mean relerr is a BULK
   statistic that is blind (E98) or anti-correlated (E101) to output damage at
   low K. **Add a tail statistic** (normalized error in the 99-100th |w|
   percentile) if you care about low-K artifacts.
2. `pack_artifact.py` then **re-check the declared size** — the index's
   `metadata.total_size` must be recomputed from the packed shards, never
   copied (E104). Derive a size from the bytes, never from a field that says
   what the bytes are.
3. `graft_vision.py` with `--copy-config-keys` (now default ON).
4. **Generate one token through the fused path the artifact will ship with**
   (III.10). Every byte-level gate we own passed an artifact that could not
   run (E100). The referee scores through the REFERENCE decode path and is
   structurally blind to this.
5. Score on the family's own instrument, and state which one.
