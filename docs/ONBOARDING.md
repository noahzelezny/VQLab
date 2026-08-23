# Onboarding a new model family — do this BEFORE fitting anything

This is a two-hour, mechanical characterisation pass. The lab spent weeks on
a fitter regression whose cause (a seeding choice interacting with layer
depth) could have been front-loaded into this pass. It tells you which knobs
matter for THIS family before any codebook is fit.

## 0. Establish the instrument BEFORE the artifact

- Build a teacher cache (`vqlab score --mode cache ...` / `kl_damage.py
  cache`) and **confirm it is deterministic** — score the same artifact
  twice. A family whose raw scores are non-deterministic is a family you
  cannot rank quants in; find that out now, not after four fits. (This is
  how the gemma-4 family was excluded.)
- Identify the comparator set and record HOW each was produced. A locally
  converted 4-bit and a downloaded 4-bit are different instruments; name
  which.
- Verify the outlier gate (`vqlab verify`) can parse the family's tensor
  layout. Adding a family entry is minutes; discovering the gate cannot read
  your artifacts after you have twenty of them is a day.

## 1. Profile the weight geometry (no fitting)

For a sample of layers spanning depth x every projection:

- **Distribution shape per layer** — fraction of weight energy in the top 1%
  of |w|, kurtosis, near-zero fraction. Shallow layers are routinely a
  different animal from body layers, measurably so in minutes.
- **Where the bytes are** — GiB per bit by layer band. On the 397B the body
  (L10–56) is 8.81 GiB/bit against the shallow band's 1.87, so a per-tensor
  effect that only hits the body dominates the artifact. Know this ratio
  before you reason about any per-tensor result.

## 2. Run the init sweep (`probe_init_sweep.py`)

Layers x all three projections, K at the low end of your intended range,
init as the only variable, evaluated on HELD-OUT experts, ≥2 seeds, verdict
requiring both tail buckets to move by more than the run-to-run spread. In
~30 minutes it answers, per family:

- Does k-means++ seeding help or hurt, and **where**? On the 397B it is
  uniformly better below L15 and sells the tail on 18 of 24 body tensors —
  a flip invisible to mean relative error.
- Is the effect depth-structured, projection-structured, or neither? Three
  tensors is not enough; sampling across an unknown boundary reads as
  "unreliable effect".
- **Which K regime is safe.** The seeding penalty shrinks or reverses as K
  rises. If the target bpw forces low K, seeding is a quality lever; at
  large K it is free.

## 3. Only then fit

Carry into the fit: which init, which K band, and whether shallow and body
layers want different treatment. If the sweep says the family is
depth-split, a flat recipe leaves quality on the table and a per-band recipe
is the first thing to try.

## Writing a new fitter? Start from the guard list

A fitter generalised by copying a working fit once silently dropped the
relative-error abort, and shipped a tensor whose codebook, codes and scales
were all exactly zero — computed relerr 1.0000, printed, not checked.
Minimum guard list:

- `--relerr-abort` with a sane per-geometry default, and a
  refit-then-fail path (healthy relerr scales with K: ~0.19 at K2048-class,
  ~0.31 at K256, ~0.46 at K128 — tune per geometry, not globally)
- per-tensor relerr printed AND checked (printing is not checking)
- code dtype chosen from K, never hardcoded (uint8 at K≤256, else uint16)
- output written to a NEW directory, never in place (fits resume)
- index `metadata.total_size` computed from what was written, never copied
- source reads under `with mx.stream(mx.cpu):` with `mx.eval` inside the
  block, then an all-zero assertion (the deferred-read hazard —
  METHODOLOGY.md)
- record the resolved interpreter stack (python, mlx version, the actual
  `.so` path) in the fit log
