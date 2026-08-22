# STATE (2026-08-21 ~18:00) — mechanism night

## WHERE THINGS STAND

**Tonight's headline: the 08-18 fitter mystery is SOLVED, and the obvious fix
for it FAILED.** Both are results; the second is a real negative, not a setback.

- **E107-E110: mechanism identified.** `kmeans++` seeding (commit 689e03c,
  labelled "robustness, NOT a quality lever") buys the bulk by selling the
  tail. Depth-structured: uniformly BETTER below L15, sells the tail on 18/24
  body tensors from L20, `down_proj` body 8/8. Explained by geometry (E110):
  shallow layers are heavy-tailed (excess kurtosis +1.25) so distance-
  proportional seeding lands ON the tail; body layers are SUB-Gaussian (-0.38)
  so it merely covers the bulk better. Body is 8.81 GiB/bit vs shallow's 1.87,
  so the body verdict is the artifact verdict — which is why K256 regressed
  while K2048/K8192 won. **Mean relerr across body tensors moves -0.00033: the
  gate is blind to all of it.**
- **E112: FALSIFIED.** Body-only magnitude-weighted k-means (p=4, from L20)
  scored **2.9945 / 2.6442** against the K256 refit's 2.8057 / 2.6447 at
  byte-identical 111.617 GiB. Wikitext +0.1888 WORSE — 4.7x the regression it
  was built to fix. Reading rule was pre-registered (d4f8da8) before any number
  existed; "code held steady" was the available spin and the pre-registration
  stopped it. **Kills "the bulk/tail trade is recoverable by reweighting the
  objective."** Strengthens laws 11-12: we engineered the exact error band
  E102 identified, succeeded in weight space, and made the model worse.
- **E113: packed d8 kernel WORKS.** Bit-identical to the unpacked d8 kernel on
  synthetic data, and byte-identical greedy text on the real artifact
  ('Paris.\nA. True\nB' both ways). Unlocks 3.0591/2.6728 @ ~101 GiB vs
  shipped 2.2's 3.1706. **SPEED UNMEASURED** — Noah saw ~17% decode tax but
  that run is disqualified twice (contended share; mismatched runtimes between
  arms). E83's device-memory-codebook warning stands unrefuted.
- **E114 (running, M3): dense 27B refit.** The first dense artifact was
  DEFECTIVE — L60 up_proj had codebook, codes AND scales all exactly zero
  against a healthy source. Fitter printed relerr 1.0000 and carried on
  because `fit_dense_vq.py` had NO abort. Fixed (129f66d). Refitting on M3 per
  standing policy (M4: 4 collapse incidents; M3: 0).

## OPEN DECISIONS — NOAH'S, NOT OURS

1. **The HuggingFace fetch.** M4's local copy of the published 2.2bpw has a
   695-line bundled `model.py`; the share copy has 717. Same weights, same
   index, same config. Fetching the published `model.py` from HF and md5-ing
   it against both would settle whether **downloaders are running different
   code than our benches** (E81 in its published form). Neither session pulled
   it — it reaches an external service and the answer could imply a
   published-artifact correction. Blocks nothing.
2. **Whether any published artifact needs re-splicing.** Downstream of (1).
   All three published bundles are older than HEAD, which is EXPECTED (they
   bundle the runtime they shipped with; line counts track publication dates).
   That is not a defect. The 695-vs-717 discrepancy is.

## NEXT WORK, IN ORDER

- **Gate + score the dense 27B refit** (M3, ~15 min after fit). The
  does-this-generalise-past-MoE question. Compare against the q2/q3/q4 ladder
  at its ACTUAL size (9.61 GiB, between q2's 7.83 and q3's 10.96). NOT
  size-matched to q4 — the readings are the ablation vs q4 and the ladder
  placement, never "VQ beats 4-bit".
- **d8 speed A/B, properly** (M4). Needs a quiet share and matched runtimes on
  both arms. Parity within ~5% means the 101 GiB rung improves at no speed
  cost; below 0.75x is a measured tradeoff and cheap-shallow stays the product.
- **Peer's SMB round-trip test.** Write a large payload M4 -> share, read back,
  compare hashes. Decides the WRITE-time corruption shape (gemma d2-K512: clean
  fit log, corrupt disk). Does NOT explain E95, which was compute-time.
- **NOT recommended: routing M4 fits through local disk.** A permanent tax on
  every fit, supported by one incident and contradicted by another. Run the
  round-trip test first.

## THE TWO FAILURE SHAPES (keep these separate)

- **Compute-time:** fit log SHOWS the bad value (E95 L60: 1.0000, printed).
  `--relerr-abort` catches these.
- **Write-time:** fit log is CLEAN and the disk bytes are not (gemma d2-K512).
  The abort is structurally blind. **Only the post-hoc outlier gate, on a
  different box, covers both.** Which one fires is diagnostic.

## DOCS

- `FINDINGS.md` — laws, through E110. Laws 11-13 are tonight's.
- `PROCESS.md` — NEW. Family-onboarding pass (profile geometry, sweep init,
  THEN fit), the new-fitter guard list, box policy, and the standing gates.
  This is the artifact Noah asked for toward shaping MoEMash practically.
- `EXPERIMENTS.md` — chronology through E114.

## FIXES THAT LANDED TODAY AND NOW FIRE AUTOMATICALLY

vision_config copied by default (graft_vision); packed index `total_size`
recomputed from the packed shards (pack_artifact, was declaring the UNPACKED
size — 37-61% overstatements that made exo refuse to place the flagship); dense
code width chosen from K (uint16 at K256 doubled an artifact); packed d8
dispatch; row-width validation on the fused packed path; `--relerr-abort` on
the dense fitter; verify_artifact reads dense artifacts (vq_linear key, 2D
codes, qwen3_8_dense family).
