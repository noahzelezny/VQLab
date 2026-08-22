# STATE (2026-08-21 ~18:35) — mechanism night

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
  shipped 2.2's 3.1706.
- **E115: speed MEASURED — ~19% decode tax.** Clean-mode decode@2048 17.24 vs
  21.22 tok/s, ratio **0.812**, matched runtimes (md5-identical model.py),
  fan on the box, corroborated by adjacent-run pairings (0.817/0.837).
  **Confirms E83's device-memory-codebook warning** — d8-K16384's 1 MB
  codebook cannot sit in threadgroup memory. So the 101 GiB rung is a real
  quality win at a real speed cost, and which artifact belongs there is Noah's
  call. **Instrument caveat: decode at ~100 GiB is BIMODAL** — the same
  artifact ran 21.14 / 12.69 / 21.27 / 21.20. Swap (swapouts never moved off
  1,343,843), thermal, and storage path are all ruled out by measurement.
  Quote ratios from one session, never absolutes, never n=1.
- **E116: SMB write path CLEAN.** Five real shards, 16 GiB, md5 local -> share
  -> back, 0 mismatches. Exonerates the write path on this sample; does NOT
  exonerate the box and does NOT explain E95 (compute-time, proven from its own
  fit log). Kills "route M4 fits through local disk" — the last reason to pay
  that permanent tax is gone. The post-hoc outlier gate stays mandatory.
- **E114 (running, M3): dense 27B refit.** The first dense artifact was
  DEFECTIVE — L60 up_proj had codebook, codes AND scales all exactly zero
  against a healthy source. Fitter printed relerr 1.0000 and carried on
  because `fit_dense_vq.py` had NO abort. Fixed (129f66d). Refitting on M3 per
  standing policy (M4: 4 collapse incidents; M3: 0).

## OPEN DECISIONS — NOAH'S, NOT OURS

1. **[RESOLVED 08-21 19:00, Noah authorized the fetch]** Published model.py
   fetched from all three HF repos: 2.2/3.1 are md5-identical to our 717-line
   copies, 2.4 to our 1093-line copy. The "695-line local copy" does not
   exist anywhere on the volume — downloaders run exactly our code. Nothing
   needs correction. Noah leans toward pushing the tested 1093 runtime to the
   2.2/3.1 repos (it is a superset, one added dense path); needs a III.10
   smoke on each repo's artifact with the 1093 bundle first. NOT pushed.
1b. **The original HuggingFace fetch question.** M4's local copy of the published 2.2bpw has a
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
- **Score E94** (35B-K8192 refresh, fit done 12:01, 17.7 GiB on the Thunderbay
  SSD, 120 tensors mean relerr 0.1323). Waiting on the M3.
- **Understand the bimodal decode instrument** before any speed number goes on
  any card. Every published card's speed table was produced by the method E115
  just showed to be unreliable. This is a hole in the instrument, not a defect
  in any artifact — but it blocks speed claims.
- **DONE, was pending here:** d8 speed A/B (E115) and the SMB round-trip
  (E116). "Route M4 fits through local disk" is dead — E116 removed the
  remaining reason.

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
- `EXPERIMENTS.md` — chronology through E118 (E117/E118 pre-registered, running).

## FIXES THAT LANDED TODAY AND NOW FIRE AUTOMATICALLY

vision_config copied by default (graft_vision); packed index `total_size`
recomputed from the packed shards (pack_artifact, was declaring the UNPACKED
size — 37-61% overstatements that made exo refuse to place the flagship); dense
code width chosen from K (uint16 at K256 doubled an artifact); packed d8
dispatch; row-width validation on the fused packed path; `--relerr-abort` on
the dense fitter; verify_artifact reads dense artifacts (vq_linear key, 2D
codes, qwen3_8_dense family).


## NIGHT OF 08-21 (M3 session, Noah asleep) — running ship

- **E95 CONCLUDED: dense VQ carries — not MoE-only.** 27B dense, flat
  d4/K256: 325.6 mnats / 76.5% top-1 / ppl 6.403 @ 9.7 GiB. Above the affine
  q2->q3 line at its size (predicts ~439 mnats). Gate PASS, III.10 smoke
  PASS. Paper session unblocked. Full entry + two runtime defects III.10
  caught (d!=2 dense dispatch; expert kernel exceeds threadgroup memory at
  27B shapes) in EXPERIMENTS E95 RESULT.
- **L60 write-corruption recurred then vanished:** first splice zeroed L60
  up_proj (all three tensors) with a CLEAN fit file — pure write-time.
  Rebuild from identical inputs: clean, 0 zeros. Intermittent. Note the
  peer's E116 found the SMB write path clean on a 16 GiB round-trip; both
  stand — E116 says the path is not ALWAYS bad, the L60 pair says it is
  SOMETIMES bad. The post-hoc gate is the only defense; it worked twice.
- **E117 running on M3** (K256 random-init, chain self-gates/packs/scores;
  logs say E115 — numbering collision, see EXPERIMENTS E117 note).
- **E118 armed** behind it via run_night_queue.sh (K512 random; logs say
  E116). Watcher launches only on a clean E117 DONE banner.
- **HF fetch RESOLVED** (Noah authorized): published model.py md5-identical
  to our copies on all three repos; no 695-line file exists; no correction
  needed. Pushing the 1093 runtime to 2.2/3.1 awaits a III.10 smoke per repo
  and Noah's go.
