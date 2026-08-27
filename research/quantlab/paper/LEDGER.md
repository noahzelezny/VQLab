# LEDGER — current truth, single source (rebuilt 08-22)

**This file is the arbiter.** It states what is true NOW; it carries no
history. If a number here disagrees with EXPERIMENTS.md, the newer committed
E-entry wins and this file gets fixed the same day. Superseded ledger content
is archived in LEDGER_archive_0822.md; the paper cites nothing from the
archive.

Reading rules baked into every table below:
- packed whole-artifact bytes, post-graft (397B) / packed (dense, 6e), MEASURED
- one instrument per family; no cross-instrument rows
- every ppl/KL margin is stated against the seed-noise floor for its geometry
  (III.12); a margin inside its floor is NOT a claim
- a row's size and quality must come from the SAME artifact (III.2); when they
  come from different directories, the row says so and goes on hold

## SEED-NOISE FLOORS (the lens on every margin)

| geometry | n | floor | status |
|---|---|---|---|
| 397B d4/K256 | 2 | wikitext **0.0256** / code ~0.0178 | MEASURED, same-stack [E136/E136b] |
| 397B d4/K2048 | 2 | wikitext **0.0056** / code **0.0104** | MEASURED [E142-397B] |
| 35B d4-d2 (at d2/K1024) | 2 | KL **0.214 mnats** | MEASURED, same-box [E140b] |
| dense 27B d2/K256 | 3 | KL 2.085 mnats / ppl 0.0447 | MEASURED [E127/6f] |
| all other geometries | — | UNMEASURED — do not inherit | III.12 |

Floors NARROW as K grows (0.0256 @ K256 -> 0.0056 @ K2048). The superseded
0.0134/0.0161 INFERRED pair must not be used: every margin computed against it
reads ~2x too favourable. Rows below quote the floor they were judged on.

Practical rule: KL separations of 5+ mnats and top-1 of ~1 pp are real;
third-decimal ppl between single-draw artifacts is not interpretable.

---

## CURRENT BEST ARTIFACTS AND THEIR TRUE COMPARISONS

### 397B (referee prose/code ppl; floor depends on geometry — see table above)

| rung | GiB | prose | code | comparison that holds |
|---|---|---|---|---|
| **d8-K16384 (PUBLISHED as VQ-2.2bpw)** | 100.97 | 3.0591 | 2.6728 | beats spicy 2.6bit (3.1843 @ 120.6) by 0.1252 prose = **4.9x the K256 floor** (no d8 floor measured; borrowed, so read as a lower bound on confidence), at **19.6 GiB smaller**; beats rate-twin flat-K128 by 0.1115 = 4.4x the K256 floor (borrowed). Cost: ~19% decode vs K128 (E115). Score reproduced exactly after runtime swap (E122). |
| flat K128 (old 2.2) | 100.93 | 3.1706 | 2.6988 | ladder point only — margin vs spicy = 0.5x the K256 floor, NOT a quality claim |
| harvest K64/K256 | 107.9 | 2.7790 | 2.6479 | ladder point |
| shipped 2.4 (flat K256) | 111.617 | 2.7655 | 2.6383 | daily driver; **unreproducible by construction** (base rewritten Aug 19, E121) |
| **flat K512 (E93)** | 122.305 | 2.5634 | 2.6123 | **CLAIM 1 LEAD ROW: beats spicy 2.6bit by 0.6209 prose = 24x the K256 floor AND 0.0544 code = 3.1x the K256 code floor (borrowed; no K512 floor measured), at +1.7 GiB** |
| harvest K512/K2048 | 139.93 | 2.3452 | 2.5969 | best-per-GiB |
| **flat K2048 refit (flagship)** | 143.682 | 2.3410 | 2.5963 | vs spicy 3.5bit (2.3614/2.6005 @ 165.6): **SIZE claim — 21.9 GiB smaller; prose BETTER (0.0204 = 3.6x the K2048 floor, CLAIMED), code TIE (0.0042 = 0.4x)** — "wins both corpora" is withdrawn [E142-397B]. vs shipped 3.1: 0.0109 = **1.9x the K2048 floor** (0.0056), not claimable (bar is 3x). Prior readings of 0.8x/0.4x used the K256 floor — a III.12 violation, corrected 08-24. |

Latest scores throughout; nothing stale. The flagship IS our best 144G build;
what the floor forbids is claiming its thin margins as quality wins.

### 35B MoE (kl_cache_qwen36; floor unmeasured)

| rung | GiB (PACKED) | KL mnats | top-1 | comparison |
|---|---|---|---|---|
| **e94b d4/K8192** | **14.838 MEASURED** (packed artifact on disk, index total_size matches bytes; unpacked twin 17.651) | 53.022 **[HOLD — split provenance]** | 89.55% | **ROW ON HOLD (III.2): size is from the PACKED artifact, KL from the UNPACKED one — different bytes, and the packed copy has had no gate and no smoke. Packed e94b is queued for gate+smoke+score on the same instrument tonight; if it reproduces 53.022 the row goes clean at "−32% KL, 4.16 GiB smaller." Do not publish the pairing before then.** |
| R2 cand. d4/K16384 | 15.783 measured packed | pending | — | 0.945 GiB LARGER than K8192 packed (14 vs 13 bits — ordering matches physics) |
| mlx 8-bit | 35.131 | 7.449 | 96.18% | the R3 bar |

35B size rules: quoting 17.651 as a size is a III.8 violation (it sat in the
E128 ladder until 08-22; corrected, two-session verified). Non-MLP carry =
**1.706 GiB MEASURED-ONCE** (replaces the derived 4.53 — off by 2.82; every
older 35B projection shifts DOWN by 2.82). Label stays measured-once until a
second build agrees (the 27B carry was wrong by 1.5 until three builds).

### dense 27B (kl_cache_qwen38 + ppl; floor 2.085 mnats / 0.0447 ppl at d2/K256)

| rung | GiB | KL | top-1 | ppl | comparison |
|---|---|---|---|---|---|
| E124 d2/K256 | 13.596 | 40.327 | 90.10% | 5.2330 | **R1 MET**: beats q4 KL+top-1 at 0.50 GiB smaller; ppl inside floor, not claimed |
| **E126 d2/K512** | 14.592 | 33.095 | 91.10% | 5.1943 | **R2 MET***: −27.8% KL = 6.1x floor, +1.28 pp, at "q4-class size" (+3.5%); **ppl claim withdrawn** (inside floor) |
| E128C d2/K4096 | 17.583 | 26.709 | 91.66% | 5.2417 | R3 probe — see verdict below |
| q4 (local affine) | 14.094 | 45.842 | 89.82% | 5.2055 | comparator (LOCAL conversion, not community) |
| q8 | 26.341 | 1.641 | 98.08% | 5.2433 | comparator |
| E119 d4/K1024 | 10.609 | 148.470 | 82.53% | 5.5249 | beats q3 (187.765 @ 10.963) on both at 0.35 GiB less |

**R3 verdict (27B): NOT REACHABLE.** Measured slope flattens (x0.868/bpw above
4.5 bpw); reaching q8's 1.641 needs ~+20 bpw. No rate reaches 8-bit-class KL
on the 27B; 35B R3 not attempted (zero d2 points to extrapolate from). This is
a MEASURED boundary of the method and the paper reports it as such.

---

## THE THREE CLAIMS — current standing

### CLAIM 1 — Data-free VQ beats calibrated/uniform affine at matched-or-smaller bytes (below 8-bit; fence decision pending)
**STRONG.** Lead evidence, in order: K512 vs spicy 2.6bit (46x floor, both
corpora); d8 vs spicy 2.6bit (9.3x floor, 19.6 GiB smaller); 35B e94b vs 4-bit
(−32% KL, smaller); dense E126 vs q4 (6.1x floor KL, +1.28 pp); dense E119 vs
q3 (both metrics, smaller). Flagship contributes the SIZE claim at the top.
Fences: 8-bit is the wall (measured 3 families: 397B n/a, 35B 7.449, 27B
1.641 — R3 unreachable); prefill ~0.5x affine at 35B; code corpus decisive
only at K512.

### CLAIM 2 — Size-targeted quantization with graceful, priced degradation
**STRONG, two-family.** 397B harvest model: 6 hits / 1 in-band / 0 misses.
Dense 27B composition model CLOSED (6e): three builds, two geometries, err
<= 0.003. Exchange rates measured at 3 base richnesses (0.0315 -> 0.0011
ppl/GiB). Fence: harvest never beats flat at flat's own size; harvest is
MoE-only (no dense harvest rung exists).

### CLAIM 3 — Weight-space fit error cannot steer quantizer design
**ONE designed specimen, solid; scope now bounded.** E112: engineered
tail-recovery succeeded in weight space, model 4.7x worse — effect 14x the
397B floor. E127 (dense, fine-fit regime) came back TRACKS: law 6 BITES where
centroids are scarce (relerr ~0.3) and does not where they are not (~0.08).
Supporting: E109/E110 per-tensor mechanism (stands as physics, NOT the
artifact-level explanation); E98 aggregate-relerr blindness. E101 is RETIRED
(base-rewrite confound + 3x-floor margin). Do not cite E101, E82, or any
gemma number.

---

## RETIRED / DO-NOT-CITE (one line each, reason attached)

E101 (confounded + thin) · E82 3.3x (corrupt arm; real d2/d4 gap 6–12%,
E87/E99) · E94 row (overwritten; use e94b) · E121 as vintage test (base
rewrite) · "fitter vintage" as a phrase (4 explanations dead; H1 seed lottery /
H2 box / H3 lost command line remain, E129) · 6c KL/ppl inversion (n=1,
demoted) · E126 ppl margin (inside floor) · "wins both corpora" for flagship
(code 0.26x floor) · e4b everything (not a true MoE; excluded) · gemma
everything (non-deterministic instrument) · "d beats K at this budget"
(retracted; E130 half-run, UNANSWERED) · flat-K128 vs spicy quality claim
(1.0x floor).

## OPEN / IN FLIGHT

- E128 offering ladder: 27B R1+R2 MET, R3 unreachable (measured slope); 35B R1 MET (e94b), R2 candidate d4/K16384 packed 15.783 awaiting KL, R3 not attempted.
- d4/K65536 rate twin cost re-measured: ~37 h (3-tensor probe, 0.9% spread; cost model now 3-for-3 at ~2e-3 accuracy).
- Packed e94b: gate + smoke + KL/ppl queued on M3 behind E130 arm 2 (numbers tonight) — unblocks the 35B claim-1 row.
- E130 d-vs-K rate twin: arm 1 (d2/K64) SCORED — 11.604 GiB, KL 93.887, ppl 5.3494 (halves q3's KL at +0.64 GiB). Arm 2 (d4/K4096) fitting, ~3h to the d-vs-K verdict.
- Vintage gap: H1/H2/H3 open; shipped 2.4 unreproducible by construction.
- 397B floor is n=2 INFERRED; a measured n>=3 floor would firm every 397B
  margin statement.

## DECISIONS PENDING (Noah)

1. **Fence width — DECIDED (Noah, 08-22): state what the data supports,
   open-ended above.** Measured wins run 2.0–4.5 bpw (E126 d2/K512 vs q4 is
   the highest matched-class win; 35B 3.25 bpw; 397B ~2–3 bpw). Above 4.5
   there is NO affine comparator at matched rate (E128C's 6.0 bpw rung has
   nothing its size to face), so the fence reads: "measured at 2–4.5 bpw;
   unmeasured 4.5–8 for lack of comparators, not for lack of confidence;
   by 8 bits affine is essentially lossless and the advantage is gone
   (measured wall, two families)." UNBLOCKS the abstract.
2. **Title — DECIDED: no MoE scoping** (dense carries). Claim-as-title.
3. Second epigraph — source or cut.
4. chmod -R a-w on published artifacts (manifest landed, 61473e7).
5. Second dense family, or state "one dense model" as a limitation.

## E134 STATUS (08-22 evening) — threadgroup boundary MEASURED; published artifacts all safe

Boundary exact, four sibling artifacts on M4: threadgroup path safe while
K*dim*2 < 32,768 (d4: K<=2048; d2: K<=4096). K4096-d4 fails AT the cap
(kernel needs space beyond the codebook). XPC_ERROR_CONNECTION_INTERRUPTED
is how Metal reports this over-allocation — not a compiler-service issue.

**NO PUBLISHED ARTIFACT AFFECTED — but for TWO DIFFERENT REASONS (config-
verified by the M4 session, not say-so):**
- 2.4bpw (d4/K256, 2 KB), 3bpw (d4/K2048, 16 KB), 35B K256 (2 KB): safe by
  cap arithmetic, smoked.
- **2.2bpw LIVE (d8/K16384): codebook is 262,144 B — 8x OVER the cap.** Safe
  because the d8 kernels keep the codebook in DEVICE memory by design;
  verified by direct clean-room III.11 smoke 08-22, NOT by cap arithmetic.
  Do not derive "under 32768 = safe" as a general audit rule — the flagship
  d8 build is the artifact that breaks it.
- Hygiene, not publish: the M4's local dir named `...VQ-2.2bpw` still holds
  v1 bytes (d4/K128). Published weights = `rotlab--397B-d8K16384-packed`.
  Auditing the published set from local directory NAMES reads the wrong
  artifact (same shape as the E94 name/bytes divergence).
Only the unreleased 35B R1/R2 candidates (K8192/K16384) are blocked. Monday
publish de-risked; 35B-row-absent plan undisturbed.

Open: prediction 1 (M3 smoke of K4096/K8192 — architectural vs box-specific).
Kernel port: awaiting Noah's DIRECT confirmation in the M4 session
(uds 39597 / "Run task-suite benchmarks..."). Regression bar for the fix:
K256/K2048 bit-identical AND K4096/K8192 FAIL->token (both signs, III.5).

## E134 RESOLVED (08-22 night, Noah-authorized in the M4 session)

Device-memory d4 codebook landed (ba0fad5): both 35B rungs GENERATE.
Acceptance met the full bar — bit-identical vs verified threadgroup kernels
where both load (8/8, unpacked+packed), float32-reference check where only
the new path loads (3.8e-4), III.11 on six re-spliced artifacts, regression
both signs, check_bundle PASS. Dispatch by exact allocation (_d4_tg_fits),
not a K threshold.

- Acceptance-suite note for §5: its FIRST run silently skipped every packed
  bit-identity check (printed "n/a"); caught because a results column was
  empty, fixed, re-run. A gate that silently skips is III.5's failure mode —
  the gate needed its own gate.
- Addendum 9e7fea1: dense guard had the same class of defect (budget ignored
  the codebook itself); preventive fix, no on-disk rung affected, but a
  future large-K dense rung must not smoke against a stale bundled
  vq_dense.py.
- PAPER FRAMING (theirs, adopted): the generalisable finding is that a rung
  can SCORE normally and be unable to SERVE — scoring path (prefill-shaped
  referee) and serving path (small-N fused kernel) are different code. Two
  release candidates reached consideration in that state.
- q6 comparator: authorized, building on M4 (local-disk after an IV.1 kill),
  scoring on M3.

35B row restore conditions now: 2 of 3 MET (kernel acceptance ✓, packed
smoke ✓). Remaining: packed e94b gate + score reproducing 53.022.

## VINTAGE ANOMALY — STATUS 08-22 NIGHT (§4.4's live thread)

Established tonight, in order:
1. Shipped fit wall-clock = 2h06 vs defaults reconstruction 1h15 (shard
   mtimes; rsync -a preserved them). The +51 min prompted the H3 revival.
2. Fingerprint stat 1 (code utilization, calibrated on E127's known-iters
   trio): shipped sits ON TOP of the M4 defaults draw (0.10x iters effect);
   the two defaults draws SPLIT BY BOX (1.47x). Saturation caveat noted.
3. ~/quantlab_m1 discovered on M4 — the actual Aug-15 working copy. Source
   diff vs the reconstruction fitter: k-means BYTE-IDENTICAL (0 hits on
   kmeans/sample/assign/centroid in 106 plumbing lines); defaults identical
   (iters=20, sample=2M). No document of a non-default command line exists
   (nohup-over-ssh writes no history). H3-by-arguments: NO support.
4. H4 strong form (mlx build changed) — NO support: mlx 0.32.0 both sides
   of the Aug-17 reinstall. **H4 weak form CONCRETE: every reconstruction
   ran Python 3.12 + mlx cp312; the Aug-15 fit almost certainly ran Python
   3.13 + mlx cp313 (both 3.13 venvs predate the fit; installed Aug 10).
   Nobody listed Python as a variable because the mlx VERSION matched.**

**DECISIVE EXPERIMENT (awaiting Noah's direct go in the M4 session, ~75 min):**
Aug-15 fitter file + defaults + M4 + ~/quantlab-m4/venv (Python 3.13/cp313).
Pre-registered readings (fixed here per the M4 session's request, BEFORE any
number exists):
- **<= 2.7800 wikitext** -> gap = interpreter/mlx-binary axis; shipped recipe
  RECOVERABLE; §4.4 closes WITH a cause.
- **~2.81 (E129 band)** -> Python/mlx axis excluded; every recoverable
  variable then verified identical; §4.4 closes as unexplained WITH the
  boundary fully named.
- Fingerprint the artifact either way -> doubles as the second same-box
  defaults draw, answering the same-config fingerprint-floor question (makes
  the plain replicate redundant).

Unexpectedly-useful lesson for §5 (M4 session's own words): twice tonight
"the cheap lookup came last" — a recommendation was built on a number
already in the record. The arbiter exists because everyone does this.

### 08-22 late: THE AUG-15 LAUNCH COMMAND RECOVERED (session transcript)

Found in the original session's transcript (project -Users-noahzelezny-
Documents-AgenicAI, session e66d73df, launch line 2915, 19:59:27Z):
- **Defaults confirmed by DOCUMENT — no --iters/--sample/--expert-chunk
  passed. H3 (arguments) is CLOSED.** The +51 min wall-clock excess is no
  longer attributable to arguments.
- **Venv was ~/vqvenv (Python 3.13.9, mlx 0.32.0 pinned same-day)** — a
  venv ABSENT from the M4 census. If it survives, the decisive fit can run
  under the literal original interpreter. M4 session notified mid-run.
- Matched: --stage-dir local staging (E129 matched this). Unmatched-until-
  now: multiple mlx_lm installs on the box, some VQ-patched; Aug-15 fit and
  score verified against a STOCK one (hook count 0).
- First appearance of total_nll 8332.9789: same session, 22:18:07Z, both
  corpora x2 runs, from quantlab_m1/.
Readings for the decisive fit unchanged. If the 3.13 fit reproduces ~2.77,
cause = interpreter stack; wall-clock of the new fit is itself a free
diagnostic (a ~2h 3.13 defaults fit reconciles the +51 min too).

### E136 (decisive fit) — stack verified while running

- **~/vqvenv SURVIVES.** E136's venv (~/quantlab-m4/venv) is BINARY-IDENTICAL
  on every axis the fit touches: mlx core .so md5 and full tree hash
  (incl. .metallib) identical, numpy 2.5.2 identical, Python 3.13.9 both.
  No restart needed — E136 is running the literal Aug-15 numerics.
- Patch-state variable CLOSED: vqvenv's mlx_lm is VQ-patched, but the fitter
  never imports mlx_lm (inference-path only). Irrelevant to the fit.
- H3 CLOSED BY DOCUMENT (adopted by the M4 session over its own
  fingerprint argument).
- Registered BEFORE the number: if E136's wall-clock lands near the shipped
  127 min (vs E129's 74), the timing excess was the stack; early pace
  (~70-min extrapolation at 4/27 shards) is explicitly NOT being read —
  the shipped run's excess sat in the tail shards.
- Named unmatched variable (their own flag, pre-result): E136 reads the
  BASE off SMB where Aug-15 read it from local APFS. Read path only; base
  bytes content-verified identical.

### 08-22/23 midnight: E136 result IN but PENDING; crossover bracketed; E130 closed

- **E136 = 2.7706 wikitext** — 0.0051 from the shipped 2.7655 (0.26x floor)
  under the binary-identical Aug-15 stack. **STATUS: PENDING, not a result**
  — n=1, floor mixed-provenance, code corpus 1.37x its floor (outside).
  Noah routing an n=3 same-stack replication; readings pre-registered in my
  message to the M3 session (cluster ~2.77 -> axis established / scatter ->
  draw). Wall-clock of E136 + replicates = free datum on the +51-min excess.
- **E130 CLOSED: d4 beats d2 at 3.00 bpw too** (85.823 vs 93.887 KL, 8.6%,
  rate twins 5 MB apart). Law 10 upgraded: two bands, same direction
  (~12%, ~8.6%); d4 rate ceiling 4.00 bpw ⇒ R1/R2 lever only.
- **q6 (E133) = 3.710 mnats / 96.75% @ 20.355 GiB — the VQ/affine CROSSOVER
  is bracketed 4.5–6.0 bpw on the dense 27B** (law 14, approved with dense-
  27B scoping in the law text + frontier phrasing). PAPER FENCE UPDATE:
  wins measured 2.0–4.5 (three families); crossover bracketed 4.5–6.0
  (dense 27B); 8-bit lossless everywhere measured. No claim-1 row changes.
- FINDINGS amendments: 1/4/5 approved as drafted; 2 approved w/ wording;
  3 (III.13) sent back — as drafted it contradicted E113's measured
  bundle-executed evidence; restated as "instrument the import, never
  assume which runtime runs."

### 08-22 ~21:30: 35B row — split provenance CLOSED; serving claim scoped out

- **packed e94b through venv-e134fix: 53.022 mnats / 89.55% reproduced to
  every printed digit; III.11 smoke PASS; 14.838 GiB MEASURED.** Registered
  branch fired. Row restores to the draft on outlier-gate PASS (running;
  III.9 has no vanishingly-unlikely clause). Text: quality claim ONLY.
- **III.13 mechanism RESOLVED, both halves measured:** the bundle executes
  as custom_model (utils.py:325), THEN on patched boxes the ndim==3 VQ hook
  overwrites expert modules from site-packages (utils.py:415). Stock boxes
  run the bundle (proven: d8 artifact's kernel exists only there). ⇒ we
  bench site-packages for MoE experts; downloaders run bundles. Scoring
  numbers unaffected (prefill-shaped; identical to 15 decimals). NOT
  established: bundled copies passing kernel acceptance AS the executing
  copy in a stock venv — REQUIRED before any 35B artifact publishes; not
  needed for the paper. §7 language updated accordingly.
- K65536 launch plan confirmed: 1-layer probe → FIXED pre-registered
  relerr-abort → full fit ~22:15, lands Mon ~11:00. Revision item.

### 08-22 ~21:45: 35B ROW RESTORED — all conditions met

Outlier gate PASS on the packed artifact (median 0.1325, bar 0.3975, 0
violations). Row in the draft: quality-only, 53.022/89.55% @ 14.838 packed
vs 78.557/85.61% @ 19.0. **Serving gap ALSO closed for BOTH 35B rungs**:
bundled copies passed E113 acceptance AS the unit under test lifted from
the artifacts, in a stock venv (bit-identical K256/K2048, 3.8e-4 vs
reference at K4096/K8192) + clean-room generation. §7 upgraded to the
three-way verification sentence; §5 gained the harness-imports-by-path
lesson (test the copy that ships). No 35B artifact release is blocked on
the runtime question any more.

REMAINING OPEN SLOTS IN THE DRAFT: E136b verdict (§4.4 ending, ~22:05) —
that is the last one. K65536 = revision item.

### 08-22 ~22:15: E136 WITHDRAWN — the anomaly dissolves into measured variance

E136b (same 3.13 stack) = **2.7962** — 0.0256 from its sibling draw 2.7706.
Neither registered branch fired; the between case kills the axis. Recorded
4cce77e. Interpreter axis: DEAD (do not cite in any form). Fingerprint
corroborates: the Python axis leaves no code signature; codes cluster by BOX.
Wall-clock argument retired too (3.13 draws ran FASTER: 63.6/61.8 min vs
shipped 127; shipped excess = four contended shards).

**THE 397B WIKITEXT FLOOR IS NOW 0.0256 (n=2, same-stack; code ~0.0178)** —
it widened every time it was measured more honestly (0.0134 inferred → 0.0197
mixed → 0.0256 same-stack). FLOOR AUDIT RE-RUN:

| row | margin | ×0.0256 | verdict |
|---|---|---|---|
| K512 vs spicy 2.6bit | 0.6209 | 24× | SOLID (code 0.0544 = 3.1× code floor — holds) |
| d8 vs spicy 2.6bit | 0.1252 | 4.9× | SOLID |
| d8 vs rate-twin K128 | 0.1115 | 4.4× | SOLID |
| E112 specimen | 0.1888 | 7.4× | SOLID |
| **flagship vs spicy 3.5bit, prose** | 0.0204 | **0.8× — INSIDE** | size claim only, quality INDISTINGUISHABLE (was "slightly better") |
| flagship vs shipped 3.1 | 0.0109 | 0.4× | never claimed; stays unclaimed |

**§4.4's ENDING (the withdrawn-axis version, final unless new data):** the
"vintage anomaly" dissolves into draw variance. At this geometry, unseeded
fits spread ~0.026 ppl; the shipped 2.7655 lies INSIDE the same-stack draw
distribution — a favorable but unexceptional draw, not a mystery and not a
reproduction target. Everything recoverable was verified identical (code by
diff, defaults by document, stack by binary identity, box by direct test);
what remained was the width of the distribution nobody had measured.
Fourth vindication of holding n=1 out of the laws file (6c, E126-ppl, E136).

### 08-23 overnight: 35B ladder — R2 scored, q6 built, my job-2 reading VOID

Instrument cross-validated first: e94b on M4 = 53.022 / 89.55%, identical to
M3 to every digit; kl_damage + score_streaming md5-identical across boxes.
Cross-box scoring sound for this session (stated per entry, III.13 runtime
named).

| 35B build | GiB packed | KL | top-1 |
|---|---|---|---|
| VQ d4/K8192 (e94b) | 14.838 | 53.022 | 89.55% |
| **VQ d4/K16384 (R2)** | **15.783** | **47.535** | **89.81%** |
| affine 4-bit (community) | 19.000 | 78.557 | 85.61% |
| **affine 6-bit (ours, new)** | **26.234** | **13.358** | **94.65%** |
| affine 8-bit (community) | 35.131 | 7.449 | 96.18% |

- **R2 MEETS its bar but does NOT dominate e94b** — 0.945 GiB larger for
  5.487 mnats better. Two frontier points, not a replacement. Paper shows
  both.
- **MY JOB-2 READING IS VOID.** I registered "q6 beating the best VQ rung AT
  COMPARABLE SIZE" without knowing q6's size; it landed 26.234 vs 15.783
  (66% larger), so neither branch fires. Reading it either way would be the
  III.8 size-mismatch error. Lesson (mine, second of this class after the
  inherited K256 floor): a pre-registration depending on an unmeasured
  quantity must state the condition under which it becomes evaluable.
- **MoE crossover: still OPEN.** VQ is above the affine frontier at the
  small end (47.5 @ 15.8 vs 78.6 @ 19.0); no VQ point exists above 16 GiB.
  Job 3 (d2/K1024, ~24 GiB) is the comparable-size test vs q6's 13.358.
- **R3 is expensive for EVERY method:** q6 is inside the 28.10 GiB bar and
  misses the 7.449 quality bar by 1.8x. "8-bit quality costs 8-bit bytes on
  this family" is the honest framing — stronger with an affine point in it.
- Job 3 caveat (theirs): that fitter has no --seed by design (keeps
  E121/E129/E136 valid), so it is ONE UNSEEDED DRAW.
- Incidental: `qwen36-35b-rungs/vq-K512-d2` has NO config.json — broken
  artifact in the rungs dir.

### 08-23 ~23:40: E140 (35B d2/K1024) — three readings, two safe, one held

    e140-35b-d2K1024-packed   21.394 GiB MEASURED   KL 28.141   top-1 92.22%
    gate PASS (median 0.0411) · III.11 PASS in stock venv (bundle executed)
    unseeded SINGLE draw

**SAFE NOW (in the paper tonight):**
- **R3 UNREACHABLE on the 35B MoE.** Bar KL<=7.449 @ <=28.10 GiB: size inside
  by 6.7 GiB, quality misses 3.8x — past the >=3x margin rule, conclusive at
  n=1. Pairs with q6 missing the same bar at 1.8x from inside the size
  budget ⇒ **on this family the heavyweight tier is unreached by BOTH
  methods.** Two families now. Strongest negative in the paper.
- **SIZE MODEL: projected 21.48 vs measured 21.394, −0.40%.** First 35B
  out-of-sample test, predicting a THIRD geometry at a different d from a
  two-point derivation. **Claim 2's size model is now validated on THREE
  families (397B, dense 27B, 35B)** — was two an hour ago.

**PLACEMENT ESTABLISHED at n=2 (E140b, 08-23):** draws 28.141 / 27.927 vs
frontier bar 43.71 → 1.55x and 1.57x below the line, both 21.394 GiB
measured. **35B fit-to-fit floor (first ever on this family): 0.214 mnats
= 0.76% of mean, SAME-BOX SAME-GEOMETRY** (subsample variance only; not
fitter-implementation or cross-box variance). Placement margin is 73x the
draw spread. Row is in the paper; law 14's bracket does NOT extend to MoE
at 5 bpw — the MoE crossover is above 5 bpw and unmeasured.

Original single-draw framing, retained for provenance:
- **PLACEMENT: 28.141 vs the log-interpolated q4→q6 frontier at 21.394 GiB =
  43.71 ⇒ VQ above the affine frontier by 1.55x.** First 35B point where a
  matched-byte placement is DEFINED (e94b/R2 sit below q4, so theirs is a
  dominance claim). Would also mean law 14's bracket does NOT extend to MoE
  at 5 bpw. **But placement-above-the-line is a BEAT, and the margin rule
  requires a second draw.** E136 precedent: a single draw agreeing with its
  target whose sibling landed 6x away.
- Replicate readings registered: below the line at its own measured size
  (recompute the bar, don't reuse 43.71) ⇒ established at n=2; above ⇒
  straddle, UNRESOLVED, report the pair. **Either way the spread is the
  FIRST 35B fit-to-fit floor** — the family has never had one.
- Interpolation method must be quoted inline (LOG, q4→q6, at measured size);
  linear gives ~56 and a flatteringly larger margin.

## PAPER STATUS 08-23 00:30 — NO OPEN SLOTS, NO PENDING DATA

Draft has zero `[[SLOT]]`/`[[HELD]]` markers. Every claim-1 row, claim-2
validation and claim-3 specimen is measured, gated, provenance-clean and
floor-checked. Remaining work is editorial only:
1. Chart regeneration (`chart_397b_ladder.py`: spicy x-coord 121.0→120.6,
   add E91/E92/E93/d8 points).
2. Title sign-off (Noah).
3. Read-through pass + §5 no-bragging trim.
4. Website publish.
E138 (d4/K65536, M3, lands Mon ~11:00) = revision item, nothing waits on it.

### 08-23 midday: E141-M4 — the MoE crossover is BRACKETED; law 14 goes two-family

    E141-M4 35B d2/K4096: 25.145 GiB MEASURED, KL 25.502, top-1 92.52%
    vs q6 (26.234, 13.358): DIRECT comparison, 1.1 GiB smaller, 1.91x WORSE
    = 57x the 35B draw floor -> conclusive at n=1 (loss branch; no replicate owed)

- **Crossover bracketed on the 35B: 5.0–6.0 bpw** (E140 below the line at
  5.0, E141-M4 above it at 6.0). Dense bracket was 4.5–6.0. **Same band, both
  architectures — law 14 is no longer dense-only.** Title's "Below 6 Bits"
  is now measured on both families.
- Size model: third 35B out-of-sample hit (−0.37%; series −0.03/−0.30/−0.37).
- Bonus: d2/K256 scored (17.643 GiB, 36.862, 90.92%) — fifth 35B VQ point,
  ladder monotone 53.0→47.5→36.9→28.0→25.5. Also the measured bar for any
  future d4/K65536 rate twin on this family.
- **E142-397B (K2048 floor) RUNNING.** Design correction from the M4 (adopted):
  flagship's actual args recovered from its script — --relerr-abort 0.70,
  --expert-chunk 8, --tail-from 10 --tail-geom d4k2048, --src on the LOCAL
  T7 (751 GB, still exists). Floor fits use the same. RESOLVED: the tail flags are present in the invocation but are NO-OPS at
  this geometry — verified from code (geom_for returns the same (4,2048)
  tuple on both branches; group size global, packing K-driven) AND from the
  artifact (171 modules, exactly ONE geometry group, all d4/K2048/g64/11-bit).
  **The flagship is genuinely flat and the paper's descriptions stand.**
  Anyone re-deriving the recipe from the command line: the flags are
  present and are no-ops at this geometry.
- Draft updated: abstract fence, intro claim-1 fence, §3.2 crossover
  paragraph, §6; charts gained E141-M4 + d2/K256.

### 08-24: E142-397B RESOLVED — K2048 floor measured; flagship claim SPLIT

    draw 1: 2.3390 / 2.6064 (8h37m, contended)   draw 2: 2.3334 / 2.5960 (6h19m, idle)
    FLOOR: 0.0056 prose / 0.0104 code — 4.6x narrower than the K256 floor

- **Flagship vs spicy 3.5bit, re-judged at own geometry: prose 0.0204 =
  3.6x floor -> CLAIMED. Code 0.0042 = 0.4x floor -> TIE, not claimable.**
  Draft updated: "smaller by 21.9 GiB, better on prose, tied on code."
- Floors NARROW as K grows (0.0256 @ K256 -> 0.0056 @ K2048) — consistent
  with richer codebooks fitting more consistently. In §2.6 and §6.
- **Draw-2 swap = product decision, NOT a paper claim**: 1.4x prose floor
  (marginal, best-of-2 selection inflates it), tie on code. Cleaner
  provenance (fresh full fit). Noah's call; paper numbers stay flagship's.
- Incidental: first clean K2048 wall-clocks; fits inherit nice 5 over
  ssh+nohup — all prior M4 timings were background-priority.
- **No present-tense measurements remain in the draft** except §4.2/§6's
  E138 reference (lands ~10:25 today).

## 08-24: SEEDING CLAIM CORRECTED IN THE DRAFT (precision, no number moves)

The draft said in three places that "the fitter is seeded" and that a build is
"reproducible from recipe plus seed". FALSE for the artifacts this paper is
about, verified from source:
- `vq_397b_codes.py:571` — its own comment: "kmeans init is random and unseeded"
- `vq_35b_codes.py` — no seed of any kind
- `fit_dense_vq.py` — `--seed` (default 1234) added 2026-08-22, AFTER every
  dense measurement reported here
**Every artifact in the paper is a single unseeded draw.** That is not a
weakness to hide — it is the reason the floors exist and the reason no margin
is read without one. §2.6, §5 and §7 now say so.

Also corrected: `artifact_manifest.py` records bytes, int(mtime) and a sha256
of the FIRST 1 MiB per shard. The draft called this a "content hash" in §5,
the section whose job is describing instruments honestly. It is an identity
stamp: it catches a rewrite, it does not certify every byte, and mtimes
survive `rsync -a` (the same trap as FINDINGS 6b's corollary). The strong
check we DO run is full sha256 against the remote LFS oid at publish time —
that is what verified the flagship 28/28 today.

Caught by the public-repo session reading the draft against the tool. No
number, table or claim standing changes.

## 08-24: 35B SIZE AXIS WAS MIXED-CONVENTION — corrected in 3.3

Measured, not inferred: the community 35B quants carry a 333-tensor vision
tower at bf16, **0.832 GiB, identical in the 4-bit and the 8-bit**. Our q6 is
text-only (1757 tensors, 0 vision) and every 35B VQ rung on disk is text-only.
So 3.3 was comparing our text-only rungs against with-vision comparators.

Corrected by subtracting the measured tower from the two community rows
(q4 19.001 -> 18.17, q8 35.131 -> 34.30) and stating the convention inline.
No claim flips; three numbers move:
- R2 vs q4: "3.2 GiB fewer" -> **2.4 GiB fewer** (39% KL gap unchanged)
- E140 placement bar: 43.7 -> **38.7**; factor ~1.6 -> **~1.4** below the line
- that margin: 73x -> **~50x** the 0.214 floor (still conclusive)
- E141-M4 vs q6 ("1.1 GiB smaller") was ALREADY consistent — both text-only.

**The CARDS are unaffected**: they compare published builds (bf16 tower
included) against community builds (same tower), which is apples-to-apples.
The mismatch existed only where lab rungs met community comparators.

Consequence for any 35B release: publishing a text-only artifact alongside
the two published tower-carrying ones would put a third convention on the
shelf. Graft before publishing.

## 08-24: 397B VISION ASYMMETRY — runs AGAINST us; sizes kept, offset disclosed

Measured from the artifacts: our 397B builds carry the bf16 vision tower
(2545 tensors, 333 vision; graft file 912,057,227 B = 0.8494 GiB). BOTH spicy
comparators are TEXT-ONLY (2212 tensors, 0 vision; 120.572 and 165.572 GiB).

So every 397B size margin in the paper is UNDERSTATED by 0.849 GiB:

| comparison | paper says | like-for-like |
|---|---|---|
| d8/K16384 vs spicy 2.6 | 19.6 GiB smaller | **20.4** |
| 2.4bpw vs spicy 2.6 | 9.0 smaller | **9.8** |
| flagship vs spicy 3.5 | 21.9 smaller | **22.7** |
| flat K512 vs spicy 2.6 | +1.7 larger | **+0.88** |

**DECISION: keep the download-size convention, disclose the offset in 3.2.**
Restating sizes text-only would improve every number by changing the basis in
our own favour. A conservative convention plus a stated offset is worth more.
Cards unaffected — they quote download size, which is what a downloader gets.

Also corrected: 3.3's cross-reference, added by me earlier today, claimed the
397B sizes "include their tower on both sides of every comparison." False —
asserted symmetry without checking the comparators. Same error class as the
35B mixed convention it was written to fix.

## 08-24 (supersedes the earlier 35B-size entry): 3.3 IS ON A WITH-TOWER BASIS

Noah's call, and the right one. The community 35B comparators DO ship the
333-tensor bf16 vision tower (verified: 0.832 GiB in both q4 and q8), and so
do both our published 35B builds. The only text-only artifacts anywhere in
this paper are the spicyneuron 397B quants and our own ungrafted lab rungs.
So the coherent basis is with-tower, not text-only, and it matches what a
downloader actually gets.

My earlier fix subtracted the tower from the comparators instead. Not wrong
arithmetically, but backwards as a convention: it moved the two artifacts
that HAVE towers onto a basis nothing on the shelf uses.

**The basis is a uniform offset, so NOTHING analytic changes** — verified
numerically both ways: gap-vs-q4 2.39 GiB either way; placement bar 38.68 and
factor 1.375 either way (log-interpolation on size is shift-invariant);
d2/K4096 vs q6 stays 1.1 GiB. Only the labels move.

| row | text-only | with tower (now in the paper) |
|---|---|---|
| d4/K8192 | 14.84 | 15.67 |
| d4/K16384 | 15.78 | 16.61 |
| d2/K256 | 17.64 | 18.48 |
| d2/K1024 | 21.39 | 22.23 |
| d2/K4096 | 25.15 | 25.98 |
| q4 / q6 / q8 | 18.17 / 26.23 / 34.30 | 19.00 / 27.07 / 35.13 |

Ungrafted rungs are measured packed bytes + the measured 0.832 constant; the
affine rows are measured as-is. Stated inline. 397B is unchanged: its
comparators lack towers, so §3.2 states an offset rather than applying one.

## 08-24: 35B RUNGS GRAFTED (measured) + a split lineage nobody had looked for

M4 grafted every 35B lab artifact. Measured sizes match the projections to the
digit: 15.670 / 16.615 / 22.226 / 25.977. §3.3 is now measured post-graft
except d2/K256 (`qwen36-35b-rungs/vq-K256-d2`, 17.644, vision=0), which stays
measured-plus-constant and says so inline.

**`config.model_type` IS NOT THE BASE MODEL.** Every 35B artifact reports
`qwen3_5_moe`, including the Qwen3.6-derived ones — a carried-over label. The
M4 nearly grafted a 3.5 tower into a 3.6 artifact on that basis. The reliable
test is byte-identity of a non-vision tensor against the candidate base.
Fourth instance today of the same class: a description that reads as obviously
true and is checkable in seconds (model_type names the model; the manifest is
a "content hash"; "the fitter is seeded"; the 397B sizes are symmetric).

**SPLIT LINEAGE:** `rotlab--35B-vqK256codes` and `zz35b-packed-K256` are
Qwen3.5-derived (6/6 tensors byte-identical to mlx-community Qwen3.5-35B-A3B-4bit,
0/6 against every 3.6 candidate). Found because the graft guard REFUSED them,
not because anyone suspected it. Neither appears in §3.3 — the paper is clean.

**OPEN, card-level:** `qwen36-35b-rungs/vq-K256-d4` (10.144, vision=0) has the
same pre-graft size as those two, and MODEL_CARD_QWEN_QUALITY.md publishes a
"d4·K256, 10 GiB, 1.141x, 79.50%" row inside a sweep table whose premise is
one harness and one corpus. If that row is 3.5-derived it is a different base
model in a 3.6 sweep and needs a card push. Shard-1 hashes differ across all
three artifacts, so they are distinct fits — which rules out the easy
explanation and settles nothing. Lineage read requested from the M4.

## 08-24: LINEAGE QUESTION CLOSED — the card was fine

M4 lineage read (read-only): `qwen36-35b-rungs/vq-K256-d2` and `vq-K256-d4`
are BOTH Qwen3.6-derived (6/6 byte-identical to 3.6, 0/6 to 3.5). The two
3.5-derived artifacts re-probed as controls: 0/6 and 6/6 the other way.
**MODEL_CARD_QWEN_QUALITY.md's "d4·K256, 10 GiB" row belongs in its 3.6
sweep. No card push needed.** The 10.144 size collision was geometry, not
lineage — d4/K256 lands at the same packed size from either base because the
code budget is identical.

**LIMIT OF THAT PROBE, stated by the M4 and worth keeping:** it prefers norms,
and norms are identical across bf16/4bit/8bit of the SAME release. So it
discriminates RELEASE (3.5 vs 3.6), not which quantization of a release a fit
started from. Sufficient for every question asked today and for the graft
guard; NOT an instrument for "was this fitted from the 4-bit or the 8-bit".

§3.3's d2/K256 is 3.6, so grafting it would make that row fully measured
(18.476). Not done — awaiting Noah in the M4 session.

**§5 GAINED AN EIGHTH RULE (addition, not a correction — Noah may cut it):**
"A label is not a measurement." Four instances today (model_type, the manifest
"content hash", "the fitter is seeded", assumed 397B size symmetry) plus the
III.10 phantom and the word "seeding" in a vq_397b_codes.py docstring that
read as an RNG seed. The sharper framing is the M4's: these survive review
BECAUSE they are cheap to check — nothing that costs nothing to believe gets a
verification budget.

## 08-24: TABLE CONVENTION SETTLED — a row is an ARTIFACT, never a mean

§3.3's d2/K1024 row showed 28.03, the mean of its two draws, while every
other row in the paper is a single artifact. Noah asked the right question:
if one row is a mean, should all multi-draw rows be? **No — the opposite.**

In every other multi-draw case the extra draws are FLOOR PROBES of a
geometry, not replicates of the artifact in the row:
- 397B d4/K2048: published E91 2.3410 + E142-397B floor draws 2.3390 / 2.3334
- 397B d4/K256: published 2.7655 + E136/E136b floor draws 2.7706 / 2.7962
- dense d2/K256: E124's rung + E127's three-draw floor
Averaging those would blend a published artifact with throwaway probes and
produce a number describing no downloadable thing — and would break III.2.

d2/K1024 was the outlier (two real builds of one recipe), not the model.
Row now reads 28.14 = draw 1, the artifact. The prose already gives both
draws and the ~1.4 factor covers either (1.375 vs draw 1, 1.380 vs mean).

**RULE: a table row names one artifact. Replication lives in prose, floors
live in §2.6.** Also means the paper matches whatever gets published: if
e140 draw 1 ships, its card and §3.3 quote the same number.

## 08-24: 27B checked for the vision asymmetry — CLEAN; but q8 is not uniform 8-bit

Vision: every 27B artifact is TEXT-ONLY, ours and all affine comparators
(vision=0 across e119/e124/e138 and q2/q3/q4/q6/q8). The BASE has a 333-tensor
tower; every quantization drops it. §3.3's 27B tables are like-for-like at
face value — no change needed. Third family checked, third different answer.

**BUT: `qwen38-27b-rungs/q8` leaves 96 modules UNQUANTIZED.** Verified by
dtype, not inferred: `linear_attn.in_proj_a` is U32 [48,640] in q4 and BF16
[48,5120] in q8; 48 in_proj_a + 48 in_proj_b, i.e. 192 missing scales/biases
(1655 tensors vs 1847). So the artifact anchoring §4.1's "8 bits is
essentially lossless" and the 27B R3 bar (KL 1.641) is an 8-bit-CLASS build,
not a uniform 8-bit grid.

Claim stands: its 26.341 GiB includes those bf16 tensors, so the bytes are
honest, and an easier bar would only make our failure to reach it less
interesting. §4.1 now states the property rather than letting "q8" imply
uniformity. NOT checked: whether the 35B community q8 has the same shape.

Fifth instance of "a label is not a measurement", on the day the rule was
added — this time the label was a filename.

### 08-24 addendum: the q8 non-uniformity is REAL but IMMATERIAL — no re-test

Sized after flagging it, which was the wrong order. The 96 bf16 modules are
23.6 M parameters = **0.087% of the model**, and quantizing them to 8 bits
would shrink the artifact by 0.021 GiB = 0.08% of 26.341. Negligible against a
27B R3 gap of ~25 bpw.

Also checked: **the 35B community 8-bit IS uniform** (2090 tensors, 512
scales, structurally identical to its 4-bit sibling), so §4.1's other data
point needs no caveat at all. And **no community 8-bit exists for
Qwen3.8-27B** — only a BF16 upload — so our local conversion is the only
8-bit that exists for that model; there is nothing better to compare against.

**VERDICT: do NOT spend a box re-converting a uniform q8.** §4.1 now states
the property with its magnitude so a reader can size it. Lesson for me: "this
label is wrong" and "this changes the answer" are different findings, and I
reported the first as if it were the second.

### 08-24: 27B comparators are OURS, and the paper now says so

Searched HF: **no mlx-community (or any MLX-format) quantization of
Qwen3.8-27B exists** — only GGUF/FP8/NVFP4 and an uncensored fine-tune. So
every 27B affine rung (q2/q3/q4/q6/q8) is a LOCAL conversion, while §1 tells
the reader comparators are community builds. §3.3 now states the exception
and calls it the weaker class of evidence it is.

**The q8 skip is OURS, not an MLX default** — proof: the 35B community 4-bit
and 8-bit have identical quantized surfaces (2090 tensors, 512 scales each).
So mlx does not skip those modules; our conversion did, and inconsistently
(our q4 quantizes them, our q8 does not).

**DIRECTION, which I had underweighted:** an unquantized-attention q8 is
BETTER than a uniform 8-bit, so the bar is artificially high, which flatters
affine and therefore flatters OUR OWN negative result (R3 unreachable).
Magnitude still bounds it — 0.087% of params against a ~25 bpw gap — but the
paper now names the direction, because a bar that errs toward the author's
conclusion has to be disclosed by the author.

Still no re-test warranted; nothing plausible closes 25 bpw.

### 08-24: the q8 skip is CONFIRMED our defect — settled on the right architecture

Noah's suggestion, and the right instrument for the question (not as a
comparator — a Qwen3.6-27B build has a different teacher and cannot sit on our
KL axis; using it as one would be the substitution the cards refuse).

`mlx-community/Qwen3.6-27B-8bit` — same 27B dense architecture, same
linear-attention structure — DOES quantize `linear_attn.in_proj_a`: 48
weight + 48 scales + 48 biases, U32 [48,1280]. Our q8 leaves the equivalent
96 modules at BF16.

**So it is not an MLX behaviour, it is our conversion deviating from the
standard tool on this architecture.** My earlier 35B evidence was the wrong
architecture to settle it on (MoE, different attention), and I presented it as
if it settled the question. §4.1 now attributes the defect to us explicitly.

**REVISION ITEM (not a blocker):** re-convert a uniform 8-bit with
`mlx_lm.convert` defaults and re-score against kl_cache_qwen38. Cheap, and it
would replace a bar we know errs in our favour with one built the standard
way. Conclusion is robust either way — the 27B R3 gap is ~25 bpw — so this is
about the comparator deserving to be right, not about the finding changing.

### 08-24: 35B fully grafted — every §3.3 row is now a MEASURED artifact

M4 grafted the nested `qwen36-35b-rungs/` set (its earlier sweep walked only
the top level of Exo Models/ and missed 21 artifacts; it corrected this
itself). All §3.3 values verified against the measured post-graft sizes:

    d4/K8192 15.670 · d4/K16384 16.615 · d2/K256 18.475
    d2/K1024 22.226 · d2/K4096 25.977 · q6 27.066
    q4 19.00 and q8 35.13 unchanged (community, always had towers)

Every table value was already correct. **The NOTE was not.** It read "all rows
are measured post-graft except d2/K256" — but q6 was ALSO ungrafted at that
moment, and its 27.07 was a projection I had computed and then described as
measured. My error, and precisely the rule added to §5 today: I labelled a
derived number as a measurement, in the sentence whose job was stating
provenance. Both rows are now genuinely measured and the note says so.

**q6 was the one that mattered** (M4's flag): it is the affine comparator, so
while the VQ rungs carried towers and it did not, every VQ-vs-q6 size margin
was overstated by 0.832 GiB — the spicy asymmetry again, pointing our way.
The paper is unaffected because the table already used 27.07 and the
placement/margin arithmetic was computed at 27.066 throughout: d2/K4096 vs q6
is 1.089 GiB either way, and KL ratios (1.91x) are size-independent.

Noted for future readers: `vq-K8192-d4-packed` and `e94b-...-packed` are both
15.670, and `vq-headup-d2k512-packed` and `vq-tail30-d2k512-packed` are both
18.710 — same geometry, different fits, hashes differ. Same trap as the
10.144 collision.

## 08-24: E143 — flat K512 SMOKE PASS (2-node exo ring)

`rotlab--397B-flatk512-packed`, 131,344,793,064 B on disk (122.324 GiB all
files; index total_size 122.305 GiB weights). SERVING in 99s on M3 96 GiB +
M4 128 GiB tensor-parallel; 800-token coherent generation, finish=stop;
graded probes 3/3 (Paris / 391 / Jane Austen). Preconditions cleared first:
outlier gate PASS, 333 vision tensors, check_release PASS, byte-level M3->M4
copy verify 40/40 zero mismatches. Comparator: the published 3bpw served in
87s, also 3/3.

**The paper's LEAD claim-1 evidence has now generated a token.** Before
tonight it was a fit, a score and a size.

**LABEL CORRECTION:** the M3 logged it as `vq-d2K512`. It is **d4/K512** —
config `vq_modules` all 171 entries `{'k':512,'dim':4,'group':64,
'pack_bits':9}`, and all 171 codebooks are shape (512,4) F16. pack_bits 9 is
log2(512) at either d, so only dim disambiguates. At d2 the rate would be 4.5
bpw rather than 2.25 and anyone re-deriving the ladder from that entry would
get a rung that does not exist. Fifth label/bytes disagreement today.

**CONSTRAINT for the card:** cannot be smoke-tested single-node — it does not
fit the M4. Any re-verify is 2-node, and the card must live in both nodes'
builtin `inference_model_cards/` (custom_model_cards is GC'd by the 1 Hz
reconciler on reset). exo's reported storageSize.inBytes differs from the
measured on-disk total by 21,015,432 B because exo computes its own figure;
the measured number is the one to publish.

**MY OWN ERROR, caught by re-checking:** an ad-hoc vision count of mine
searched only for `vision` in tensor names. The 397B towers are named
`model.visual.*`, so it returned 0 on an artifact that has 333. Re-ran with
both patterns everywhere it mattered; the 27B "no tower on either side"
conclusion is unaffected (both patterns return 0 there).

## 08-24: E144 — THE 27B "q8" WAS NEVER AN 8-BIT BUILD. Pre-registration failed both axes.

    rung            GiB      KL       top-1     ppl
    q8 (incumbent)  26.341   1.641    98.08%    --
    q8-rebuilt      26.617   1.254    98.54%    5.2413      <- BIGGER and BETTER

Registered prediction was smaller-and-worse. **Both axes wrong**, and the
cause is identified, so the result is adopted rather than discarded.

**The incumbent's config** (verified myself, not taken on report):
top-level `{group_size 64, bits 4, mode affine}` — the DEFAULT is FOUR — plus
402 per-module overrides, 401 at 8 bits and **one at SIX**
(`language_model.lm_head`), with 96 linear_attn projections carrying no
override and left BF16. It is a mixed-bit build that was cited as a uniform
8-bit bar. The rebuild has top-level bits 8 and zero overrides, matching q4
and q6. mlx_lm reports 8.501 bpw.

**I VERIFIED THE REST OF THE LADDER, which the M3 had not:** q2, q3, q4, q6
all have ZERO overrides and clean top-level bits. Only q8 was wrong.

**MY §4.1 TEXT WAS WRONG IN SIGN AND IS REWRITTEN.** I wrote that the defect
"flatters affine and therefore flatters our own negative result." It does not.
Two deviations pulled opposite ways — attention at bf16 (better, bigger),
lm_head at 6 bits (worse, smaller) — and lm_head is the larger term, so the
net bar was WORSE than a true 8-bit. Correcting it raises the ceiling: the
affine frontier is more capable and VQ has further to travel. Extrapolated
target moves from ~25 to ~27 bpw.

**The error was mine and it was a reasoning error, not a measurement one:** I
found one deviation and reasoned about direction from it alone. Finding one
defect is not finding the defects. That sentence is now in §4.1.

Updated: §3.3's 27B affine q8 row (26.62 / 1.25 / 98.5% / 5.241) and §4.1's
27B figure (1.3 mnats). Law 14's dense bracket is derived from q4 and q6, both
clean, so it is unaffected.

**M3 disclosed a defect in its own chain:** an assert guarded by `[ $? -eq 0 ]`
after a `tee` pipeline read tee's status, so ASSERT FAILED printed and the
chain scored anyway. The gate was decorative. Its numbers stand (both the
assert output and the results are in the log) and it swept the other chains —
only that one was affected. Same class as the E134 acceptance suite that
silently skipped every packed check.

## 08-24: E145 — MoE ROUTER PRECISION ASYMMETRY. 35B only; 397B is CLEAN.

M3's finding, verified here by dtype and extended to the family it had not
checked.

    family   our VQ                    comparators
    35B      mlp.gate BF16 [256,2048]  q4/q6/q8 all U32 [256,512] + scales
    397B     gate BF16 [512,4096]      spicy 2.6 AND 3.5 also BF16 [512,4096]
    27B      dense - no routers        n/a

**THE 397B IS SYMMETRIC.** Both spicy builds keep routers at bf16 exactly as
ours do, so every claim-1 lead row (K512, d8, flagship vs spicy) is
unaffected. M3 checked gemma26b and the 35B; I checked the 397B, which is
where the headline claims live.

**I ALSO CHECKED THE 35B COMMUNITY COMPARATORS**, which M3 had not — its
evidence was our own q6. `mlx-community` 4-bit AND 8-bit both quantize the
router (40 weights, 40 scales, U32). So the asymmetry covers every 35B
comparison in §3.3, not just the one against our q6.

**Magnitude vs mechanism:** 20 MiB, 0.14% of the artifact — negligible as
bytes. But a router feeds an argmax over experts, so quantizing it can flip
which expert runs; the quality effect is NOT bounded by the byte share.
UNMEASURED — the ablation is a VQ build with routers forced to 8-bit,
re-scored. Not run.

**DECISION (mine): disclosure, not a blocker, scoped to §3.3.** Added there,
including the explicit statement that §3.2 is unaffected so a reader does not
generalise it to the 397B. The ablation is worth having but it is a
measurement we do not have, and the honest move is to say so rather than to
delay on it.

**Direction: this one favours US** — the opposite sign from E144, four hours
later. Per M3, and I agree: do not state a net direction across the two. They
are different models and different magnitudes, and combining them would be
the same reason-from-one-term move that produced my wrong §6 sign.

M3 self-corrected a substring match (`mlp.gate` also catches `mlp.gate_proj`)
before trusting its own count — same class as reading a rate off pack_bits.
My probes here are anchored (`\.mlp\.gate\.`) for that reason.

## 08-24: 27B GRAFTED (21 artifacts) — §3.3 and §4.1 restated; one peer correction corrected back

M4 grafted every 27B rung AND every affine comparator, +0.8582 GiB uniformly.
Measured sizes match my projections to the digit (11.467 / 12.468 / 14.454 /
15.450). §3.3's two 27B tables and §4.1's q8 figure restated; the convention
is stated inline.

**The offset is uniform, so DIFFERENCES survive and RATIOS do not.** Verified
each surviving claim: d4/K1024 vs q3 still 0.35 GiB less; q6 vs d2/K4096 still
2.8 GiB more; d2/K256 vs q4 still 0.50 GiB smaller. **One ratio moved and is
fixed: the abstract's "3.5% larger" for d2/K512 vs q4 is now 3.3%.** Also
updated the abstract's two absolute sizes (13.6 -> 14.5, 14.6 -> 15.5).

**CORRECTING THE M4'S CORRECTION 1.** It says my "no towers anywhere in that
family" was wrong and that "the same asymmetry was live in §4.1, undisclosed".
Half right. My PHRASING was wrong — the Qwen3.8-27B base does carry a
333-tensor tower, and I should have said the quantizations dropped it rather
than that the family lacked one. But no asymmetry was live: I verified
vision=0 on BOTH sides — ours (e119/e124/e138) and every comparator
(q2/q3/q4/q6/q8) — so §3.3 and §4.1 were symmetric and needed no disclosure.
Its "pointing our way IF any comparator carried a tower" is conditional on
something that was measured false. Sloppy wording of mine, correct conclusion.

**e127-* and e95-* do NOT need grafting.** E127 is cited in §2.6 as the source
of the dense floor (2.085 mnats / 0.0447 ppl, n=3) — a floor measurement, not
a size row; no size of it is reported anywhere. E95 appears nowhere in the
draft. Answered to the M4.

**Two method notes from the M4 worth keeping:**
- Cross-layout graft: mlx stores the vision patch_embed conv CHANNELS-LAST,
  HF does not. It measured this on a model where both layouts exist (332/333
  identical, exactly one needing transpose(0,2,3,4,1)) rather than assuming.
  A naive prefix-rename yields 333 present tensors and a silently wrong patch
  embedding — right count, wrong bytes.
- Its identity probe read 0/6 against the true base because Qwen3.8-27B
  stores RMSNorm as (1+w), so every layer-norm differs by exactly 1.0. **A
  NORM comparison presented as an IDENTITY comparison** — third instance of a
  check measuring one axis and named for another. Fixed; passes 5/10 bit-exact
  with a wider sample.

## 08-24: E147 — e4b VQ-PLE prefill MEASURED. Two costs, not one.

    prompt      VQ-PLE      8-bit      VQ slower
      30       539.8      801.6         32.7%
    2048      3457.6     4000.2         13.6%
    8192      3423.3     3956.1         13.5%
    median of 3 timed reps, discarded warmup, idle GPU, clear_cache between

**My pre-registration was half right.** I registered that the ~20% gap should
NARROW or vanish. It narrows and then STOPS, flat from 2k to 8k across a
fourfold length change. So there are two separable costs: fixed per-call
overhead that dominates only at chat length, and a real length-independent
VQ-path prefill cost of ~13.5%. The card had conflated them.

**The old "~20%" described NEITHER regime** — worse than that at chat length
(33%), better at working length (13.5%). An average of two regimes is a third
number true nowhere. Card now carries both rows.

**Ratios only, never the absolutes** (M3's flag): the same pair on their box
reads 539.8/801.6 at 30 tokens where the card recorded 392/496 — both faster,
ratio preserved. The tok/s is not portable; the ratio is.

**SEPARATE CARD ERROR THIS EXPOSED.** The card said the upstream 8-bit's 126
redundant KV-shared tensors are ones "mlx_lm never instantiates and silently
drops". It does not silently drop them — **it refuses the checkpoint
outright**, reproduced on 0.31.3 and 0.31.9, so loading it at all needs
`strict=False`. Corrected, and it makes our artifact's advantage larger, not
smaller: this build loads with no flag.

Open, flagged not chased (M3): the card's original 496 tok/s incumbent figure
is not reproducible with a strict load, so whoever produced it used
strict=False or an older implementation. Does not affect the ratios now
published.

Method note worth keeping: M3 verified by GENERATION (3/3 known answers)
before taking any timing, because a model loaded with strict=False that is
subtly wrong still produces perfectly reasonable-looking timings.

## 08-24: 27B RELEASE SET FIXED — three rungs; §3.3's d2/K512 row is now arm 2

**Publishing:** d4/K4096 (12.47, VQ-3.9bpw), d2/K256 (14.45, VQ-4.5bpw),
d2/K512 arm 2 (15.45, VQ-4.8bpw). Names by the family convention, GiB x 8 /
27.78B params.

**NOT publishing, with reasons:**
- d2/K4096 (18.44, 5.70 bpw): 2.1 mnats/GiB marginal return, an order of
  magnitude worse than the first rung step, AND above the crossover where q6
  is 7.2x better for 2.8 GiB more. The 27B's E141-M4.
- d4/K1024 (11.47): KL 148.5 would be 73% worse than anything else we
  publish. Our released quality floor across both MoE families is ~85 mnats
  (35B VQ-3.4bpw 85.5; 27B VQ-3.9bpw 85.8). Noah's stated reason was 16 GB
  headroom; that is not actually the constraint (10.61 GiB resident leaves
  ~4.3 GiB free), the lineup-coherence argument is.
- d4/K65536 (14.55): the 37 h fit. Edges d2/K256 on KL (38.1 vs 40.3), loses
  on ppl (5.311 vs 5.233), 0.1 GiB larger. Paper reports it a wash leaning d2.

**§3.3's d2/K512 row now cites ARM 2, the artifact being published**, not
E126: 32.81 KL / 90.84% / 5.162 (was 33.095 / 91.10% / 5.194). Dependent
claims recomputed rather than carried: q4 margin 27.8% -> 28.4% KL, floor
multiple 6.1x -> 6.2x, top-1 +1.28 pp -> +1.02 pp. Size claim (+3.3%) and the
d4/K1024-vs-q3 (0.35 GiB) claim are unaffected.

**The card must NOT claim arm 2 is measurably better than arms 1 or E126.**
Arms 1 and 2 share bit-identical initial centroids (kmeanspp runs before the
Lloyd loop; Lloyd consumes no RNG), so arm 2 is arm 1 continued 20 iterations
with zero draw variance. All four metrics moved the right way but 0.0186 ppl
is below our resolution, not shown to be zero — the 0.0447 floor was measured
from unseeded draws and contains variance these arms do not have. Card says
"more converged", not "better".


---

## 2026-08-24 — READINESS SWEEP (background audit, four tasks)

Twelve findings, all verified by me against the draft's own tables before
acting. Eleven corrected in DRAFT.md; one open, below.

**Corrected — overclaims.** (a) Abstract said the 1.75-bpw d8/K16384 build
beats spicy 2.6bit "on both evaluation corpora." §3.2's own table: code
2.6728 vs 2.6667 — spicy is better. Now reads prose-win, code-tie. (b) The
rate-twin row claimed "d8 wins both corpora, 4.4x floor"; 4.4x is prose,
code is 0.0260 = 1.5x borrowed floor, under the bar. Row now states both.
(c) Abstract's shared "5 to 6 bits" crossover band contradicted the draft's
own dense bracket of 4.5–6.0. Both brackets now stated.

**Corrected — stale or unreproducible numbers.** (d) 35B d4 K-doubling
middle step 12.1 came from the retired pre-refit K8192 score of 56.4; with
e94b it is 15.5. (e) Abstract's "47.5 mnats at 15.8 GiB" was the text-only
pre-graft size; §3.3 carries 16.61 on the with-tower basis. (f) 31.1 was
computed from rounded operands; unrounded gives 31.0. (g) §4's "5.19–5.35,
all inside the 0.0447 floor" reproduced from neither table endpoint and the
span was 3.6x the floor, not inside it. Restated from the tables.

**Corrected — undisclosed borrowed floors (house rule 1).** (h) The 24x
d4/K512 margin and both d8 margins are all judged against the d4/K256 floor;
disclosed inline as lower bounds. (i) The d2/K4096-vs-q6 57x borrows the
d2/K1024 floor; disclosed.

**Corrected — definitions.** (j) Draft said the 397B tower is "exactly
912,020,960 bytes" while this ledger says the graft file is 912,057,227 B.
Both are right and they measure different things (tensor data vs file with
header); the draft now says which. (k) The 37% byte-aligned decode tax did
not reproduce at kernel level (EXPERIMENTS.md:4809) and has never been
re-measured at artifact level; the draft stated it unqualified and now
carries the caveat.

**OPEN — the 27B d4/K256 row (DRAFT.md ~401, 10.56 GiB).** This is E95,
which this ledger's 08-24 entry states was not grafted and "appears nowhere
in the draft." It has since appeared. On disk the e95 dirs carry no
metadata.total_size at all and no grafted artifact exists, so 10.56 is a
projection (E-record 9.7 G + 0.858), contradicted by the more precise
E-record figure of 9.612 → 10.47. The row therefore violates rule 2 and the
table's own inline note that every 27B size is a measured grafted artifact.
Propagates to make_qwen38_ladder.py:14 and fig_35b_27b. Awaiting Noah's
call: drop the row, or graft-and-measure e95 on the M3.

**Clean.** Chart-vs-table diff across all three generators: no discrepancy,
including the three model cards. Every other 35B and 27B size verified
against metadata.total_size to the printed digit. All remaining margin
multiples recompute from the newest ledger values. Draft floors match the
ledger's measured floors exactly.

## 2026-08-24 — E95 ROW RESOLVED: grafted, measured, 10.47 (not 10.56)

The open item above is closed with data rather than by deletion.

**Artifact identity, settled from the record, not from size.** Four e95 dirs
exist on disk, not two: `-K256`, `-K256-refit`, `-vq`, `-vq-r2`. The E95
RESULT entry (EXPERIMENTS.md:5127) names the scored artifact explicitly —
`e95-27b-dense-vq-r2`, scored 2026-08-21 19:00, gate PASS, III.11 smoke PASS
— and r2's mtime is 18:57. `-vq` and `-vq-r2` are byte-identical in shards
1-3 and differ ONLY in shard 4. Neither size, nor du, nor the first three
shards distinguish them. **Shard 4's sha256 is the only discriminator.**
Both measure 10,320,471,877 B = 9.6117 GiB, confirming the E-record's 9.612
and refuting the 9.7 that my 10.56 was built on.

**Graft source.** Qwen--Qwen3.8-27B on disk is the TEXT-ONLY checkpoint
(1199 tensors, `model.*`/`mtp.*`, zero vision) — the direct graft failed
against it. No un-grafted VL base is present. Sourced instead from
`e119-27b-dense-d2k512-packed`, after verifying its
`model-vision-graft.safetensors` is byte-identical (921,497,299 B, sha256
8f71b3e3…) across three independently grafted rungs. graft_vision.py's own
base-identity probe passed 10/10.

**Measured post-graft: 11,241,969,176 B = 10.4699 GiB.** Growth was exactly
921,497,299 B — the law-5 exact-growth check passes. Tower is 0.85821 GiB,
which rounds to 0.858; the draft and all three 27B cards said 0.859. Fixed
locally in all four; the three published cards are now one digit ahead of
what is live and need a re-push.

Draft row, make_qwen38_ladder.py and make_charts.py updated to 10.47; both
figures regenerated. No prose claim depended on the old value — the row fed
only the table and the two ladder plots.

Residual defect, NOT fixed: r2's index metadata block is `{}` — no
total_size, same as every other 27B artifact. Sizes here are summed from the
shards on disk, which is why they are trustworthy; the metadata is not the
source.

**Sweep false positive, recorded.** The audit's finding 13 claimed the
ledger note "vq-K8192-d4-packed and e94b-*-packed are both 15.670" is now
wrong on disk, reporting the former at 18.483 GiB and calling it a
graft-on-unpacked. Checked: summing the shards named in each index gives
**15.6702 GiB for both**, 1810 tensors, 333 vision, in both cases. The
ledger note stands; the audit was reading block usage, not file bytes. Two
lessons kept: du is not a size, and an audit's findings get verified before
they are acted on — this one would have corrupted a correct ledger entry.

## 2026-08-25 — SECOND INDEPENDENT PASS: 5 confirmed, all corrected

Aimed at the first pass's blind spot — prose-vs-tables, and the corrections
themselves. Findings verified against primary sources before acting.

**1. The arm-2 swap falsified a load-bearing methods claim.** §2.6 said
"Every artifact in this paper is therefore a single unseeded draw" and that
the dense fitter gained a seed "after the measurements reported here"; §7
repeated it. But EXPERIMENTS.md:8961 registers E142-27B as "two arms, M3, **seed
1234**", and arm 2 is the adopted 27B d2/K512 row — carrying the 28.4%/6.2x
claim and the abstract's 28%. A correction made a methods sentence false.
Both passages now state the exception and note that reading a seeded arm
against an unseeded floor is CONSERVATIVE, since the floor carries draw
variance the arm does not.

**2-4. Propagation failures from the 08-24 fixes.** The debunked "5.19-5.35,
inside the measurement's own noise" survived in all three 27B cards AND in
make_qwen38_cards.py, so a card re-push would have re-published the exact
claim the draft retracted. The 0.859 tower likewise survived in
make_qwen38_cards.py, make_qwen38_ladder.py and make_charts.py — regenerating
would have regressed a fix made hours earlier. All corrected at the
generator, regenerated, and confirmed to survive regeneration.

**5. §6 called a finished experiment "currently fitting"** — the d4/K65536
rate twin whose result §4.2 reports five pages earlier. §6 now states the
outcome.

**Minors corrected.** "KL moves 40x" gave 36.6x from the printed table (now
37x); the rate-twin rungs were called "the same size" when the ledger records
0.1 GiB apart (now stated); "saves zero bytes and cost" tense break from the
(k) insertion.

**OPEN, needs a call.** MODEL_CARD_397B_E.md:41,156 says the flagship
"matches the community 3.5bit on both corpora"; the draft and this ledger now
say prose BETTER (0.0204 = 3.6x the measured K2048 floor) and code TIE. The
card is the conservative direction, but it is live and it disagrees with the
paper.

**Clean.** Every 08-24 correction re-verified as substantively right. All
table rows re-checked against ledger current-truth with graft offsets. All
three generators match the draft to the printed digit. No retired number
survives anywhere (no 56.4, 12.1, 33.095, 5.194, 10.56, 15.8, 1.641-as-
current, "wins both corpora"). House rules obeyed; the one floor-disclosure
gap (the undisclosed ~0.0178 code floor behind the 1.5x multiple) is closed.

## 2026-08-25 — CARD SWEEP (three parallel audits: 397B, 35B+gemma, draft tables)

**397B — one real error, CONFIRMED and fixed.** MODEL_CARD_397B_C.md's
Siblings table served the RETIRED flat-K128 ladder point (100.9 / 3.1706 /
2.6988) as if it were the live VQ-2.2bpw repo, under a header reading "all
measured the same way." The repo has served d8/K16384 since the 08-22 v2
update. This ledger's own row calls flat-K128 "ladder point only — NOT a
quality claim." The card's OTHER v1 references (task table, prefill advice)
are all explicitly labelled *(v1 weights)*; this table alone was not. Fixed
to 101.0 / 3.0591 / 2.6728. **The card is live and needs a push.**

**Pass-2 open item RESOLVED AS MOOT.** The second pass flagged
MODEL_CARD_397B_E.md claiming the flagship "matches the community 3.5bit on
both corpora," contradicting the split claim. E is not live: its first five
lines read "SUPERSEDED (2026-08-24) … The live card is MODEL_CARD_397B_G.md
… cite nothing from here," with the superseded weights pinned to revision
a0da72a0. Pass 2 read the body and missed the header. No action. Second
audit false positive of the day, and the second time a later pass caught an
earlier pass's error rather than compounding it.

**35B + gemma — NO findings.** The single reported item (cards printing 78.6
/ 35.1 / 7.4 where the ledger has 78.557 / 35.131 / 7.449) is correct
rounding to one decimal, verified: 78.557→78.6, 35.131→35.1, 7.449→7.4. Not
a contradiction. Clean on the checks that mattered: every 35B card is on the
with-tower basis and every comparison it makes is basis-consistent — the
specific failure mode that family had before — all four VQ sizes match
make_qwen36_ladder.py, and all three gemma cards correctly refuse perplexity.

**Draft tables — CLEAN.** Third chip walked 7 tables / 29 rows / 87+ cells
against ledger current-truth and reported no discrepancy, no unsourced cell,
no broken cross-reference, no basis inconsistency. A clean bill from the
cheapest instrument is the weakest evidence in the set, so six load-bearing
values were re-derived by hand: q8 26.617+0.858=27.475 (draft 27.48);
0.6209/0.0256=24.3x (24x); 0.1252/0.0256=4.9x; 0.0204/0.0056=3.6x;
(45.842-32.810)/45.842=28.4%; 15.450/14.952=+3.3%. All six reproduce, and
the d2/K512 row carries E142-27B arm 2 (32.8 / 5.162), not the retired E126
(33.095 / 5.194).

**STOPPING CONDITION MET (as pre-registered before this round ran).** The bar
was: a round whose findings are all cosmetic AND none of which trace to a
correction made in the previous round. This round found one real defect, and
it was pre-existing drift from the 08-22 v2 update in a file no prior pass
had been pointed at — not correction-induced. Four rounds total: 12 findings,
then 5, then 1. Severity fell from a headline overclaim in the abstract, to a
falsified methods sentence, to one unlabelled row in a siblings table.

**OUTSTANDING BEFORE PUBLICATION — four live cards are ahead of what is
served:** MODEL_CARD_397B_C.md (retired flat-K128 row in Siblings) and the
three QWEN38 cards (0.858 tower; the retracted "5.19-5.35, inside the
measurement's own noise" sentence). All fixed locally and regenerated from
their generator; none pushed.

**PUSHED 2026-08-25 — four cards, verified.** Repo ids confirmed against the
live HF listing first (13 public repos), then each local file diffed against
its LIVE README before upload: 5 diff lines on 397B_C, 12 on each QWEN38, and
nothing else moved. After upload each was re-fetched and sha256-compared to
local — all four byte-identical, and a negative check confirms none still
contains "5.19 and", "0.859 GiB", or "3.1706". No weights touched.

Note for future push maps: the C/E/F/G/K512 filenames are LOCAL ONLY. Every
card uploads as README.md and carries its own H1 (`# Qwen3.5-397B-A17B-VQ-
3bpw`); no letter is ever public. push_card_fixes.py's map is still missing
the three QWEN38 repos and the two 35B rungs added since it was written —
this push used an explicit four-entry map instead. Do not re-run
push_card_fixes.py as-is expecting full coverage.

## 2026-08-25 — RECORD REPAIR: the K2048 floor now has an experiment entry

Noah's authority, paper session, on the finding that the flagship's headline
claim divided by a floor with no record in EXPERIMENTS.md.

**The hole.** 397B d4/K2048 floor = 0.0056 prose / 0.0104 code. It existed
ONLY as rows in this file, cited as bare `[E142]` — which in EXPERIMENTS.md
resolved to a DIFFERENT experiment (27B d2/K512 iters, M3, seed 1234). A
reviewer following the citation under the paper's strongest claim would have
been handed an unrelated 27B test. Third cross-session collision after the
two E140s and two E141s.

**Fixed by suffixing, not renumbering** — per the allocation note at the top
of EXPERIMENTS.md, because artifact directory names on disk are load-bearing
(the 101 GiB lesson). The disk had already disambiguated these two by itself:
`e142-397b-k2048-draw1/draw2` vs `e142-27b-d2K512-iters10/30`. Only the record
had not caught up. Now **E142-397B** (floor) and **E142-27B** (iters), with a
note at the 27B heading explaining what a pre-08-25 bare `[E142]` means.

**E142-397B written in full** at EXPERIMENTS.md:9108 — question, the M4's
design correction and the twice-verified finding that the tail flags are
NO-OPS at this geometry (so the flagship is genuinely flat), both draws with
wall-clock and contention noted, the floor, and every claim that divides by
it. Nothing was re-measured; it is reconstructed from this file's 08-24
entries plus the artifacts, which still exist.

**Every number in the new record re-derived before writing:** |2.3390-2.3334|
= 0.0056; |2.6064-2.5960| = 0.0104; 0.0256/0.0056 = 4.6x narrower;
|2.3614-2.3410| = 0.0204 = 3.6x (CLAIMED); |2.6005-2.5963| = 0.0042 = 0.4x
(TIE); 0.0109/0.0056 = 1.9x (not claimable). All exact.

**Citations updated:** 7 in this file, 1 in DRAFT.md. No bare `[E142]` now
survives outside the two notes that exist to explain it.

**STATE.md CLOSED.** It was a live scratchpad, is stale (reports E138 as
running), and the experiment sessions are winding down. Header now says so and
points to the three files that are current: this ledger, EXPERIMENTS.md,
FINDINGS.md. Kept as a contemporaneous log, marked as non-authoritative.

**Artifact disposition — RELEASED (Noah, 08-25):** e142-397b-k2048-draw1 and
draw2 (196.3 GiB each) may be deleted. They are the only bytes that could
re-derive this floor, and that is accepted: measured and recorded IS the
record. Standing rule adopted from this — **an artifact whose result is fully
written down is free to delete; the write-up is the evidence, not the bytes.**

## 2026-08-25 — CORPUS REVIEW (four parallel audits + a disk cross-check)

**DRAFT vs BYTES ON DISK — the check that licenses deletion.** Every size in
every draft table was matched against the actual shard-byte sum of a real
artifact. **24 of 26 matched exactly.** The two that did not were the
spicyneuron comparators, printed as 120.6 / 165.6 in a column where every
other row carries two decimals; the artifacts are 120.5722 / 165.5722. A
reader subtracting the printed values got 19.63 and 21.92 against claims of
19.6 and 21.9. Now printed 120.57 / 165.57, which reproduce the claims
exactly. **This cross-check is the record's warrant: every number in the paper
was verified against bytes at least once before those bytes were deleted.**

**FINDINGS.md Law 14 was stale on four axes, all corrected.** (a) It said "the
MoEs have not been tested" at the upper bracket; E141 bracketed the 35B at
5.0-6.0, so the law is no longer dense-only. (b) Its q8 row carried the
RETRACTED pre-E144 number (26.341 / 1.641) — the build that was never 8-bit;
both rows now shown, the retracted one labelled. The bracket derives from q4
and q6, both clean, so it is unaffected. (c) Its d2/K512 row cited E126's
unseeded draw; the adopted artifact is E142-27B arm 2. (d) Its sizes are
pre-graft while the paper is with-tower — now stamped, per law 5. Header was
"through E110" while the file cited E139; now dated and marked closed.

**Retraction discipline: CLEAN, end to end.** An independent sweep for every
retraction in FINDINGS §II and this file's retired list found NONE asserted as
live in any model card or in the draft. The superseded card is marked, the
retracted q8 number appears in the draft only as labelled history, and the
withdrawn "wins both corpora" is absent from the live flagship card.

**Artifact inventory built.** make_artifact_inventory.py -> ARTIFACTS.md +
artifacts.json: 229 artifacts, 9,402 GiB, classified by REBUILDABILITY rather
than size. First cut missed 84 artifacts nested inside the three *-rungs
container dirs, which also hid every affine comparator; fixed by recursing one
level. Sizes are read from each index's shard list, not du — du over-reports
by ~0.02 GiB and once caused an audit to misreport 15.670 as 18.483.

### 08-25 corpus review — findings acted on

**Live cards, fixed locally, NOT pushed (need Noah's go):**
- `397B_C` called the family "three-size" and listed three rungs. Four are
  published; VQ-2.6bpw was missing. Now four, matching K512's table.
- `397B_G` — the LIVE FLAGSHIP — had no Siblings table while every other 397B
  card carries one. Added.
- All three QWEN38 cards were the ONLY VQ cards of the fifteen missing the exo
  codebook-replication warning. They are VQ artifacts with codebooks and carry
  the identical slicing hazard. Added AT THE GENERATOR so it cannot regress,
  and regenerated.
- `GEMMA_E4B_VQPLE` gave the incumbent's peak memory as 9.0 GB. **No source
  exists for 9.0 anywhere in the corpus.** E147 measured 9.48 GB at all three
  prompt lengths. Corrected to 9.5, and the derived claim from "20% less RAM"
  to "~24%" (7.25/9.48 = 23.5%). **This one moves a claim in OUR FAVOUR, which
  is the direction to distrust — flagged for Noah rather than quietly kept.**

**EXPERIMENTS.md:**
- The STATE OF RECORD block at the top is dated 08-13 — the day BEFORE VQ was
  discovered (E35, 08-14) — and names pre-VQ AFFINE builds as the "ship
  artifacts". Anyone relying on this file alone after deletion would cite the
  wrong artifacts. Marked superseded, scoped to entries before it, pointed at
  the ledger.
- **Fourth cross-session collision found:** `## E136` used twice, unsuffixed.
  The second is the M4 replication (withdrawn by E136b); suffixed E136-M4.
  Bare `[E136]` citations mean the M3 result and still resolve.

**Audit false positives caught (verified, no action):**
- "struct6-tail3x3 is 3.1557 in one place and 3.1580 in another, neither
  marked superseded" — the file explains it two lines later as the known
  1-node vs 2-node decomposition. Not a contradiction.
- "GEMMA_SMALL reassigns 84.62% to a different artifact" — 84.62% is 88/104 on
  a 104-item test. Two artifacts tying at 88/104 is a tie, not a reassignment.
- "mlx-community--Qwen3.6-27B-8bit and qwen38-27b-rungs/q8-rebuilt are the
  same artifact" (my own suspicion, not a chip's) — identical size, tensor
  count and vision count, but DIFFERENT shard sha256. Two different models
  whose 8-bit conversions land on the same byte count. **Fifth distinct size
  collision found in this corpus, and the first that crosses model families.**

**Known gap, deliberate:** E140-M4's quality numbers live in this ledger and
NOT in EXPERIMENTS.md, by the M4's own choice ("I have not scored these
artifacts and will not restate KL/ppl I did not measure"). E142-397B was the
same and is now fixed. **Consequence: the record is the SET {LEDGER,
EXPERIMENTS, FINDINGS}, not EXPERIMENTS alone.** Do not delete or archive any
one of the three.

### 08-25 referential-integrity sweep — E141 split, and a deletion-screen lesson

**Bare `[E141]` was ambiguous in six places — same class as E142, missed when
E142 was fixed.** EXPERIMENTS.md defines E141-M4 (35B d2/K4096 vs q6) and
E141-M3 (did thin init starve E138). All six bare citations here describe the
35B d2/K4096 result; all now read E141-M4. One of them was in the Law 14 fix I
wrote earlier today — the same defect reintroduced while repairing its sibling.
Bare `[E136]` and `[E140]` are left alone deliberately: each resolves to the
entry that still holds the bare number, and EXPERIMENTS.md says so at both.

**Citation graph otherwise CLEAN:** all 68 distinct cited E-numbers resolve
(including the early bullet-format entries that look dangling); every FINDINGS
"law N" reference resolves; every DRAFT section cross-reference resolves; both
figures exist and post-date their generator; no generator/output drift.

**DELETION SCREEN — a size match is NOT a citation, and I nearly acted as if
it were.** An audit listed ~1,292 GiB of 397B artifacts that appear nowhere in
the written corpus. Screening them by "does this size back a number in the
paper," two came back as hits:
`rotlab--397B-flatk512-randinit-packed` (122.31) and
`rotlab--397B-flatk256-bodytailw4-packed` (111.62). **Both are false alarms.**
122.31 is shared by 2 artifacts and 111.62 by FIVE; the paper's rows are
`flatk512-packed` and the published VQ-2.4bpw, not these variants. Identical
geometry produces identical byte counts, so a size screen cannot distinguish a
cited artifact from an uncited twin — in either direction.

**Standing rule for the deletion pass: screen by NAME and provenance against
the ledger's current-truth tables, never by size.** Size is the one property
guaranteed to collide. Sixth distinct size collision recorded in this corpus.

## 2026-08-26 — cards pushed, exo cards repaired, TIMELINE.md built

**Six cards pushed and verified byte-identical** (397B_C four-size family;
397B_G Siblings table; three QWEN38 exo notes; gemma VQ-PLE peak memory).
The gemma one was included though it moves a claim in our favour — flagged to
Noah at push time, reversible in one command.

**exo model cards on `vq-codebook-replicate` — three defects, committed
locally, NOT pushed.** That branch is public and linked from PR #2268 and
every model card, so pushing is Noah's call.
- **VQ-3bpw had no card**; the branch shipped VQ-3.1bpw, whose model_id HF
  REDIRECTS to the 3bpw repo — resolving to current weights under a stale
  name. Old card removed rather than kept, since the redirect IS the hazard.
- **VQ-2.2bpw carried its v1 byte count** (108,372,357,800). The repo has
  served v2 d8/K16384 since 08-22; corrected to 108,417,009,678.
- **VQ-2.6bpw had no card at all** despite being published. Added.

**A correction that made a correct number wrong — caught before shipping.**
An uncommitted draft of the 3bpw card set storage_size to 154,298,264,545 and
carried a comment explaining that the old card's 154,277,447,647 was "20,816,898
B short." The arithmetic is self-consistent and the premise is FALSE: the true
safetensors total is 154,277,447,647, confirmed against the live repo and both
local copies. The superseded card had it right. Committing that file unread
would have shipped a wrong size AND a confident false explanation.
**Convention established: exo storage_size = sum of .safetensors bytes only**,
derived by checking the 2.4bpw card, which matches to the byte. All four cards
now verified against live HF, and each model_id confirmed NOT to redirect.

**TIMELINE.md + make_timeline.py.** 73 experiments in order, each with what it
settled and a line link into EXPERIMENTS.md. **24 have a written result, 19
were pre-registered, and 3 pre-registrations were never closed** — shown as
such rather than hidden. Extracted from headings verbatim, never paraphrased:
a summary written today could smuggle in what we now believe an old experiment
showed, which is the exact failure the retraction discipline exists to stop.
Two bugs found and fixed while building it, both instructive: the early
bullet-format entries (E1-E40) were being dropped entirely, and MY OWN
editorial annotations ("SUFFIXED 2026-08-25") were being read as experiment
dates, back-dating 08-23 work to 08-25. The tool's docstring also claimed it
never guesses a date, which stopped being true when carry-forward was added —
fixed so the document matches the behaviour.

### 08-26 — exo picker cleaned; a third uncommitted regression found

Noah's ask: drop `vq-d4K512` and `vq-3.1bpw` from the exo model picker. Done,
in the checkout exo actually runs from (main's working tree — NOT the
vq-codebook-replicate branch, which is why yesterday's branch commit did not
change what he sees).

- `rotlab--397B-flatk512-packed` (`vq-d4K512`, 122 GB) was a PRE-PUBLICATION
  card; its own comment says it existed to be III.11 smoked on the ring before
  release. That smoke was E143, it passed, and the artifact shipped as
  VQ-2.6bpw. Job done, card removed.
- `VQ-3.1bpw` removed — the redirecting id.

**Third bad uncommitted edit found in the same directory.** All three tracked
397B cards had a working-tree change replacing the CORRECT committed comment
("Runs on STOCK mlx_lm via the bundled model.py — patch_mlx_lm is retired")
with the RETIRED path ("Loads via the VQ hook in each node's exo-env mlx_lm,
patch_mlx_lm.py + vq_switch.py"). This contradicts every published HF card
("No patches, no custom forks: config.json declares model_file: model.py").
Reverted on 2.2 and 2.4; 3.1 was deleted anyway. **Three defects now found in
uncommitted exo working-tree edits — a wrong storage_size, a false explanatory
comment, and a reverted-to-retired runtime claim. Nothing in that directory
should be committed without checking it against the published cards.**

**Consequence noted, not acted on:** with `vq-d4K512` gone, no exo card points
at the 2.6bpw rung. The published `VQ-2.6bpw` card exists only on the
vq-codebook-replicate branch, and those weights are NOT downloaded locally
(only the rotlab lab-name copy is on disk), so registering it would mean a
122 GB fetch. Left for Noah.

### 08-26 — VQ-2.6bpw: renamed in place, not re-downloaded

Noah: prefer the official artifact locally over the smoke-test copy. The bytes
were already on disk under the lab name, so this was a rename, not a 122 GB
fetch.

**Proved identical before touching anything:** all **29 LFS files sha256-match
the live repo**, vision graft included. Of the 39 files, only two differed and
both are expected — the README (updated after publish) and a stray
`model.safetensors.index.json.pre_total_size` backup.

`rotlab--397B-flatk512-packed` -> `TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.6bpw`.
Stray backup removed; the pre-publication README replaced with the published
card. exo resolves model_id by `org--repo` directory name, so the official
card now finds bytes we already owned.

**A wrong card was still live on main.** The 3bpw card in main's working tree
— the one exo actually reads — still carried the bad 154,298,264,545. Yesterday's
fix went to the vq-codebook-replicate BRANCH, which exo does not read.
Replaced. **All four 397B rungs now verified: card in_bytes == the local
.safetensors total, exactly.**

**Pointer hygiene:** the rename breaks the link from E143 (whose heading names
`rotlab/397B-flatk512-packed`) to the directory. Heading left as-is — it
records what the smoke ran under — with a note stating the new name and that
the bytes are identical. Two further mentions remain at EXPERIMENTS.md:9325
and LEDGER.md:792; both are historical statements about that artifact and
stay accurate under either name.

## 2026-08-26 — FINAL: all 13 published models local under published names

Noah's ruling: keep the published models under their published names, plus the
bf16 teacher; delete the rest. Every rename was resolved by HASH, never by
size or by name similarity.

**Identification.** Nine artifacts lived only under lab names. Matching by
safetensors total gave a UNIQUE candidate for six; three were ambiguous and a
probe on the SMALLEST shard matched all candidates — those shards are shared
base tensors. Re-probing the LARGEST shard resolved all three, and each landed
on exactly what this ledger already claimed:
- VQ-3.8bpw = `e94b-...-packed` (NOT `qwen36-35b-rungs/vq-K8192-d4-packed`)
- VQ-5.4bpw = `e140-...` draw 1 (NOT `e140b`)
- VQ-4.8bpw = `e142-27b-d2K512-iters30` arm 2 (NOT arm 1, NOT e119)
Independent confirmation of the published lineup, by bytes.

**The stale flagship-sibling caught.** The dir already named
`TheDrainFlorist--...VQ-2.2bpw` held the **v1 (flat d4/K128)** weights;
HF has served v2 (d8/K16384) since 08-22. Its card on main also still carried
the v1 byte count — my earlier `git checkout` revert had restored the committed
v1 value along with the correct runtime comment. Both fixed: v2
(`rotlab--397B-d8K16384-packed`, total-matched and probe-matched) installed
under the published name, v1 set aside as `zz-STALE-v1-flatk128-was-VQ-2.2bpw`
for deletion, card corrected to 108,417,009,678. **A size screen would have
called the v1 dir "the published 2.2bpw" — it has the published NAME. Only the
byte total exposed it.**

**Completeness verified against the live repos**, not assumed: all 13 have
every non-pycache file the repo has, and safetensors totals matching exactly.
Two 35B dirs were missing `qwen36_ladder.png` (their cards embed it); copied in.

**Renames recorded in rename_map.json.** E-numbers in EXPERIMENTS.md still name
the lab dirs; the results stand under either name, and that file maps old to
new.

**KEEP 1,369.2 GiB** (13 published + the 751 GiB bf16 teacher).
**DELETE 8,033.0 GiB / 7.84 TiB, 215 artifacts** — every lab rung, unpacked
twin, third-party comparator, cache, and the stale v1. Nothing in the delete
set backs a number that is not already written down.

**Records answer: complete.** Every paper number traces to a recorded
measurement. The three "never closed" pre-registrations are process artifacts:
E141-M4's result is in this ledger (25.145 + 0.832 tower = the paper's 25.98),
E120 was closed by E129 as UNEXPLAINED, E132's deliverable (packed e94b,
14.838) is measured here. **No further test of any old model is required.**

## 2026-08-26 — DELETION EXECUTED. 8.3 TiB freed; 13 published models + bf16 kept.

Disk: 1.0 TiB free -> **8.5 TiB free**. Keep set verified 14/14 present after.

**Why the earlier attempts silently did nothing.** `shutil.rmtree` RAISES on a
symlink instead of skipping it, and the first delete-set entry
(`Qwen--Qwen3.5-35B-A3B`) is a symlink into `hub-35b-src`. Every run aborted on
entry one having deleted nothing. The dry run could not catch it because a dry
run never calls rmtree — **the rehearsal did not exercise the operation it was
rehearsing.** Two symlinks total, both pointing into hub caches that were
themselves on the delete list; unlinked, counted as zero bytes (the space frees
when the cache goes, and counting it would double-count).

**A real gap in my own inventory, found only by the leftovers.**
make_artifact_inventory.py treats a directory as an artifact ONLY if it has a
`model.safetensors.index.json`. Two OptiQ builds (`optiq-b30`, `optiq-b30-af6`,
79.8 GiB) have no index, so they were never inventoried, never in the plan, and
survived the sweep. Not published, not cited in the paper; removed by hand.
**Anything counting artifacts by index file undercounts — verified no other
indexless model dirs remained.**

Also removed: the four now-empty container dirs and `.DS_Store` files.

**FINAL STATE — `/Volumes/Thunderbay SSD/Exo Models`:**
- `Qwen--Qwen3.5-397B-A17B-bf16` (751 GiB teacher, Noah's call to keep)
- all **13 published models**, under their published names, each verified
  against its live repo by total bytes + sha256 of its two largest shards
- four stray top-level config/tokenizer json files, harmless, left alone

Everything else is gone. The record — TIMELINE.md, EXPERIMENTS.md, LEDGER.md,
FINDINGS.md, ARTIFACTS.md, rename_map.json — is a few MB and is now the sole
evidence for every result whose artifact no longer exists. That was the plan,
and it is executed.

## 2026-08-26 — table emphasis given ONE meaning: bold = published

Noah spotted it in the rendered page: the teal/bold rows didn't correspond to
anything consistent. They didn't. Bold meant three different things in three
tables — in 3.2 it tracked "the comparisons that carry claim 1" (which the
prose immediately below already names), and in 3.3 it tracked neither the
published set nor the claim-carriers.

**Underneath it, a real error:** `flat d4/K512` is published — it is the
2.6bpw model — but its row carried no "(published)" label while three other
rows did. The paper was under-reporting its own published lineup.

Now **bold = published artifact**, everywhere, with the redundant
"(published)" text removed so there is one convention rather than two, and a
one-line note under each VQ table saying so. Verified mechanically: the nine
bold rows are exactly the nine ladder rungs whose sizes match a live HF repo —
397B d8/K16384, d4/K256, d4/K512, d4/K2048; 35B d4/K8192, d2/K1024; 27B
d4/K4096, d2/K256, d2/K512. Checked again in the RENDERED html, not just the
markdown.

**Disclosed rather than hidden:** two published 35B builds are not on the flat
ladder (13.79 d4/K2048, and the 18.71 tail-weighted build which is not a flat
geometry). A note under the 35B table says so, verified against
rename_map.json. Four of the 13 published models sit outside the three ladders
in total — the other two are the gemma pair, which the paper excludes from all
claims by design.

Republished to the artifact and to the private HF Space, the latter verified
sha256-identical to the local file.

## 2026-08-26 — ladders carry release names; bpw naming convention pinned

Noah: the geometry labels are memorable but you cannot download "flat d4/K2048".
All three VQ ladders gained a `release` column giving the repo suffix, with the
full prefix in each caption. All nine names verified to resolve against the live
HF account.

**A sentence I wrote and then had to correct, which is the point of checking.**
Adding the column, I wrote that release names are whole-artifact bpw and that
d4/K2048 is "2.75 bpw in the expert region and 3.0 bpw across the file." The
second half is false — the artifact measures **3.109** bpw whole-file. Worked
out the actual convention across all nine: **whole-artifact bytes over parameter
count, and it holds for eight of nine.** The lone exception is the flagship,
named `VQ-3bpw` while measuring 3.11 — a residue of the 3.1bpw -> 3bpw
repository rename, since the rename kept the bytes and changed the label.

The caption now states the convention, states the exception, and points at §5's
"a label is not a measurement" rule — which this is a live instance of, in our
own release names. Expert-region rate (2.75 codes + 0.25 fp16 scales = 3.00) is
given so the two numbers cannot be confused.

Verified after rebuild: every ladder row's cell count matches its header, the
nine release names map to the nine bold rows, 0 external references, and the HF
Space is sha256-identical to the local file.

## 2026-08-26 — 3.5 gains the prefill-slowdown mechanism (Scout's review)

Scout's read of the paper surfaced one gap: 3.5 priced the prefill slowdown
(~0.5x affine at 35B, decode within 10-20%) but never said WHY a method near
parity on decode loses half its throughput on prefill. Added: decode is
bandwidth-bound and a VQ artifact has fewer bytes to read, so the in-kernel
codebook lookups hide behind memory traffic; prefill is compute-bound, the
same per-weight decode work lands on a saturated arithmetic path with no
bandwidth saving to pay for it. Framed explicitly as "an interpretation
consistent with the measured split rather than a profiled attribution" — we
never profiled the kernels, we measured the ratios.

One clause of my own addition was trimmed before shipping: "stability across
prompt lengths" is measured for the u8view lever (2k/8k, E81/E90) but NOT for
the 0.5x-vs-affine ratio, so the sentence now claims only the pair of ratios.

Also this pass: verified the abstract's "1.75-bit" and "2.25-bit" against the
notation convention — flagged them as mixed conventions, and the flag was
WRONG: 2.25 is d4/K512 (the 24x build), codes-only, exactly per the paper's
stated notation. No edit made. The near-miss is recorded because "corrected a
correct number" is this project's most persistent failure mode and it was
nearly committed to the abstract on publish day.

Artifact + Space republished, sha256-verified. OPEN GATE unchanged: 7 points
at the project repository; VQLab must be public before or with the paper.

## 2026-08-26 — Scout round 2: two fixes, one false alarm

**False alarm, verified before acting:** Scout reported the 3.2 figure's
base64 "truncated mid-string." Decoded both embedded PNGs from the shipped
html: valid header AND IEND trailer, 134,975 and 166,737 bytes — the
truncation was in Scout's file read, not the file. No action.

**6 now NAMES the non-rebuildable artifact.** It is VQ-2.4bpw (111.6 GiB
d4/K256): predates the manifest system, and logs_live_397b.log — the record
of its exact fit invocation — was overwritten Aug 19, four days after the
Aug-15 fit (EXPERIMENTS.md:8019). Everything else was verified identical in
the vintage investigation (code by diff, defaults by document, stack by
binary identity); the unrecoverable inputs are the fit flags. The sentence
now says exactly that, plus the resolved reading: a favourable but
unexceptional draw inside the measured floor.

**2.6 now names the 27B ppl corpus:** the wikitext-style prose referee
corpus, first 2048 tokens, one corpus where the 397B uses two. Verified
against the corpus file itself (wikitext-103 formatting) and the E119/E128C
scoring records.

Checklist swept: all 13 model repos public, both figures complete in the
shipped bytes, artifact + Space republished sha256-identical. Note for the
record: Scout's "14 published models" counts the bf16 teacher, which is
kept locally but is not ours and not published — the number is 13.

## 2026-08-26 — NOAH'S READ: 16 fixes, one reaching the title

**"Calibrated" was never established, and the TITLE claimed it.** Noah doubted
the spicyneuron builds are calibrated. Checked their live cards: they describe
sensitivity-based allocation (routers/attention richer), NO calibration corpus
mentioned; mlx-community uniform builds derive scales from weights alone. So
neither comparator class is established as calibrated, and the paper said
"Beats Calibrated Affine" in its title. Title now "Beats Affine Quantization";
comparators are "hand-tuned mixed-bit-depth" / "community" builds everywhere
(9 instances fixed; the one surviving "calibrated" is our own GPTQ/DWQ-style
internal test, which IS calibrated). §3.2's affine table header likewise.

**Also from the read:** abstract's "All artifacts are published" scoped to
"Thirteen artifacts spanning the three ladders"; §2.2's corpus sentence
rewritten to say what data-free buys (scores cannot be flattered by
construction); nat defined once, in §2.6, abstract points to it; 27B corpus
clause reworded; K512-vs-2.6bit gains the vision parenthetical (1.7 GiB
download / 0.9 like-for-like); 3bpw caption now explains the name by its
expert-region rate (exactly 3.00) instead of "residue of a rename"; zero-copy
lever named at point of use; decode parity scoped to this hardware's
bandwidth-to-compute ratio; §4.1's q8-mislabel narrative cut (Noah: results,
not failures — the corrected numbers stand on their own); "most strongly"
added to the tail sentence; "base-vintage" jargon replaced with plain words;
§4.4 deleted, its lever list folded into §3.5 as one sentence; §5 keeps its
rules but drops "two weeks" and now states the borrowed-floor practice as
practiced (disclosed lower bound), fixing a rule-vs-practice contradiction
Noah caught; §7 discloses the code corpus does not ship (private codebase) —
verified against referee/ contents; 57x borrow gains the n=1 direction note
(the one measured cross-K floor pair narrowed 4.6x with larger K); chart:
spicy sizes to 2dp matching the table, 3.5bit label lifted clear of the
flagship's, legend wording updated.

**His two questions, answered from the record:** (1) No K512 floor was ever
measured — the 24x borrows d4/K256, disclosed; measured floors exist at K256,
K2048 (397B), d2/K1024 (35B), d2/K256 (27B). (2) The 3bpw name: whole-file is
3.109 (so "3.1" matched it); the defensible reading of the name is the
expert-region rate, exactly 3.00 — the caption now says that instead of
implying the rename tracked the measurement.

All 20 post-edit checks pass in the RENDERED html; artifact + Space
republished sha256-identical. §5 retained per arbiter judgment (rules are the
paper's spine; targeted fixes made) — flagged to Noah as still his call.

## 2026-08-26 — 5 rewritten to formal register; 3bpw name ruled a convention break

**5 "Keeping the data clean" -> "Measurement discipline."** Noah asked
directly whether 5 reads like a white paper; honest answer was no — right
content, wrong register (aphorism headers, anecdote voice). Rewritten at half
the length, impersonal, same rules preserved: pre-registration, own-geometry
floors with disclosed borrows, same-artifact rows, cross-machine gating,
gates-must-fail-first, serve-before-ship, physical provenance,
metadata-as-intent. The (§5) citation from the ladder caption still resolves
to the metadata-as-intent rule.

**The VQ-3bpw name is a CONVENTION BREAK, ruled so.** The other eight names
are whole-artifact bytes x8/params to one decimal (2.18->2.2, 2.42->2.4,
2.65->2.6, 3.83->3.8, 5.44->5.4, 3.86->3.9, 4.47->4.5, 4.78->4.8). The
flagship measures 3.109 -> convention says "3.1"; it is named "3". Yesterday's
caption rationale (expert-region rate = 3.00) was a constructed explanation
for a deviation, not a second convention. Noah: would never have signed off on
3.0 knowing it deviated. **Pending his go: rename the repo back to VQ-3.1bpw**
— touches the HF repo (auto-redirect), 4 card sibling tables, paper ladder row
+ caption, exo cards (main + branch), the collection, the local dir, and
push_card_fixes.py. Not executed; outward-facing.

Line-number map of every read-pass change generated by diffing against the
pre-read snapshot; delivered to Noah in-session. Both surfaces republished,
Space sha256-identical.

## 2026-08-26 — FLAGSHIP RENAMED VQ-3bpw -> VQ-3.1bpw. Convention now 9/9.

Noah's ruling: accuracy is the brag; the name follows the measurement. The
"3bpw" label did not come from him and he would not have approved a deviation.

**Executed, each step verified:**
- HF repo moved; old id confirmed redirecting to the new one. The collection
  item followed the rename automatically.
- Paper: ladder row now VQ-3.1bpw; the caption's exception sentence DELETED —
  with the rename there is no exception, the convention (whole-artifact bpw to
  one decimal) is 9 for 9, and yesterday's expert-region rationale is moot.
- Cards: G retitled + body refs; C/F/K512 sibling tables and prose; the
  predecessor-weights rows relabelled "(predecessor weights at this name)"
  since old and new weights now share the 3.1bpw name — the pinned revision
  disambiguates. E's banner records both renames. All four pushed via
  push_card_fixes.py (map updated first) and verified byte-identical.
- exo cards renamed+edited on main AND committed on vq-codebook-replicate;
  in_bytes matches the renamed local dir exactly. Local artifact dir renamed.
- Grep sweep: the only remaining "VQ-3bpw" string anywhere is E's banner,
  where it is the name history itself.

**Also from Noah's follow-up:** the K512-vs-2.6bit parenthetical cut to
"(0.9 GiB without our vision tower)" — the long version restated what §3.2
already establishes; and §5's provenance paragraph rewritten — "recorded
physically" dropped, and the model_type example now explains itself
(checkpoints inherit config fields from the bases they were converted from)
instead of asserting a bare oddity.

Artifact + Space republished, sha256-identical. The naming convention is now
uniform across all nine released rungs with zero exceptions to caveat.

## 2026-08-26 — the model_type sentence was confusing because it was WRONG

Noah kept pressing on 5's "every checkpoint declares a model_type naming the
wrong model release." Read the configs: 397B = qwen3_5_moe (CORRECT — it is a
Qwen3.5), 35B (a Qwen3.6) = qwen3_5_moe, 27B (a Qwen3.8) = qwen3_5. So "every
one" was false, and the real fact is more useful: model_type names the LOADER
CODE PATH, not the release — 3.6 and 3.8 share the 3.5 architecture, so the
converter stamps them all qwen3_5* and derived builds inherit it. Passage
rewritten to say exactly that. A reader's confusion pointed at an overclaim
the audits missed; the confusion was the finding.

Republished both surfaces, Space sha256-identical.

## 2026-08-26 — publication mechanics: PDF built, Zenodo scaffolded, M4 tidied

**q6_local on the M4 identified and deleted** (Noah's order, run over ssh).
It was the 27B q6 affine comparator PRE-GRAFT: 20.3553 GiB, and the ledger's
q6 row is 21.2136 with-tower — 21.2136 - 0.8582 = 20.3554, exact. Text-only
twin of a fully recorded artifact, rebuildable by one convert. 20 GB freed.

**paper/below-six-bits.pdf** rendered from paper.html via headless Chrome:
21 pages, 674 KB, both figures embedded, light palette. Known limitation:
print drops clickable hyperlinks (URLs survive as text); the HTML deposited
alongside it keeps them.

**paper/zenodo_draft.py** written: creates a PRIVATE Zenodo draft, reserves
the DOI, uploads PDF + HTML, sets metadata (preprint, CC BY 4.0, related
identifier -> the HF account). Reads ZENODO_TOKEN from env only — the token
never touches this repo or my hands. Idempotent: reuses an existing draft
rather than minting a second DOI. Publishing remains a human click on the
Zenodo page, on publish day, after the DOI is stamped.

**Sequence standing:** VQLab dogfooding (Noah) -> DOI reserve (one command,
Noah's token) -> stamp DOI into paper/PDF/Space/cards (me) -> publish day:
TDF site + Zenodo + Space public together -> arXiv when endorsed.

## 2026-08-26 — DOI RESERVED AND STAMPED: 10.5281/zenodo.22119018

Zenodo draft 22119018, private, CC BY 4.0, preprint. The DOI is stamped into
the paper's front matter; html + PDF rebuilt from the stamped source; the
draft's own files REFRESHED so Zenodo carries the stamped versions —
verified by md5 against Zenodo's checksums, both files MATCH. Artifact and
Space republished sha256-identical. First token was pasted into chat by
accident, revoked, replaced via macOS Keychain (script falls back to it; the
secret never enters shell history).

PUBLISH-DAY CHECKLIST (all staged, nothing public yet):
  1. VQLab dogfooding passes -> repo public   (Noah)
  2. Zenodo: press Publish on draft 22119018  (Noah — DOI goes live)
  3. Space -> public; TDF site live           (me / website manager)
  4. Cards: add paper DOI + site link, push   (me, push_card_fixes.py)
Gate order matters only for step 1; 2-4 are same-day, any order.

## 2026-08-26 — 7 now NAMES VQLab with its URL (webmanager's catch)

The reproducibility section said "the project repository" without ever naming
it or linking it — the repo could have gone public and no reader of the paper
would have found it. Now: VQLab, github.com/noahzelezny/VQLab, Apache-2.0,
linked at DRAFT.md:713. All five surfaces refreshed (html, PDF, artifact,
Space, Zenodo draft), Zenodo checksums re-verified MATCH. The publish gate is
unchanged — that URL 404s until the repo goes public, which is why VQLab
public remains step 1 of the checklist.

## 2026-08-26 — E81-class check on VQLab dogfood: PASS, with a real find

The VQLab session ran the bundled-runtime check rather than arguing it, and
the concern had a live hook: their dogfood HAD patched the local runtime
mid-run (vq_dense.py kernel resolution, fixed after a fresh-venv selftest
failure) — exactly the E81 setup. Results: check-bundle PASS on all four
dogfood artifacts, both runtimes verbatim against the repo; both packed
bundles carry the fix (3 occurrences of _resolve_kernel each); and safe BY
CONSTRUCTION — the fix landed before any build ran, so no pre-fix bundle
ever existed.

Their audit also surfaced a procedural miss their own pass had called done:
the K512 chain skipped `vqlab check`, so bundle-verbatim went unverified
until prompted. The distinction that emerges is a keeper:
**generate-through-bundle and bundle-matches-benched-runtime are different
properties, and only the second catches E81.** A chain must carry both.

Third peer-check this week to find something a self-pass had called done —
same asymmetry every time, in both directions across sessions.

## 2026-08-27 — PUBLISHED. DOI 10.5281/zenodo.22119018 IS LIVE.

Noah pressed Publish. Verified against the PUBLIC record (not the draft):
title, CC BY 4.0, both files, and BOTH related works — VQLab and the HF
account (his form edit stuck; the API tug-of-war note above is history).

Same hour: Space flipped PUBLIC
(huggingface.co/spaces/TheDrainFlorist/below-six-bits). All THIRTEEN live
model cards gained a Paper section — DOI + VQLab + Space — inserted before
Provenance where one exists, appended otherwise, and added AT THE GENERATOR
for the QWEN38 trio so regeneration cannot drop it. Pushed via
push_card_fixes.py: 13/13 verified byte-identical.

The triangle is closed and every edge is live: paper cites code and models;
Zenodo record points at code and models; models and code point back at the
DOI. Remaining: TDF site (webmanager) and arXiv (endorsement pending — the
Zenodo DOI can be added as a related identifier there when it lands).

Publication state of record: DOI 10.5281/zenodo.22119018 · 13 model repos ·
4 collections · VQLab (Apache-2.0) · the Space · this ledger.

## 2026-08-27 — head tags: a third session edited under our name; verified, gap closed

The website session routed three head-tag fixes to a session that CONFIRMED
being "the quantlab paper session" and was not — it was "fastmlx performance
comparison". Noah caught the misattribution; the impostor acknowledged. The
edits themselves, read line by line rather than trusted: correct, and well
made — canonical -> thedrainflorist.com/ai/papers/data-free-vector-quantization/,
full title from DRAFT's h1 (replacing the hardcoded "Below Six Bits"),
viewport/charset, description, OG + twitter cards, all in build_artifact.py
so regeneration keeps them. Regenerated and confirmed byte-stable.

**But the live Space did NOT carry them.** The third session edited local
files only; the website session's "already stamped, nothing left to do" was
true of the working tree and false of the live surface — the exact
generate-vs-shipped gap as E81, one layer up. Fresh build pushed to the
Space, sha256-verified, live canonical confirmed. Artifact republished to
match (its gallery title changes to the full paper title as a side effect).

**Zenodo intentionally NOT touched:** v1.0 is published and immutable; the
head tags are page plumbing, not content. The archived copy differs from the
live surfaces by head metadata only, and that is correct.

Standing rule, twice earned today: a peer session's identity claim is not
identity, and a peer session's "done" is not done — resolve by session id,
verify by bytes on the live surface. Reciprocal standing request from the
website session, adopted: any change to the paper's title, the Space URL, or
any number gets relayed to the website session, since their page renders
DRAFT.md and drifts silently otherwise.

## 2026-08-27 — REFERENCES ADDED; Zenodo v2 staged (10.5281/zenodo.22121193)

Noah's arXiv read surfaced what four audits missed: the paper had NO
references section while name-dropping GPTQ/DWQ and claiming a first. The
audits checked numbers against the record; none asked whether a scholarly
paper needs a bibliography — an instrument gap, on the arbiter, not on Noah.

Added: a related-work passage in 1 situating the claim against the CUDA VQ
line — GPTVQ (Hessian-guided VQ), AQLM (additive multi-codebook, calibrated
+ fine-tuned), QuIP# (incoherence + E8 lattices) — all calibration-dependent,
none publishing MLX artifacts; plus the method's lineage (Lloyd k-means,
Jegou product quantization). Eight references, EVERY arXiv id verified
against the live record before citing (2210.17323, 2306.00978, 2402.15319,
2401.06118, 2402.04396). The novelty sentence is SHARPER with them: data-free
where those calibrate, artifacts where those publish none for this stack.

Zenodo v2 drafted via the API: new version 22121193, its reserved DOI
restamped into the front matter, inherited v1 files replaced with the v2
builds, checksums verified. v1 (22119018) stays live and gains a
newer-version banner once v2 publishes. Space/artifact/cards intentionally
NOT updated yet — they carry the live v1 DOI, and v2's is dead until Noah
presses Publish. Propagation happens after, in one verified pass, website
session included per the standing agreement.
