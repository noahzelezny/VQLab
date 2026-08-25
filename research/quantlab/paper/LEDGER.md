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
| 397B d4/K2048 | 2 | wikitext **0.0056** / code **0.0104** | MEASURED [E142] |
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
| **flat K2048 refit (flagship)** | 143.682 | 2.3410 | 2.5963 | vs spicy 3.5bit (2.3614/2.6005 @ 165.6): **SIZE claim — 21.9 GiB smaller; prose BETTER (0.0204 = 3.6x the K2048 floor, CLAIMED), code TIE (0.0042 = 0.4x)** — "wins both corpora" is withdrawn [E142]. vs shipped 3.1: 0.0109 = **1.9x the K2048 floor** (0.0056), not claimable (bar is 3x). Prior readings of 0.8x/0.4x used the K256 floor — a III.12 violation, corrected 08-24. |

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

### 08-23 midday: E141 — the MoE crossover is BRACKETED; law 14 goes two-family

    E141 35B d2/K4096: 25.145 GiB MEASURED, KL 25.502, top-1 92.52%
    vs q6 (26.234, 13.358): DIRECT comparison, 1.1 GiB smaller, 1.91x WORSE
    = 57x the 35B draw floor -> conclusive at n=1 (loss branch; no replicate owed)

- **Crossover bracketed on the 35B: 5.0–6.0 bpw** (E140 below the line at
  5.0, E141 above it at 6.0). Dense bracket was 4.5–6.0. **Same band, both
  architectures — law 14 is no longer dense-only.** Title's "Below 6 Bits"
  is now measured on both families.
- Size model: third 35B out-of-sample hit (−0.37%; series −0.03/−0.30/−0.37).
- Bonus: d2/K256 scored (17.643 GiB, 36.862, 90.92%) — fifth 35B VQ point,
  ladder monotone 53.0→47.5→36.9→28.0→25.5. Also the measured bar for any
  future d4/K65536 rate twin on this family.
- **E142 (K2048 floor) RUNNING.** Design correction from the M4 (adopted):
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
  paragraph, §6; charts gained E141 + d2/K256.

### 08-24: E142 RESOLVED — K2048 floor measured; flagship claim SPLIT

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
- E141 vs q6 ("1.1 GiB smaller") was ALREADY consistent — both text-only.

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
- 397B d4/K2048: published E91 2.3410 + E142 floor draws 2.3390 / 2.3334
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

