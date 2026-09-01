# CLAIM LEDGER — what the paper asserts and exactly what carries it

Companion to OUTLINE.md (section map) and DRAFT.md (prose). This file is the
grounding layer: **if a number is not in this ledger, it does not get a
paragraph in §2 or §3.** Every row names its instrument and date per rule
III.2, and every size is stamped pre-/post-graft per law 5.

Status: thesis locked (framing A, 08-21). Five slots PENDING today.
Reconciliations applied: spicy-2.6bit = 3.1843 @ 120.6 (not 3.1830 @ 120.3);
shipped 3.1 = 143.7 post-graft (not 144.0); size model = 6 hits / 1 in-band /
0 misses. Rationale for each in OUTLINE §Ambiguities + the 08-21 report.

---

## THESIS

In the 2–3.5 bits-per-weight regime where large mixture-of-experts models
actually fit on Apple Silicon, quantization quality is bought by **codebook
richness applied flatly across the expert surface**, and it can be bought
**with no calibration data at all**: data-free vector quantization beats
calibrated affine quantization at matched packed bytes on MoE experts. Flat
allocation at the target width is the peak of that ladder — and because it is
the peak, the useful role of mixed allocation is not to beat it but to
**price the sizes between flat rungs**, which yields a size-targeting method:
name a byte budget, get the best artifact at it in a single fit.

Two claims follow. Both are fenced hard; neither is claimed outside its
measured regime.

---

## CLAIM 1 — Data-free VQ beats calibrated affine at matched bytes on MoE
experts, 2–3.5 bpw.

### Carrying rows

| # | comparison | ours | incumbent | verdict | instrument / date |
|---|---|---|---|---|---|
| 1.1 | 397B flagship vs spicyneuron 3.5bit | flat-K2048-refit **143.682 GiB post-graft**, prose **2.3410**, code **2.5963** | 165.6 GiB, 2.3614 / 2.6005 | **wins both corpora, 21.9 GiB smaller** | referee prose+code, E91, 08-21 03:18; refs re-scored same day |
| 1.2 **REFRESHABLE — HOLD** | 397B low rung vs spicyneuron 2.6bit | flat-K128 **100.9 GiB post-graft**, prose **3.1706**, code 2.6988 | 120.6 GiB, 3.1843 / 2.6667 | **prose +0.43%, code −1.20%** — wins prose at 19.7 GiB smaller, LOSES code | referee, E74 refs re-scored 08-20. **OLD FITTER — and now outclassed at its own size: d8-K16384 = 3.0591 / 2.6728 @ 100.970 GiB beats it on both corpora. Note the E92 regression means a K128 refit can no longer be assumed to help (vintage effect hurts at small K). Row likely replaced by the d8 rung, pending Noah's d8 accept decision.** |
| 1.3 | 35B vs mlx-community 4-bit | d4-K8192 (3.25 bpw analytic), **56.4 mnats**, 89.4% top-1 | 78.557 mnats, 85.61% | **wins, at smaller size** | kl_cache_qwen36, E84 + addendum; comparator two-box verified, digit-identical |

### Fences (all stated in the paper, none omitted)

| fence | evidence |
|---|---|
| **At 8-bit the advantage vanishes.** mlx 8-bit is essentially lossless (7.449 mnats / 96.18%); the qwen3.6 8→4-bit cliff is 10.5x. Claim lives strictly at 2–3.5 bpw. | E84 addendum |
| **Row 1.2 is not a clean sweep** — the 100.9 rung wins prose and loses code. The both-corpora win exists only at the top of the ladder (row 1.1). | E-record L2099 |
| **Dense is an open question.** | `[[PENDING: E95]]` — registered before fitting |
| **The e4b "VQ beats 8-bit affine on embeddings" result is retracted** — accidental fp32 path, not the codebook. Not cited as evidence anywhere. | E76 |
| **VQ pays a real prefill cost**: 35B VQ prefill ~0.5x affine even with u8view. Goes in §6, not buried. | E81 |
| **Fitter vintage: per-tensor mechanism measured; artifact-level cause REOPENED (E117).** What stands, replicated and depth-structured: k-means++ seeding trades tail for bulk on sub-Gaussian body tensors at scarce K (E107–E110), invisibly to mean relerr; and reweighting the objective to recover it is FALSIFIED (E112, 4.7x worse). What FELL: the leap from that mechanism to E92's artifact-level regression — E117 ran today's fitter with random init at K256 and scored 2.8158, WORSE than ++ (2.8057), both worse than shipped (2.7655). Today's fitter cannot reproduce the 08-16 result with EITHER init; seeding is excluded as primary cause; two other 08-18/19 k-means commits remain unisolated. Loose thread: code moved the OTHER way (2.6347, best of the three) — the change trades corpora at K256. E101 (better fit, worse model) stands untouched. "Refit rungs are the honest frontier" stays retired | E91, E92, E94 result |
| **Single vendor stack** (MLX/Metal), two MoE families. | §6 |

### What would falsify it
A matched-packed-bytes affine build beating a VQ build on both corpora at
2–3.5 bpw on the same instrument. E95 landing negative does not falsify
claim 1; it fixes the word "MoE experts" in the thesis as load-bearing.

---

## CLAIM 2 — A quantization can be tuned to a particular size target while
retaining quality gracefully.

**Statement (Noah's framing, 08-21).** The finding is not that a fit is fast
or that some rung wins; it is that **the size axis becomes continuous**.
Given a byte budget anywhere in the ladder's range, a build can be produced
at that budget whose quality degrades gracefully and *predictably* rather
than falling to the next rung down. Three measured components make "tune"
and "gracefully" concrete:
  1. **flat rungs are the quality peaks** — the reference points the tuning
     works between;
  2. **the exchange rate is measured and small at rich bases** — harvesting
     shallow bits costs 0.0315 ppl/GiB off K128 falling to 0.0011 off
     K2048, ~2x the byte-efficiency of stepping down the flat ladder;
  3. **the size model prices the target before the fit runs** — 6 hits,
     1 in-band, 0 misses.

NOT the claim: fit wall-clock (hardware-dependent, not a finding); that
harvest ever beats flat at flat's own size (it does not).

### Foundation: flat is the peak

| era | evidence | numbers |
|---|---|---|
| affine | matched-byte shape sweep at **identical 141.42 GiB**, 3 shapes | flat/tail30 **2.3982** vs ramp 2.5042 vs spike 2.7224 — 0.32 ppl spanned by allocation shape alone [E29] |
| affine, mechanism | a 4-bit layer costs exactly two 3-bit promotions ⇒ under a byte budget you never buy the expensive width while any layer sits at the floor. Provably optimal, not preferred. | E29 |
| VQ | ladder is **monotone**: no harvest rung beats the flat rung at or above its size | E79 (after the proxy-score retraction), E78 dose-response |

### The mechanism and its price

| base richness | harvest cost (prose ppl/GiB) | source |
|---|---|---|
| K128 | 0.0315 (1st bit), 0.0238 (2nd) | E78 |
| K256 | 0.0033 | E79 addendum |
| K2048 | **0.0011** (−3.72 GiB for +0.0042) | E91 |
| *reference — flat ladder slope* | *0.0365 (K128→K256), 0.0129 (K256→K2048)* | E79 addendum |

⇒ harvest cost falls ~10–30x as base richness rises; ~2x the byte-efficiency
of stepping down the flat ladder.

### The size model (the pricing tool)

`new_size = base_size − 1.87 GiB × shallow_bits_harvested`
(397B; shallow L0–9 = 1.87 GiB/bit, body L10–56 = 8.81 GiB/bit)

**Scorecard: 6 hits, 1 in-band, 0 misses.** Every point stamped; vision tower
= exactly 912,020,960 bytes (0.849 GiB), measured, byte-identical across two
independent grafts.

| # | predicted | measured | err | note |
|---|---|---|---|---|
| 1 | 100.93 | 100.9 | −0.03 | |
| 2 | 108.3 | 107.9 | −0.40 | |
| 3 | 99.0 | 99.05 | +0.05 | E78 |
| 4 | 140.3 ± 0.5 | 139.93 | −0.37 | E80 |
| 5 | 111.6 | 110.768 pre-graft → +0.02 corrected | +0.02 | E92; best in series |
| 6 | 143.7 | 143.65 post-graft | −0.05 | E91 |
| 7 | 122.6 | 121.456 pre-graft → −0.29 corrected | −0.29 | E93; **in band** |
| — | graft growth must be EXACTLY +912,020,960 bytes | | | `[[PENDING: E92/E93 graft]]` |

### Fences

| fence | evidence |
|---|---|
| **Harvest is NEVER free and never a quality win at a flat rung's own size.** | E78, E79 |
| **The one apparent counter-example was a confound.** E80's harvest rung (2.3452 @ 139.93) beat the shipped 3.1 (2.3519 @ 143.7) — but the shipped rung was old-fitter. E91 held the fitter constant and the flat build won (2.3410). The harvest cost is +0.0042, positive. | E80 → E91 |
| **The counter-design is priced out, not tested.** Shallow 1.87 vs body 8.81 GiB/bit = **4.7:1**; harvesting the whole shallow region cannot fund one body-width step. This is an argument from measured coefficients, not a fitted experiment — state it as such. | E74 addendum (which records the original version of this argument as *answering the wrong question*) |
| **The +0.0042 harvest cost contains fit noise.** Bodies are block-identical EXCEPT L10–11 — four tensors refit at matched geometry with an unseeded k-means draw. At this effect size that is a known contaminant, not negligible. | E91 correction |
| **No interaction was measured.** The fitter/harvest contrasts share an endpoint; they are a decomposition *by construction*. Detecting interaction needs the 4th cell of the 2x2, unfit. | E91 correction |

### Scope note: claim 2 may be the MoE-specific half (Noah, 08-21)

Claim 1 and claim 2 are not fenced at the same width, and the paper should
not imply they are.

- **Claim 1 (flat VQ) has a live generality test today**: E95 fits a TRUE
  dense 27B, flat d4/K256.
- **Claim 2 (harvest) has none.** Per the E95 registration the dense fit is
  **FLAT ONLY, no tail complications** — so E95 is structurally silent on
  harvest. A dense win widens claim 1 and leaves claim 2 measured on MoE
  only. The paper must say that explicitly; a reader will otherwise read
  "the recipe carries."
- **Measured only on MoE.** Cheap-shallow lands decisively on qwen 35B and
  on 397B. (An earlier draft of this note cited gemma 26B as a third,
  weaker data point; WITHDRAWN 08-21 — the gemma instrument is
  non-deterministic and cannot carry a quality claim. See admissibility
  table below.)
- **Mechanism this is consistent with**: the position law (law 2) prices
  shallow-layer redundancy, and a 512-expert tensor has redundancy a dense
  tensor does not. If that is the mechanism, harvest is expected to weaken
  on dense — which is a prediction, NOT a result, and is registered here as
  untested. Testing it needs a dense harvest rung nobody has fit.

### What would falsify it
A mixed-allocation build beating the flat rung at or above its own size on
one instrument; or a size-model miss outside ±0.5 GiB on a stamped,
post-graft point.

---

## SLOT RESULTS (08-21 afternoon — four of five landed; numbers from EXPERIMENTS E92–E99)

| slot | result | vs pre-registered grid |
|---|---|---|
| **E89** d8-K16384 | **3.0591 / 2.6728 @ 100.970 GiB post-graft** — beats rate-twin flat-K128 (3.1706 / 2.6988) on both corpora | BETTER than the 3.10–3.15 point estimate. Lands 0.009 ABOVE the <3.05 scrutiny bar, so the memorization check did not fire — **flag: accept deliberately, not silently. No formal E89 verdict entry exists yet; number appears only in the ladder table.** Cross-d effect persists at product scale. **d8 COMPLETE (E89/E113/E115): quality win at matched bytes (3.0591/2.6728 vs 3.1706/2.6988); packed kernel correct (byte-identical greedy, 73cdaf7); decode tax MEASURED ~19% (clean-mode ratio 0.812, matched runtimes, adjacent-pair corroborated) — E83's device-memory warning confirmed. Paper frames it per Noah: at ~101 GiB, the only size class a 128 GB box can hold, the choice is quality-per-byte vs decode speed at fixed bytes — a measured trade, presented without deciding it (the rung assignment is Noah's product call).** |
| **E92** flat-K256 refit | **2.8057 / 2.6447 @ 111.617 GiB — worse than shipped 2.4 on both at byte-identical size** | **RE-TEST RESOLVED: stands, and is now the entry point of the mechanism chain E101→E110 (not corrupt; fits BETTER, scores worse; k-means++ bulk/tail trade, depth-structured). Shipped 2.4 stays shipped. See vintage fence.** |
| **E93** flat-K512 | **2.5634 / 2.6123 @ 122.305 GiB** — beats interpolation (2.628 predicted at its size) by 0.064 | Confirmed. Also confirms Noah's registered 120.6-class prediction: vs spicy 2.6bit (3.1843 / 2.6667 @ 120.6) at +1.7 GiB, prose **−19.5%**, code better too. Bar was ≤3.025; landed 2.5634. |
| **E94** 35B K8192 refresh | **53.022 mnats / 89.55%** vs standing 56.413 / 89.37% = **−6.0%** | Confirmed. Instrument identity verified field-by-field; both arms outlier-gated (E98). Vintage effect measured at TWO families, once each — not "repeatedly" at either. |
| **E95** dense 27B | **RESOLVED (08-21 19:00): DENSE VQ CARRIES — the recipe is NOT an MoE-expert phenomenon.** e95-27b-dense-vq-r2: 325.575 mnats / 76.46% / ppl 6.4032 @ 9.7 GiB, sitting ~26% ABOVE the affine q2→q3 line at its size (interp predicts ~439 mnats / 65.5%). Flat d4/K256, no tail — the weakest reasonable configuration, so a LOWER bound (and it used ++ seeding at K256, inside the 397B penalty band). Gates: outlier PASS, III.10 smoke PASS. **Still banned: "VQ beats 4-bit" (not size-matched). Ladder next: E119 K512/1024/2048 pre-registered; the "usable 27B quant" bar is K2048 beating q3 (187.8 mnats @ 11 GiB) at ≤11 GiB.** |

## NEW SINCE THE SLOTS (E96–E99, all paper-relevant)

- **E99 — the d2/d4 margin, on clean data: d4 wins by 6–11%**, three
  independent estimates now agreeing (6.4% @3.25 bpw, 11.4% @3.75, E87's
  ~12% @2.00). The corrupt d2 arm had INFLATED d4's margin ~3x — the
  correction moves against our preferred result and the paper says so.
  §4's d2-vs-d4 item now cites E87 + E99, not E87 alone.
- **E97 — the monotonicity screen: at fixed d, KL must fall as K rises;
  any inversion is a broken artifact or a broken law.** Goes in §2/§5 as a
  GATE (one line), NOT as a narrated incident. Noah's directive (08-21):
  §5 must not read as bragging about catching our own mistakes — the
  incidents earn their space only as the source of reusable gates, and the
  section stays at four exhibits.
- **E98 — law 6 in its sharpest form: two K8192 fits with aggregate relerr
  identical to four decimals (n=40 means) differ 6.0% in output KL.**
  Corollary: relerr cannot detect the vintage effect; the gate catches
  broken artifacts, not better ones. Wording care: aggregate convergence,
  NOT pointwise-identical tensors (codebooks differ, max abs 1.95).
- **E96 — duration-prediction rule** (schedule hygiene, not paper content):
  a duration is measured only from a completed run of the same shape.


## NEW SINCE E99 (E100–E113 digest, 08-21 evening)

- **Mechanism chain, RE-SCOPED after E117 (E101→E102→E107→E110→E112→E117)**
  — still a §3/§4 centerpiece, but the honest arc now ENDS on an open
  question, which is arguably stronger: a real, replicated per-tensor
  mechanism (tail/bulk trade, depth-structured) that turned out NOT to be
  the artifact-level story; the engineered fix falsified (E112); the
  isolation experiment falsified (E117); cause unisolated among two
  remaining commits. Keeps the two law-6 specimens (E101, E112) intact.
  The paper must NOT attribute E92's regression to ++ seeding.
- **E103** — flagship serves on the 2-node exo ring, coherent on graded
  probes. Claim: "serves and is coherent." NEVER "sharding is bit-exact"
  (single-box comparison impossible at 143.68 GiB). §7 material.
- **E104** — metadata.total_size declared unpacked sizes on packed artifacts
  (flagship +37%). Published three verified unaffected. Principle for §5
  one-liner: derive sizes from bytes, never from a self-describing field.
- **New gates:** III.10 (one token through the shipping fused path before
  release — would have saved the whole d8 chain); E97 monotonicity screen;
  duration rule (E96). All one-line §2/§5 mentions per the no-bragging
  directive.
- **E115 — speed-instrument law for the whole paper:** decode at ~100 GiB is
  BIMODAL on the measuring box (same artifact: 21.14/12.69/21.27/21.20; swap,
  thermal, path all ruled out by measurement). Every §3/§6 speed number is a
  same-session ratio between arms, never an absolute, never n=1. §6 states
  that the published cards' speed tables predate this rule.
- **E114/E116 — the two failure shapes, one incident each:** compute-time
  (E95's zeroed tensor, visible in the fit log, caught by --relerr-abort) vs
  write-time (gemma d2-K512, clean log, corrupt bytes, only the cross-box
  outlier gate sees it). §5-adjacent material; one paragraph, not narrated.
- **E105/E106** (tail-weighted k-means screen + seeding blow-up) — internal
  process material; the paper keeps only E112's falsification, which
  subsumes them.

## RECONCILED SIZES (whole-artifact post-graft bytes, from the E92/E93 chain)

| artifact | old cite | NOW |
|---|---|---|
| shipped 2.4 | 112.0 | **111.617** |
| shipped 3.1 | 143.7 | **143.682** |
| flagship refit | 143.65 | **143.682** (byte-identical to shipped 3.1, as the design requires) |
| shipped 2.2 | 100.9 | **100.930** |

**Rule for E95: ask Noah for the number. Do not guess.**

---

## NOT CLAIMED (kept out on purpose)

- e4b embedding VQ win [E76 — retracted, artifact private]
- "cheap-shallow beats the rung above it" [E79 — proxy score]
- d4 beats d2 by 3.3x [E82 — corrupt arm; real effect 12.2% KL, E87]
- "exactly additive" decomposition [E91 correction — algebraic identity]
- law 10 as a general dimension law [E87 scope: one pair, at 2.00 bpw]
- any gemma perplexity number [model property, HF-verified invalid]
- d8 anything, until E89 lands

---

## INSTRUMENT ADMISSIBILITY (Noah's directive, 08-21) — "do not build a house on sand"

| family | admissible for | why |
|---|---|---|
| **Qwen3.5-397B-A17B** (MoE) | claim 1, claim 2, all headline rows | referee prose+code ppl, deterministic, reproduces to total_nll |
| **Qwen3.6-35B-A3B** (MoE) | claim 1, geometry laws | kl_cache_qwen36, deterministic, two-box verified |
| **Qwen3.8-27B** (dense) | claim 1 generality only | `[[PENDING: E95]]`, kl_cache_qwen38 |
| **gemma-4 e4b** | **NOTHING.** Excluded entirely (incl. §5 incidents). | not a true MoE — different structure; findings cannot generalize to the claimed families |
| **gemma-4 26B** (MoE) | at most a passing mention; **preferred: omit** | gemma ppl is invalid as a model property (HF-verified) ⇒ scoring is non-deterministic ⇒ no quality claim can rest on it |

**Consequences applied:**
- §6 no longer offers "gemma-4 side evidence" as support. Two MoE families,
  full stop, plus the dense result when it lands.
- **E60 (gemma 26B cheap-shallow, +0.94 pt vs a ~1 pt pre-registered bar) is
  WITHDRAWN** as support for harvest family-dependence. It was cited in the
  08-21 scope-note discussion and is retracted here. The MoE-specificity of
  claim 2 stands as a registered prediction with no gemma under it.
- **RESOLVED (Noah, 08-21): e4b is cut entirely, §5.3 included.** The
  disqualifier is the surface, not the instrument — e4b is not a true MoE
  and its findings cannot generalize to the families the paper claims.
  §5 now carries four incidents, not five; E76's fair-test lesson survives
  only as internal discipline, not as a paper exhibit. Gemma4-26b IS a
  comparable MoE but stays excluded on instrument grounds (non-deterministic
  scoring).

---

## PRE-REGISTERED PREDICTION (Noah, 08-21, before E93 scores)

> "If we made a model that was 120.6 GiB (same size as spicy 2.6bit) I'm
> confident that it would beat the affine scores by a meaningful margin."

**This is nearly in hand: E93 flat-K512 is predicted at ~122.31 GiB
post-graft — 1.7 GiB above spicy 2.6bit's 120.6.** Registered NOW, before
the number exists, per rule III.1.

- Comparator: spicyneuron 2.6bit, 120.6 GiB, prose 3.1843 / code 2.6667.
- Nearest measured neighbours: K256 @112.0 = 2.7655 / 2.6383 (already beats
  it on both corpora at 8.6 GiB smaller); K2048 @143.7 = 2.3519 / 2.5987.
- Registered reading: "meaningful margin" is operationalized as **≥5% prose
  vs 3.1843**, i.e. ≤3.025.
- **RESOLVED same day: E93 = 2.5634 / 2.6123 @ 122.305 GiB — prose 19.5%
  better and code better, at +1.7 GiB. Prediction confirmed, decisively.**

---

## PROVENANCE BREAK (E121, 08-21 23:45) — read before citing any 397B refit

The `struct6-tail3x3` BASE was silently rewritten Aug 19, three days AFTER the
shipped 2.4bpw was built from it. Source bf16 unchanged (Aug 8). Original base
does not survive (five locations searched, never published to HF).

**Every 397B refit since Aug 19 used different input bytes than the shipped
rungs**: E92, E93, E117, E118, E121, and the flagship flatk2048-refit.

| consequence | status |
|---|---|
| "Fitter vintage" as a phrase | **DEAD.** Four in-algorithm explanations proposed; three falsified (E117/E118/E120), fourth voided. Actual cause was provenance, found with `ls -l`. |
| **E101 (better fit, worse model)** | **CONFOUNDED — do not cite as a clean pair.** Relerr is vs the unchanged bf16 source (valid), but end-to-end ppl carries the base's non-expert tensors, which differ between arms. Untestable: the 08-16 base is gone. Claim 3 loses this specimen. |
| **E112** | **SURVIVES.** Both arms built 08-21 from the same base, byte-identical size, identical geometry, pre-registered rule. Now claim 3's sole clean designed specimen. |
| E107–E110 mechanism | Stands as per-tensor physics; definitively NOT the artifact-level explanation. |
| E94 "two families" | Artifact overwritten; **cite e94b** (fresh fit, reproduced 53.022 / 89.55% exactly). Fits are statistically but not bitwise reproducible — MLX RNG unseeded across processes [E125, 6d]. |
| shipped 2.4bpw | **Unreproducible by construction.** §7 must say so. |
| 6c KL/ppl inversion | DEMOTED by its own replication test — n=1, do not carry as a phenomenon. |

**§7 impact:** two proven silent in-place overwrites, neither caught by a gate.
Publish-time manifest + `chmod -R a-w` proposed, not yet landed. Until it does,
§7 states the limitation rather than claiming clean pinning.

---

## DENSE 27B LADDER (E119/E124/E126) — claim 1's dense half, now a curve

| rung | size GiB | KL mnats | top-1 | ppl |
|---|---|---|---|---|
| q2 (affine) | 7.9 | 1426.891 | 46.07% | 16.4349 |
| E95 d4/K256 | 9.7 | 325.575 | 76.46% | 6.4032 |
| **E119 d4/K1024** | **10.609** | **148.470** | **82.53%** | **5.5249** |
| q3 (affine) | 10.963 | 187.765 | 79.48% | 5.8323 |
| E124 d2/K256 | 13.596 | 40.327 | 90.10% | 5.2330 |
| q4 (affine) | 14.094 | 45.842 | 89.82% | 5.2055 |
| **E126 d2/K512** | **14.592** | **33.095** | **91.10%** | **5.1943** |

- **E126 beats q4 on ALL THREE metrics** (−27.8% KL, +1.28 pp, lower ppl) at +3.5% size.
- **E119 K1024 beats q3 on both metrics at 0.35 GiB LESS.**
- Dense size model closed: `total = codes + 0.498 + 5.129 GiB` — three builds,
  two geometries [6e]. **Quote PACKED size and assert it** (d2/K512: 21.565
  unpacked vs 14.590 packed).
- RETRACTED by the lab session: "d beats K at this budget" — d2/K256 and
  d4/K1024 are 2.99 GiB apart. Exact rate twin (d4/K65536) untested.
- **Fence consequence for the paper:** we now beat 4-bit affine on dense at
  ~4-bit-equivalent, and d4-K8192 beat mlx 4-bit on the 35B. The "2–3.5 bpw"
  fence in the thesis is likely TOO NARROW — revisit before the abstract.

---

## IN FLIGHT / OPEN (as of 08-22)

**E127 (running) — claim 3's replacement specimen.** Dense 27B, d2/K256, one
knob: `--iters 10` (A) vs `--iters 30` (B). Same Aug-8 source, same q4 base,
both fit within the hour, both gated, ppl AND KL end-to-end. Branches
registered before the first fit: INVERSION / TRACKS / FLAT / VOID-as-specimen
(if B fails to achieve lower relerr). Different model AND geometry from both
397B specimens. **Caveat raised to the lab session: A and B differ in init
draw as well as iters (6d — MLX RNG unseeded across processes); a third arm C
at iters=10 with a different draw would measure the seed-noise floor at this
geometry. If C is not run, the effect must be reported with the floor stated
as UNMEASURED.**

**§7 tooling landed:** `artifact_manifest.py` (61473e7) — per shard: bytes,
mtime, sha256 of first 1 MiB; manifests stored outside the artifact; III.5
verified BOTH directions (fails on a rewritten shard, passes on an untouched
one, on a synthetic artifact). 11 artifacts stamped incl. the 3 published
397B rungs and the base. §7 can now say: stamped and checkable going forward;
two historical in-place overwrites documented with consequences; shipped
2.4bpw unreproducible by construction.

## DECISIONS OUTSTANDING (Noah's)

| # | decision | recommendation |
|---|---|---|
| 1 | **Fence width.** Thesis says 2–3.5 bpw, but E126 beats q4 at 4-bit-equivalent and d4-K8192 beat mlx 4-bit at 35B. Only 8-bit is genuinely lossless. | Widen to "below 8-bit"; it is a stronger claim the evidence already supports. Blocks the abstract. |
| 2 | **Second dense family.** Only true dense model in the lab is Qwen3.8-27B; gemma excluded. | Accept "one dense model" as a stated limitation, OR acquire a second dense model. Limitation is defensible. |
| 3 | **Rate twin d4/K65536** (~13.594 GiB, exact twin of E124). | Only if you want a d-vs-K claim. Justification: MoE says raise-K-first, dense band's best is d2/K512 — if the optimum differs by architecture that is a real finding. Expensive; below E127. |
| 4 | **`chmod -R a-w` on published artifacts.** Outward-facing; lab session correctly left it. | Yes, after the manifest pass — it converts overwrite forensics into a lookup. |
| 5 | **Title.** Unblocked since E95; dense-inclusive options now live, MoEMash name off hold. | Claim-as-title, without "MoE" in the scope. |
| 6 | **Second epigraph** — still unsourced anywhere in the repo. | Attribute to an unlogged conversation, or cut. |

---

## THE NOISE FLOOR PASS (08-22) — every margin re-read against seed noise

Fits are unseeded across processes (6d), so two artifacts of identical
geometry differ. Floors measured:

- **dense 27B d2/K256, n=3:** KL range **2.085 mnats**, ppl range **0.0447** [E127/6f]
- **397B d4/K256, n=2, INFERRED** (rests on E120's ~2.4e-6 scatter-add ≈
  one-hot): wikitext **0.0134**, code **0.0161** [E129]

**Rule (III.12, approved 08-22): a floor belongs to the geometry it was
measured at — never inherited.** III.12 is III.4's twin (n≥3 for speed)
extended to quality metrics.

### Claim 1 rows re-scored against the 397B floor

| row | prose margin | ×floor | verdict |
|---|---|---|---|
| **K512 (E93) vs spicy 2.6bit** | **0.6209** | **46×** | **SOLID — now the paper's lead row** |
| — same row, code | 0.0544 | 3.4× | holds |
| d8-K16384 vs rate-twin K128 | 0.1115 | 8.3× | SOLID |
| flagship vs spicy 3.5bit | 0.0204 | 1.5× | **THIN** |
| — same row, code | 0.0042 | 0.26× | **INSIDE FLOOR — not a claim** |
| K128 (100.9) vs spicy 2.6bit | 0.0137 | 1.0× | **INSIDE FLOOR — dead as a quality claim** |

**Restructuring forced by this:**
1. **Claim 1 leads with E93/K512**, not the flagship. 46× the floor vs 1.5×.
2. **The flagship row becomes a SIZE claim**: 21.9 GiB smaller at
   indistinguishable-to-slightly-better quality. Size is not subject to seed
   noise; "wins both corpora" is withdrawn (code margin 0.0042).
3. **Row 1.2 (100.9) is removed as a quality comparison.** Prose margin equals
   the floor exactly. It survives only as a size/ladder point.
4. **E126 dense**: KL (6.1× floor) and top-1 (+1.28 pp) hold; **ppl claim
   withdrawn** by the lab session before I cited it.

### Claim 3 after E127

**E127 = TRACKS, not INVERSION.** B (iters 30) had lower relerr AND better ppl
(−0.1256, 2.8× floor). So no second specimen. Two consequences:
- **Claim 3 rests on E112 alone** — but E112's effect is 0.1888 = **14× the
  397B floor**, comfortably outside noise. One designed specimen, solid.
- **E127 BOUNDS law 6 rather than refuting it**: d2/K256 dense is a far finer
  fit (relerr 0.084) than the scarce-centroid regime where E102 measured the
  tail/bulk crossover (relerr 0.31). Law 6 was demonstrated where centroids
  compete; E127 shows it does not bite where they do not. That is a scoping
  result worth stating in the paper.
- E101 stays retired: confounded by the base rewrite AND only 3× floor.

**Standing consequence for every table: third-decimal ppl differences between
single-draw artifacts are not interpretable. KL separations of 5+ mnats and
top-1 separations of ~1 pp are.**
