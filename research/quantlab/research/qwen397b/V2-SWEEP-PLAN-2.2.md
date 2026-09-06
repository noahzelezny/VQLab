# Qwen3.5-397B-A17B — v2 RE-SCOPED at the 2.2bpw base

Scoped 2026-09-05, superseding the base choice in `V2-SWEEP-PLAN.md`
(everything about method there still stands; this document changes the
base and re-derives every number that depended on it). **Nothing in this
plan has been run.** Numbers are read from committed artifacts/configs
(cited) or derived arithmetically; anything needing a run is **[TO
VERIFY]**.

`E` = `/Volumes/Thunderbay SSD/Exo Models` (read-only for this work).

---

## Why the base moved (Noah's ruling, 2026-09-05)

The v1 daily driver is the **2.2bpw rung (~101 GiB)**. A v2 at the 2.4
base's 111.6 GiB is bigger than v1 and is therefore disqualified — same
principle as the BEST6 ruling: **size is part of quality**. Re-pointing
at 2.2 also resolves the MTP contradiction the old plan flagged
(`V2-SWEEP-PLAN.md` "head-compatible daily driver"): 2.2 trunk 100.12 +
head 3.13 = 103.25 GiB resident honours the head reservation; a 2.4
base cannot.

**Iso-size target: exactly the 2.2 artifact's 100.971 GiB disk**
(`E/TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw/model.safetensors.index.json`
→ `total_size` 108,414,760,257).

## What does NOT carry over

The 57-row single-layer sweep table in `sweep.tsv` was measured **at the
2.4 base** and layer effects are base-specific (a single-layer control
got the wrong sign across bases, `research/glm53-flash/LEDGER.md:460-478`;
constraint #2 of the old plan). BEST6 {40,42,44,45,47,48} is a
*hypothesis* about where the leverage lives (late band L37–L55), not a
license. **The promotion sweep re-runs at 2.2.** Only the method, the
driver skeleton, the scorer, and the control discipline carry over.

---

## The 2.2 base, verified from the artifact (2026-09-05)

| | value | source |
|---|---|---|
| VQ modules | 171 (57 layers × 3 proj), **flat** | `config.json` → `vq_modules` |
| geometry | **d=8, K=16384, group=64, pack_bits=14** (1.75 bit/wt) | same |
| affine side | 287 @6-bit, 90 @4-bit, 9 @3-bit (same census as 2.4) | `config.json` → `quantization` |
| disk | **100.971 GiB** | index `total_size` |
| index keys | 2545, **set-equal with 2.4** (checked 2026-09-05) | both indexes |
| shard assignment vs 2.4 | **identical** — 0 of 2545 keys differ (checked 2026-09-05) | both `weight_map`s |

Per-layer expert payload: 6.4437B params/layer (derived in the old plan
from the 2.4→2.6 delta; geometry-independent). Handy constant:
**+0.25 bit/weight = +0.1875 GiB per layer**.

## The dim problem, and the two ways through

Every existing richer artifact is **d4** (2.4 = d4/K256 at 2.00 bpw,
2.6 = d4/K512 at 2.25, 3.1 = d4/K2048 at 2.75). The 2.2 base is **d8**.
`v2_sweep.py` refuses cross-dim splices by design ("changing d changes
the reconstruction"). Two options:

### Option A — cross-dim splice from shipped donors (RECOMMENDED)

Promote a layer by swapping its three modules to the **2.4 rung's
d4/K256** tensors (+0.25 bpw → **+0.1875 GiB/layer**). Mixed-d artifacts
are runtime-legal and *shipped*: Flash-Next 2.1 is d8/K16384 base with
d2/K256 promoted modules, `vq_modules` is per-module, and the same
`model.py` runtime family executes it (`research/flash-next/
V2-SWEEP-PLAN.md` Q1). The donor tensors come from a shipped, fully
scored artifact — the promotion level's quality is exactly "the 2.4
rung's version of that layer," which is known good.

- Driver change: relax the dim check to per-module `vq_modules`
  bookkeeping; drop shard-hardlink assumption for promoted shards
  (codes/codebook/vq_scales shapes differ across d — per-key rewrite +
  index `total_size` rebuild, the METHOD.md:193-199 fallback that the
  driver already names but doesn't implement). Scope: modest, contained
  to `splice()` + `preflight()`.
- **[TO VERIFY]** one splice smoke: build L00 promoted, load, generate,
  confirm the qwen3_5 bundle honours per-module dim the way qwen4_exp
  does. This is gate #1 before the sweep.
- New fitting needed for promotions: **none**.

### Option B — stay d8, fit richer-K codebooks

Fit d8/K32768 (pack 15, +0.125 bpw → +0.094 GiB/layer) from the bf16
(`E/Qwen--Qwen3.5-397B-A17B-bf16`, on-volume). Driver unchanged. Cost: a
fresh k-means fit per candidate layer at K32768 over 6.44B params —
the most expensive fitting we'd have ever run per layer, for an
untested quality level. Fallback if Option A's runtime smoke fails.

## Demotions (needed under BOTH options)

Nothing below d8/K16384 exists. The pay-side levels must be fitted from
the bf16 — this is the re-based version of the old plan's "stage 0":

| level | pack_bits | Δ bpw | Δ GiB/layer |
|---|---|---|---|
| d8/**K8192** | 13 | −0.125 | **−0.09375** |
| d8/**K4096** | 12 | −0.25 | **−0.1875** |

Fit **both levels** for ~10 demotion candidates (bottom of the new 2.2
promotion sweep is a HINT, not a license — the surface is
non-monotonic, same rule as before). K-means at K8192/K4096 is cheaper
than any fit we've done for this model (fewer centroids than the
shipped K16384). Disk: 2.2 TiB free on the volume — a non-issue.

## Iso-size composition arithmetic (Option A)

Promote N layers at +0.1875, pay with demotions:

| shape | bytes | demotions needed |
|---|---|---|
| promote 6 | +1.125 GiB | **6 × K4096** (−0.1875) — the clean 6-and-6 |
| promote 6 | +1.125 GiB | 12 × K8192 (−0.09375) — shallower damage, wider blast |
| promote 4 | +0.75 GiB | 4 × K4096 or 8 × K8192 |

Which pay-shape wins is an empirical question the demotion mini-sweep
answers (measured per-layer demotion damage, same driver + TSV). Default
recommendation: 6-and-6 with K4096, falling back to 12×K8192 if any
K4096 demotion row shows outsized damage.

## Stages

0. **Driver adaptation + splice smoke** (Option A gate). Relax
   `v2_sweep.py` dim/shard checks to the per-key-rewrite contract; build
   one promoted candidate; load + generate. No sweep until this passes.
   *(Parallel, GPU-free: fit d8/K8192 + d8/K4096 codebooks for the ~10
   demotion candidates from the bf16.)*
1. **Promotion sweep at 2.2**: 57 single-layer candidates, d4/K256
   donor, prose ppl + KL vs the bf16 teacher cache, same TSV schema.
   Resumable; overnight-arc shaped. Per-row cost is splice (median ~4.2
   GiB shard rewrite at 2.4 — similar here **[TO VERIFY]**) + streaming
   score (~15 GiB flat resident, reads ~101 GiB/row).
2. **Demotion mini-sweep**: ~10 single-layer demotion candidates ×
   (K8192, K4096) — ~20 rows, same driver.
3. **Compose** promote-best-N + demote-to-pay at **exactly ≤100.971
   GiB**, plus a shuffled control at identical bytes.
4. **Score & gate**: prose AND code vs the shipped 2.2, control must
   lose. Ship only if better-or-equal on BOTH corpora (the standing
   bar). MTP head compatibility is preserved by construction (size
   unchanged; head is trunk-independent... **[TO VERIFY]** head was
   validated against 2.2 activations — `docs/MTP.md:389-392` says the
   2.2 pairing is the reserved convention, so this should be the
   already-tested pairing).

## Decision points for Noah

1. **Option A (cross-dim, recommended) vs Option B (fit K32768)** — A
   needs one runtime smoke to be proven; B needs no driver work but the
   priciest fitting yet.
2. **Overnight scheduling**: stage 1 owns the scorer box for a night
   (57 rows). Stage 0 fitting can run any time (CPU/GPU-light,
   Thunderbay-bound).
3. Demotion candidate count (default ~10) and pay-shape default
   (6-and-6 K4096) — fine to defer until the sweep table exists.

## Out of scope (unchanged from the old plan)

PLE tables, affine-side reallocation, the leverage probe (falsified —
do not reach for it), any byte-heavier shape (BEST6-class results are
evidence only).

---

## RESULTS (2026-09-05, measured on the M4, mlx-lm 0.31.3)

Gate #1 PASSED: cross-dim L0 candidate loads and generates coherently
(27 tok/s, peak 108 GB) and scores sanely. Full 57-layer promotion sweep
COMPLETE (~65 s/row; table in the M4's `~/v2sweep22/sweep-2.2.tsv`,
to be synced here).

- BASE (shipped 2.2): prose ppl **3.0568** (8192-token instrument).
- Top-6 by measured effect: **L43 +0.0342, L45 +0.0219, L47 +0.0183,
  L29 +0.0175, L44 +0.0171, L48 +0.0152**. Hot zone is L41–L49 + L29.
  The 2.4-base best-6 would have been the WRONG set here (its #1, L40,
  is +0.0060 mid-pack) — base-specificity confirmed a third time.
- All of L1–L11 are NET-NEGATIVE when promoted (GLM's "some promotions
  hurt" reproduces).
- **BEST6 {29,43,44,45,47,48}: 2.9624 = +0.0944** (76% of the naive
  +0.1242 sum — composition holds). **CTRL6 (bottom-6, identical
  +1.1208 GiB): 3.0606 = −0.0038 — control loses.** Both evidence rows;
  both over iso-size and NOT shippable shapes.
- Demotion candidates (least promotion-sensitive, all early):
  L9, L3, L8, L16, L24, L31, L6, L26, L5, L1.

Demotion pipeline (next): `fit-moe --dim 8 --k 4096` (and 8192) for the
10 candidates from the bf16 (M4, `--stage-dir` one 8-GB shard at a time —
the M4 data volume has only ~17 GiB free) → `pack_artifact` to 12/13-bit
block packing → splice as demotion donor → mini-sweep → compose
promote-6 + pay at ≤100.971 GiB → prose+code vs base with control.

## Demotion mini-sweep (2026-09-05 evening, measured, K4096 pack-12)

`demote-sweep.tsv` (M4, demote_fit.py — fit-moe math verbatim, standalone
because the affine skeleton was deleted; vintage gate: L9 refit at K16384
scored 3.0605 vs base 3.0568, in-family, and refit noise is HONEST cost
since v2 ships these tensors). All rows exactly −0.1875 GiB (100.783).

Damage vs base 3.0568: L31 −0.0002 (FREE), L24 −0.0001 (FREE), L3 0.0036,
L1 0.0044, L5 0.0065, L26 0.0067, L9 0.0079... L16 0.0097 (worst).

Cheapest-6 pay = ~0.021. COMPOSED CANDIDATE (in flight): promote
{29,43,44,45,47,48} + demote {31,24,3,1,5,26} = `397b-v2-iso100` at
~100.97 GiB, projected prose ≈ +0.073 net; code corpus is the open
question (never measured per-layer). Ship bar: >= shipped 2.2 on BOTH.

## FINAL VERDICT (2026-09-05 night) — ALL GATES PASS

`397b-v2-iso100` = promote {29,43,44,45,47,48} (d4/K256 from the 2.4
rung) + demote {31,24,3,1,5,26} (d8/K4096, fresh fits, archived to HDD
vqlab-fits/). **100.964 GiB** (7 MiB under the shipped 2.2).

| build (100.964 GiB) | prose | code |
|---|---|---|
| shipped 2.2 (100.971) | 3.0568 | 2.6728 |
| **v2-iso100** | **2.9730 (+0.0838)** | 2.6729 (even) |
| ctrl100 (shuffled: promote {0,4,8,10,35,49} seed-1234, same pay) | 3.0501 | 2.6692 |

Candidate beats the byte-identical shuffled control by 0.077 prose —
selection is causal. Closes 28% of the 2.2->2.4 gap (2.4 rung: 2.7624 @
111.62 GiB) at zero bytes. MTP head reservation intact by construction.
Artifacts: M4 ~/v2sweep22/work/{397b-v2-iso100,397b-v2-ctrl100}; iso100
rsyncing to E/v2sweep397b/. Fits archived (HDD vqlab-fits, 6/6 archive
hits on the control rebuild — convention already paying).

SHIP DECISION: Noah's (nothing published without his go-ahead).
Remaining pre-publish work if GO: runtime smoke via shipping bundle,
check_release gates, card update (rung-honest: v2 replaces the 2.2
slot; 2.4 unchanged), HF upload (Noah runs credentials).
