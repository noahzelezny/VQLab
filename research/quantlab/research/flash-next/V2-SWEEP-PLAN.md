# Qwen3.8-Flash-Next — v2 by MEASURED single-layer effect

Scope: the **2.1bpw / 45.0 GiB rung** first (the 64 GB cliff), then 3.2 and
4.4. Nothing in this document loads a model; it is the scoping pass that
lets the sweep start the moment the GPUs free.

## Why this sweep exists

Every shipped Flash-Next mix was chosen by `vqlab layer-leverage` — the
hot-2 (L0,L1) at 2.1bpw and the familial hot-6 (L0,L1,L31,L35,L36,L39) at
3.2 / 4.4 / 5.5 (`LEDGER.md:235-307`). The probe looked trustworthy: its
picks were confirmed causal at the 64 GB rung (`LEDGER.md:245-246`) and its
ranking transferred across geometries at Pearson r=0.905
(`LEDGER.md:283-297`).

Then GLM-5.3 falsified the same instrument for its family. The probe's
top-8 **lost to a bottom-8 control**, and three of its eight picks were
actively harmful (`research/glm53-flash/LEDGER.md:265-271, 391-394`). Only
direct measurement worked.

Flash's probe has **never been checked against measured single-layer
effects.** Two facts make it suspicious even here, on Flash's own evidence:

1. **The probe's cold end is already known to be wrong.** Downgrading its
   10 quietest layers made the model *worse than flat* — KL 460.27 against
   the flat rung's 419.88 (`LEDGER.md:267-282`). The ledger's own verdict:
   "Upgrading hot layers (validated causal) and downgrading cold ones
   (refuted) are NOT symmetric operations." Half the probe's ranking is
   already falsified.
2. **The causal confirmation was never a comparison.** `mixL01` beat flat
   (`LEDGER.md:245`), but no control was ever built — no other 2-layer pair
   was scored at the same bytes. "L0+L1 helps" was demonstrated; "L0+L1 is
   the best pair" was assumed. r=0.905 is the probe agreeing with *itself*
   across rungs, which is consistency, not accuracy.

So this sweep's **first-class output is a verdict on the probe**, not just a
better mix. Design below makes the shipped artifact one row in its own
sweep, so it is either validated or beaten — both publishable.

---

## Q1 — The 2.1 rung's geometry, current mix, and promotion budget

### It is NOT flat, and it is hot-2, not hot-6

Verified from the shipped artifact's own `config.json` (`vq_modules`, 144
entries):

| | dim | K | pack_bits | modules |
|---|---|---|---|---|
| base geometry | 8 | 16384 | 14 | 138 |
| promoted | 2 | 256 | — (u8) | 6 |

The 6 promoted modules are exactly `layers.{0,1}.mlp.switch_mlp.{gate,up,down}_proj`.
**The 2.1bpw rung carries a hot-2 mix (L0+L1), not hot-6.** `TABLE.md:8`
says so ("hot-2 mix") and the artifact agrees; the brief's premise that
2.1's note mentions no mix is what the *cards* say, and the cards are the
thing that is thin here — the mix is real and is in the config.

PLE is d8/K256 throughout (the reallocation win, `LEDGER.md:217-234`); this
sweep does not touch PLE. See "Deliberately out of scope".

### The three artifacts, and that the flat base survived

| dir (under `/Volumes/Thunderbay SSD/Exo Models/`) | index `total_size` | role |
|---|---|---|
| `qwen4exp_vq_packed_d8k16384` | **44.531 GiB** | FLAT base — the sweep's reference |
| `TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw` | **45.780 GiB** | shipped hot-2 |
| `qwen4exp_vq_packed_mixL01` | 45.780 GiB | same artifact, pre-release name |

The flat rung was marked "superseded, cleanup is Noah's call"
(`LEDGER.md:231-233`) — **it was never cleaned up.** That is what makes this
sweep cheap: the reference row already exists at the right bytes.

> `du -sh` reports `mixL01` as **0B**. That is the hardlink accounting, not
> a missing artifact — `mixL01` and the shipped dir share inodes with the
> flat base for every untouched shard. Never size these dirs with `du`; the
> index `total_size` is the number.

### The 64 GB bar

Card/ledger figures for this rung, measured from outside the process:

- trunk resident **44.96 GiB** (`LEDGER.md:494`, `CARD-UPDATE-DRAFT.md:136` —
  "one 44.96 GiB load")
- everything on: 44.96 trunk + 0.84 vision + 2.13 MTP head + 0.75 KV@32k =
  **48.68 GiB** (`LEDGER.md:499-500`)
- measured peak on an 8192-token request: **47.34 → 47.60 GiB**
  (`docs/MTP.md:414`, `docs/MTP.md:91`)

So the operative bar is **peak ≈ resident + ~2.6 GiB**, and the shipped rung
sits at ~47.6 GiB peak against a 64 GB box. The ladder's own sizing rule is
"build to the box, not the bpw — headroom ratio ~0.69 of RAM"
(`LEDGER.md:31-33`); 0.69 × 64 = 44.2 GiB of weights, which is where the
flat base (44.53) sits and which the shipped rung (45.78) already exceeds
slightly.

### Budget: three tiers, and the primary one is byte-matched

Promotion costs **0.6243 GiB/layer** (derived below). From the flat base:

| tier | N layers | total GiB | peak est. | purpose |
|---|---|---|---|---|
| **byte-matched** | **2** | **45.78** | ~47.6 | **the probe verdict** — head-to-head vs shipped hot-2 |
| stretch | 4 | 47.03 | ~48.8 | still inside the shipped headroom story |
| ceiling | 6 | 48.277 | ~50.1 | matches `mixL01p4` (48.3 GiB, KL 361.51) — byte-neutral |

**The primary result is the N=2 row.** Best-2-by-measurement against
shipped-L0L1 at *identical bytes* is the cleanest possible test of the
probe, and it is free — the shipped artifact is already scored
(KL 390.09, `TABLE.md:8`).

---

## Q2 — Promotion geometry, bytes/layer, and fit coverage

### The step

One layer, d8/K16384 (14-bit packed) → d2/K256 (u8, unpacked). Measured
from the safetensors headers of the two packed artifacts, not assumed:

| tensor (per projection) | flat base | donor / shipped-promoted |
|---|---|---|
| `codes` | U32 packed 14-bit | U8 |
| `codebook` | F16 [16384, 8] | F16 [256, 2] |
| `vq_scales` | F16, unchanged | F16, unchanged |

`switch_mlp` bytes per layer: **0.6208 GiB flat → 1.2451 GiB promoted**.

> **Step = +0.6243 GiB/layer**, and it is **uniform across all 48 layers**
> (`len(set(per_layer_bytes)) == 1` on both sides).

Cross-check: 2 × 0.6243 = 1.2486, and 44.531 + 1.249 = **45.780** — the
shipped artifact's index `total_size` to the byte. The step is right. The
driver asserts this at preflight and refuses to run if it drifts.

> **A trap this sweep must not fall into, caught by the driver's dry-run.**
> The 397B driver derives its step as `(donor_total - base_total) /
> n_layers`, which is valid there because base and donor differ in expert
> geometry *alone*. **It is invalid here.** `qwen4exp_vq_fit_d2k256` is a
> whole d2/K256 model whose **PLE bank is a different geometry too**, so the
> whole-index delta gives **0.9968 GiB/layer — 60% too high**. Only
> `switch_mlp` is spliced. The step must be derived from the `switch_mlp`
> tensor headers, scoped; anything else silently corrupts every budget and
> total-size claim downstream. This is why the byte-matched N=2 tier is
> 45.78 and not 46.5.

**A ledger arithmetic discrepancy, checked and closed.** `LEDGER.md:255-258`
reports `mixL01p4` (6 promoted layers) at 48.3 GiB, i.e. "+4.6 GiB" over the
flat 43.7 — which would be 0.77 GiB/layer, not 0.6243. The header-derived
step predicts 44.531 + 6 × 0.6243 = **48.277 GiB**, and
`qwen4exp_vq_packed_mixL01p4`'s index `total_size` is **48.277 GiB**. The
step is confirmed a third time; the ledger's "+4.6 from 43.7" was loose
rounding of the base (43.7 is the card figure, 44.531 the index), not a
different step. **No open question here.**

### Donor map across the lineup — the 2.1 sweep does not generalize blindly

Read from each shipped artifact's `vq_modules`. Every mix promotes the same
six layers (L0,1,31,35,36,39) except 2.1, which promotes two:

| rung | base geometry | donor geometry | promoted layers | index GiB |
|---|---|---|---|---|
| 2.1 | d8/K16384 | **d2/K256** | 0,1 | 45.780 |
| 3.2 (`31mix6`) | d4/K2048 | **d2/K256** | 0,1,31,35,36,39 | 69.550 |
| 4.4 (`92mix6`) | d2/K256 | **d2/K1024** | 0,1,31,35,36,39 | 94.136 |

So the brief's "d2/K1024 donors" is correct **for the 4.4 rung** and wrong
for 2.1 and 3.2, which use d2/K256. Each rung's sweep therefore needs its
own `--donor`, and the driver re-derives the byte step per pair rather than
carrying 0.6243 across (see the PLE trap above). Coverage for all three
donors is full 48/48.

### Fit coverage: COMPLETE. No refit, and NO T7.

The brief asks whether full-coverage donor fits exist for every candidate
layer. They do:

| fit dir | tensors | layers with `switch_mlp` VQ | geometry |
|---|---|---|---|
| `qwen4exp_vq_fit_d2k256` | 3671 | **48 / 48 (0–47)** | d2/K256 |
| `qwen4exp_vq_fit_d8k16384` | 3338 | 48 / 48 | d8/K16384 |
| `qwen4exp_vq_fit_full` | 3338 | 48 / 48 | (base struct) |

`qwen4exp_vq_fit_d2k256` carries `codes` + `codebook` + `vq_scales` for
`gate_proj`, `up_proj` and `down_proj` on **all 48 layers** — 432
`switch_mlp` tensors, no gaps. It is the 92.4 GiB rung's fit, so it was
always full-coverage; the shipped hot-2 used 6 of its 432 tensors.

**Correction to the brief:** the promoted layers use **d2/K256**, not
d2/K1024 (`LEDGER.md:243-244`; confirmed in the shipped `vq_modules`, K=256).
`qwen4exp_vq_fit_d2k1024` exists on disk and is the 5.5 rung's fit — a
*second, richer* donor, available if a step-size control is wanted (Q4).

### The splice contract HOLDS — hardlink-legal

Checked directly, base vs donor:

- key sets **identical** (3671 each, 0 base-only, 0 donor-only)
- all 432 `switch_mlp` keys present on both sides
- **shard assignment identical for all 432** (138 shards each)

So a promotion rewrites only the shard(s) carrying that layer and hardlinks
the other ~137. This is the strong form of the contract at
`research/allocation/METHOD.md:193-199` — no per-key index rebuild needed.

Shard groups per layer (base): min 2.01, **median 4.41**, max 8.83 GiB.
43 of 48 layers touch **one** shard; layers 7, 20, 26, 39, 45 touch two.

### Where a refit — and the T7 — WOULD be needed

**Not for this sweep.** The Flash bf16 being deleted costs nothing here,
because every donor tensor the sweep can ask for is already fitted on the
Thunderbay. The T7 would be required only to:

- fit a donor geometry that does **not** exist on disk (anything other than
  d2/K256, d2/K1024, d8/K16384, d8/K4096-era leftovers, or the PLE fits) —
  fitting reads the bf16 weights;
- rebuild the MTP sidecar (Q5) — it is extracted from the original
  checkpoint's `mtp.*` tensors (`src/vqlab/mtp_extract.py:1-10`);
- re-cut the teacher top-64 cache (Q3) — it is a bf16 forward.

None of those is on the critical path. **Do not hook up the T7 for stage 1.**

---

## Q3 — Instrument

### Scorer: `vqlab.stream_score`, KL vs the cached teacher

This family has what the 397B lacks: **a teacher cache already on disk.**

`/Volumes/Thunderbay SSD/Exo Models/flashnext_teacher_topk_prose/meta.json`:

```json
{"model": ".../models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d...",
 "corpus": "src/vqlab/referee/referee_corpus.txt", "top_k": 64, "tokens": 2049}
```

And `qwen4_exp` is a **validated** scorer in the family registry —
`SCORERS["qwen4_exp"]["validated"] = True` (`src/vqlab/stream_score.py:181-183`),
meaning its streamed pass has reproduced a direct forward to all printed
decimals (house rule 5, `stream_score.py:178-182`). Unknown families are a
hard error by design (`stream_score.py:215-218`); this one is not unknown.

**Exact invocation — the one that produced every `TABLE.md` cell:**

```
<ladder python> -m vqlab.stream_score \
  --model <artifact> \
  --corpus src/vqlab/referee/referee_corpus.txt \
  --tokens 2048 \
  --kl-cache "/Volumes/Thunderbay SSD/Exo Models/flashnext_teacher_topk_prose"
```

It emits one JSON line: `ppl`, `mean_kl_millinats`, `top1_agreement`,
`captured_mass` (`stream_score.py:243-269`).

**`--tokens 2048` is load-bearing.** The scorer refuses outright if the
token ids differ from the cache — *"FAIL: token ids differ from the cache —
the KL would compare different positions"* (`stream_score.py:255-258`). The
cache holds 2049 ids. There is no cheap-prefix option here: a shorter
prefix does not give a degraded KL, it gives no KL at all. (This is
*better* than the 397B's situation, which had to reason about prefix
comparability — the guard makes it impossible to get wrong.)

**Ranking column: `mean_kl_millinats`.** `TABLE.md:22, 25` — "KL is the
ranking column", stated twice, because both q6 prose and d2/K1024 literary
read below the teacher as slice artifacts. Prose ppl is recorded but does
not rank.

### Streaming residency: confirmed

`stream_score.py:1-8` — *"score a model that cannot be RESIDENT on any
box… materialize one DecoderLayer at a time… Flat memory: peak is one layer
plus activations, so a 598 GiB bf16 teacher scores on a 96 GB box."*
The 598.5 GiB GLM teacher pass on a 96 GB box (`LEDGER.md:316`) is the
existence proof. A 45 GiB candidate is far inside that.

So **scoring never needs the GPUs to be free of the cluster campaign in the
memory sense** — but it does contend for them, which is why Q4 chunks.

### The interpreter rule

**Every number must name its interpreter.** The 2026-09-03 GLM incident:
the same artifact, same corpus, same token ids produced K512 literary
1.616585 on the ladder venv and 1.628498 on the exo env — both
deterministic, both reproducible, disagreeing with each other
(`research/glm53-flash/LEDGER.md:98-116`). Six other hypotheses were
eliminated; the only remaining variable was the MLX build
(`glm53-flash/LEDGER.md:144-156`). The divergence is **not uniform** —
+0.74% on one corpus, −0.18% on another (`glm53-flash/LEDGER.md:158-164`) —
so it cannot be corrected out with an offset, and an entire ledger section
had to be marked SUPERSEDED (`glm53-flash/LEDGER.md:45-49`).

For `qwen4_exp` the runtime of record is **`~/.venvs/qwen4exp` on both M3
and M4** — mlx-lm 0.32.0 from the unmerged PR ml-explore/mlx-lm#1788
(`docs/MTP.md:377`; a clean install is described at
`docs/DENSE-VQ-DECODE.md:186`). It must **never** be an exo env
(`/opt/anaconda3/envs/exo`, `/opt/homebrew/anaconda3/envs/exo`): grafted
mlx-lm plus a jaccl mlx fork.

> **[TO VERIFY] — there are two candidates and I did not resolve which cut
> `TABLE.md`.** `~/.venvs/qwen4exp` (named in `docs/MTP.md:377`) and
> `/Volumes/Thunderbay SSD/venvs/qwen4exp` (exists on the volume; its
> sibling `glm5vlm` is the GLM ladder instrument). **Before the first sweep
> number counts, re-score the shipped 2.1bpw and confirm it reproduces
> KL 390.09 / top-1 78.8% / prose 5.9033.** The driver does this
> automatically (`--verify-instrument`) and refuses to continue on a
> mismatch. This is cheap insurance against re-running the GLM incident on
> a second family.

The driver records the interpreter path and `captured_mass` on **every**
TSV row, so no number is ever orphaned from its instrument.

### Significance bar

Re-scores of fixed weights are deterministic, so **splice-vs-splice deltas
are exact** — the property that made GLM's sweep deltas exact
(`glm53-flash/LEDGER.md:371`). The bar is therefore not measurement noise.

It *is* the unseeded-fit confound: a promoted layer's measured effect mixes
"richer codebook" with "that layer's particular draw". The known
whole-model reference points on this family are the lever's own curve —
−24%, −16%, −16%, −15%, ~0% up the ladder (`LEDGER.md:326-333`) — and the
quiet-layer scoop's +99 mnats blowup (`LEDGER.md:274`), which is the scale
at which effects here are unambiguous.

**[TO VERIFY]** no fit-to-fit reproducibility floor has been measured for
d2/K256 on this family. Rule III.12 forbids borrowing a floor across
geometries. Until one exists, treat single-layer effects under ~5 mnats as
unranked.

### The control is MANDATORY

Build and score a **bottom-N control** at identical bytes alongside best-N.
On GLM *the control was the result*. Here the control set is doubly
motivated: the probe's cold end is already falsified on Flash's own
evidence (`LEDGER.md:267-282`).

---

## The probe verdict — how this sweep answers it

Four rows, all at **N=2, byte-identical at 45.78 GiB**:

| row | layers | source | status |
|---|---|---|---|
| `SHIPPED` | 0,1 | probe's hot-2 | **already scored: KL 390.09** |
| `BEST2` | top 2 by measured effect | this sweep | to build |
| `CTRL2` | bottom 2 by measured effect | this sweep | to build |
| `BASE` | — | flat | already scored: KL 419.88 |

Read it as:

- **`BEST2` == `SHIPPED`** (same two layers fall out) → the probe is
  *vindicated on Flash*, and the GLM falsification is family-specific. That
  is a real, publishable result and it costs one sweep.
- **`BEST2` beats `SHIPPED`** → the probe left quality on the table at the
  hardware cliff; ship the new mix, amend the card.
- **`CTRL2` ≥ `SHIPPED`** → the probe carries no usable signal on Flash
  either, exactly as on GLM, and the r=0.905 self-consistency was
  measuring nothing. The strongest outcome of the three.

Note the sweep produces the full 48-layer measured ranking as a by-product,
so the probe-vs-measurement **rank correlation across all 48 layers** falls
out for free — a far stronger statement than the top-2 comparison alone.
That correlation, against the probe's published hot/quiet sets
(hot L0,1,31-33,35-39; quiet L3,8-11,13,16,23,45,46 — `LEDGER.md:285-287`),
is the headline number.

---

## Q4 — Sweep size, cost, and the interleave with the 397B

**48 expert layers → 48 single-layer candidates + 1 BASE row = 49 scoring
runs** for a full stage 1. (`config.json` `vq_modules` has 144 entries =
48 layers × 3 projections, no gaps.)

Per candidate: rewrite a median 4.41 GiB shard group (hardlink ~137
shards), then one streamed KL pass over 48 layers reading 44.5 GiB.

> **[TO VERIFY] — there is no committed wall-clock for `stream_score` on a
> Flash artifact anywhere in the ledgers.** I looked. The timings that
> exist are fit timings (K256 PLE ~9 min, K4096 ~80 min, `LEDGER.md:229`;
> hot-6 mini-fit 32 min, `LEDGER.md:326`) and the hot-6 splice's "~30 min"
> end-to-end including gates (`LEDGER.md:303`) — which is the closest
> anchor and covers splice + score + gates for a 6-layer build.

**Planning band: 12–25 min per candidate** (splice ~2–4 min at median shard
size, score dominating). That is **10–20 h for all 48** — so **2–3 nights
of ~6–8 h, and the sweep does need chunking.** At ~18 min a 6-GPU-hour
night clears **~20 candidates**.

**Do not plan on the band — calibrate it.** The driver prints a running
per-candidate rate and a projected time-to-finish after every candidate,
and `--max-candidates N` stops cleanly inside a night. The first candidate
settles the estimate; re-plan from it.

### Interleave with the 397B sweep

Both want the same two boxes. They are not symmetric, and that asymmetry is
the schedule:

| | Flash v2 | 397B v2 |
|---|---|---|
| scorer footprint | flat, streamed | flat ~15 GiB, streamed |
| artifact read/pass | 44.5 GiB | 111.6 GiB |
| shard rewrite | median 4.4 GiB | median 4.2 GiB |
| box that fits it | **M3 (96 GB) or M4** | M4 preferred |

Proposal: **Flash runs on the M3, 397B on the M4**, concurrently, because
both scorers are streaming and neither is resident-bound. The binding
constraint is **not RAM, it is the SMB link** — *"Everything on the
Thunderbay reaches the M4 over SMB, and that link is the single biggest
source of bad measurements in this project… Do not run two model loads at
once and then trust a wall-clock number from either"* (`docs/MTP.md:405-410`).

So: **concurrent is safe for the KL numbers** (deterministic, link-speed
independent) but **poisons every wall-clock**. Since neither sweep's
deliverable is a timing, that is an acceptable trade — with one rule:

> If either sweep needs a throughput number, it runs **alone**, and the
> other is paused. Wall-clock rows produced concurrently are marked
> `contended` in the TSV (the driver sets this from `--contended`).

If the cluster campaign wants both boxes exclusively, Flash yields first:
its per-pass I/O is 2.5x cheaper, so it recovers a lost night faster.

---

## Q5 — MTP sidecar: unchanged, and the brief's premise needs correcting

**v2 ships the SAME `mtp-head-q6.safetensors`, unmodified. No rebuild.**

The brief says "the Flash head grafts from base-model layers". **That is
GLM's mechanism, not Flash's.** `src/vqlab/mtp_extract.py:26-33` is
explicit: Qwen-style checkpoints name the head `mtp.*` / `nextn.*`, and
*"GLM-5.3 stores its MTP layer as plain `...layers.<num_hidden_layers>.*`
(index 45 on Flash…), which this regex cannot see"*. Flash's head is
extracted from the original bf16 checkpoint's own `mtp.*` tensors, which
MLX conversion strips (`mtp_extract.py:5-8`).

Consequences for v2, all favourable:

1. **The head is independent of expert codebook geometry.** This sweep
   changes `switch_mlp` codes/codebooks on some layers and nothing else.
   The head shares the trunk's `embed_tokens` and `lm_head`
   (`LEDGER.md:2230-2232`, `mtp_use_dedicated_embeddings: false`) — neither
   is touched. The sidecar remains valid by construction.
2. **Head precision cannot affect quality at all** — *"the trunk verifies
   every drafted token, so a worse draft costs a rejection (speed), never a
   wrong token"* (`LEDGER.md:485-487`). A v2 trunk can only change
   *acceptance*, and acceptance should be re-measured on the winner, not on
   every candidate.
3. **It does not consume the promotion budget by default.** mlx-lm globs
   `model*.safetensors` and does not consult the index
   (`LEDGER.md:490-493`), so `mtp-head-q6.safetensors` is invisible to the
   stock loader — optional *residency*, though not optional download.

**Naming trap, load-bearing.** THREE different files on the store share the
name `mtp-head-q6.safetensors` and they are **not interchangeable**
(`LEDGER.md:2390-2397`):

| family | size | hidden |
|---|---|---|
| **root (Flash)** | **2.140 GiB** | **2560** |
| 397B 2.2bpw | 5.412 GiB | 4096 |
| GLM 2.7bpw | 6.094 GiB | 4096 (inter 12288) |

Any copy must be geometry-checked against the target config first, as was
done before the 2026-09-02 staging.

**Housekeeping flag for Noah (not actioned — SSD is read-only tonight):**
the sidecar is **staged again** in the shipped 2.1bpw dir, mtime
2026-09-02 10:52, after the 2026-08-30 decision to remove staged sidecars
from all four dirs so no upload could sweep them
(`research/flash-next-recipe/LEDGER.md:218-224`). It is not in the index so
it affects no number above, but it is one `hf upload` away from shipping an
unmeasured feature — the exact thing the 2026-08-30 MTP pull was about
(`LEDGER.md:383-396`). Decide before any v2 upload.

**If the sidecar ever needed rebuilding, the T7 would be required** —
extraction reads the original checkpoint. It does not need rebuilding.

---

## Staged execution order

**Stage 0 — free, run now, loads nothing.**
`./flash_v2_sweep.py --dry-run` — validates paths, splice contract,
geometry, budget, RAM/disk preflight, and prints the plan.

**Stage 1 — instrument verification (1 scoring run, ~15 min).**
`--verify-instrument` re-scores the shipped 2.1bpw and refuses to continue
unless it reproduces **KL 390.09 / top-1 78.8% / prose 5.9033**
(`TABLE.md:8`). This closes the [TO VERIFY] on which venv cut the table,
*before* 48 candidates are spent on the wrong interpreter.

**Stage 2 — the 48-layer sweep (2–3 nights).**
Single-layer promotion from the flat base, ranked by `mean_kl_millinats`.
Resumable; `--max-candidates` fits a night.

**Stage 3 — the verdict rows (3 builds).**
`BEST2`, `CTRL2` at byte-matched 45.78 GiB, plus the probe-vs-measurement
rank correlation over all 48 layers against `SHIPPED`'s known 390.09.

**Stage 4 — budget tiers, only if stage 3 says targeting works.**
`BEST4` (47.03) and `BEST6` (48.277, byte-identical to `mixL01p4`'s bytes so the
comparison against KL 361.51 is byte-neutral).

**Stage 5 — transfer to 3.2 and 4.4.**
The probe's claim to test there is *familial transfer* (`LEDGER.md:283-297`).
If measurement disagrees with the probe at 2.1, the hot-6 on 3.2/4.4 is
unsupported and those rungs need their own sweeps — donor coverage exists
for both (`qwen4exp_vq_fit_d2k256` full, `d2k1024` full).

**Stage 6 — gates before anything ships.** `smoke` runs EARLY, not last
(`glm53-flash/LEDGER.md` on check-release/check-bundle not executing the
bundle). Nothing reaches HF without a smoke, including metadata-only
changes (`LEDGER.md:374-382`).

### Deliberately out of scope

- **PLE reallocation.** The K256→K16 probe found a hard floor: +26 mnats
  for −3.0 GiB (`LEDGER.md:259-266`), and at this margin PLE-vs-expert
  trades move *along* the frontier, not off it. Nothing to win.
- **Downgrading cold layers.** Refuted (`LEDGER.md:267-282`).
- **New fits.** Not needed (Q2); would require the T7.
- **Any GPU work tonight.** This document is the whole of tonight's output.

---

## Driver

`flash_v2_sweep.py`, beside this file. Mirrors the 397B driver's
conventions and **shares its TSV schema** (`candidate, layers, donor, gib,
d_gib, prose_ppl, d_prose, code_ppl, d_code, tokens, wall_s, utc`) plus
Flash-specific trailing columns (`kl_mnats, d_kl, top1, captured_mass,
interpreter, contended`) so one set of tooling reads both ledgers.

It loads no model: splicing is a CPU-only byte transform, scoring is
delegated to `vqlab.stream_score`.

### Commands

```bash
cd research/quantlab/research/flash-next

# 0. validate everything, load nothing  (SAFE TO RUN NOW, GPUs untouched)
./flash_v2_sweep.py --dry-run

# 1. confirm the interpreter reproduces TABLE.md before spending a night
./flash_v2_sweep.py --verify-instrument \
    --python ~/.venvs/qwen4exp/bin/python

# 2. the sweep — first night, 20 candidates then stop cleanly
./flash_v2_sweep.py --layers 0-47 --max-candidates 20 \
    --python ~/.venvs/qwen4exp/bin/python --contended

# 2b. later nights: identical command, resumes from state.json
# 3. the verdict rows (layers from the ranked TSV)
./flash_v2_sweep.py --combo <a>,<b> --tag BEST2 --python ~/.venvs/qwen4exp/bin/python
./flash_v2_sweep.py --combo <y>,<z> --tag CTRL2 --python ~/.venvs/qwen4exp/bin/python

# self-test: reproduces the SHIPPED artifact byte-for-byte from the flat base
./flash_v2_sweep.py --combo 0,1 --tag REPRO_SHIPPED --keep
```

The `REPRO_SHIPPED` self-test is the driver's own falsification check: if
promoting L0+L1 from the flat base does not reproduce the shipped
artifact's 45.780 GiB and its scores, the splice is wrong and every sweep
number is void.
