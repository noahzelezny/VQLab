# Qwen3.5-397B-A17B — v2 daily driver by MEASURED single-layer effect

Scoped 2026-09-03. **Nothing in this plan has been run.** No GPU work and no
model load happened while writing it; every number below is either read from
a committed artifact/config/ledger (cited file:line) or derived arithmetically
from those, and anything that needs a run to settle is marked **[TO VERIFY]**.

Paths are relative to `research/quantlab/` unless absolute. `E` =
`/Volumes/Thunderbay SSD/Exo Models` (read-only for this work).

Driver: [`v2_sweep.py`](v2_sweep.py) in this directory.

---

## The recipe being ported

GLM's result, which this mirrors: promoting the **8 best expert layers by
MEASURED single-layer effect** captured **37% of the full uniform upgrade's
gain for 19% of its bytes — 1.93x the efficiency of simply buying bits**
(`research/glm53-flash/LEDGER.md:424-434`).

Two hard-won constraints come with it and are binding here:

1. **The leverage probe is falsified for MoE allocation.** `vqlab
   layer-leverage`'s top-8 lost to a *bottom-8 control* at identical bytes
   (`LEDGER.md:255-289`), and three of its eight picks were among the
   actively harmful layers (`LEDGER.md:391-394`). Selection must be by
   measurement on the assembled model. The probe is still wired into the CLI
   with no deprecation guard (`src/vqlab/cli.py:52`) — do not reach for it.
2. **Layer effects are base-specific.** A sweep table transfers to no other
   rung; a single-layer control got the **wrong sign** across bases
   (`LEDGER.md:460-478`). So this sweep must run at the 2.4bpw base itself.
   Nothing from GLM's layer table carries over — only the method.

Also inherited: **16 of GLM's 42 promotions were net-NEGATIVE**
(`LEDGER.md:386-390`). Expect some of the 57 here to hurt. That is the
finding, not a bug.

---

## Q1 — Base rung and promotion budget

**Base confirmed: `VQ-2.4bpw`, and its expert layers ARE flat d4/K256.**

The card's "mixed precision by layer sensitivity" language
(`MODEL_CARD_397B_C.md:81-87`) describes the **affine** tensors, not the VQ
experts. Read from the shipped artifact tonight:

| | value | source |
|---|---|---|
| VQ modules | 171 | `E/…VQ-2.4bpw/config.json` → `vq_modules` |
| geometry | **exactly one group**: `k=256, dim=4, group=64` | same |
| expert layers | **57**, layers 0–56, no gaps | 171 / 3 projections |
| projections | `gate_proj`, `up_proj`, `down_proj` | same |
| affine side | 287 tensors @6-bit, 90 @4-bit, 9 @3-bit | `config.json` → `quantization` |
| disk | **111.617 GiB** (`total_size` 119,848,110,921) | `model.safetensors.index.json` |

So the base is genuinely flat and the default assumption in the ask holds.
(Precedent for the same trap: the flagship's `--tail-*` flags "LOOK like the
flagship is not flat" and are inert no-ops — `EXPERIMENTS.md:9144-9151`.)

### Budget

| quantity | GiB | source |
|---|---|---|
| usable memory, "128 GB" machine | **119.2** | `MODEL_CARD_397B_F.md:56-57` |
| 2.4bpw resident | 110.8 | `MODEL_CARD_397B_C.md:41` |
| 2.4bpw **peak @ 30k context** | **117.7** | `MODEL_CARD_397B_C.md:41` |
| **headroom before it stops fitting** | **1.5** | 119.2 − 117.7 |

Promotions are codes, i.e. weights, so they add to resident **1:1**. The
budget is therefore **1.5 GiB, and that is the whole of it** — it is already
the tightest rung of the three on a 128 GB box
(`MODEL_CARD_397B_C.md:178-180`, "~118 GiB peak against ~120 GiB usable
leaves little", `:224`).

**Conservative recommendation: 6 layers (+1.125 GiB → 112.742 GiB disk,
peak ≈118.8), keeping ~0.4 GiB of margin and the 30,031-token context claim
(`MODEL_CARD_397B_C.md:42`) intact.**

**Best-8 costs +1.5000 GiB → 113.117 GiB, peak ≈119.2 — the entire budget,
zero margin.** It is shippable only if a measured 30k-context resident run
re-confirms the peak; otherwise the context claim drops. Do not ship best-8
on the arithmetic alone. The driver warns above `--budget-gib` (default 1.5).

> **Convention conflict, flagged for Noah, not resolved here.** The standing
> V2 rule is *published size = trunk + reserved MTP head bytes*
> (`research/allocation/METHOD.md:218-236`). The 397B head is **3.19 GiB**
> (`docs/MTP.md:382`), and 397B + head is measured to fit a 128 GB box **only
> at 2.2bpw** (trunk 100.12 + head 3.13 = 103.25 GiB resident,
> `docs/MTP.md:389-392`). A 2.4-based v2 therefore **cannot** honour the
> reservation convention. Either v2 ships trunk-only at 2.4 (an explicit
> exception), or the head-compatible daily driver is the 2.2 rung and this
> sweep should be re-pointed there. **This is a product decision and it
> should be made before stage 3, not after.**

---

## Q2 — Promotion geometry, and whether the fits already exist

**"One step richer" = K256 → K512, d unchanged at 4.** The donor is the
shipped **2.6bpw** rung.

Read from the three artifacts' configs tonight — all flat, all d4, `group`
64 throughout:

| rung | K | pack_bits | disk GiB | per-layer promotion cost from 2.4 |
|---|---|---|---|---|
| **2.4bpw (base)** | 256 | *(none — byte-aligned)* | 111.617 | — |
| **2.6bpw (donor A)** | 512 | 9 | 122.305 | **+0.187501 GiB** |
| **3.1bpw (donor B)** | 2048 | 11 | 143.682 | **+0.562535 GiB** |

Per-layer cost is `(donor total_size − base total_size) / 57`, and the
geometry model closes **exactly**: 57 × 0.187501 = 10.688 takes 111.617 →
**122.305**, the 2.6bpw card's own figure (`MODEL_CARD_397B_K512.md:19`);
57 × 0.562535 = 32.06 takes it → **143.682**, the 3.1bpw card's
(`MODEL_CARD_397B_G.md:42`). Both land on the published sizes to three
decimals, which is the check that the donors really are uniform promotions of
the same base.

### Stage-1 fitting is FREE — verified, not assumed

The four properties a splice needs, all checked tonight against the artifacts:

- **Same key set**: all three indexes carry **2545** tensors, sets equal.
- **Same shard assignment**: `weight_map` is *identical* between 2.4 and 2.6
  (not merely same-keyed). So untouched shards **hardlink**; only the 1–2
  shards carrying a promoted layer are rewritten. This is strictly better
  than the affine case, where 2654 of 2998 keys moved shards and forced an
  index rebuild (`research/allocation/METHOD.md:193-199`).
- **Same `vq_modules` coverage**: identical 171 keys; entries differ only in
  `k` and `pack_bits` — the same relationship GLM's donors had
  (`LEDGER.md:243-245`).
- **Same 3 tensors per module**: `codebook`, `codes`, `vq_scales`. A
  one-layer promotion swaps exactly **9 tensors**.

**Same fitter vintage, too:** E91 (K2048), E92 (K256 → the shipped 2.4) and
E93 (K512 → the shipped 2.6) were fit **in one M4 queue, all 171/171 at the
intended flat geometry, peer config-audited** (`EXPERIMENTS.md:5003`), with
sizes predicted and confirmed through graft (`EXPERIMENTS.md:5014`). So the
donor is not a different-vintage confound.

Per-candidate cost on disk: **1–2 shards, median 4.20 GiB rewritten** (max
8.31), against 2221 GiB free on the volume. Disk is a non-issue.

### The honest caveat on step size

GLM's promotion was **K512 → K2048 = +2 bits per code**. Ours is **K256 →
K512 = +1 bit per code — half the step.** Expect **smaller per-layer effects
than GLM's**, and do not import GLM's effect magnitudes as expectations.

The alternatives were considered and rejected:

- **K2048 donor (+3 bits/code)**: only **2 layers** fit the 1.5 GiB budget.
  GLM measured a 3-layer mix at **0.2 mnats = "nothing"; mixing needs scale
  before it registers at all** (`LEDGER.md:297-299`). Two layers is below
  that bar.
- **Fit a K1024 donor** (+2 bits, matching GLM's step exactly): does not
  exist, needs a full 57-layer fit (~1h at this scale per
  `EXPERIMENTS.md:4999`), and at +0.375 GiB/layer buys only **4** layers.

**K512 is the only option with enough LAYERS to register.** The driver
supports `--donor k2048` so a 2-layer step-size control can be run cheaply
alongside, and that control is worth having: it is the only way to tell "the
step is too small" apart from "targeting does not work here".

> **Stage-0 option, ~1h GPU, worth pricing.** A *better* K512 fit was
> measured — E118's random-init draw scored **2.5249 vs E93's 2.5634
> wikitext, better by 0.0385, past the 0.005 bar** (`EXPERIMENTS.md:6350-6353`)
> — but its artifact is **an empty directory** on the volume today
> (`E/rotlab--397B-flatk512-packed`, 0 B). Refitting it
> (`--init random`) would improve *every* promotion in the sweep. Recommended
> only if the GPUs are free early; the sweep is valid without it, just
> against a slightly weaker donor. **[TO VERIFY]** that E118's recipe is
> otherwise identical to E93's.

---

## Q3 — Instrument

**There is NO bf16-teacher KL cache for this family, and building one is out
of scope for v2 stage 1.** Verified on the volume: the only teacher caches
present are `glm53_teacher_topk_prose`, `glm53_teacher_topk_prose_512` and
`flashnext_teacher_topk_prose`. Nothing for the 397B, and the older
per-family caches referenced in the scripts (`kl_cache_qwen36`,
`kl_cache_qwen38`, e.g. `run_e95_score.sh:50`) are for the 35B/27B and are
not on disk either.

Two further blockers make the GLM KL route a multi-night project, not an
option for tonight's plan:

- The KL scorer's family registry has **only** `qwen4_exp` and `glm5_next`
  (`src/vqlab/stream_score.py:182-186`), and it **refuses** an unknown
  `model_type` outright — "a generic loop would silently mis-run this family"
  (`stream_score.py:215-218`). A `qwen3_5_moe` scorer would have to be
  written **and validated against a direct forward to all printed decimals**
  before any number counts (house rule 5, `stream_score.py:178-182`).
- The other KL path, `vqlab kl` (`kl_damage.py`), loads the student
  **resident** (`src/vqlab/kl_damage.py:99-100, 237`) — 111.6 GiB, which is
  why its own docstring notes "the 397B teacher cannot be resident"
  (`kl_damage.py:31-33`).

### Instrument of record: `referee/score_streaming.py`

This is the named instrument for this ladder — *"Instrument of record:
`referee/score_streaming.py` (single-box, raw wikitext, 8192-token prefix)"*
(`EXPERIMENTS.md:45-46`) — and it produced every published 397B number
(`run_ladder_397b.sh:34-35`).

- **It streams, bounded**: it evaluates one block at a time, frees it
  (`score_streaming.py:67-78`), flat **~15 GiB** resident regardless of model
  size (`score_streaming.py:8-10`). Explicitly **exempt** from the
  resident-memory guard (`preflight_ram.py:16`). So no candidate ever needs a
  100 G load, and the sweep can run on either box.
- **It is deterministic**: "scored twice to an identical total negative
  log-likelihood" (`MODEL_CARD_397B_E.md:33-35`), "bit-identical reruns"
  (`EXPERIMENTS.md:2129`). **Splice-vs-splice deltas are therefore exact**,
  the same property that made GLM's sweep deltas exact
  (`LEDGER.md:371`).

**Exact invocation that produced the published numbers** (`run_ladder_397b.sh:34-35`):

```
<stock-mlx-lm python> referee/score_streaming.py \
    --model <artifact> --corpus referee/referee_corpus.txt        # wikitext, 2.7655
<stock-mlx-lm python> referee/score_streaming.py \
    --model <artifact> --corpus referee/referee_corpus_code.txt   # code,     2.6383
```

`--max-tokens` defaults to 8192 (`score_streaming.py:28`), which is the
published prefix. Reference values for the base: **wikitext 2.7655 / code
2.6383** (`MODEL_CARD_397B_C.md:218`).

> **Environment, load-bearing.** The scorer needs a **stock** `mlx-lm`
> (`MODEL_CARD_397B_E.md:33-34`, "an unmodified `mlx-lm` install"). On this
> M3 that is `/opt/anaconda3/bin/python3` (mlx_lm 0.31.3 — the same version
> as the audited fresh-venv run, `EXPERIMENTS.md:6805`). It must **NEVER** be
> an exo env (`/opt/anaconda3/envs/exo`, `/opt/homebrew/anaconda3/envs/exo`):
> those carry a grafted `mlx-lm` and a jaccl `mlx` fork, so a number from
> them is off-instrument. The old `quantlab/venv` in the shipped scripts'
> shebangs **no longer exists** (that repo is merged into vqlab); the driver
> defaults to `sys.executable` and refuses if the interpreter is missing.

### Sweep economics: rank on wikitext only

Score **wikitext only** during the sweep and both corpora on the finalists.
This halves stage-1 cost and is defensible: wikitext is the ranking column,
and the 2.4 rung's wikitext margin is a real result at 16x the noise floor
while its code margin is inside it (`MODEL_CARD_397B_C.md:47-51`). The
driver's `--code` flag adds the code corpus when wanted.

### The significance bar

Deltas are exact (deterministic re-scores of fixed weights), so the bar is
**not** the fit-to-fit floor. But a promoted layer's measured effect
confounds "richer codebook" with "that layer's particular unseeded draw" —
the fits are unseeded by design (`MODEL_CARD_397B_K512.md:123`). The
measured whole-model floors are **d4/K256: 0.0256 prose**
(`EXPERIMENTS.md:9135-9136`) and **d4/K2048: 0.0056 prose / 0.0104 code**
(`EXPERIMENTS.md:9155-9159`); a floor is never borrowed across geometries
(rule III.12, same lines), so **no K512 floor exists yet** — **[TO VERIFY]**.

**Therefore the GLM control is mandatory, not optional.** Build and score a
**bottom-N control** at identical bytes alongside the best-N. On GLM *the
control was the result* (`LEDGER.md:265-271`); a best-N that does not clearly
beat its own control has not demonstrated targeting, whatever its absolute
number.

---

## Q4 — Sweep size and cost

**57 expert layers → 57 single-layer candidates + 1 base row = 58 referee
runs** for a full stage 1. (The bf16 config has `num_hidden_layers` 60,
`num_experts` 512, `num_experts_per_tok` 10; layers 57–59 carry no VQ
modules.)

Per candidate: splice (median 4.20 GiB of shard rewrite, hardlink the rest)
+ one streaming referee pass over 60 blocks.

**[TO VERIFY] — there is no committed wall-clock for `score_streaming.py` on
a 397B artifact anywhere in the ledgers.** I looked; the `seconds` fields in
`results_tasks/*.json` (e.g. 5396.1) are the lm-eval task bench, a different
instrument. The honest anchors are: **load time ~60 s**
(`MODEL_CARD_397B_C.md:40`), a 130 s load measured for a comparable 397B
artifact (`EXPERIMENTS.md:6808`), and the fact that the pass reads ~111.6 GiB
off the SSD once.

**Planning band: 10–30 min per candidate**, i.e. **10–29 h for all 57** —
so **2–3 nights of ~8 h, and the sweep does need chunking.** At the midpoint
(~20 min) a night clears ~24 candidates.

**Do not plan on the band — calibrate it.** The driver prints a running
per-candidate rate and a projected time-to-finish after every candidate, and
`--max-candidates N` stops cleanly inside a night. The first candidate
settles the estimate; re-plan from it.

Cheaper if needed: `--tokens 2048` cuts prefill compute ~4x (weight I/O is
unchanged, so the saving is partial). **Reduced-prefix numbers rank
correctly but are NOT comparable to the published 8192 figures** — GLM hit
exactly this and any shipped artifact still had to be scored at full length
(`LEDGER.md:493-500`). Use it only if the calibrated rate makes 8192
untenable.

---

## Q5 — MTP sidecar: unchanged, and it is not `mtp-head-q6`

**v2 ships the SAME sidecar, unmodified. No rebuild.**

Two independent reasons, both measured:

1. **The head is rung-independent by construction.** It is "grafted from the
   upstream bf16 MTP tensors, **not derived from the trunk's quantization**,
   so ONE head file serves every rung of a model" (`docs/MTP.md:230-232`;
   same statement in `research/allocation/METHOD.md:233-236`). `vqlab
   mtp-extract` pulls it from the **source checkpoint**, not from our
   artifact (`README.md:195`, `src/vqlab/cli.py:56`). v2 changes only trunk
   expert codes, which the head never touched.
2. **Trunk quantization does not move acceptance — tested, negative.**
   Across three rungs, 12 paired prompts each: 0.8151 / 0.7823 / 0.8057,
   **not monotonic, nothing significant, not even ordered by trunk quality**
   (`docs/MTP.md:302-322`). "Trunk improvements and MTP are independent"
   (`docs/MTP.md:326-329`).

**Naming correction:** `mtp-head-q6.safetensors` is the **Flash-Next** file
(2.14 GiB, `research/flash-next/LEDGER.md:618`, shipped in the 2.1bpw
artifact, `:218`). The **397B** sidecar is
`~/heads/mtp-397b-e3q8.safetensors` — **3.19 GiB, experts 3-bit / rest
8-bit, `fc_order=eh`, `norm_shift=1.0`**, present on both M3 and M4
(`docs/MTP.md:382`), from the graft
`~/heads/mtp-graft-397b-bf16.safetensors` (1553 tensors, 12.29 GiB,
`docs/MTP.md:381`).

Carry forward, unchanged by v2: the 397B head has **0.9023 acceptance — our
best — but a 1.000x speedup** (`docs/MTP.md:101`). v2 does not change that
either way. And per Q1, the 2.4 trunk + 3.19 GiB head does not co-fit a
128 GB box regardless.

---

## Staged execution order

Stage 0 and 1 are the only ones that need the GPUs for long.

| stage | what | needs | cost |
|---|---|---|---|
| **0 (optional)** | refit the K512 donor with `--init random` to recover E118's better draw (`EXPERIMENTS.md:6350-6353`) | GPU | ~1 h **[TO VERIFY]** |
| **1a** | `--dry-run`, then score the BASE row | GPU (1 run) | ~10–30 min |
| **1b** | 57 single-layer candidates, chunked by `--max-candidates` | GPU | **10–29 h over 2–3 nights [TO VERIFY]** |
| **2** | rank by measured `d_prose`; build **best-6**, **best-8** and a **bottom-8 control** at identical bytes; score all three on **both** corpora | GPU | ~6 runs, 2–4 h |
| **3** | gates on the winner: `verify_artifact.py`, `check_vision.py`, `check_release.py` (the ladder's own chain, `run_ladder_397b.sh:26-31`); **measured 30k-context resident run**; card | GPU | 2–3 h |

**Stage 2 gates stage 3.** If best-N does not clearly beat the bottom-N
control, the honest outcome is a negative result recorded in the ledger — the
same shape GLM's morning result took before the measured sweep reversed it
(`LEDGER.md:271-289`, then `:424-434`). Ship nothing on a projection.

Do not project a combination from the single-layer sums. GLM's
superadditivity ratio was **1.74x / 1.82x for weak sets but 0.84x (SUB) for
the strong set** — "the ratio cannot be used as a fixed correction factor"
(`LEDGER.md:436-443`), and best-N is greedy selection on a non-monotonic,
interacting surface with **no optimality guarantee** (`LEDGER.md:445-447`).

### Deliberately out of scope

- A bf16-teacher KL cache + a validated `qwen3_5_moe` scorer (Q3).
- The DP optimum over a four-level space. GLM's DP beat greedy by 2.15 mnats
  (`LEDGER.md:481-484`), but **the solver was never committed and no longer
  exists**, and the demotion level here would need a d8 397B artifact that
  the ladder does not have.
- Anything touching the exo API, supervisors, or writes under `E`.

---

## Driver

[`v2_sweep.py`](v2_sweep.py). Validated tonight **without loading any model**:
the splice path was exercised end-to-end against a synthetic pair of
artifacts built to the same key/shard/`vq_modules` contract, asserting
promoted tensors swap, non-VQ tensors pass through untouched, `vq_modules.k`
updates, `total_size` matches bytes on disk, untouched shards stay
hardlinked, and **touched shards do not** (a hardlinked rewrite would corrupt
the shipped base). That test caught a real bug: `mx.save_safetensors` forces
a `.safetensors` extension, so the temp-then-swap wrote to
`…​.tmp.safetensors` and the swap failed on a missing file — fixed.

What it does:

- **Refuses to start** unless key sets match, shard assignments match,
  `vq_modules` coverage matches, both sides are flat, `d` is unchanged, the
  donor's K is actually richer, every requested layer is a VQ module, the
  splice working set fits under `--headroom` of physical RAM, scratch disk is
  available, and the interpreter/scorer/corpus all exist.
- **Checkpoints**: `state.json` records every completed candidate; re-running
  skips them. Ctrl-C safe. It refuses to mix donors in one state file, since
  effects are base- and donor-specific (`LEDGER.md:460-478`).
- **Logs** a ledger-ingestible TSV (`sweep.tsv`): candidate, layers, donor,
  GiB, ΔGiB, prose ppl, Δprose, code ppl, Δcode, tokens, wall seconds, UTC.
- **Splices CPU-only** (`mx.set_default_device(mx.cpu)`) for the reason
  `splice_ple.py:18` does: a GPU command buffer around a slow external-volume
  read trips the Metal watchdog, which is the exact 397B failure recorded at
  `EXPERIMENTS.md:4727`.
- Deletes each candidate after scoring unless `--keep`.

### Commands

```bash
cd research/quantlab/research/qwen397b

# 0. validate everything, load nothing  (SAFE TO RUN NOW)
./v2_sweep.py --dry-run

# 1. the sweep — first night, 24 candidates then stop cleanly
./v2_sweep.py --layers 0-56 --max-candidates 24

# 1b. later nights: identical command, resumes from state.json
./v2_sweep.py --layers 0-56 --max-candidates 24

# 2. finalists, both corpora (layers from the ranked TSV)
./v2_sweep.py --combo <best6>  --tag best6   --code --keep
./v2_sweep.py --combo <best8>  --tag best8   --code --keep --budget-gib 1.6
./v2_sweep.py --combo <worst8> --tag control --code --keep   # the control

# step-size control, 2 layers at the K2048 donor, separate state file
./v2_sweep.py --combo <best2> --tag k2048x2 --donor k2048 \
              --state state_k2048.json --tsv sweep_k2048.tsv --code --keep
```


---

## RULING (Noah, 2026-09-04): size is part of quality — BEST6 does not ship

"Being any bigger invalidates it. There's no qualitative improvement if it
requires more bpw." BEST6 (+0.94 GiB) is therefore EVIDENCE, not a release:
it proved targeting transfers to this family (prose +0.0667, code
2.6383->2.6289, additive composition, bottom-6 control loses at identical
bytes) and located the leverage (late band L37-L55).

THE SHIPPABLE SHAPE — iso-size v2 at exactly the base's 111.805 GiB:
promote best-6, pay with ~5 demotions (K256->K128, -0.1875 GiB/layer).
Stages: (0) fit K128 codebooks from the bf16 for ~10 demotion candidates
(bottom of the promotion sweep is a HINT, not a license — the surface is
non-monotonic); (1) demotion mini-sweep, one layer at a time, same driver
+ TSV; (2) compose promote+demote at iso-bytes; (3) score prose+code vs
base with a shuffled control; ship only if better-or-equal on BOTH.
Same bar applies to any Flash v2 (its BEST2 was also byte-heavier and is
likewise shelved; the shipped byte-matched hot-2 already won anyway).
