# AGENTS.md — vqlab

Instructions for AI coding agents (Claude Code, Codex, Cursor, Cline, Aider,
any other) working in this repo. Agent-agnostic by design.

**This repo is a TOOLKIT, not a pile of experiments.** Nearly every question
you are about to answer already has an instrument here. Building a one-off
script in a scratch directory when a CLI command exists is the single most
common failure mode in this repo's history, and it costs hours and produces
results that nobody can reproduce.

## Do these three things before writing any code

```bash
# NOTE: the exo env (the one that loads qwen4_exp / Flash) does not have
# vqlab installed — run its CLI with PYTHONPATH=src from the repo root:
#   PYTHONPATH=src /opt/anaconda3/envs/exo/bin/python -m vqlab.cli <cmd>
python -m vqlab.cli --help          # 41 commands. Read the list. Twice.
sed -n 1,60p docs/INDEX.md          # what every doc is FOR + whether it still holds
sed -n 1,40p docs/ONBOARDING.md     # the mechanical pass before fitting ANY new family
```

Then grep for the thing you were about to build:

```bash
grep -rli "<the concept>" src/vqlab/ docs/ research/quantlab/
```

## The instruments that get rebuilt by accident

| question | use this | NOT |
|---|---|---|
| which layers deserve more bits? | `vqlab layer-leverage` — **rank by the JUMP in `traj_rel`, NOT by `local_rel`** (F95: local_rel is isolation damage and is anti-signal; jump-ranked beat it by 0.7-1.0 pt on every corpus) | hand-rolled band ablations |
| how much damage does this artifact carry? | `vqlab score` / `kl_damage.py` | ad-hoc KL scripts |
| ppl on the house corpora | `scripts/score_ppl_resident.py` + the THREE corpora in `src/vqlab/referee/` (prose / code-public / literary) | your own corpus files |
| task benchmarks | `research/quantlab/score_tasks_streaming.py` (layer-streamed; scores models larger than RAM) | a new eval harness |
| is this artifact releasable? | `vqlab check-release` / `check-bundle` / `selftest` | eyeballing |
| fit a mixed-geometry rung | `vqlab fit-moe --vq-layers` (scatter fits) | bespoke build scripts |

If an instrument genuinely does not exist, **add it to `src/vqlab/` with a CLI
entry** rather than leaving a script in scratch. That is why the toolkit is
good: every past agent who needed something added it here.

## Authority order when sources disagree

1. **The shipped artifact's own `config.json` and `README.md`** — the record
   of what actually shipped. A card's methodology section says how the mix
   was chosen; the config says what the mix IS.
2. `docs/FINDINGS-LOG.md` — the measured record (F-numbers, corrections
   applied in place).
3. `research/quantlab/EXPERIMENTS.md` — a LAB NOTEBOOK. It narrates attempts,
   including ones whose verdicts the same arc later overturned. **Never
   characterize a released artifact from an experiment entry** (this error
   was made twice in one hour on 2026-09-13; both times a 30-second config
   read would have prevented it).

## Standing rules that have each been paid for

* **Perplexity is deterministic.** Re-scoring an artifact returns the same
  number; it cannot estimate a noise floor. A fit-to-fit floor requires a
  SECOND INDEPENDENT FIT of the same recipe.
* **Isolation probes are anti-signal for allocation.** Measuring one
  layer's damage against an intact network is the condition where
  downstream laundering hides it (quantlab E12 on GLM/affine; F93-F95 on
  Flash/VQ). Use compounding/trajectory measures.
* **Rank allocation by KL, not ppl.** Ppl aggregates and absorbs offsetting
  errors. Ppl is the RELEASE GATE; KL is the ranking instrument.
* **One harness.** Never compare a number from one scoring path against
  another. qwen4_exp loglikelihoods even shift 0.1-0.7 nats with batch
  composition (F87) — same path AND same batching.
* **Depth/geometry laws are family-local.** GLM, 397B and Flash each measured
  a different shape. Never inherit an allocation across families.
* **Generate one token through the shipping runtime** before calling anything
  releasable (rule III.11 — an unservable artifact once scored perfectly).
* **Artifacts NEVER go on the internal disk.** Scratch:
  `/Volumes/Thunderbay SSD/vqlab-scratch/`; fits:
  `/Volumes/Thunderbay HDD/vqlab-fits/`; teachers:
  `/Volumes/Thunderbay HDD/Teacher Models/` (archived, not re-downloaded).
* **Version vocabulary:** v1 = first codebooks; v2 = mixed codebooks
  (measured per-layer allocation); v3 reserved for gradient-tuned. Artifacts
  never carry campaign letters.

## The law book — `research/quantlab/FINDINGS.md`

**Read it before proposing any quantization idea.** Five sections:
I settled laws, II retracted leads (do NOT re-chase), III instrument rules,
IV MLX/Metal rules, V open questions. Each law survived at least one attempt
to kill it, and each Metal rule cost at least one run. The most load-bearing,
distilled — the file itself is authoritative:

**Quality / allocation laws**
* **Position law (I.2)** — early layers tolerate cheap bits; enrichment pays
  only in the BACK of the network; knee ≈ layer 30 of 60 (affine) / L10 (VQ
  shallow-harvest). It TRANSFERS across quantization families. The file says
  "do not rediscover it a third time"; it was rediscovered a fourth time on
  2026-09-13 (F83/F93) by probing bands by hand. Read the law first.
* **Escape the cheapest width broadly before enriching narrowly (I.3)** —
  under a byte budget, maximize non-cheapest layers; never buy the expensive
  width while any layer sits at the floor.
* **Fit error != output damage (I.6)** — weight-space relerr does not rank
  output quality across geometries. Only KL/ppl on the ASSEMBLED model counts.
  (Re-confirmed the hard way in F78/F94: every proxy inverted.)
* **Higher d wins at MATCHED rate (I.10)** — d4 > d2 by ~12% KL at 2.00 bpw,
  confirmed on a second exact rate twin at 3.00 bpw. Modest, not a landslide,
  and only meaningful at MATCHED rate.
* **Quality tracks total bytes (I.1)** — packaging washes across K at d4;
  whether it washes across d is UNSETTLED.
* **Price a rung before fitting it (I.5)** — the size model predicts, and
  every data point must be stamped pre- or post-vision-graft (the tower is a
  fixed 0.849 GiB; mixing the two is a units mismatch that once looked like
  a geometry effect).
* **Operational sweet spot d4/K256 (I.8)** — and **healthy relerr ranges
  SCALE WITH K** (K2048 ~0.19, K256 ~0.31, K128 ~0.46). Set
  `--relerr-abort` PER GEOMETRY; a threshold tuned at one K wrongly aborts
  healthy fits at another.
* **Decode is a wash across geometries; PREFILL is where geometry shows
  (I.9)** — non-byte-aligned code widths pay bit-extraction.

**Instrument rules (III) — violations produced every false result in that file**
* Pre-register predictions before fitting or scoring; a falsified prediction
  is recorded as falsified, never reframed.
* **A comparison row must name the ARTIFACT and the INSTRUMENT that produced
  it.** A number older than the artifact it faces gets RE-MEASURED, not
  cited. Never compare a real artifact against a proxy score.
* Comparators must pass `check_comparator.py` before their row is believed —
  a comparator that loads short scores worse and FLATTERS us.
* Speed: n>=3 with scatter, prompt length stated, one process per arm, never
  on a contended box. **Quote a RATIO between arms from the same session,
  never an absolute** — at ~100 GiB the decode instrument is BIMODAL
  (21.1 / 12.7 / 21.3 / 21.2 tok/s, same artifact, back to back; cause
  unknown).

**Metal rules (IV)**
* **Fused d4/d2 kernels cache the codebook in threadgroup memory:
  `K * dim * 2 < 32768` is a HARD architectural ceiling** (d4 safe to K2048,
  d2 to K4096; K4096@d4 fails ON the cap). Compute `K*dim*2` FIRST.
  `XPC_ERROR_CONNECTION_INTERRUPTED` is how Metal reports this
  over-allocation — it is NOT a compiler-service fault.
* **Dense and MoE are DIFFERENT RUNTIMES** — `vq_dense.py`'s fused path is
  gated on `codebook.shape[1] == 2`. A smoke on one path says NOTHING about
  the other.
* Load under `with mx.stream(mx.cpu):` with `mx.eval` INSIDE the block — a
  lazy read still pending when a save forces evaluation is paid inside a GPU
  command buffer and gets watchdog-killed "at the write step".

**Retracted (II) — do not re-chase without new evidence:** "cheap-shallow
beats the rung above it" (proxy-score artifact), "VQ beats 8-bit affine on
embeddings" (confounded by an fp32 path), the fused row-gather prefill lever
(MLX already fuses it), byte-aligned packing (0 bytes saved, 37% decode cost).

## Before scoping ANY engineering change: audit the fleet

One query over every shipped `config.json` costs 20 seconds and routinely
shows the problem already solved elsewhere. F96 concluded "this needs a
kernel change" by reasoning about a format in isolation; F97's fleet audit
found 18 of 19 artifacts packing exactly and reduced it to one rung's
geometry choice — a refit, not kernel work. Corollary: if one artifact is
the only one with a problem, suspect that artifact's config, not the
shared machinery.

## Long runs

Overnight/multi-hour work needs `nohup ... & disown` plus per-module
checkpoints — a chain tied to the session dies with it. Geometry refits crash
on GPU timeouts under disk contention; wrap them in a retry supervisor that
resumes from checkpoints.
